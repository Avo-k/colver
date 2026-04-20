"""Extract hidden-layer activations for every sample in /tmp/probe_data.bin.

Saves to /tmp/probe_activations.npz containing:
  scenario_id  (N,) uint8
  position     (N,) uint8
  nn_action    (N,) uint8
  nn_bids      (N,) bool  (action in 1..40)
  obs          (N, obs_dim) float32
  h0           (N, 512) float16  — layer 0 post-ReLU
  h1           (N, 512) float16  — layer 1 post-ReLU
  h2           (N, 512) float16  — layer 2 post-ReLU
  features     (N, 17) float32   — hand features
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from bid_net_torch import load_bid_net


PROBE_BIN = "/tmp/probe_data.bin"
OUT_PATH = "/tmp/probe_activations.npz"


def load_probe_data(path: str):
    with open(path, "rb") as f:
        data = f.read()
    obs_dim = int(np.frombuffer(data[0:4], "<u4")[0])
    n = int(np.frombuffer(data[4:8], "<u4")[0])
    n_feat = int(np.frombuffer(data[8:12], "<u4")[0])
    # Per sample: 4B (ids) + obs_dim*4 + n_feat*4
    sample_size = 4 + obs_dim * 4 + n_feat * 4
    print(f"Loading {n:,} samples, obs_dim={obs_dim}, n_features={n_feat}")
    body = np.frombuffer(data[12:12 + n * sample_size], dtype=np.uint8).reshape(n, sample_size)
    scenario_id = body[:, 0].copy()
    position = body[:, 1].copy()
    nn_action = body[:, 2].copy()
    obs = np.frombuffer(body[:, 4:4 + obs_dim * 4].tobytes(), "<f4").reshape(n, obs_dim)
    feats = np.frombuffer(body[:, 4 + obs_dim * 4:].tobytes(), "<f4").reshape(n, n_feat)
    return scenario_id, position, nn_action, obs, feats


def main():
    t0 = time.time()
    scenario_id, position, nn_action, obs, feats = load_probe_data(PROBE_BIN)
    print(f"loaded in {time.time()-t0:.1f}s")

    net = load_bid_net("models/bid_v5_isdd/bid_nn_final.bin").cuda().eval()

    N = obs.shape[0]
    batch = 16384
    h0 = np.empty((N, net.hidden), dtype=np.float16)
    h1 = np.empty((N, net.hidden), dtype=np.float16)
    h2 = np.empty((N, net.hidden), dtype=np.float16)

    t0 = time.time()
    with torch.no_grad():
        for s in range(0, N, batch):
            e = min(s + batch, N)
            x = torch.from_numpy(obs[s:e].copy()).cuda()
            q, acts = net(x, return_hidden=True)
            h0[s:e] = acts[0].half().cpu().numpy()
            h1[s:e] = acts[1].half().cpu().numpy()
            h2[s:e] = acts[2].half().cpu().numpy()
    dt = time.time() - t0
    print(f"extracted {N:,} activations in {dt:.1f}s ({N/dt:,.0f}/s)")

    nn_bids = (nn_action >= 1) & (nn_action <= 40)
    np.savez_compressed(
        OUT_PATH,
        scenario_id=scenario_id.astype(np.uint8),
        position=position.astype(np.uint8),
        nn_action=nn_action.astype(np.uint8),
        nn_bids=nn_bids,
        obs=obs.astype(np.float32),
        h0=h0,
        h1=h1,
        h2=h2,
        features=feats.astype(np.float32),
    )
    sz = Path(OUT_PATH).stat().st_size / (1024**2)
    print(f"Saved: {OUT_PATH} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
