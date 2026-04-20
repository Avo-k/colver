"""Verify PyTorch forward matches Rust forward on 500 sample (obs, q) pairs."""
import numpy as np
import torch

from bid_net_torch import load_bid_net


def load_dump(path: str):
    with open(path, "rb") as f:
        data = f.read()
    obs_dim = int(np.frombuffer(data[0:4], "<u4")[0])
    num_actions = int(np.frombuffer(data[4:8], "<u4")[0])
    n = int(np.frombuffer(data[8:12], "<u4")[0])
    sample_size = (obs_dim + num_actions) * 4
    body = np.frombuffer(data[12:12 + n * sample_size], "<f4").reshape(n, obs_dim + num_actions)
    obs = body[:, :obs_dim]
    q = body[:, obs_dim:]
    return obs, q, obs_dim, num_actions, n


def main():
    obs, q_rust, obs_dim, num_actions, n = load_dump("/tmp/dump_obs_q.bin")
    print(f"Loaded {n} samples: obs_dim={obs_dim}, num_actions={num_actions}")

    net = load_bid_net("models/bid_v5_isdd/bid_nn_final.bin")
    assert net.obs_dim == obs_dim

    net = net.cuda()
    obs_t = torch.from_numpy(obs).cuda()
    with torch.no_grad():
        q_torch = net(obs_t).cpu().numpy()

    abs_diff = np.abs(q_torch - q_rust)
    print(f"Max abs diff: {abs_diff.max():.2e}")
    print(f"Mean abs diff: {abs_diff.mean():.2e}")
    print(f"p99 abs diff: {np.percentile(abs_diff, 99):.2e}")

    # Action agreement: argmax match
    agree = (q_torch.argmax(axis=1) == q_rust.argmax(axis=1)).mean()
    print(f"Argmax agreement: {agree:.4f}")

    # Show one mismatch if any
    max_idx = abs_diff.max(axis=1).argmax()
    if abs_diff[max_idx].max() > 1e-3:
        print(f"\nWorst sample idx={max_idx}:")
        print(f"  rust q[0:5]  = {q_rust[max_idx, :5]}")
        print(f"  torch q[0:5] = {q_torch[max_idx, :5]}")
        print(f"  diff[0:5]    = {q_torch[max_idx, :5] - q_rust[max_idx, :5]}")
    else:
        print("[ok] PyTorch matches Rust within tolerance")


if __name__ == "__main__":
    main()
