"""Export DMC Q-network weights from PyTorch checkpoint to raw f32 binary for Rust inference.

Usage:
    python scripts/export_dmc_weights.py models/dmc_final.pt models/dmc_final.bin

Weight file layout (contiguous little-endian f32):
  For each of 3 hidden layers:
    W: in_dim × hidden (row-major), b: hidden, gamma: hidden, beta: hidden
  Final output layer:
    W: hidden × 32 (row-major), b: 32

PyTorch Sequential indices:
  net.0 = Linear(obs_dim, hidden)    net.1 = LayerNorm(hidden)
  net.2 = ReLU()
  net.3 = Linear(hidden, hidden)     net.4 = LayerNorm(hidden)
  net.5 = ReLU()
  net.6 = Linear(hidden, hidden)     net.7 = LayerNorm(hidden)
  net.8 = ReLU()
  net.9 = Linear(hidden, num_actions)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def export_dmc_weights(checkpoint_path: str, output_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict and hidden size
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint
        hidden = checkpoint.get("hidden", 1024) if isinstance(checkpoint, dict) else 1024
    else:
        state = checkpoint
        hidden = 1024

    # Infer hidden from first layer weight shape
    w0 = state["net.0.weight"]
    actual_hidden = w0.shape[0]
    obs_dim = w0.shape[1]
    if actual_hidden != hidden:
        print(f"Warning: checkpoint hidden={hidden} but weight shape says {actual_hidden}, using {actual_hidden}")
        hidden = actual_hidden

    num_actions = state["net.9.weight"].shape[0]
    print(f"Architecture: {obs_dim} → {hidden} (LN+ReLU) ×3 → {num_actions}")

    # Layer indices: (linear_idx, layernorm_idx)
    layers = [(0, 1), (3, 4), (6, 7)]

    all_weights = []
    total_params = 0

    for linear_idx, ln_idx in layers:
        w = state[f"net.{linear_idx}.weight"].cpu().numpy()   # (hidden, in_dim)
        b = state[f"net.{linear_idx}.bias"].cpu().numpy()     # (hidden,)
        gamma = state[f"net.{ln_idx}.weight"].cpu().numpy()   # (hidden,)
        beta = state[f"net.{ln_idx}.bias"].cpu().numpy()      # (hidden,)

        in_dim = w.shape[1]
        print(f"  Layer net.{linear_idx}: Linear({in_dim}, {hidden}) + LayerNorm({hidden})")
        print(f"    W: {w.shape}, b: {b.shape}, gamma: {gamma.shape}, beta: {beta.shape}")

        # PyTorch Linear stores (out_features, in_features) which is already row-major
        all_weights.append(w.flatten())
        all_weights.append(b.flatten())
        all_weights.append(gamma.flatten())
        all_weights.append(beta.flatten())
        total_params += w.size + b.size + gamma.size + beta.size

    # Final output layer (no LayerNorm)
    w_out = state["net.9.weight"].cpu().numpy()   # (num_actions, hidden)
    b_out = state["net.9.bias"].cpu().numpy()     # (num_actions,)
    print(f"  Layer net.9: Linear({hidden}, {num_actions})")
    print(f"    W: {w_out.shape}, b: {b_out.shape}")

    all_weights.append(w_out.flatten())
    all_weights.append(b_out.flatten())
    total_params += w_out.size + b_out.size

    combined = np.concatenate(all_weights).astype(np.float32)
    assert len(combined) == total_params

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.tofile(output_path)
    print(f"\nExported {total_params:,} weights ({total_params * 4:,} bytes) to {output_path}")
    print(f"Hidden size: {hidden} (needed for Rust loader)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DMC weights for Rust inference")
    parser.add_argument("input", help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("output", help="Path for output binary (.bin)")
    args = parser.parse_args()

    export_dmc_weights(args.input, args.output)
