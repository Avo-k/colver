#!/usr/bin/env python3
"""Live plot of belief network training progress.

Reads the training log output and plots loss, accuracy, and LR curves.
Auto-refreshes every 10 seconds.

Usage:
    # Follow a running training process:
    python scripts/plot_belief_training.py /tmp/claude-1000/-home-avok-code-colver/tasks/b0fd3d5.output

    # Or a saved log file:
    python scripts/plot_belief_training.py training.log

    # One-shot (no live refresh):
    python scripts/plot_belief_training.py training.log --once
"""

import re
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG_PATTERN = re.compile(
    r"Epoch\s+(\d+)/(\d+):\s+"
    r"train_loss=([\d.]+)\s+"
    r"val_loss=([\d.]+)\s+"
    r"val_acc=([\d.]+)%"
    r"(?:\s+lr=([\d.eE+-]+))?"
)


def parse_log(path):
    epochs, train_loss, val_loss, val_acc, lrs = [], [], [], [], []
    try:
        with open(path) as f:
            for line in f:
                m = LOG_PATTERN.search(line)
                if m:
                    epochs.append(int(m.group(1)))
                    train_loss.append(float(m.group(2 + 1)))
                    val_loss.append(float(m.group(3 + 1)))
                    val_acc.append(float(m.group(4 + 1)))
                    lr = float(m.group(6)) if m.group(6) else None
                    lrs.append(lr)
    except FileNotFoundError:
        pass
    return epochs, train_loss, val_loss, val_acc, lrs


def plot(path, once=False):
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Belief Network Training", fontsize=14, fontweight="bold")

    while True:
        epochs, train_loss, val_loss, val_acc, lrs = parse_log(path)
        if not epochs:
            print("No data yet, waiting...")
            time.sleep(5)
            continue

        for ax in axes.flat:
            ax.clear()

        # --- Loss ---
        ax = axes[0, 0]
        ax.plot(epochs, train_loss, "b-", alpha=0.7, linewidth=1.5, label="Train")
        ax.plot(epochs, val_loss, "r-", alpha=0.9, linewidth=1.5, label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Baseline: ln(3) ≈ 1.0986 for 3-class, ln(4) ≈ 1.3863 for 4-class
        ax.axhline(y=1.3863, color="gray", linestyle="--", alpha=0.5, label="random (4-class)")
        ax.axhline(y=1.0986, color="gray", linestyle=":", alpha=0.5, label="random (3-class)")

        # --- Accuracy ---
        ax = axes[0, 1]
        ax.plot(epochs, val_acc, "g-", linewidth=1.5, label="Val Accuracy")
        ax.axhline(y=33.33, color="gray", linestyle="--", alpha=0.5, label="Random (33.3%)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Val Accuracy (latest: {val_acc[-1]:.2f}%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))

        # --- Train/Val gap ---
        ax = axes[1, 0]
        gap = [v - t for t, v in zip(train_loss, val_loss)]
        ax.plot(epochs, gap, "m-", linewidth=1.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Loss - Train Loss")
        ax.set_title(f"Generalization Gap (latest: {gap[-1]:.4f})")
        ax.grid(True, alpha=0.3)

        # --- Learning Rate ---
        ax = axes[1, 1]
        has_lr = any(lr is not None for lr in lrs)
        if has_lr:
            lr_vals = [lr for lr in lrs if lr is not None]
            lr_epochs = [e for e, lr in zip(epochs, lrs) if lr is not None]
            ax.plot(lr_epochs, lr_vals, "orange", linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate")
            ax.set_title("LR Schedule")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No LR data\n(constant LR)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="gray")
            ax.set_title("LR Schedule")

        fig.tight_layout()
        plt.draw()
        plt.pause(0.1)

        if once:
            plt.ioff()
            plt.show()
            return

        time.sleep(10)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log_file> [--once]")
        sys.exit(1)

    path = sys.argv[1]
    once = "--once" in sys.argv
    plot(path, once=once)
