#!/usr/bin/env python3
"""Live training monitor — reads training.log and plots metrics.

Usage: uv run python scripts/plot_training.py [--interval 10]
"""
import re
import sys
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

LOG_FILE = "training.log"
INTERVAL = int(sys.argv[sys.argv.index("--interval") + 1]) if "--interval" in sys.argv else 10

def parse_log(path):
    steps, losses, speeds, eps_list = [], [], [], []
    eval_steps, deal_wrs, rand_wrs = [], [], []
    last_step = 0

    with open(path) as f:
        for line in f:
            # Metric lines: "   10000 | 0.250 | 0.400 | 1403378 |   0.0725 |    68749 |     425"
            m = re.match(r'\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*([\d.]+)', line)
            if m:
                step = int(m.group(1))
                eps = float(m.group(2))
                loss = float(m.group(3))
                speed = float(m.group(4))
                steps.append(step)
                losses.append(loss)
                speeds.append(speed)
                eps_list.append(eps)
                last_step = step

            # Eval lines: "  [EVAL] deals 67% | rand 88% (20s)"
            m = re.match(r'\s*\[EVAL\]\s*deals\s+(\d+)%(?:\s*\|\s*rand\s+(\d+)%)?', line)
            if m:
                eval_steps.append(last_step)
                deal_wrs.append(int(m.group(1)))
                rand_wrs.append(int(m.group(2)) if m.group(2) else None)

    return steps, losses, speeds, eps_list, eval_steps, deal_wrs, rand_wrs

plt.ion()
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("DMC Training Monitor", fontsize=14, fontweight='bold')
fig.canvas.manager.window.wm_geometry("+100+100")

while True:
    try:
        steps, losses, speeds, eps_list, eval_steps, deal_wrs, rand_wrs = parse_log(LOG_FILE)
    except (FileNotFoundError, ValueError):
        time.sleep(INTERVAL)
        continue

    if not steps:
        time.sleep(INTERVAL)
        continue

    for ax in axes.flat:
        ax.clear()

    # Top-left: Loss
    ax = axes[0, 0]
    ax.plot(steps, losses, color='#e74c3c', linewidth=0.8)
    ax.set_title("Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Top-right: Speed
    ax = axes[0, 1]
    ax.plot(steps, speeds, color='#3498db', linewidth=0.8)
    ax.set_title("Throughput (steps/s)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Steps/s")
    ax.grid(True, alpha=0.3)

    # Bottom-left: Eval win rates
    ax = axes[1, 0]
    if eval_steps:
        ax.plot(eval_steps, deal_wrs, 'o-', color='#2ecc71', label='Deal WR%', markersize=6)
        valid_rand = [(s, r) for s, r in zip(eval_steps, rand_wrs) if r is not None]
        if valid_rand:
            ax.plot([s for s, _ in valid_rand], [r for _, r in valid_rand],
                    's-', color='#e67e22', label='Match vs Rand%', markersize=6)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
        ax.legend(fontsize=8)
    ax.set_title("Eval Win Rates")
    ax.set_xlabel("Step")
    ax.set_ylabel("Win %")
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Epsilon
    ax = axes[1, 1]
    ax.plot(steps, eps_list, color='#9b59b6', linewidth=0.8)
    ax.set_title("Epsilon (exploration)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Epsilon")
    ax.grid(True, alpha=0.3)

    # Progress info
    current = steps[-1]
    total = 20_000_000
    pct = current / total * 100
    avg_speed = sum(speeds[-10:]) / len(speeds[-10:])
    eta_h = (total - current) / avg_speed / 3600
    fig.suptitle(f"DMC Training — {current/1e6:.1f}M / 20M ({pct:.1f}%) | {avg_speed:.0f} steps/s | ETA {eta_h:.1f}h",
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.pause(INTERVAL)

    if current >= total:
        print("Training complete!")
        plt.ioff()
        plt.show()
        break
