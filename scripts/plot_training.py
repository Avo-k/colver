#!/usr/bin/env python3
"""Live training monitor — reads training.log and plots metrics.

Handles multiple concatenated runs with overlapping steps by deduplicating
(keeps last occurrence for each step).

Usage: uv run python scripts/plot_training.py [--interval 10] [--total 47000000]
"""
import re
import sys
import time
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
matplotlib.use('TkAgg')

LOG_FILES = ["training.log"]
if "--extra" in sys.argv:
    LOG_FILES.append(sys.argv[sys.argv.index("--extra") + 1])
INTERVAL = int(sys.argv[sys.argv.index("--interval") + 1]) if "--interval" in sys.argv else 10

def parse_log(path):
    # Raw data — may have duplicates from overlapping runs
    raw_metrics = {}  # step -> (eps, loss, speed)
    raw_evals = {}    # step -> (deal_wr, rand_wr, ckpt_wr)
    last_step = 0
    detected_total = None

    for p in (path if isinstance(path, list) else [path]):
      with open(p) as f:
        for line in f:
            # Detect new run boundary: the dashed separator before metric lines
            # When a new run starts, flush stale data from killed runs
            if re.match(r'^-{40,}', line):
                # Peek at what step the new run will start at — flush on first metric
                continue

            # Detect total from step-offset header: "steps will display as 32000001..47000000"
            m = re.search(r'steps will display as (\d+)\.\.(\d+)', line)
            if m:
                run_start = int(m.group(1))
                detected_total = int(m.group(2))
                # New run starting at run_start: discard stale data from previous killed runs
                raw_metrics = {s: v for s, v in raw_metrics.items() if s < run_start}
                raw_evals = {s: v for s, v in raw_evals.items() if s < run_start}

            # Detect resumed run without step-offset (overnight run style)
            m = re.search(r'Resumed from', line)
            if m and detected_total is None:
                # Will be handled by first metric line being lower than last_step
                pass

            # Metric lines: "   10000 | 0.250 | 0.400 | 1403378 |   0.0725 |    68749 |     425"
            m = re.match(r'\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*([\d.]+)', line)
            if m:
                step = int(m.group(1))
                eps = float(m.group(2))
                loss = float(m.group(3))
                speed = float(m.group(4))
                # Detect run restart: step jumped backward — flush stale future data
                if step < last_step:
                    raw_metrics = {s: v for s, v in raw_metrics.items() if s < step}
                    raw_evals = {s: v for s, v in raw_evals.items() if s < step}
                raw_metrics[step] = (eps, loss, speed)
                last_step = step

            # Eval lines: "  [EVAL] deals 67% | rand 88% | ckpt 55% (20s)"
            m = re.match(r'\s*\[EVAL\]\s*deals\s+(\d+)%(?:\s*\|\s*rand\s+(\d+)%)?(?:\s*\|\s*ckpt\s+(\d+)%)?', line)
            if m:
                deal = int(m.group(1))
                rand_wr = int(m.group(2)) if m.group(2) else None
                ckpt_wr = int(m.group(3)) if m.group(3) else None
                raw_evals[last_step] = (deal, rand_wr, ckpt_wr)

    # Sort by step (dedup already handled by dict — last write wins)
    sorted_steps = sorted(raw_metrics.keys())
    steps = sorted_steps
    eps_list = [raw_metrics[s][0] for s in sorted_steps]
    losses = [raw_metrics[s][1] for s in sorted_steps]
    speeds = [raw_metrics[s][2] for s in sorted_steps]

    sorted_eval_steps = sorted(raw_evals.keys())
    eval_steps = sorted_eval_steps
    deal_wrs = [raw_evals[s][0] for s in sorted_eval_steps]
    rand_wrs = [raw_evals[s][1] for s in sorted_eval_steps]
    ckpt_wrs = [raw_evals[s][2] for s in sorted_eval_steps]

    return steps, losses, speeds, eps_list, eval_steps, deal_wrs, rand_wrs, ckpt_wrs, detected_total

def millions_formatter(x, pos):
    return f'{x/1e6:.0f}M'

plt.ion()
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("DMC Training Monitor", fontsize=14, fontweight='bold')
fig.canvas.manager.window.wm_geometry("+100+100")

while True:
    try:
        steps, losses, speeds, eps_list, eval_steps, deal_wrs, rand_wrs, ckpt_wrs, detected_total = parse_log(LOG_FILES)
    except (FileNotFoundError, ValueError):
        time.sleep(INTERVAL)
        continue

    if not steps:
        time.sleep(INTERVAL)
        continue

    for ax in axes.flat:
        ax.clear()

    fmt = ticker.FuncFormatter(millions_formatter)

    # Top-left: Loss
    ax = axes[0, 0]
    ax.plot(steps, losses, color='#e74c3c', linewidth=0.8)
    ax.set_title("Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.xaxis.set_major_formatter(fmt)
    ax.grid(True, alpha=0.3)

    # Top-right: Speed
    ax = axes[0, 1]
    ax.plot(steps, speeds, color='#3498db', linewidth=0.8)
    ax.set_title("Throughput (steps/s)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Steps/s")
    ax.xaxis.set_major_formatter(fmt)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Eval win rates
    ax = axes[1, 0]
    if eval_steps:
        ax.plot(eval_steps, deal_wrs, 'o-', color='#2ecc71', label='Deal WR%', markersize=6)
        valid_rand = [(s, r) for s, r in zip(eval_steps, rand_wrs) if r is not None]
        if valid_rand:
            ax.plot([s for s, _ in valid_rand], [r for _, r in valid_rand],
                    's-', color='#e67e22', label='Match vs Rand%', markersize=6)
        valid_ckpt = [(s, r) for s, r in zip(eval_steps, ckpt_wrs) if r is not None]
        if valid_ckpt:
            ax.plot([s for s, _ in valid_ckpt], [r for _, r in valid_ckpt],
                    'D-', color='#e74c3c', label='Match vs Ckpt%', markersize=6)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
        ax.legend(fontsize=8)
    ax.set_title("Eval Win Rates")
    ax.set_xlabel("Step")
    ax.set_ylabel("Win %")
    ax.set_ylim(40, 100)
    ax.xaxis.set_major_formatter(fmt)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Epsilon
    ax = axes[1, 1]
    ax.plot(steps, eps_list, color='#9b59b6', linewidth=0.8)
    ax.set_title("Epsilon (exploration)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Epsilon")
    ax.xaxis.set_major_formatter(fmt)
    ax.grid(True, alpha=0.3)

    # Progress info
    current = steps[-1]
    if "--total" in sys.argv:
        total = int(sys.argv[sys.argv.index("--total") + 1])
    elif detected_total:
        total = detected_total
    else:
        total = current  # no target known, just show current

    avg_speed = sum(speeds[-10:]) / len(speeds[-10:])
    if current < total:
        pct = current / total * 100
        eta_h = (total - current) / avg_speed / 3600
        fig.suptitle(f"DMC Training — {current/1e6:.1f}M / {total/1e6:.0f}M ({pct:.1f}%) | {avg_speed:.0f} steps/s | ETA {eta_h:.1f}h",
                     fontsize=13, fontweight='bold')
    else:
        fig.suptitle(f"DMC Training — {current/1e6:.1f}M (complete) | {avg_speed:.0f} steps/s",
                     fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.pause(INTERVAL)

    if current >= total and total > 0:
        print("Training complete!")
        plt.ioff()
        plt.show()
        break
