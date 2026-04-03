#!/usr/bin/env python3
"""Live training monitor — reads training.log and plots metrics.

Handles both old-style eval (deals/rand/ckpt) and new v5 eval (rand/ckpt/isdd/nn_bid).
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
    raw_evals = {}    # step -> dict of metric_name -> value
    last_step = 0
    detected_total = None

    for p in (path if isinstance(path, list) else [path]):
      with open(p) as f:
        for line in f:
            # Detect new run boundary: the dashed separator before metric lines
            if re.match(r'^-{40,}', line):
                continue

            # Detect total from step-offset header: "steps will display as 32000001..47000000"
            m = re.search(r'steps will display as (\d+)\.\.(\d+)', line)
            if m:
                run_start = int(m.group(1))
                detected_total = int(m.group(2))
                raw_metrics = {s: v for s, v in raw_metrics.items() if s < run_start}
                raw_evals = {s: v for s, v in raw_evals.items() if s < run_start}

            # Detect resumed run without step-offset
            m = re.search(r'Resumed from', line)
            if m and detected_total is None:
                pass

            # Metric lines: "   10000 | 0.250 | 0.400 | 1403378 |   0.0725 |    68749 |     425"
            m = re.match(r'\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*([\d.]+)', line)
            if m:
                step = int(m.group(1))
                eps = float(m.group(2))
                loss = float(m.group(3))
                speed = float(m.group(4))
                if step < last_step:
                    raw_metrics = {s: v for s, v in raw_metrics.items() if s < step}
                    raw_evals = {s: v for s, v in raw_evals.items() if s < step}
                raw_metrics[step] = (eps, loss, speed)
                last_step = step

            # Eval lines — flexible parser for both old and new formats:
            #   Old: "  [EVAL] deals 67% | rand 88% | ckpt 55% (20s)"
            #   New: "  [EVAL] rand 85% | ckpt 55% | isdd 35% | nn_bid 80% (210s)"
            #   Retro: "1000000 [EVAL] rand 88% | ckpt 55% | isdd 45% (210s)"
            if '[EVAL]' in line:
                # Check for step-prefixed format (from retro_eval)
                m = re.match(r'\s*(\d+)\s+\[EVAL\]', line)
                if m:
                    eval_step = int(m.group(1))
                else:
                    eval_step = last_step
                evals = {}
                for em in re.finditer(r'(\w+)\s+(\d+)%', line):
                    name = em.group(1)
                    if name == 'nn':  # skip "nn" from "nn_bid"
                        continue
                    evals[name] = int(em.group(2))
                if evals:
                    raw_evals[eval_step] = evals

    # Sort by step
    sorted_steps = sorted(raw_metrics.keys())
    steps = sorted_steps
    eps_list = [raw_metrics[s][0] for s in sorted_steps]
    losses = [raw_metrics[s][1] for s in sorted_steps]
    speeds = [raw_metrics[s][2] for s in sorted_steps]

    sorted_eval_steps = sorted(raw_evals.keys())
    eval_steps = sorted_eval_steps
    eval_data = [raw_evals[s] for s in sorted_eval_steps]

    return steps, losses, speeds, eps_list, eval_steps, eval_data, detected_total

def millions_formatter(x, pos):
    if abs(x) >= 1e6:
        return f'{x/1e6:.0f}M'
    elif abs(x) >= 1e3:
        return f'{x/1e3:.0f}K'
    else:
        return f'{x:.0f}'

# Eval metric display config: (key, label, marker, color)
EVAL_METRICS = [
    ('deals',   'Deal WR%',       'o', '#2ecc71'),
    ('rand',    'Match vs Rand%',  's', '#e67e22'),
    ('ckpt',    'Match vs Ckpt%',  'D', '#e74c3c'),
    ('isdd',    'Match vs IS-DD%', '^', '#8e44ad'),
    ('nn_bid',  'NN Bid %',        'v', '#3498db'),
]

plt.ion()
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("DMC Training Monitor", fontsize=14, fontweight='bold')
fig.canvas.manager.window.wm_geometry("+100+100")

while True:
    try:
        steps, losses, speeds, eps_list, eval_steps, eval_data, detected_total = parse_log(LOG_FILES)
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

    # Bottom-left: Eval win rates (all except nn_bid)
    ax = axes[1, 0]
    has_eval = False
    for key, label, marker, color in EVAL_METRICS:
        if key == 'nn_bid':
            continue  # plotted separately in bottom-right
        valid = [(s, d[key]) for s, d in zip(eval_steps, eval_data) if key in d]
        if valid:
            ax.plot([s for s, _ in valid], [v for _, v in valid],
                    f'{marker}-', color=color, label=label, markersize=6)
            has_eval = True
    if has_eval:
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Waiting for first eval...', transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='gray')
    ax.set_title("Eval Win Rates")
    ax.set_xlabel("Step")
    ax.set_ylabel("Win %")
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(fmt)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Epsilon + NN bid fraction
    ax = axes[1, 1]
    ax.plot(steps, eps_list, color='#9b59b6', linewidth=0.8, label='Epsilon')
    nn_bid_data = [(s, d['nn_bid'] / 100.0) for s, d in zip(eval_steps, eval_data) if 'nn_bid' in d]
    if nn_bid_data:
        ax.plot([s for s, _ in nn_bid_data], [v for _, v in nn_bid_data],
                'v-', color='#3498db', label='NN Bid frac', markersize=5)
    ax.set_title("Epsilon & NN Bid Fraction")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.xaxis.set_major_formatter(fmt)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Progress info
    current = steps[-1]
    if "--total" in sys.argv:
        total = int(sys.argv[sys.argv.index("--total") + 1])
    elif detected_total:
        total = detected_total
    else:
        total = current

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
