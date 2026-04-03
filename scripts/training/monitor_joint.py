#!/usr/bin/env python3
"""Live monitoring dashboard for train_joint.

Usage:
    python scripts/monitor_joint.py [--csv models/joint/training_log.csv] [--refresh 30]

Generates an HTML file with interactive Plotly charts, auto-refreshes.
Open models/joint/dashboard.html in a browser.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

def generate_html(df: pd.DataFrame, output: Path):
    """Generate a self-contained HTML dashboard with Plotly."""

    # Filter out rows with zero loss (before training starts)
    df_loss = df[df["play_loss"] > 0].copy()
    df_eval = df[df["rand_wr"] > 0].copy()

    html = """<!DOCTYPE html>
<html><head>
<title>Joint Training Monitor</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<meta http-equiv="refresh" content="30">
<style>
  body { font-family: system-ui; background: #1a1a2e; color: #eee; margin: 20px; }
  h1 { color: #e94560; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .chart { background: #16213e; border-radius: 8px; padding: 8px; }
  .stats { display: flex; gap: 24px; margin-bottom: 16px; flex-wrap: wrap; }
  .stat { background: #16213e; padding: 12px 20px; border-radius: 8px; }
  .stat .val { font-size: 1.6em; font-weight: bold; color: #e94560; }
  .stat .label { font-size: 0.85em; color: #888; }
</style>
</head><body>
"""

    # Stats summary
    if len(df) > 0:
        last = df.iloc[-1]
        elapsed_steps = int(last["step"])
        elapsed_m = elapsed_steps / 1_000_000
        eps = int(last["episodes"])
        sps = last.get("steps_per_sec", 0)
        eta_h = (35_000_000 - elapsed_steps) / max(sps, 1) / 3600 if sps > 0 else 0
        html += f"""
<h1>Joint Bid+Play Training</h1>
<div class="stats">
  <div class="stat"><div class="val">{elapsed_m:.1f}M</div><div class="label">Steps</div></div>
  <div class="stat"><div class="val">{eps:,}</div><div class="label">Episodes</div></div>
  <div class="stat"><div class="val">{sps:.0f}</div><div class="label">Steps/sec</div></div>
  <div class="stat"><div class="val">{eta_h:.1f}h</div><div class="label">ETA (35M)</div></div>
  <div class="stat"><div class="val">{last.get('nn_pct', 0):.0f}%</div><div class="label">NN Bid %</div></div>
"""
        if len(df_eval) > 0:
            last_eval = df_eval.iloc[-1]
            html += f"""  <div class="stat"><div class="val">{last_eval['rand_wr']*100:.0f}%</div><div class="label">vs Random</div></div>
  <div class="stat"><div class="val">{last_eval['isdd_wr']*100:.0f}%</div><div class="label">vs IS-DD</div></div>
"""
        html += "</div>\n"

    html += '<div class="grid">\n'

    # Chart 1: Losses
    if len(df_loss) > 0:
        steps = df_loss["step"].tolist()
        play_loss = df_loss["play_loss"].tolist()
        bid_loss = df_loss["bid_loss"].tolist()
        html += f"""
<div class="chart" id="loss"></div>
<script>
Plotly.newPlot('loss', [
  {{x: {steps}, y: {play_loss}, name: 'Play Loss', line: {{color: '#e94560'}}}},
  {{x: {steps}, y: {bid_loss}, name: 'Bid Loss', line: {{color: '#0f3460'}}}}
], {{
  title: 'Training Loss', paper_bgcolor: '#16213e', plot_bgcolor: '#16213e',
  font: {{color: '#eee'}}, xaxis: {{title: 'Step', gridcolor: '#333'}},
  yaxis: {{title: 'Loss', gridcolor: '#333', type: 'log'}}, margin: {{t: 40}}
}});
</script>
"""

    # Chart 2: Eval win rates
    if len(df_eval) > 0:
        steps_e = df_eval["step"].tolist()
        rand_wr = [x * 100 for x in df_eval["rand_wr"].tolist()]
        isdd_wr = [x * 100 for x in df_eval["isdd_wr"].tolist()]
        html += f"""
<div class="chart" id="eval"></div>
<script>
Plotly.newPlot('eval', [
  {{x: {steps_e}, y: {rand_wr}, name: 'vs Random', line: {{color: '#e94560'}}}},
  {{x: {steps_e}, y: {isdd_wr}, name: 'vs IS-DD', line: {{color: '#53d8fb'}}}}
], {{
  title: 'Eval Win Rate (%)', paper_bgcolor: '#16213e', plot_bgcolor: '#16213e',
  font: {{color: '#eee'}}, xaxis: {{title: 'Step', gridcolor: '#333'}},
  yaxis: {{title: 'Win %', gridcolor: '#333', range: [0, 100]}}, margin: {{t: 40}},
  shapes: [{{type: 'line', y0: 50, y1: 50, x0: 0, x1: 1, xref: 'paper', line: {{dash: 'dot', color: '#555'}}}}]
}});
</script>
"""

    # Chart 3: Buffer sizes
    if len(df) > 0:
        steps_all = df["step"].tolist()
        pbuf = df["play_buf"].tolist()
        bbuf = df["bid_buf"].tolist()
        html += f"""
<div class="chart" id="buf"></div>
<script>
Plotly.newPlot('buf', [
  {{x: {steps_all}, y: {pbuf}, name: 'Play Buffer', line: {{color: '#e94560'}}}},
  {{x: {steps_all}, y: {bbuf}, name: 'Bid Buffer', line: {{color: '#0f3460'}}}}
], {{
  title: 'Replay Buffer Size', paper_bgcolor: '#16213e', plot_bgcolor: '#16213e',
  font: {{color: '#eee'}}, xaxis: {{title: 'Step', gridcolor: '#333'}},
  yaxis: {{title: 'Entries', gridcolor: '#333'}}, margin: {{t: 40}}
}});
</script>
"""

    # Chart 4: Exploration (epsilon + NN bid %)
    if len(df) > 0:
        play_eps = df["play_eps"].tolist()
        bid_eps = df["bid_eps"].tolist()
        nn_pct = df["nn_pct"].tolist()
        html += f"""
<div class="chart" id="explore"></div>
<script>
Plotly.newPlot('explore', [
  {{x: {steps_all}, y: {play_eps}, name: 'Play Epsilon', yaxis: 'y'}},
  {{x: {steps_all}, y: {bid_eps}, name: 'Bid Epsilon', yaxis: 'y'}},
  {{x: {steps_all}, y: {nn_pct}, name: 'NN Bid %', yaxis: 'y2', line: {{dash: 'dot', color: '#53d8fb'}}}}
], {{
  title: 'Exploration Schedule', paper_bgcolor: '#16213e', plot_bgcolor: '#16213e',
  font: {{color: '#eee'}}, xaxis: {{title: 'Step', gridcolor: '#333'}},
  yaxis: {{title: 'Epsilon', gridcolor: '#333'}},
  yaxis2: {{title: 'NN %', overlaying: 'y', side: 'right', gridcolor: '#333'}},
  margin: {{t: 40}}
}});
</script>
"""

    # Chart 5: Steps/sec throughput
    if len(df) > 0:
        sps_list = df["steps_per_sec"].tolist()
        html += f"""
<div class="chart" id="throughput"></div>
<script>
Plotly.newPlot('throughput', [
  {{x: {steps_all}, y: {sps_list}, name: 'Steps/sec', fill: 'tozeroy', line: {{color: '#e94560'}}}}
], {{
  title: 'Training Throughput', paper_bgcolor: '#16213e', plot_bgcolor: '#16213e',
  font: {{color: '#eee'}}, xaxis: {{title: 'Step', gridcolor: '#333'}},
  yaxis: {{title: 'Steps/sec', gridcolor: '#333'}}, margin: {{t: 40}}
}});
</script>
"""

    # Chart 6: Cumulative transitions
    if len(df) > 0:
        pt = df["play_trans"].tolist()
        bt = df["bid_trans"].tolist()
        html += f"""
<div class="chart" id="trans"></div>
<script>
Plotly.newPlot('trans', [
  {{x: {steps_all}, y: {pt}, name: 'Play Transitions', line: {{color: '#e94560'}}}},
  {{x: {steps_all}, y: {bt}, name: 'Bid Transitions', line: {{color: '#0f3460'}}}}
], {{
  title: 'Cumulative Transitions', paper_bgcolor: '#16213e', plot_bgcolor: '#16213e',
  font: {{color: '#eee'}}, xaxis: {{title: 'Step', gridcolor: '#333'}},
  yaxis: {{title: 'Count', gridcolor: '#333'}}, margin: {{t: 40}}
}});
</script>
"""

    html += '</div>\n<p style="color:#555;margin-top:16px;">Auto-refreshes every 30s. Last update: ' + time.strftime("%H:%M:%S") + '</p>\n'
    html += "</body></html>"

    output.write_text(html)


def main():
    parser = argparse.ArgumentParser(description="Monitor joint training")
    parser.add_argument("--csv", default="models/joint/training_log.csv")
    parser.add_argument("--output", default=None, help="Output HTML path (default: same dir as CSV)")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh interval (seconds)")
    parser.add_argument("--watch", action="store_true", help="Regenerate on loop")
    parser.add_argument("--step-offset", type=int, default=0, help="Add to step values (for resumed runs)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output = Path(args.output) if args.output else csv_path.parent / "dashboard.html"

    if args.watch:
        print(f"Watching {csv_path} → {output} (every {args.refresh}s, offset={args.step_offset})")
        while True:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if args.step_offset:
                    df["step"] = df["step"] + args.step_offset
                generate_html(df, output)
                print(f"  Updated ({len(df)} rows) at {time.strftime('%H:%M:%S')}")
            else:
                print(f"  Waiting for {csv_path}...")
            time.sleep(args.refresh)
    else:
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}")
            return
        df = pd.read_csv(csv_path)
        if args.step_offset:
            df["step"] = df["step"] + args.step_offset
        generate_html(df, output)
        print(f"Dashboard: {output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
