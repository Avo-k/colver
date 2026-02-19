#!/usr/bin/env python3
"""Head-to-head tournament between DMC checkpoints.

Evaluates relative strength of DMC checkpoints via:
1. H2H deals (pairwise win rates and margins)
2. H2H matches (pairwise match win rates, first to 2000)
3. Vs random deals (absolute strength baseline)
4. Vs random matches (absolute match-play strength)
5. Vs naive IS-MCTS deals (strength vs search-based agent)

Usage:
    PYTHONPATH=scripts uv run python scripts/checkpoint_tournament.py
    PYTHONPATH=scripts uv run python scripts/checkpoint_tournament.py --skip-naive --workers 8
    PYTHONPATH=scripts uv run python scripts/checkpoint_tournament.py --checkpoints models/dmc_2000000.bin models/dmc_8000000.bin
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import numpy as np
import colver


# --- Two-env DMC-vs-DMC deal play ---

def play_h2h_deal(env_a, env_b, team_a):
    """Play one deal: model_a controls team_a, model_b controls the other.
    Both envs must be synced to the same deal state.
    Returns (a_score, b_score, a_outcome, b_outcome).
    """
    team_b = 1 - team_a
    while not env_a.is_terminal():
        phase = env_a.phase()
        player = env_a.current_player()
        team = player % 2

        if phase == 0:
            action = env_a.bid_improved()
        elif team == team_a:
            action = env_a.action_dmc_with_stats()["best_action"]
        else:
            action = env_b.action_dmc_with_stats()["best_action"]

        env_a.step(action)
        env_b.step(action)

    rewards_a = env_a.rewards()
    outcome_a = env_a.deal_outcome()
    return rewards_a[team_a], rewards_a[team_b], outcome_a[team_a], outcome_a[team_b]


def play_h2h_match(env_a, env_b, team_a, max_deals=50):
    """Play a match to 2000 between two DMC models. Returns result dict."""
    a_total, b_total = 0.0, 0.0
    deal_count = 0
    for _ in range(max_deals):
        # Reset and sync
        env_a.reset()
        env_b.reset()
        env_b.set_state_bytes(env_a.get_state_bytes().tolist())

        a_score, b_score, _, _ = play_h2h_deal(env_a, env_b, team_a)
        a_total += a_score
        b_total += b_score
        deal_count += 1
        if a_total >= 2000 or b_total >= 2000:
            break
    return {
        "a_won": a_total >= 2000,
        "a_total": a_total,
        "b_total": b_total,
        "margin": a_total - b_total,
        "deals": deal_count,
    }


# --- Vs baseline deal/match play (reuses eval_dmc patterns) ---

def play_vs_baseline_deal(env, baseline, time_ms, q_team):
    """Play a single deal vs baseline. Returns (q_score, opp_score, q_out, opp_out)."""
    env.reset()
    while not env.is_terminal():
        phase = env.phase()
        player = env.current_player()
        team = player % 2

        if phase == 0:
            action = env.bid_improved()
        elif team == q_team:
            action = env.action_dmc_with_stats()["best_action"]
        else:
            if baseline == "random":
                action = env.action_random()
            else:
                action = env.action_naive_ismcts(time_ms)

        env.step(action)

    rewards = env.rewards()
    outcome = env.deal_outcome()
    return rewards[q_team], rewards[1 - q_team], outcome[q_team], outcome[1 - q_team]


def play_vs_baseline_match(env, baseline, time_ms, q_team, max_deals=50):
    """Play a match to 2000 vs baseline."""
    q_total, opp_total = 0.0, 0.0
    deal_count = 0
    for _ in range(max_deals):
        q_score, opp_score, _, _ = play_vs_baseline_deal(env, baseline, time_ms, q_team)
        q_total += q_score
        opp_total += opp_score
        deal_count += 1
        if q_total >= 2000 or opp_total >= 2000:
            break
    return {
        "q_won": q_total >= 2000,
        "q_total": q_total,
        "opp_total": opp_total,
        "margin": q_total - opp_total,
        "deals": deal_count,
    }


# --- Parallel worker functions ---

def _worker_h2h_deals(path_a, path_b, hidden, team_a, count):
    """Worker: play count H2H deals between two checkpoints."""
    env_a = colver.Env()
    env_b = colver.Env()
    env_a.load_dmc_model(path_a, hidden)
    env_b.load_dmc_model(path_b, hidden)

    results = []
    for _ in range(count):
        env_a.reset()
        env_b.reset()
        env_b.set_state_bytes(env_a.get_state_bytes().tolist())
        a_score, b_score, a_out, b_out = play_h2h_deal(env_a, env_b, team_a)
        results.append({
            "a_outcome": a_out, "b_outcome": b_out,
            "a_reward": a_score, "b_reward": b_score,
        })
    return results


def _worker_h2h_matches(path_a, path_b, hidden, team_a, count):
    """Worker: play count H2H matches between two checkpoints."""
    env_a = colver.Env()
    env_b = colver.Env()
    env_a.load_dmc_model(path_a, hidden)
    env_b.load_dmc_model(path_b, hidden)

    results = []
    for _ in range(count):
        result = play_h2h_match(env_a, env_b, team_a)
        results.append(result)
    return results


def _worker_vs_baseline_deals(path, hidden, baseline, time_ms, q_team, count):
    """Worker: play count deals vs baseline."""
    env = colver.Env()
    env.load_dmc_model(path, hidden)

    results = []
    for _ in range(count):
        q_score, opp_score, q_out, opp_out = play_vs_baseline_deal(
            env, baseline, time_ms, q_team)
        results.append({
            "q_outcome": q_out, "opp_outcome": opp_out,
            "q_reward": q_score, "opp_reward": opp_score,
        })
    return results


def _worker_vs_baseline_matches(path, hidden, baseline, time_ms, q_team, count):
    """Worker: play count matches vs baseline."""
    env = colver.Env()
    env.load_dmc_model(path, hidden)

    results = []
    for _ in range(count):
        result = play_vs_baseline_match(env, baseline, time_ms, q_team)
        results.append(result)
    return results


# --- Utility ---

def short_name(path):
    """Extract short checkpoint name from path."""
    base = os.path.basename(path).replace(".bin", "")
    # dmc_2000000 -> 2M, dmc_final -> final
    if base.startswith("dmc_"):
        suffix = base[4:]
        if suffix.isdigit():
            n = int(suffix)
            if n >= 1_000_000 and n % 1_000_000 == 0:
                return f"{n // 1_000_000}M"
            elif n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            elif n >= 1_000 and n % 1_000 == 0:
                return f"{n // 1_000}K"
            else:
                return f"{n}"
        return suffix
    return base


def distribute_work(total, workers):
    """Split total into per-worker batch sizes."""
    per = total // workers
    rem = total % workers
    batches = [per + (1 if i < rem else 0) for i in range(workers)]
    return [b for b in batches if b > 0]


def print_matrix(names, matrix, fmt=".1%", title=""):
    """Print a square matrix with row/column labels."""
    if title:
        print(f"\n{title}")
    col_w = max(len(n) for n in names)
    col_w = max(col_w, 6)
    header = " " * (col_w + 2) + "  ".join(f"{n:>{col_w}}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = f"{name:>{col_w}}  "
        for j in range(len(names)):
            if i == j:
                cell = "-"
            else:
                val = matrix[i][j]
                if fmt == ".1%":
                    cell = f"{val:.1%}"
                elif fmt == "+.0f":
                    cell = f"{val:+.0f}"
                else:
                    cell = f"{val:{fmt}}"
            row += f"{cell:>{col_w}}  "
        print(row)


# --- Tournament sections ---

def run_h2h_deals(checkpoints, args):
    """Section 1: Head-to-head deal evaluation."""
    names = [short_name(p) for p in checkpoints]
    n = len(checkpoints)
    deals_per_side = args.h2h_deals
    workers = args.workers

    print(f"\n{'=' * 60}")
    print(f"  H2H DEALS: {deals_per_side} deals/side, {workers} workers")
    print(f"{'=' * 60}")

    # win_rate[i][j] = fraction of deals checkpoint i wins vs j
    win_rate = [[0.0] * n for _ in range(n)]
    margin = [[0.0] * n for _ in range(n)]

    pairs = list(combinations(range(n), 2))
    start = time.time()

    # Submit all jobs
    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, j in pairs:
            for team_a in [0, 1]:
                batches = distribute_work(deals_per_side, workers)
                for batch_size in batches:
                    fut = pool.submit(
                        _worker_h2h_deals,
                        checkpoints[i], checkpoints[j],
                        args.hidden, team_a, batch_size
                    )
                    futures[fut] = (i, j, team_a)

        # Collect results
        pair_results = {}  # (i, j, team_a) -> list of results
        for fut in as_completed(futures):
            key = futures[fut]
            pair_results.setdefault(key, []).extend(fut.result())

    # Aggregate
    for i, j in pairs:
        a_wins = 0
        a_total_margin = 0.0
        total = 0
        for team_a in [0, 1]:
            results = pair_results.get((i, j, team_a), [])
            for r in results:
                if r["a_outcome"] > r["b_outcome"]:
                    a_wins += 1
                a_total_margin += r["a_reward"] - r["b_reward"]
                total += 1

        if total > 0:
            win_rate[i][j] = a_wins / total
            win_rate[j][i] = 1.0 - a_wins / total
            margin[i][j] = a_total_margin / total
            margin[j][i] = -margin[i][j]

    elapsed = time.time() - start
    total_deals = len(pairs) * 2 * deals_per_side

    print_matrix(names, win_rate, fmt=".1%", title="Win Rate (row vs col):")
    print_matrix(names, margin, fmt="+.0f", title="Avg Margin (row vs col):")

    # Rankings by average win rate
    avg_wr = [sum(win_rate[i][j] for j in range(n) if j != i) / (n - 1) for i in range(n)]
    ranked = sorted(range(n), key=lambda i: avg_wr[i], reverse=True)
    print("\nRankings (by avg win rate):")
    for rank, i in enumerate(ranked, 1):
        avg_m = sum(margin[i][j] for j in range(n) if j != i) / (n - 1)
        print(f"  {rank}. {names[i]:>8}  win {avg_wr[i]:.1%}  margin {avg_m:+.0f}")

    print(f"\n  ({total_deals} deals in {elapsed:.1f}s)")
    return win_rate, margin


def run_h2h_matches(checkpoints, args):
    """Section 2: Head-to-head match evaluation."""
    names = [short_name(p) for p in checkpoints]
    n = len(checkpoints)
    matches_per_side = args.h2h_matches
    workers = args.workers

    print(f"\n{'=' * 60}")
    print(f"  H2H MATCHES: {matches_per_side} matches/side (first to 2000), {workers} workers")
    print(f"{'=' * 60}")

    win_rate = [[0.0] * n for _ in range(n)]
    margin = [[0.0] * n for _ in range(n)]

    pairs = list(combinations(range(n), 2))
    start = time.time()

    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, j in pairs:
            for team_a in [0, 1]:
                batches = distribute_work(matches_per_side, workers)
                for batch_size in batches:
                    fut = pool.submit(
                        _worker_h2h_matches,
                        checkpoints[i], checkpoints[j],
                        args.hidden, team_a, batch_size
                    )
                    futures[fut] = (i, j, team_a)

        pair_results = {}
        for fut in as_completed(futures):
            key = futures[fut]
            pair_results.setdefault(key, []).extend(fut.result())

    for i, j in pairs:
        a_wins = 0
        a_total_margin = 0.0
        total = 0
        for team_a in [0, 1]:
            results = pair_results.get((i, j, team_a), [])
            for r in results:
                if r["a_won"]:
                    a_wins += 1
                a_total_margin += r["margin"]
                total += 1

        if total > 0:
            win_rate[i][j] = a_wins / total
            win_rate[j][i] = 1.0 - a_wins / total
            margin[i][j] = a_total_margin / total
            margin[j][i] = -margin[i][j]

    elapsed = time.time() - start
    total_matches = len(pairs) * 2 * matches_per_side

    print_matrix(names, win_rate, fmt=".1%", title="Match Win Rate (row vs col):")
    print_matrix(names, margin, fmt="+.0f", title="Match Avg Margin (row vs col):")

    avg_wr = [sum(win_rate[i][j] for j in range(n) if j != i) / (n - 1) for i in range(n)]
    ranked = sorted(range(n), key=lambda i: avg_wr[i], reverse=True)
    print("\nRankings (by avg match win rate):")
    for rank, i in enumerate(ranked, 1):
        avg_m = sum(margin[i][j] for j in range(n) if j != i) / (n - 1)
        print(f"  {rank}. {names[i]:>8}  win {avg_wr[i]:.1%}  margin {avg_m:+.0f}")

    print(f"\n  ({total_matches} matches in {elapsed:.1f}s)")


def run_vs_random_deals(checkpoints, args):
    """Section 3: Deal evaluation vs random baseline."""
    names = [short_name(p) for p in checkpoints]
    deals_per_side = args.random_deals
    workers = args.workers

    print(f"\n{'=' * 60}")
    print(f"  VS RANDOM DEALS: {deals_per_side} deals/side, {workers} workers")
    print(f"{'=' * 60}")

    start = time.time()
    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for idx, path in enumerate(checkpoints):
            for q_team in [0, 1]:
                batches = distribute_work(deals_per_side, workers)
                for batch_size in batches:
                    fut = pool.submit(
                        _worker_vs_baseline_deals,
                        path, args.hidden, "random", 0, q_team, batch_size
                    )
                    futures[fut] = (idx, q_team)

        cp_results = {}
        for fut in as_completed(futures):
            key = futures[fut]
            cp_results.setdefault(key, []).extend(fut.result())

    elapsed = time.time() - start

    print(f"\n  {'Checkpoint':>12}  {'Win%':>7}  {'Margin':>8}")
    print(f"  {'-' * 12}  {'-' * 7}  {'-' * 8}")
    for idx, (path, name) in enumerate(zip(checkpoints, names)):
        all_results = []
        for q_team in [0, 1]:
            all_results.extend(cp_results.get((idx, q_team), []))
        wins = sum(1 for r in all_results if r["q_outcome"] > r["opp_outcome"])
        total = len(all_results)
        avg_margin = sum(r["q_reward"] - r["opp_reward"] for r in all_results) / total if total else 0
        print(f"  {name:>12}  {wins/total:>7.1%}  {avg_margin:>+8.0f}")

    print(f"\n  ({len(checkpoints) * 2 * deals_per_side} deals in {elapsed:.1f}s)")


def run_vs_random_matches(checkpoints, args):
    """Section 4: Match evaluation vs random baseline."""
    names = [short_name(p) for p in checkpoints]
    matches_per_side = args.random_matches
    workers = args.workers

    print(f"\n{'=' * 60}")
    print(f"  VS RANDOM MATCHES: {matches_per_side} matches/side (first to 2000), {workers} workers")
    print(f"{'=' * 60}")

    start = time.time()
    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for idx, path in enumerate(checkpoints):
            for q_team in [0, 1]:
                batches = distribute_work(matches_per_side, workers)
                for batch_size in batches:
                    fut = pool.submit(
                        _worker_vs_baseline_matches,
                        path, args.hidden, "random", 0, q_team, batch_size
                    )
                    futures[fut] = (idx, q_team)

        cp_results = {}
        for fut in as_completed(futures):
            key = futures[fut]
            cp_results.setdefault(key, []).extend(fut.result())

    elapsed = time.time() - start

    print(f"\n  {'Checkpoint':>12}  {'Win%':>7}  {'Margin':>8}  {'Deals/Match':>12}")
    print(f"  {'-' * 12}  {'-' * 7}  {'-' * 8}  {'-' * 12}")
    for idx, (path, name) in enumerate(zip(checkpoints, names)):
        all_results = []
        for q_team in [0, 1]:
            all_results.extend(cp_results.get((idx, q_team), []))
        wins = sum(1 for r in all_results if r["q_won"])
        total = len(all_results)
        avg_margin = sum(r["margin"] for r in all_results) / total if total else 0
        avg_deals = sum(r["deals"] for r in all_results) / total if total else 0
        print(f"  {name:>12}  {wins/total:>7.1%}  {avg_margin:>+8.0f}  {avg_deals:>12.1f}")

    print(f"\n  ({len(checkpoints) * 2 * matches_per_side} matches in {elapsed:.1f}s)")


def run_vs_naive_deals(checkpoints, args):
    """Section 5: Deal evaluation vs naive IS-MCTS."""
    names = [short_name(p) for p in checkpoints]
    deals_per_side = args.naive_deals
    time_ms = args.naive_time_ms
    workers = args.workers

    print(f"\n{'=' * 60}")
    print(f"  VS NAIVE IS-MCTS DEALS: {deals_per_side} deals/side, {time_ms}ms/move, {workers} workers")
    print(f"{'=' * 60}")

    start = time.time()
    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for idx, path in enumerate(checkpoints):
            for q_team in [0, 1]:
                batches = distribute_work(deals_per_side, workers)
                for batch_size in batches:
                    fut = pool.submit(
                        _worker_vs_baseline_deals,
                        path, args.hidden, "naive", time_ms, q_team, batch_size
                    )
                    futures[fut] = (idx, q_team)

        cp_results = {}
        for fut in as_completed(futures):
            key = futures[fut]
            cp_results.setdefault(key, []).extend(fut.result())

    elapsed = time.time() - start

    print(f"\n  {'Checkpoint':>12}  {'Win%':>7}  {'Margin':>8}")
    print(f"  {'-' * 12}  {'-' * 7}  {'-' * 8}")
    for idx, (path, name) in enumerate(zip(checkpoints, names)):
        all_results = []
        for q_team in [0, 1]:
            all_results.extend(cp_results.get((idx, q_team), []))
        wins = sum(1 for r in all_results if r["q_outcome"] > r["opp_outcome"])
        total = len(all_results)
        avg_margin = sum(r["q_reward"] - r["opp_reward"] for r in all_results) / total if total else 0
        print(f"  {name:>12}  {wins/total:>7.1%}  {avg_margin:>+8.0f}")

    print(f"\n  ({len(checkpoints) * 2 * deals_per_side} deals in {elapsed:.1f}s)")


# --- Main ---

def main():
    default_checkpoints = [
        "models/dmc_2000000.bin",
        "models/dmc_4000000.bin",
        "models/dmc_8000000.bin",
        "models/dmc_12000000.bin",
    ]

    parser = argparse.ArgumentParser(
        description="Head-to-head tournament between DMC checkpoints")
    parser.add_argument("--checkpoints", nargs="+", default=default_checkpoints,
                        help="Checkpoint .bin paths (default: 2M 4M 8M 12M)")
    parser.add_argument("--hidden", type=int, default=1024,
                        help="Hidden layer size (default 1024)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default 4)")
    parser.add_argument("--h2h-deals", type=int, default=100,
                        help="H2H deals per side per pair (default 100)")
    parser.add_argument("--h2h-matches", type=int, default=20,
                        help="H2H matches per side per pair (default 20)")
    parser.add_argument("--random-deals", type=int, default=200,
                        help="Deals per side vs random (default 200)")
    parser.add_argument("--random-matches", type=int, default=30,
                        help="Matches per side vs random (default 30)")
    parser.add_argument("--naive-deals", type=int, default=50,
                        help="Deals per side vs naive IS-MCTS (default 50)")
    parser.add_argument("--naive-time-ms", type=int, default=10,
                        help="Naive IS-MCTS time budget ms (default 10)")
    parser.add_argument("--skip-naive", action="store_true",
                        help="Skip vs-naive section")
    parser.add_argument("--skip-matches", action="store_true",
                        help="Skip match sections (H2H matches + random matches)")
    parser.add_argument("--skip-h2h", action="store_true",
                        help="Skip H2H sections (deals and matches between checkpoints)")
    args = parser.parse_args()

    # Validate checkpoints exist
    missing = [p for p in args.checkpoints if not os.path.exists(p)]
    if missing:
        print(f"Error: missing checkpoints: {missing}")
        return

    names = [short_name(p) for p in args.checkpoints]
    print(f"Checkpoint Tournament: {' vs '.join(names)}")
    print(f"Workers: {args.workers}, Hidden: {args.hidden}")
    total_start = time.time()

    # Section 1: H2H deals
    if not args.skip_h2h:
        run_h2h_deals(args.checkpoints, args)

    # Section 2: H2H matches
    if not args.skip_h2h and not args.skip_matches:
        run_h2h_matches(args.checkpoints, args)

    # Section 3: Vs random deals
    run_vs_random_deals(args.checkpoints, args)

    # Section 4: Vs random matches
    if not args.skip_matches:
        run_vs_random_matches(args.checkpoints, args)

    # Section 5: Vs naive IS-MCTS
    if not args.skip_naive:
        run_vs_naive_deals(args.checkpoints, args)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  TOTAL TIME: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
