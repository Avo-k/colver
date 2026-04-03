#!/usr/bin/env python3
"""Head-to-head tournament between DMC checkpoints.

Evaluates relative strength of DMC checkpoints via:
1. H2H matches (pairwise match win rates, first to 2000)
2. Vs random matches (absolute match-play strength)

Usage:
    # Smoke test (2 matches/side, 3 models)
    PYTHONPATH=scripts uv run python scripts/checkpoint_tournament.py \
      --checkpoints models/dmc_27.bin models/dmc_35.bin models/dmc_43000000.bin \
      --bid-model models/bid_nn_final.bin --h2h-matches 2 --skip-random

    # Full tournament (500 matches/side = 1000 mirrored per pair)
    PYTHONPATH=scripts uv run python scripts/checkpoint_tournament.py \
      --bid-model models/bid_nn_final.bin --h2h-matches 500 --workers 8
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import colver


# --- Global bid model path (set by main, read by workers) ---
_bid_model_path = None


# --- Two-env DMC-vs-DMC play ---

def _do_bid(env):
    """Bid using NN if loaded, else improved."""
    if _bid_model_path:
        return env.action_bid_nn()["best_action"]
    return env.bid_improved()


def play_h2h_deal(env_a, env_b, team_a):
    """Play one deal: model_a controls team_a, model_b controls the other.
    Both envs must be synced to the same deal state.
    Returns (a_score, b_score, a_outcome, b_outcome).
    """
    while not env_a.is_terminal():
        phase = env_a.phase()
        player = env_a.current_player()
        team = player % 2

        if phase == 0:
            action = _do_bid(env_a)
        elif team == team_a:
            action = env_a.action_dmc_with_stats()["best_action"]
        else:
            action = env_b.action_dmc_with_stats()["best_action"]

        env_a.step(action)
        env_b.step(action)

    rewards_a = env_a.rewards()
    outcome_a = env_a.deal_outcome()
    return rewards_a[team_a], rewards_a[1 - team_a], outcome_a[team_a], outcome_a[1 - team_a]


def play_h2h_match(env_a, env_b, team_a, max_deals=50):
    """Play a match to 2000 between two DMC models. Returns result dict."""
    a_total, b_total = 0.0, 0.0
    deal_count = 0
    for _ in range(max_deals):
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


# --- Vs baseline ---

def play_vs_baseline_match(env, baseline, time_ms, q_team, max_deals=50):
    """Play a match to 2000 vs baseline."""
    q_total, opp_total = 0.0, 0.0
    deal_count = 0
    for _ in range(max_deals):
        env.reset()
        while not env.is_terminal():
            phase = env.phase()
            player = env.current_player()
            team = player % 2

            if phase == 0:
                action = _do_bid(env)
            elif team == q_team:
                action = env.action_dmc_with_stats()["best_action"]
            else:
                if baseline == "random":
                    action = env.action_random()
                else:
                    action = env.action_naive_ismcts(time_ms)

            env.step(action)

        rewards = env.rewards()
        q_total += rewards[q_team]
        opp_total += rewards[1 - q_team]
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


# --- Workers ---

def _init_env(path, hidden):
    """Create and configure an Env with DMC + optional bid model."""
    env = colver.Env()
    env.load_dmc_model(path, hidden)
    if _bid_model_path:
        env.load_bid_model(_bid_model_path)
    return env


def _worker_h2h_matches(path_a, path_b, hidden, team_a, count, bid_model):
    """Worker: play count H2H matches between two checkpoints."""
    global _bid_model_path
    _bid_model_path = bid_model
    env_a = _init_env(path_a, hidden)
    env_b = _init_env(path_b, hidden)

    results = []
    for _ in range(count):
        result = play_h2h_match(env_a, env_b, team_a)
        results.append(result)
    return results


def _worker_vs_random_matches(path, hidden, q_team, count, bid_model):
    """Worker: play count matches vs random."""
    global _bid_model_path
    _bid_model_path = bid_model
    env = _init_env(path, hidden)

    results = []
    for _ in range(count):
        result = play_vs_baseline_match(env, "random", 0, q_team)
        results.append(result)
    return results


# --- Utility ---

def short_name(path):
    """Extract short checkpoint name from path."""
    base = os.path.basename(path).replace(".bin", "")
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

def run_h2h_matches(checkpoints, args):
    """Head-to-head match evaluation (first to 2000)."""
    names = [short_name(p) for p in checkpoints]
    n = len(checkpoints)
    matches_per_side = args.h2h_matches
    workers = args.workers

    print(f"\n{'=' * 60}")
    print(f"  H2H MATCHES: {matches_per_side} matches/side (first to 2000)")
    print(f"  Bidding: {'NN (Le Bide a Dede)' if args.bid_model else 'improved'}")
    print(f"  Workers: {workers}")
    print(f"{'=' * 60}")

    win_rate = [[0.0] * n for _ in range(n)]
    margin = [[0.0] * n for _ in range(n)]
    avg_deals = [[0.0] * n for _ in range(n)]

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
                        args.hidden, team_a, batch_size,
                        args.bid_model,
                    )
                    futures[fut] = (i, j, team_a)

        done = 0
        total_jobs = len(futures)
        pair_results = {}
        for fut in as_completed(futures):
            key = futures[fut]
            pair_results.setdefault(key, []).extend(fut.result())
            done += 1
            if done % max(1, total_jobs // 10) == 0:
                elapsed = time.time() - start
                print(f"  ... {done}/{total_jobs} jobs done ({elapsed:.0f}s)")

    for i, j in pairs:
        a_wins = 0
        a_total_margin = 0.0
        a_total_deals = 0
        total = 0
        for team_a in [0, 1]:
            results = pair_results.get((i, j, team_a), [])
            for r in results:
                if r["a_won"]:
                    a_wins += 1
                a_total_margin += r["margin"]
                a_total_deals += r["deals"]
                total += 1

        if total > 0:
            win_rate[i][j] = a_wins / total
            win_rate[j][i] = 1.0 - a_wins / total
            margin[i][j] = a_total_margin / total
            margin[j][i] = -margin[i][j]
            avg_deals[i][j] = a_total_deals / total
            avg_deals[j][i] = avg_deals[i][j]

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

    print(f"\n  ({total_matches} matches in {elapsed:.0f}s = {elapsed / 60:.1f} min)")


def run_vs_random_matches(checkpoints, args):
    """Match evaluation vs random baseline."""
    names = [short_name(p) for p in checkpoints]
    matches_per_side = args.random_matches
    workers = args.workers

    print(f"\n{'=' * 60}")
    print(f"  VS RANDOM MATCHES: {matches_per_side} matches/side (first to 2000)")
    print(f"{'=' * 60}")

    start = time.time()
    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for idx, path in enumerate(checkpoints):
            for q_team in [0, 1]:
                batches = distribute_work(matches_per_side, workers)
                for batch_size in batches:
                    fut = pool.submit(
                        _worker_vs_random_matches,
                        path, args.hidden, q_team, batch_size,
                        args.bid_model,
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
        avg_d = sum(r["deals"] for r in all_results) / total if total else 0
        print(f"  {name:>12}  {wins/total:>7.1%}  {avg_margin:>+8.0f}  {avg_d:>12.1f}")

    print(f"\n  ({len(checkpoints) * 2 * matches_per_side} matches in {elapsed:.0f}s)")


# --- Main ---

def main():
    default_checkpoints = [
        "models/dmc_27.bin",       # DouDou27 (v0.3.1 release, old Python run)
        "models/dmc_35.bin",       # DouDou35 (v0.3.0 release, old Python run)
        "models/dmc_2000000.bin",  # 2M steps (candle run, early)
        "models/dmc_5000000.bin",  # 5M
        "models/dmc_10000000.bin", # 10M
        "models/dmc_20000000.bin", # 20M
        "models/dmc_30000000.bin", # 30M
        "models/dmc_43000000.bin", # 43M (candle run, latest)
    ]

    parser = argparse.ArgumentParser(
        description="Head-to-head tournament between DMC checkpoints")
    parser.add_argument("--checkpoints", nargs="+", default=default_checkpoints,
                        help="Checkpoint .bin paths")
    parser.add_argument("--bid-model", type=str, default=None,
                        help="Path to bid NN model (.bin) for Le Bide a Dede bidding")
    parser.add_argument("--hidden", type=int, default=1024,
                        help="Hidden layer size (default 1024)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default 4)")
    parser.add_argument("--h2h-matches", type=int, default=500,
                        help="H2H matches per side per pair (default 500, x2 mirrored = 1000)")
    parser.add_argument("--random-matches", type=int, default=100,
                        help="Matches per side vs random (default 100)")
    parser.add_argument("--skip-random", action="store_true",
                        help="Skip vs-random section")
    args = parser.parse_args()

    # Set global for workers
    global _bid_model_path
    _bid_model_path = args.bid_model

    # Validate
    missing = [p for p in args.checkpoints if not os.path.exists(p)]
    if missing:
        print(f"Error: missing checkpoints: {missing}")
        return
    if args.bid_model and not os.path.exists(args.bid_model):
        print(f"Error: bid model not found: {args.bid_model}")
        return

    names = [short_name(p) for p in args.checkpoints]
    n = len(args.checkpoints)
    n_pairs = n * (n - 1) // 2
    total_matches = n_pairs * 2 * args.h2h_matches

    print(f"DMC Checkpoint Tournament")
    print(f"  Models:    {' vs '.join(names)}")
    print(f"  Bidding:   {'NN (Le Bide a Dede)' if args.bid_model else 'improved'}")
    print(f"  Workers:   {args.workers}")
    print(f"  H2H:       {args.h2h_matches}/side x2 mirrored = {2*args.h2h_matches} matches/pair")
    print(f"  Pairs:     {n_pairs} ({n} models)")
    print(f"  Total:     {total_matches} matches")

    total_start = time.time()

    run_h2h_matches(args.checkpoints, args)

    if not args.skip_random:
        run_vs_random_matches(args.checkpoints, args)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  TOTAL TIME: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
