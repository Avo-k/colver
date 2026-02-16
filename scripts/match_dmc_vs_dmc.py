#!/usr/bin/env python3
"""Quick match: two DMC models head-to-head, first to 2000 points.

Usage:
    PYTHONPATH=scripts uv run python scripts/match_dmc_vs_dmc.py \
        models/dmc_17000000.bin models/dmc_final.bin --games 50
"""

import argparse
import time
import colver
import numpy as np


def play_match(env, model_a, model_b, hidden, a_team=0, max_deals=50):
    """Play one match to 2000 between two DMC models.

    Swaps the loaded model depending on whose turn it is.
    """
    a_total, b_total = 0.0, 0.0
    deal_count = 0
    current_loaded = None

    for _ in range(max_deals):
        env.reset()

        while not env.is_terminal():
            phase = env.phase()
            player = env.current_player()
            team = player % 2

            if phase == 0:
                action = env.bid_improved()
            else:
                # Load the right model for this team
                needed = model_a if team == a_team else model_b
                if current_loaded != needed:
                    env.load_dmc_model(needed, hidden)
                    current_loaded = needed
                action = env.action_dmc_with_stats()["best_action"]

            env.step(action)

        rewards = env.rewards()
        a_total += rewards[a_team]
        b_total += rewards[1 - a_team]
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


def main():
    parser = argparse.ArgumentParser(description="DMC vs DMC match play")
    parser.add_argument("model_a", help="Path to model A (.bin)")
    parser.add_argument("model_b", help="Path to model B (.bin)")
    parser.add_argument("--games", type=int, default=50, help="Number of matches")
    parser.add_argument("--hidden", type=int, default=1024, help="Hidden size")
    parser.add_argument("--both-sides", action="store_true",
                        help="Play both as NS and EW (doubles match count)")
    args = parser.parse_args()

    name_a = args.model_a.split("/")[-1]
    name_b = args.model_b.split("/")[-1]
    print(f"Model A: {name_a}")
    print(f"Model B: {name_b}")
    print(f"Matches: {args.games} (first to 2000)")
    if args.both_sides:
        print("Playing both sides (doubles total)")
    print()

    env = colver.Env()
    sides = [0, 1] if args.both_sides else [0]
    all_results = []

    for a_team in sides:
        team_name = "NS" if a_team == 0 else "EW"
        print(f"--- {name_a} as {team_name} vs {name_b} ({args.games} matches) ---")

        wins = 0
        total_margin = 0.0
        total_deals = 0
        start = time.time()

        for i in range(args.games):
            result = play_match(env, args.model_a, args.model_b,
                                args.hidden, a_team)
            all_results.append(result)

            if result["a_won"]:
                wins += 1
            total_margin += result["margin"]
            total_deals += result["deals"]

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start
                print(f"    [{i + 1}/{args.games}] A wins {wins}/{i + 1} "
                      f"({wins / (i + 1):.0%}), "
                      f"margin {total_margin / (i + 1):+.0f}, "
                      f"{total_deals / (i + 1):.1f} deals/match, "
                      f"{elapsed:.0f}s")

        elapsed = time.time() - start
        n = args.games
        print(f"  A wins: {wins}/{n} ({wins / n:.0%})")
        print(f"  B wins: {n - wins}/{n} ({(n - wins) / n:.0%})")
        print(f"  Avg margin: {total_margin / n:+.0f}")
        print(f"  Avg deals/match: {total_deals / n:.1f}")
        print(f"  Time: {elapsed:.1f}s ({n / elapsed:.2f} matches/s)")
        print()

    total = len(all_results)
    total_a_wins = sum(1 for r in all_results if r["a_won"])
    total_margin = sum(r["margin"] for r in all_results)
    total_deals = sum(r["deals"] for r in all_results)
    print(f"=== FINAL: {name_a} wins {total_a_wins}/{total} "
          f"({total_a_wins / total:.0%}), "
          f"margin {total_margin / total:+.0f}, "
          f"{total_deals / total:.1f} deals/match ===")


if __name__ == "__main__":
    main()
