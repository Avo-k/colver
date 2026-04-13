"""Baseline: bid_v2 vs bid_v2 with DMC play, using same arena_eval framework."""
import time
import numpy as np
from colver import Env


def play_game(env):
    while env.phase() == 0 and not env.is_terminal():
        r = env.action_bid_nn()
        env.step(r["best_action"])
    while not env.is_terminal():
        r = env.action_dmc_with_stats()
        env.step(r["best_action"])
    return env.rewards()


def main():
    n_matches = 200
    wins_a = 0
    wins_b = 0
    total_diff = 0.0
    t0 = time.time()

    for i in range(n_matches):
        env_a = Env()
        env_a.reset()
        env_a.load_bid_model("models/bid_v2/bid_nn_final.bin", 512)
        env_a.load_dmc_model("models/dmc_50.bin")

        hands = env_a.get_hands()
        dealer = env_a.get_dealer()

        ns_a, ew_a = play_game(env_a)

        env_b = Env.deal_with_hands(dealer, hands)
        env_b.load_bid_model("models/bid_v2/bid_nn_final.bin", 512)
        env_b.load_dmc_model("models/dmc_50.bin")

        ns_b, ew_b = play_game(env_b)

        # Team A = NS in game A, EW in game B
        a_score = ns_a + ew_b
        b_score = ew_a + ns_b
        total_diff += (a_score - b_score)
        if a_score > b_score:
            wins_a += 1
        elif b_score > a_score:
            wins_b += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_matches}] A={wins_a} B={wins_b} "
                  f"avg_diff={total_diff/(i+1):+.1f}")

    elapsed = time.time() - t0
    print(f"\nbid_v2 vs bid_v2 baseline ({n_matches} duplicate matches):")
    print(f"  A wins: {wins_a}  B wins: {wins_b}  Draws: {n_matches-wins_a-wins_b}")
    print(f"  Avg diff: {total_diff/n_matches:+.1f} pts/match (should be ~0)")
    print(f"  Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
