#!/usr/bin/env python3
"""Evaluate a trained DMC Q-network against various baselines.

Usage:
    uv run python scripts/eval_dmc.py models/dmc_final.pt
    uv run python scripts/eval_dmc.py models/dmc_final.pt --games 200 --baseline naive --time-ms 20
    uv run python scripts/eval_dmc.py models/dmc_final.pt --games 100 --baseline smart --time-ms 20
"""

import argparse
import time

import numpy as np
import torch

import colver
from dmc_model import QNetwork

NUM_CARDS = 32


def play_match_simple(
    env: colver.Env,
    q_net: QNetwork,
    device: torch.device,
    q_team: int,
    baseline: str,
    time_ms: int,
) -> dict:
    """Play a single deal (random/naive baselines — no belief tracking needed)."""
    obs_list, _ = env.reset()
    obs = np.array(obs_list, dtype=np.float32)

    while not env.is_terminal():
        phase = env.phase()
        player = env.current_player()
        team = player % 2

        if phase == 0:  # Bidding
            action = env.bid_improved()
        elif team == q_team:
            # Q-network plays
            mask_full = np.array(env.legal_action_mask(), dtype=np.float32)
            mask = mask_full[:NUM_CARDS]
            if int(mask.sum()) <= 1:
                action = env.legal_actions()[0]
            else:
                obs_t = torch.tensor(obs, device=device).unsqueeze(0)
                mask_t = torch.tensor(mask, device=device).unsqueeze(0)
                with torch.no_grad():
                    q = q_net(obs_t)
                    q[mask_t == 0] = -1e9
                    action = q.argmax(dim=1).item()
        else:
            # Baseline plays
            if baseline == "naive":
                action = env.action_naive_ismcts(time_ms)
            else:  # random
                legal = env.legal_actions()
                action = legal[np.random.randint(len(legal))]

        obs_list, _, done, _ = env.step(action)
        obs = np.array(obs_list, dtype=np.float32)

    outcome = env.deal_outcome()
    rewards = env.rewards()
    return {
        "q_outcome": outcome[q_team],
        "opp_outcome": outcome[1 - q_team],
        "q_reward": rewards[q_team],
        "opp_reward": rewards[1 - q_team],
    }


def play_match_smart(
    env: colver.Env,
    q_net: QNetwork,
    device: torch.device,
    q_team: int,
    time_ms: int,
) -> dict:
    """Play a single deal with Smart IS-MCTS (needs belief tracking via smart_ismcts_step)."""
    obs_list, _ = env.reset()
    obs = np.array(obs_list, dtype=np.float32)
    env.smart_ismcts_init()

    while not env.is_terminal():
        phase = env.phase()
        player = env.current_player()
        team = player % 2

        if phase == 0:  # Bidding
            action = env.bid_improved()
        elif team == q_team:
            # Q-network plays
            mask_full = np.array(env.legal_action_mask(), dtype=np.float32)
            mask = mask_full[:NUM_CARDS]
            if int(mask.sum()) <= 1:
                action = env.legal_actions()[0]
            else:
                obs_t = torch.tensor(obs, device=device).unsqueeze(0)
                mask_t = torch.tensor(mask, device=device).unsqueeze(0)
                with torch.no_grad():
                    q = q_net(obs_t)
                    q[mask_t == 0] = -1e9
                    action = q.argmax(dim=1).item()
        else:
            # Smart IS-MCTS plays
            action = env.action_smart_ismcts(time_ms)

        # Use smart_ismcts_step to track beliefs for all actions
        obs_list, _, done, _ = env.smart_ismcts_step(action)
        obs = np.array(obs_list, dtype=np.float32)

    outcome = env.deal_outcome()
    rewards = env.rewards()
    return {
        "q_outcome": outcome[q_team],
        "opp_outcome": outcome[1 - q_team],
        "q_reward": rewards[q_team],
        "opp_reward": rewards[1 - q_team],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DMC agent")
    parser.add_argument("model", type=str, help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=500, help="Number of games")
    parser.add_argument("--baseline", type=str, default="random",
                        choices=["random", "naive", "smart"],
                        help="Baseline opponent")
    parser.add_argument("--time-ms", type=int, default=20,
                        help="IS-MCTS time budget per move in ms")
    parser.add_argument("--both-sides", action="store_true",
                        help="Play both as NS and EW (doubles game count)")
    parser.add_argument("--hidden", type=int, default=None,
                        help="Hidden layer size (auto-detected from checkpoint)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint and detect architecture
    ckpt = torch.load(args.model, weights_only=False, map_location=device)
    hidden = args.hidden or ckpt.get("hidden", 1024)

    q_net = QNetwork(hidden=hidden).to(device)
    q_net.load_state_dict(ckpt["model"])
    q_net.eval()
    step = ckpt.get("step", "?")
    param_count = sum(p.numel() for p in q_net.parameters())
    print(f"Loaded model from {args.model} (step {step}, {param_count:,} params, hidden={hidden})")
    baseline_desc = args.baseline
    if args.baseline in ("naive", "smart"):
        baseline_desc += f" IS-MCTS ({args.time_ms}ms/move)"
    print(f"Baseline: {baseline_desc}, Games: {args.games}")
    if args.both_sides:
        print("Playing both sides (NS and EW)")
    print()

    env = colver.Env()
    all_results = []

    sides = [0, 1] if args.both_sides else [0]
    for q_team in sides:
        team_name = "NS" if q_team == 0 else "EW"
        print(f"--- Q-network as {team_name} vs {baseline_desc} ---")

        wins = 0
        losses = 0
        draws = 0
        total_margin = 0.0
        void_deals = 0
        start = time.time()

        for i in range(args.games):
            if args.baseline == "smart":
                result = play_match_smart(env, q_net, device, q_team, args.time_ms)
            else:
                result = play_match_simple(
                    env, q_net, device, q_team, args.baseline, args.time_ms)
            all_results.append(result)

            if result["q_outcome"] > result["opp_outcome"]:
                wins += 1
            elif result["q_outcome"] < result["opp_outcome"]:
                losses += 1
            else:
                draws += 1

            total_margin += result["q_reward"] - result["opp_reward"]

            if result["q_outcome"] == 0.5 and result["opp_outcome"] == 0.5:
                void_deals += 1

            # Progress for slow baselines
            if args.baseline in ("naive", "smart") and (i + 1) % 50 == 0:
                elapsed = time.time() - start
                wr = wins / (i + 1)
                print(f"    [{i + 1}/{args.games}] {wr:.1%} win, "
                      f"{elapsed:.0f}s ({(i + 1) / elapsed:.1f} games/s)")

        elapsed = time.time() - start
        n = args.games
        print(f"  Wins: {wins}/{n} ({wins / n:.1%}), "
              f"Losses: {losses}/{n} ({losses / n:.1%}), "
              f"Draws: {draws}/{n}")
        print(f"  Avg margin: {total_margin / n:+.1f}")
        print(f"  Void deals: {void_deals}/{n}")
        print(f"  Time: {elapsed:.1f}s ({n / elapsed:.1f} games/s)")
        print()

    # Summary
    if args.both_sides:
        total_games = len(all_results)
        total_wins = sum(1 for r in all_results if r["q_outcome"] > r["opp_outcome"])
        total_margin = sum(r["q_reward"] - r["opp_reward"] for r in all_results)
        print(f"=== Overall: {total_wins}/{total_games} ({total_wins / total_games:.1%}), "
              f"margin {total_margin / total_games:+.1f} ===")


if __name__ == "__main__":
    main()
