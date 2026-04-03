#!/usr/bin/env python3
"""Evaluate a trained DMC Q-network against various baselines.

Supports both Rust inference (.bin) and PyTorch inference (.pt).
Rust inference uses the colver Env's built-in DmcNet — no torch needed.

Usage:
    # Rust inference (recommended, no torch dependency)
    PYTHONPATH=scripts uv run python scripts/eval_dmc.py models/dmc_final.bin --both-sides
    PYTHONPATH=scripts uv run python scripts/eval_dmc.py models/dmc_final.bin --mode match --baseline smart --time-ms 50 --games 60

    # PyTorch inference (.pt checkpoint)
    PYTHONPATH=scripts uv run python scripts/eval_dmc.py models/dmc_800000.pt --both-sides
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import colver

NUM_CARDS = 32


# --- Inference backends ---

class RustEngine:
    """Rust-based DMC inference via colver.Env.load_dmc_model()."""

    def __init__(self, path: str, hidden: int = 1024):
        self.path = path
        self.hidden = hidden

    def load_into(self, env: colver.Env):
        env.load_dmc_model(self.path, self.hidden)

    def action(self, env: colver.Env) -> int:
        return env.action_dmc_with_stats()["best_action"]

    @property
    def desc(self) -> str:
        return f"Rust inference ({self.path})"


class TorchEngine:
    """PyTorch-based DMC inference."""

    def __init__(self, path: str, hidden: int = 1024):
        import torch
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from dmc_model import QNetwork

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, weights_only=False, map_location=self.device)
        h = hidden or ckpt.get("hidden", 1024)
        self.hidden = h
        self.obs_dim = ckpt.get("obs_dim", 372)

        self.q_net = QNetwork(obs_dim=self.obs_dim, hidden=h).to(self.device)
        self.q_net.load_state_dict(ckpt["model"])
        self.q_net.eval()
        self.step = ckpt.get("step", "?")
        self.param_count = sum(p.numel() for p in self.q_net.parameters())
        self.torch = torch

    def load_into(self, env: colver.Env):
        pass  # PyTorch model lives in Python, not in env

    def action(self, env: colver.Env, obs: np.ndarray = None) -> int:
        mask_full = np.array(env.legal_action_mask(), dtype=np.float32)
        mask = mask_full[:NUM_CARDS]
        if int(mask.sum()) <= 1:
            return env.legal_actions()[0]
        # Truncate obs to match model's expected obs_dim (backward compat)
        obs_used = obs[:self.obs_dim]
        obs_t = self.torch.tensor(obs_used, device=self.device).unsqueeze(0)
        mask_t = self.torch.tensor(mask, device=self.device).unsqueeze(0)
        with self.torch.no_grad():
            q = self.q_net(obs_t)
            q[mask_t == 0] = -1e9
            return q.argmax(dim=1).item()

    @property
    def desc(self) -> str:
        return f"PyTorch inference (step {self.step}, {self.param_count:,} params, hidden={self.hidden}, obs_dim={self.obs_dim})"


# --- Game play ---

def play_deal(env, engine, q_team, baseline, time_ms):
    """Play a single deal. Returns (q_score, opp_score, q_outcome, opp_outcome)."""
    obs_list, _ = env.reset()
    obs = np.array(obs_list, dtype=np.float32)
    use_smart = (baseline == "smart")
    if use_smart:
        env.smart_ismcts_init()

    while not env.is_terminal():
        phase = env.phase()
        player = env.current_player()
        team = player % 2

        if phase == 0:
            action = env.bid_improved()
        elif team == q_team:
            if isinstance(engine, RustEngine):
                action = engine.action(env)
            else:
                action = engine.action(env, obs)
        else:
            if baseline == "random":
                legal = env.legal_actions()
                action = legal[np.random.randint(len(legal))]
            elif baseline == "naive":
                action = env.action_naive_ismcts(time_ms)
            else:
                action = env.action_smart_ismcts(time_ms)

        if use_smart:
            obs_list, _, done, _ = env.smart_ismcts_step(action)
        else:
            obs_list, _, done, _ = env.step(action)
        obs = np.array(obs_list, dtype=np.float32)

    rewards = env.rewards()
    outcome = env.deal_outcome()
    return rewards[q_team], rewards[1 - q_team], outcome[q_team], outcome[1 - q_team]


def play_match_to_2000(env, engine, q_team, baseline, time_ms, max_deals=50):
    """Play a match (first to 2000). Returns match result dict."""
    q_total, opp_total = 0.0, 0.0
    deal_count = 0
    for _ in range(max_deals):
        q_score, opp_score, _, _ = play_deal(env, engine, q_team, baseline, time_ms)
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


# --- Parallel worker ---

def _worker_match_batch(model_path, hidden, q_team, baseline, time_ms, count):
    """Worker function for parallel match evaluation. Each process gets its own Env."""
    env = colver.Env()
    env.load_dmc_model(model_path, hidden)
    engine = RustEngine(model_path, hidden)

    results = []
    for _ in range(count):
        result = play_match_to_2000(env, engine, q_team, baseline, time_ms)
        results.append(result)
    return results


def _worker_deal_batch(model_path, hidden, q_team, baseline, time_ms, count):
    """Worker function for parallel deal evaluation. Each process gets its own Env."""
    env = colver.Env()
    env.load_dmc_model(model_path, hidden)
    engine = RustEngine(model_path, hidden)

    results = []
    for _ in range(count):
        q_score, opp_score, q_out, opp_out = play_deal(
            env, engine, q_team, baseline, time_ms)
        results.append({"q_outcome": q_out, "opp_outcome": opp_out,
                         "q_reward": q_score, "opp_reward": opp_score})
    return results


# --- Evaluation modes ---

def run_deal_eval(env, engine, args):
    """Deal-level evaluation, with optional parallelism."""
    baseline_desc = args.baseline
    if args.baseline in ("naive", "smart"):
        baseline_desc += f" IS-MCTS ({args.time_ms}ms/move)"

    workers = args.workers
    use_parallel = workers > 1 and isinstance(engine, RustEngine)

    all_results = []
    sides = [0, 1] if args.both_sides else [0]

    for q_team in sides:
        team_name = "NS" if q_team == 0 else "EW"
        print(f"--- Q-network as {team_name} vs {baseline_desc}"
              f"{f' ({workers} workers)' if use_parallel else ''} ---")

        start = time.time()

        if use_parallel:
            per_worker = args.games // workers
            remainder = args.games % workers
            batch_sizes = [per_worker + (1 if i < remainder else 0) for i in range(workers)]
            batch_sizes = [b for b in batch_sizes if b > 0]

            side_results = []
            with ProcessPoolExecutor(max_workers=len(batch_sizes)) as pool:
                futures = [
                    pool.submit(_worker_deal_batch, engine.path, engine.hidden,
                                q_team, args.baseline, args.time_ms, bs)
                    for bs in batch_sizes
                ]
                for future in as_completed(futures):
                    side_results.extend(future.result())
            all_results.extend(side_results)
        else:
            side_results = []
            for i in range(args.games):
                q_score, opp_score, q_out, opp_out = play_deal(
                    env, engine, q_team, args.baseline, args.time_ms)
                r = {"q_outcome": q_out, "opp_outcome": opp_out,
                     "q_reward": q_score, "opp_reward": opp_score}
                side_results.append(r)
                all_results.append(r)

                if args.baseline in ("naive", "smart") and (i + 1) % 50 == 0:
                    elapsed = time.time() - start
                    wins = sum(1 for r in side_results if r["q_outcome"] > r["opp_outcome"])
                    print(f"    [{i + 1}/{args.games}] {wins / (i + 1):.1%} win, "
                          f"{elapsed:.0f}s ({(i + 1) / elapsed:.1f} games/s)")

        elapsed = time.time() - start
        n = len(side_results)
        wins = sum(1 for r in side_results if r["q_outcome"] > r["opp_outcome"])
        losses = sum(1 for r in side_results if r["q_outcome"] < r["opp_outcome"])
        draws = n - wins - losses
        total_margin = sum(r["q_reward"] - r["opp_reward"] for r in side_results)
        void_deals = sum(1 for r in side_results
                         if r["q_outcome"] == 0.5 and r["opp_outcome"] == 0.5)

        print(f"  Wins: {wins}/{n} ({wins / n:.1%}), "
              f"Losses: {losses}/{n} ({losses / n:.1%}), "
              f"Draws: {draws}/{n}")
        print(f"  Avg margin: {total_margin / n:+.1f}")
        print(f"  Void deals: {void_deals}/{n}")
        print(f"  Time: {elapsed:.1f}s ({n / elapsed:.1f} games/s)")
        print()

    if args.both_sides:
        total_games = len(all_results)
        total_wins = sum(1 for r in all_results if r["q_outcome"] > r["opp_outcome"])
        total_margin = sum(r["q_reward"] - r["opp_reward"] for r in all_results)
        print(f"=== Overall: {total_wins}/{total_games} ({total_wins / total_games:.1%}), "
              f"margin {total_margin / total_games:+.1f} ===")


def run_match_eval(env, engine, args):
    """Match-play evaluation (first to 2000), with optional parallelism."""
    baseline_desc = args.baseline
    if args.baseline in ("naive", "smart"):
        baseline_desc += f" IS-MCTS ({args.time_ms}ms/move)"

    workers = args.workers
    use_parallel = workers > 1 and isinstance(engine, RustEngine)
    if workers > 1 and not isinstance(engine, RustEngine):
        print("Warning: --workers > 1 requires Rust engine (.bin). Falling back to sequential.")
        use_parallel = False

    all_results = []
    sides = [0, 1] if args.both_sides else [0]

    for q_team in sides:
        team_name = "NS" if q_team == 0 else "EW"
        print(f"--- Match play: Q-network as {team_name} vs {baseline_desc} "
              f"({args.games} matches, first to 2000"
              f"{f', {workers} workers' if use_parallel else ''}) ---")

        start = time.time()

        if use_parallel:
            # Split games across workers
            per_worker = args.games // workers
            remainder = args.games % workers
            batch_sizes = [per_worker + (1 if i < remainder else 0) for i in range(workers)]
            batch_sizes = [b for b in batch_sizes if b > 0]

            side_results = []
            with ProcessPoolExecutor(max_workers=len(batch_sizes)) as pool:
                futures = [
                    pool.submit(_worker_match_batch, engine.path, engine.hidden,
                                q_team, args.baseline, args.time_ms, bs)
                    for bs in batch_sizes
                ]
                for future in as_completed(futures):
                    side_results.extend(future.result())

            all_results.extend(side_results)
            wins = sum(1 for r in side_results if r["q_won"])
            total_margin = sum(r["margin"] for r in side_results)
            total_deals = sum(r["deals"] for r in side_results)
        else:
            wins = 0
            total_margin = 0.0
            total_deals = 0

            for i in range(args.games):
                result = play_match_to_2000(env, engine, q_team,
                                            args.baseline, args.time_ms)
                all_results.append(result)

                if result["q_won"]:
                    wins += 1
                total_margin += result["margin"]
                total_deals += result["deals"]

                if (i + 1) % 5 == 0:
                    elapsed = time.time() - start
                    print(f"    [{i + 1}/{args.games}] {wins / (i + 1):.1%} win, "
                          f"margin {total_margin / (i + 1):+.0f}, "
                          f"{total_deals / (i + 1):.0f} deals/match, "
                          f"{elapsed:.0f}s ({(i + 1) / elapsed:.2f} matches/s)")

        elapsed = time.time() - start
        n = args.games
        print(f"  Wins: {wins}/{n} ({wins / n:.1%})")
        print(f"  Avg margin: {total_margin / n:+.0f}")
        print(f"  Avg deals/match: {total_deals / n:.1f}")
        print(f"  Time: {elapsed:.1f}s ({n / elapsed:.2f} matches/s)")
        print()

    total_matches = len(all_results)
    total_wins = sum(1 for r in all_results if r["q_won"])
    total_margin = sum(r["margin"] for r in all_results)
    total_deals = sum(r["deals"] for r in all_results)
    print(f"=== Overall: {total_wins}/{total_matches} ({total_wins / total_matches:.1%}), "
          f"margin {total_margin / total_matches:+.0f}, "
          f"{total_deals / total_matches:.1f} deals/match ===")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DMC agent")
    parser.add_argument("model", type=str,
                        help="Path to model (.bin for Rust inference, .pt for PyTorch)")
    parser.add_argument("--mode", type=str, default="deal",
                        choices=["deal", "match"],
                        help="Evaluation mode: deal (single deals) or match (first to 2000)")
    parser.add_argument("--games", type=int, default=500, help="Number of games/matches")
    parser.add_argument("--baseline", type=str, default="random",
                        choices=["random", "naive", "smart"],
                        help="Baseline opponent")
    parser.add_argument("--time-ms", type=int, default=20,
                        help="IS-MCTS time budget per move in ms")
    parser.add_argument("--both-sides", action="store_true",
                        help="Play both as NS and EW (doubles game count)")
    parser.add_argument("--hidden", type=int, default=1024,
                        help="Hidden layer size (default 1024)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (Rust engine only, default 1)")
    args = parser.parse_args()

    # Auto-detect engine from file extension
    if args.model.endswith(".bin"):
        engine = RustEngine(args.model, args.hidden)
        print(f"Engine: {engine.desc}")
    else:
        engine = TorchEngine(args.model, args.hidden)
        print(f"Device: {engine.device}")
        print(f"Engine: {engine.desc}")

    baseline_desc = args.baseline
    if args.baseline in ("naive", "smart"):
        baseline_desc += f" IS-MCTS ({args.time_ms}ms/move)"
    workers_desc = f", Workers: {args.workers}" if args.workers > 1 else ""
    print(f"Mode: {args.mode}, Baseline: {baseline_desc}, Games: {args.games}{workers_desc}")
    if args.both_sides:
        print("Playing both sides (NS and EW)")
    print()

    env = colver.Env()
    engine.load_into(env)

    if args.mode == "deal":
        run_deal_eval(env, engine, args)
    else:
        run_match_eval(env, engine, args)


if __name__ == "__main__":
    main()
