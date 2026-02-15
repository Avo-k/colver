#!/usr/bin/env python3
"""DouZero-style Deep Monte-Carlo training for Belote Contree card play.

Trains a Q-network to play cards using binary deal outcomes as targets.
Bidding is handled by improved_bid (not learned).

Usage:
    uv run python scripts/train_dmc.py
    uv run python scripts/train_dmc.py --num-envs 256 --steps 1000000
    uv run python scripts/train_dmc.py --resume models/dmc_latest.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import colver
from dmc_model import QNetwork, ReplayBuffer, _sample_legal

NUM_CARDS = 32


def evaluate(q_net: QNetwork, device: torch.device, num_games: int = 200) -> dict:
    """Evaluate Q-network vs random play.

    Q-network plays team 0 (NS), random plays team 1 (EW).
    Returns win rate and average margin.
    """
    q_net.eval()
    env = colver.Env()
    wins = 0
    total_margin = 0.0
    completed = 0

    for _ in range(num_games):
        obs_list, _ = env.reset()
        obs = np.array(obs_list, dtype=np.float32)

        while not env.is_terminal():
            phase = env.phase()
            player = env.current_player()
            team = player % 2

            if phase == 0:  # Bidding
                action = env.bid_improved()
            elif team == 0:  # Playing, Q-network team
                mask_full = np.array(env.legal_action_mask(), dtype=np.float32)
                mask = mask_full[:NUM_CARDS]
                n_legal = int(mask.sum())
                if n_legal <= 1:
                    action = env.legal_actions()[0]
                else:
                    obs_t = torch.tensor(obs, device=device).unsqueeze(0)
                    mask_t = torch.tensor(mask, device=device).unsqueeze(0)
                    with torch.no_grad():
                        q = q_net(obs_t)
                        q[mask_t == 0] = -1e9
                        action = q.argmax(dim=1).item()
            else:  # Playing, random team
                legal = env.legal_actions()
                action = legal[np.random.randint(len(legal))]

            obs_list, _, done, _ = env.step(action)
            obs = np.array(obs_list, dtype=np.float32)

        outcome = env.deal_outcome()
        if outcome[0] > outcome[1]:
            wins += 1
        total_margin += outcome[0] - outcome[1]
        completed += 1

    q_net.train()
    return {
        "win_rate": wins / max(completed, 1),
        "margin": total_margin / max(completed, 1),
        "games": completed,
    }


def main():
    parser = argparse.ArgumentParser(description="DMC training for Belote card play")
    parser.add_argument("--num-envs", type=int, default=256, help="Parallel environments")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total env steps")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eps-start", type=float, default=0.3, help="Initial exploration rate")
    parser.add_argument("--eps-end", type=float, default=0.02, help="Final exploration rate")
    parser.add_argument("--eps-decay-steps", type=int, default=500_000, help="Epsilon decay steps")
    parser.add_argument("--buffer-size", type=int, default=500_000, help="Replay buffer capacity")
    parser.add_argument("--min-buffer", type=int, default=5_000, help="Min buffer size before training")
    parser.add_argument("--train-freq", type=int, default=4, help="Train every N env steps")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Evaluate every N steps")
    parser.add_argument("--eval-games", type=int, default=200, help="Games per evaluation")
    parser.add_argument("--save-freq", type=int, default=100_000, help="Save checkpoint every N steps")
    parser.add_argument("--save-dir", type=str, default="models", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Envs: {args.num_envs}, Steps: {args.steps}, LR: {args.lr}")

    # Initialize
    q_net = QNetwork().to(device)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    replay_buffer = ReplayBuffer(capacity=args.buffer_size)
    param_count = sum(p.numel() for p in q_net.parameters())
    print(f"Q-Network: 213 -> 512 -> 512 -> 512 -> 32 ({param_count:,} params)")

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        q_net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {args.resume} at step {start_step}")

    # VecEnv
    venv = colver.VecEnv(args.num_envs)
    obs, masks = venv.reset()
    obs = np.array(obs)      # (N, 213)
    masks = np.array(masks)   # (N, 43)

    # Episode buffers: per-env list of (obs, mask32, action, team_idx)
    episode_bufs: list[list] = [[] for _ in range(args.num_envs)]

    # Stats
    total_transitions = 0
    total_episodes = 0
    total_train_steps = 0
    total_loss = 0.0
    loss_count = 0
    step_start = time.time()
    last_log_step = start_step

    print(f"\n{'Step':>10} | {'Eps':>5} | {'Buffer':>7} | {'Loss':>8} | {'Episodes':>8} | {'Trans':>7} | {'Steps/s':>7}")
    print("-" * 80)

    for step in range(start_step, args.steps):
        # Epsilon schedule
        progress = min(1.0, (step - start_step) / args.eps_decay_steps)
        eps = args.eps_start + (args.eps_end - args.eps_start) * progress

        phases = np.array(venv.phases())
        players = np.array(venv.current_players())
        actions = np.zeros(args.num_envs, dtype=np.uint8)

        # --- Bidding envs: use improved_bid ---
        bid_envs = (phases == 0)
        if bid_envs.any():
            bid_actions = np.array(venv.bid_improved())
            actions[bid_envs] = bid_actions[bid_envs]

        # --- Playing envs: Q-network epsilon-greedy ---
        play_envs = (phases == 1)
        if play_envs.any():
            play_idx = np.where(play_envs)[0]
            play_obs = obs[play_idx]
            play_masks = masks[play_idx][:, :NUM_CARDS]

            # Q-network forward pass
            obs_t = torch.tensor(play_obs, dtype=torch.float32, device=device)
            mask_t = torch.tensor(play_masks, dtype=torch.float32, device=device)
            with torch.no_grad():
                q = q_net(obs_t)
                q[mask_t == 0] = -1e9
                greedy = q.argmax(dim=1).cpu().numpy()

            # Epsilon-greedy
            random_actions = _sample_legal(play_masks)
            use_random = np.random.rand(len(play_idx)) < eps
            chosen = np.where(use_random, random_actions, greedy).astype(np.uint8)
            actions[play_idx] = chosen

            # Store transitions (only play-phase, skip forced moves)
            teams = players[play_idx] % 2  # 0=NS, 1=EW
            n_legal = play_masks.sum(axis=1)
            for local_i, env_i in enumerate(play_idx):
                if n_legal[local_i] > 1:
                    episode_bufs[env_i].append((
                        play_obs[local_i].copy(),
                        play_masks[local_i].copy(),
                        int(chosen[local_i]),
                        int(teams[local_i]),
                    ))

        # --- Done envs (phase == 2): shouldn't happen with auto-reset, but handle ---
        done_envs = (phases == 2)
        if done_envs.any():
            # These are stuck in Done state — give them action 0 (will be ignored by reset)
            actions[done_envs] = 0

        # --- Step all envs ---
        obs, _rewards, dones, masks, outcomes = venv.step(actions.tolist())
        obs = np.array(obs)
        masks = np.array(masks)
        dones = np.array(dones)
        outcomes = np.array(outcomes)  # (N, 2): [NS_outcome, EW_outcome]

        # --- Flush completed episodes ---
        done_idx = np.where(dones)[0]
        for env_i in done_idx:
            buf = episode_bufs[env_i]
            if len(buf) > 0:
                ns_outcome = outcomes[env_i, 0]
                ew_outcome = outcomes[env_i, 1]
                # Batch push to replay buffer
                n_trans = len(buf)
                b_obs = np.array([t[0] for t in buf], dtype=np.float32)
                b_masks = np.array([t[1] for t in buf], dtype=np.float32)
                b_actions = np.array([t[2] for t in buf], dtype=np.int64)
                b_teams = np.array([t[3] for t in buf], dtype=np.int64)
                # Each transition's return depends on its team
                b_returns = np.where(
                    b_teams == 0, ns_outcome, ew_outcome
                ).astype(np.float32)
                replay_buffer.push_batch(b_obs, b_masks, b_actions, b_returns)
                total_transitions += n_trans
            episode_bufs[env_i] = []
            total_episodes += 1

        # --- Train ---
        if (replay_buffer.size >= args.min_buffer
                and step % args.train_freq == 0):
            b_obs, b_masks, b_actions, b_returns = replay_buffer.sample(args.batch_size)
            b_obs = b_obs.to(device)
            b_masks = b_masks.to(device)
            b_actions = b_actions.to(device).long()
            b_returns = b_returns.to(device)

            # Q-values for taken actions
            q_values = q_net(b_obs)
            q_taken = q_values.gather(1, b_actions.unsqueeze(1)).squeeze(1)

            loss = F.mse_loss(q_taken, b_returns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_count += 1
            total_train_steps += 1

        # --- Logging ---
        if (step + 1) % 10_000 == 0:
            elapsed = time.time() - step_start
            steps_done = step + 1 - last_log_step
            sps = steps_done / max(elapsed, 1e-6)
            avg_loss = total_loss / max(loss_count, 1)
            print(
                f"{step + 1:>10,} | {eps:>5.3f} | {replay_buffer.size:>7,} | "
                f"{avg_loss:>8.4f} | {total_episodes:>8,} | {total_transitions:>7,} | "
                f"{sps:>7.0f}"
            )
            total_loss = 0.0
            loss_count = 0
            step_start = time.time()
            last_log_step = step + 1

        # --- Evaluate ---
        if (step + 1) % args.eval_freq == 0:
            result = evaluate(q_net, device, num_games=args.eval_games)
            print(
                f"  [EVAL] Win rate: {result['win_rate']:.1%}, "
                f"Margin: {result['margin']:+.2f} "
                f"({result['games']} games)"
            )

        # --- Save checkpoint ---
        if (step + 1) % args.save_freq == 0:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / f"dmc_{step + 1}.pt"
            torch.save({
                "model": q_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
            }, ckpt_path)
            # Also save as latest
            latest_path = save_dir / "dmc_latest.pt"
            torch.save({
                "model": q_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
            }, latest_path)
            print(f"  [SAVE] {ckpt_path}")

    # Final eval and save
    print("\n--- Final Evaluation ---")
    result = evaluate(q_net, device, num_games=args.eval_games)
    print(f"Win rate vs random: {result['win_rate']:.1%}, Margin: {result['margin']:+.2f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    final_path = save_dir / "dmc_final.pt"
    torch.save({
        "model": q_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": args.steps,
    }, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
