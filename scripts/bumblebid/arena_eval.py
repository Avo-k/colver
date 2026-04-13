"""Arena-style evaluation: Bumblebid (NS bid) vs bid_v2 (EW bid), full games.

Both teams use the same play engine (DMC DouDou50) so the only difference is bidding.
Runs N matches with duplicate dealing (same hands, swapped sides) for variance reduction.
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "scripts")

from bumblebid.model import Bumblebid, MAX_SEQ_LEN, P_NONE
from bumblebid.data import (
    ACTION_PRIMARY_LUT, ACTION_SUIT_LUT, ACTION_SUIT2_LUT,
)

from colver import Env


def load_bumblebid(path, d_model, n_layers, n_heads, device):
    model = Bumblebid(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


@torch.no_grad()
def bumblebid_action(model, env, device):
    """Get Bumblebid's bid action for the current state."""
    seat = env.current_player()
    dealer = env.get_dealer()
    bid_history = env.get_bid_history()

    # Get hand from env — get_hands() returns list of 8 card indices per seat
    hands = env.get_hands()
    hand_cards_list = hands[seat]

    # Build tokens: [CLS] [POS] [8 cards] [bid tokens...]
    from bumblebid.model import P_CLS, P_POS0, P_RANK0, S_NULL

    cards = [(c % 8, c // 8) for c in hand_cards_list]  # (rank, suit)
    cards.sort(key=lambda c: c[1] * 8 + c[0])

    pos = (seat + 4 - dealer) % 4
    primary = [P_CLS, P_POS0 + pos]
    suits = [S_NULL, S_NULL]

    for rank, suit in cards:
        primary.append(P_RANK0 + rank)
        suits.append(suit)

    for _seat, action in bid_history:
        if len(primary) + 2 > MAX_SEQ_LEN:
            break
        primary.extend([int(ACTION_PRIMARY_LUT[action]), P_NONE])
        suits.extend([int(ACTION_SUIT_LUT[action]), int(ACTION_SUIT2_LUT[action])])

    seq_len = len(primary)
    while len(primary) < MAX_SEQ_LEN:
        primary.append(0)
        suits.append(0)

    p_t = torch.tensor(primary[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
    s_t = torch.tensor(suits[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)

    mask = env.legal_action_mask()
    m_t = torch.from_numpy(mask[:43]).unsqueeze(0).to(device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(p_t, s_t)
    logits = logits.float().masked_fill(m_t == 0, -1e9)
    return logits.argmax(dim=-1).item()


def play_game(env, bumblebid_model, bumblebid_team, device, play_method, play_time_ms):
    """Play a full game. bumblebid_team: 0=NS, 1=EW."""
    # Bidding phase
    while env.phase() == 0 and not env.is_terminal():
        seat = env.current_player()
        team = seat % 2

        if team == bumblebid_team:
            action = bumblebid_action(bumblebid_model, env, device)
        else:
            result = env.action_bid_nn()
            action = result["best_action"]

        env.step(action)

    # Playing phase
    while not env.is_terminal():
        if play_method == "dmc":
            result = env.action_dmc_with_stats()
            action = result["best_action"]
        elif play_method == "heuristic":
            action = env.action_heuristic_play()
        else:
            action = env.action_smart_ismcts(play_time_ms)

        env.step(action)

    return env.rewards()


def make_env(bid_model, play_model, play_method):
    """Create and configure an Env with models loaded."""
    env = Env()
    env.reset()
    env.load_bid_model(bid_model, 512)
    if play_method == "dmc" and play_model:
        env.load_dmc_model(play_model)
    return env


def main():
    p = argparse.ArgumentParser(description="Arena eval: Bumblebid vs bid_v2")
    p.add_argument("--model", required=True, help="Bumblebid .pt checkpoint")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--bid-model", default="models/bid_v2/bid_nn_final.bin",
                   help="bid_v2 .bin weights for opponent")
    p.add_argument("--play-model", default=None,
                   help="DMC play model (required if --play-method=dmc)")
    p.add_argument("--play-method", default="heuristic", choices=["dmc", "heuristic", "ismcts"])
    p.add_argument("--play-time-ms", type=int, default=20)
    p.add_argument("--matches", type=int, default=200,
                   help="Number of deals (each played twice with sides swapped)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_bumblebid(args.model, args.d_model, args.n_layers, args.n_heads, device)
    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"Bumblebid: {n_params:,} params  d={args.d_model} L={args.n_layers} H={args.n_heads}")
    print(f"Play: {args.play_method}" + (f" ({args.play_model})" if args.play_model else ""))
    print(f"Opponent bid: {args.bid_model}")
    print(f"Matches: {args.matches} x 2 (duplicate)")

    bb_wins = 0
    v2_wins = 0
    bb_total = 0.0
    v2_total = 0.0
    n_played = 0

    t0 = time.time()

    for i in range(args.matches):
        # Direction A: Bumblebid = NS
        env_a = make_env(args.bid_model, args.play_model, args.play_method)
        hands_a = env_a.get_hands()
        dealer_a = env_a.get_dealer()

        ns_a, ew_a = play_game(env_a, model, 0, device, args.play_method, args.play_time_ms)

        # Direction B: same hands, Bumblebid = EW
        env_b = Env.deal_with_hands(dealer_a, hands_a)
        env_b.load_bid_model(args.bid_model, 512)
        if args.play_method == "dmc" and args.play_model:
            env_b.load_dmc_model(args.play_model)

        ns_b, ew_b = play_game(env_b, model, 1, device, args.play_method, args.play_time_ms)

        # Bumblebid score = NS in game A + EW in game B
        bb_score = ns_a + ew_b
        v2_score = ew_a + ns_b

        bb_total += bb_score
        v2_total += v2_score
        if bb_score > v2_score:
            bb_wins += 1
        elif v2_score > bb_score:
            v2_wins += 1

        n_played += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            avg_bb = bb_total / n_played
            avg_v2 = v2_total / n_played
            print(f"  [{i+1}/{args.matches}] BB={bb_wins} V2={v2_wins} "
                  f"avg_diff={avg_bb - avg_v2:+.1f} "
                  f"({elapsed:.0f}s, {n_played/elapsed:.1f} deals/s)")

    elapsed = time.time() - t0
    avg_bb = bb_total / n_played
    avg_v2 = v2_total / n_played
    diff = avg_bb - avg_v2

    print(f"\n{'='*50}")
    print(f"Results: {n_played} duplicate matches")
    print(f"  Bumblebid wins: {bb_wins} ({100*bb_wins/n_played:.1f}%)")
    print(f"  bid_v2 wins:    {v2_wins} ({100*v2_wins/n_played:.1f}%)")
    print(f"  Draws:          {n_played - bb_wins - v2_wins}")
    print(f"  Avg BB score:   {avg_bb:.1f}")
    print(f"  Avg V2 score:   {avg_v2:.1f}")
    print(f"  Avg diff:       {diff:+.1f} pts/match")
    print(f"  Time: {elapsed:.0f}s ({n_played/elapsed:.1f} deals/s)")


if __name__ == "__main__":
    main()
