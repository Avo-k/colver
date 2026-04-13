"""Baseline: bid_v2 (NS) vs bid_v2 (EW) to establish what 'good' looks like."""
import numpy as np
import torch

import sys
sys.path.insert(0, "scripts")

from bumblebid.data import load_bumblebid_pool, load_dd_pool, BiddingEnv, compute_deal_score
from bumblebid.opponent import BidNetV2, build_bid_obs_batch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pool
    dealers_bb, hp, hs, dd = load_bumblebid_pool("data/pools/bumblebid_2.5M.bin")
    dealers_dd, hands, dd_pts = load_dd_pool("data/pools/dd_2.5M.bin")

    # Load bid_v2
    model = BidNetV2.load_from_bin("models/bid_v2/bid_nn_final.bin").to(device)
    model.eval()

    rng = np.random.default_rng(99)
    n_deals = 5000
    total_reward = 0.0
    n_void = 0
    n_ns_take = 0
    n_ew_take = 0
    ns_contracts = []
    ew_contracts = []
    ns_made = 0
    ew_made = 0

    env = BiddingEnv()

    for _ in range(n_deals):
        idx = rng.integers(len(dealers_dd))
        dealer = int(dealers_dd[idx])
        dd_p = dd_pts[idx].tolist()

        env.dealer = dealer
        env.dd_pts = dd_p
        env.bid_history = []
        env.current_player_seat = (dealer + 1) % 4
        env.consecutive_passes = 0
        env.current_bid = None
        env.coinche_level = 0
        env.done = False

        while not env.done:
            seat = env.current_player_seat
            mask = env.legal_actions_mask()

            obs = build_bid_obs_batch(
                hands[idx:idx+1],
                np.array([seat]),
                np.array([dealer]),
                [env.bid_history],
                np.array([0]),
            )
            obs_t = torch.from_numpy(obs).to(device)
            m_t = torch.from_numpy(mask).unsqueeze(0).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                q = model(obs_t)
            q = q.float().masked_fill(m_t == 0, -1e9)
            action = q.argmax(dim=-1).item()

            env.step(action)

        if env.current_bid is None:
            n_void += 1
        else:
            val_enc, suit, taker_team = env.current_bid
            cv = val_enc * 10
            if taker_team == 0:
                n_ns_take += 1
                ns_contracts.append(cv)
            else:
                n_ew_take += 1
                ew_contracts.append(cv)

            # Check if contract was made
            ns_pts = int(dd_p[suit])
            taker_pts = ns_pts if taker_team == 0 else (252 - ns_pts if ns_pts in (0, 252) else 162 - ns_pts)
            if cv == 250:
                made = (taker_pts == 252)
            else:
                made = (taker_pts >= cv)
            if taker_team == 0 and made:
                ns_made += 1
            elif taker_team == 1 and made:
                ew_made += 1

        ns_r, _ = env.compute_reward()
        total_reward += ns_r

    mean_r = total_reward / n_deals
    void_pct = 100.0 * n_void / n_deals
    ns_take_pct = 100.0 * n_ns_take / n_deals
    ew_take_pct = 100.0 * n_ew_take / n_deals
    ns_made_pct = 100.0 * ns_made / max(n_ns_take, 1)
    ew_made_pct = 100.0 * ew_made / max(n_ew_take, 1)
    ns_mean_cv = np.mean(ns_contracts) if ns_contracts else 0
    ew_mean_cv = np.mean(ew_contracts) if ew_contracts else 0

    print(f"=== Baseline: bid_v2 vs bid_v2 ({n_deals} deals) ===")
    print(f"  Mean NS reward: {mean_r:+.4f}")
    print(f"  Void: {void_pct:.1f}%")
    print(f"  NS takes: {ns_take_pct:.1f}% (avg cv={ns_mean_cv:.0f}, made={ns_made_pct:.1f}%)")
    print(f"  EW takes: {ew_take_pct:.1f}% (avg cv={ew_mean_cv:.0f}, made={ew_made_pct:.1f}%)")
    print()
    print("A trained Bumblebid should aim for:")
    print(f"  - Reward near {mean_r:+.4f} (balanced) or better")
    print(f"  - NS taking contracts ~{ns_take_pct:.0f}% of deals with ~{ns_made_pct:.0f}% make rate")
    print(f"  - Low void rate (~{void_pct:.0f}%)")

if __name__ == "__main__":
    main()
