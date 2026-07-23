#!/usr/bin/env python3
"""Benchmark de crédibilité des mondes échantillonnés en phase d'enchère.

Principe (auto-supervisé, proposé par l'utilisateur 2026-07-23) : on génère des
positions mid-enchère réalistes (auctions bid v6 self-play), on échantillonne K
mondes par sampler (playgen conditionné vs uniforme), puis on demande au juge
(bid v6) si, avec la main cachée que le monde attribue à chaque bidder observé,
il rejouerait l'annonce observée. Un bon posterior doit être cohérent avec le
processus qui a généré l'enchère.

Usage:
  uv run python scripts/analysis/bench_world_credibility.py \
      [--positions 30] [--worlds 12] [--seed 42] \
      [--playgen models/playgen_v2/playgen_v2_half.bin]

Résultats de référence (playgen v2 @60K, 30 positions, 564 jugements) :
  playgen : argmax 59%  top3 91%
  uniform : argmax  9%  top3 27%
"""
import argparse
import random
import time

from colver import Env

BID_MODEL = "models/bid_v6_isdd.bin"


def fresh_deal(rng):
    cards = list(range(32))
    rng.shuffle(cards)
    return [sorted(cards[i * 8:(i + 1) * 8]) for i in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=30)
    ap.add_argument("--worlds", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--playgen", default="models/playgen_v2/playgen_v2_half.bin")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stats = {s: {"argmax": 0, "top3": 0, "n": 0} for s in ("playgen", "uniform")}
    t_start = time.time()
    pos_done = 0

    while pos_done < args.positions:
        dealer = rng.randrange(4)
        hands = fresh_deal(rng)
        env = Env.deal_with_hands(dealer, hands)
        env.load_bid_model(BID_MODEL)
        n_steps = rng.randrange(2, 5)
        actions = []
        ok = True
        for _ in range(n_steps):
            a = env.action_bid_nn()["best_action"]
            actions.append(a)
            env.step(a)
            if env.is_terminal() or env.phase() != 0:
                ok = False
                break
        if not ok or not any(a > 0 for a in actions):
            continue
        observer = rng.randrange(4)
        speakers = [(dealer + 1 + i) % 4 for i in range(len(actions))]
        targets = [(i, sp, a) for i, (sp, a) in enumerate(zip(speakers, actions))
                   if a > 0 and sp != observer]
        if not targets:
            continue

        penv = Env.deal_with_hands(dealer, hands)
        penv.load_playgen_model(args.playgen)
        penv.dede_init()
        for a in actions:
            penv.dede_step(a)
        pg_worlds = penv.playgen_sample_auction_deals(
            observer, args.worlds, args.temperature) or []

        obs_hand = hands[observer]
        rest = [c for c in range(32) if c not in obs_hand]
        un_worlds = []
        for _ in range(args.worlds):
            r = list(rest)
            rng.shuffle(r)
            w = [None] * 4
            w[observer] = obs_hand
            for j, p in enumerate([x for x in range(4) if x != observer]):
                w[p] = sorted(r[j * 8:(j + 1) * 8])
            un_worlds.append(w)

        for label, worlds in (("playgen", pg_worlds), ("uniform", un_worlds)):
            for w in worlds:
                e2 = Env.deal_with_hands(dealer, [list(map(int, h)) for h in w])
                e2.load_bid_model(BID_MODEL)
                ti = 0
                for i, a in enumerate(actions):
                    if ti < len(targets) and targets[ti][0] == i:
                        r = e2.action_bid_nn()
                        qs = sorted(r["q_values"], key=lambda x: -x[1])
                        rank = next(
                            (k for k, (act, _) in enumerate(qs) if act == a), 99)
                        stats[label]["argmax"] += rank == 0
                        stats[label]["top3"] += rank < 3
                        stats[label]["n"] += 1
                        ti += 1
                    e2.step(a)
        pos_done += 1

    dt = time.time() - t_start
    print(f"{pos_done} positions, {dt:.0f}s")
    for label, s in stats.items():
        n = max(1, s["n"])
        print(f"{label:8s}: argmax {s['argmax']/n*100:.0f}%  "
              f"top3 {s['top3']/n*100:.0f}%  (n={s['n']} jugements)")


if __name__ == "__main__":
    main()
