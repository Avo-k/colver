"""Do playgen worlds make IS-DD play stronger than uniform worlds, at equal
determinization count?

The two bots differ in **one line** of their spec — `[worlds] source` — so any
difference is the world sampler and nothing else.

Paired design (duplicate matching at the deal level): each deal is played twice
with the same deterministic bidder, once with playgen as NS and once as EW.
Averaging playgen's net points over the two cancels deal luck, which otherwise
swamps the effect at any sample size we can afford.

    export COLVER_PLAYGEN_GPU_URL=http://localhost:8003
    uv run python scripts/analysis/playgen_vs_uniform.py --dets 32 --deals 500

Result (2026-07-24, 500 deals @ 32 dets): playgen +10.3 pts/deal,
95% CI [+0.8, +19.7] — significant. The edge is in reaching a good decision
with *fewer* worlds, which is exactly the regime production runs in.
"""

import argparse
import math
import os

import colver

BID, DONE = 0, 2

SPEC = """
[bid]
strategy = "improved_v2"

[play]
method = "isdd"
time_ms = 0
determinizations = {dets}

[worlds]
source = "{source}"
temperature = {temp}
"""


def seat_players(dets, temp, playgen_team):
    """Four agents; `playgen_team` (0=NS, 1=EW) samples worlds from playgen."""
    specs = {
        team: SPEC.format(
            dets=dets,
            temp=temp,
            source="sidecar" if team == playgen_team else "uniform",
        )
        for team in (0, 1)
    }
    return [colver.Agent(specs[seat % 2], seat) for seat in range(4)]


def play_deal(dealer, hands, dets, temp, playgen_team):
    """One full deal. Returns (ns_points, ew_points)."""
    env = colver.Env.deal_with_hands(dealer, [list(h) for h in hands])
    agents = seat_players(dets, temp, playgen_team)
    for a in agents:
        a.init_deal(env)

    while not env.is_terminal():
        seat = int(env.current_player())
        action = int(agents[seat].decide(env)["action"])
        for a in agents:
            a.observe(env, action)
        env.step(action)

    ns, ew = env.rewards()
    return float(ns), float(ew)


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p * 100, (c - h) * 100, (c + h) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", type=int, default=32)
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    args = ap.parse_args()

    if not os.environ.get("COLVER_PLAYGEN_GPU_URL"):
        raise SystemExit(
            "Set COLVER_PLAYGEN_GPU_URL to a running playgen_gpu_server — "
            "without it the 'playgen' side would not exist."
        )

    diffs = []  # playgen's net points per deal, averaged over the two seatings
    wins = ties = 0
    for i in range(args.deals):
        seed_env = colver.Env()
        seed_env.reset()
        dealer = int(seed_env.get_dealer())
        hands = [list(h) for h in seed_env.get_hands()]

        nets = []
        for playgen_team in (0, 1):
            ns, ew = play_deal(dealer, hands, args.dets, args.temp, playgen_team)
            nets.append((ns - ew) if playgen_team == 0 else (ew - ns))

        d = (nets[0] + nets[1]) / 2.0
        diffs.append(d)
        if d > 0:
            wins += 1
        elif d == 0:
            # Both seatings reached the same score: usually a passed-out deal or
            # a contract both samplers play identically. Common, and not a draw
            # in any meaningful sense — just no signal from this deal.
            ties += 1
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{args.deals}] mean playgen net = "
                  f"{sum(diffs)/len(diffs):+.1f} pts/deal", flush=True)

    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    winp, lo, hi = wilson(wins, n)

    print("\n===== RESULT =====")
    print(f"dets={args.dets}  deals={n} (paired, 2 games each)  temp={args.temp}")
    print(f"playgen net points/deal: {mean:+.1f}  "
          f"95% CI [{mean-1.96*se:+.1f}, {mean+1.96*se:+.1f}]")
    print(f"playgen deal win-rate:   {winp:.1f}%  95% CI [{lo:.1f}, {hi:.1f}]  "
          f"({wins}W/{ties}T/{n-wins-ties}L)")
    significant = (mean - 1.96 * se) > 0 or (mean + 1.96 * se) < 0
    print(f"=> playgen {'STRONGER' if mean > 0 else 'WEAKER'} than uniform "
          f"({'significant' if significant else 'NOT significant'} at 95%)")


if __name__ == "__main__":
    main()
