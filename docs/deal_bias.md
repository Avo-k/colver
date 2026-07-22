# Deal Bias: Traditional Gather-Cut Dealing vs Competition Shuffling

How biased are the deals when a table plays "à la maison" — tricks gathered into
team piles, one single cut, deal 3-3-2, never a real shuffle — compared to
competition practice (full shuffle before every deal)?

**TL;DR: the traditional procedure is measurably and strongly biased toward
distributional deals.** Voids are +42% more frequent (10.0% vs 7.0% of
suit-holdings), 5+ card suits +68% (1.95% vs 1.16%), and the bias fully installs
after a **single deal** — the deck reaches its stationary "clumped" regime
immediately and stays there for the whole match. Games are sharper: +8.7pp
coinche rate, +4.4pp chute rate, bigger per-deal score gaps, and matches end
~5% sooner. Surprisingly, *realized capots go down* (7.4% vs 8.6%): both sides
get wilder hands, so the defense can cut early rushes more often.

Binary: [`colver-core/src/bin/deal_bias.rs`](../colver-core/src/bin/deal_bias.rs)

```bash
cargo run --bin deal_bias --release -- --soirees 5000 [--mode both] [--csv out.csv]
```

## Protocol

A **soirée** = 3 matches to 2000 points. The deck is fully shuffled at the start
of each match (as at a real table when a game ends). Deals are played by the
champion bots — **Bid v6** (score-aware, fed the real cumulative match scores)
+ **DouDou50** — on all 4 seats, with dealer rotation, so trick composition and
auction dynamics are realistic. Both modes use identical bots; only deck
preparation between deals differs.

**Tradition mode** (between consecutive deals of a match):
1. Each team keeps one pile: every trick won is stacked on the team's pile in
   the order it was won (cards within a trick in play order).
2. The two team piles are stacked — coin flip for which ends on top.
3. One single cut, uniform position, at least 3 cards from either edge.
4. New dealer (rotation) deals 3-3-2 starting left of the dealer.
5. Passed-out deals: the four hands are tossed back suit-grouped in seat order,
   then cut and redealt.

**Shuffle mode**: full Fisher-Yates shuffle before every deal, then the same
3-3-2 deal.

Run: 5000 soirées per mode = 15 000 matches / mode, 148k (tradition) and 156k
(shuffle) deals. Errors below are match-level (cluster) standard errors — deals
within a match are correlated in tradition mode and must not be treated as iid.
Sanity check: shuffle-mode void rate is 7.05%, matching the theoretical uniform
value C(24,8)/C(32,8) ≈ 6.98% (and deal #0 of tradition matches, always freshly
shuffled, measures 7.00%).

## Hand structure

% of the 16 suit-holdings per deal (4 hands × 4 suits), by suit length:

| len       | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-----------|------|------|------|------|------|------|------|------|
| tradition | **9.93** | 25.25 | 32.03 | 22.66 | 8.20 | **1.71** | **0.217** | 0.010 |
| shuffle   | 6.98 | 26.30 | 35.89 | 22.64 | 7.05 | 1.07 | 0.071 | 0.001 |

The distribution polarizes: middling 2-card holdings are drained (−3.9pp) and
pushed to both extremes. A 6-card suit is **3× more likely** at a traditional
table; a 7-card suit 10× (still rare).

| metric (match-level mean ± SE) | tradition | shuffle | Δ |
|---|---|---|---|
| void suits | 9.996 ± 0.016 % | 7.047 ± 0.016 % | **+2.95pp (+42%)** |
| suits of 5+ cards | 1.954 ± 0.009 % | 1.160 ± 0.007 % | **+0.79pp (+68%)** |
| longest suit per hand | 3.4450 ± 0.0010 | 3.3178 ± 0.0009 | +0.127 |
| hands holding K+Q of a suit | 22.30 ± 0.05 % | 21.50 ± 0.05 % | +0.80pp |
| hand plain-points std | 9.91 | 10.31 | **−0.40** |

Why: a trick is mostly same-suit (everyone must follow), and the 3-card packets
of the 3-3-2 deal map those blocks into single hands. One cut is just a
rotation — it destroys almost nothing. The only entropy injected per deal is
the cut position (~5 bits) plus the pile coin flip.

Two second-order effects are more subtle:

- **Belote is slightly *more* frequent** (+0.8pp): K and Q of a suit often fall
  in the same trick (Q is played on K, or both dumped on a master), stay
  adjacent through the gather, and land in the same 3-packet.
- **Honor points are slightly *less* concentrated** (std 9.91 vs 10.31): a
  typical trick = one winner card + fillers, so each 3-packet tends to carry
  exactly one high card — suits clump, but points spread out.

## Effect on the game

| metric | tradition | shuffle | Δ |
|---|---|---|---|
| mean contract | 115.99 ± 0.05 | 116.41 ± 0.05 | −0.4 pts |
| coinche rate (of contracts) | 35.38 ± 0.13 % | 26.71 ± 0.11 % | **+8.7pp** |
| chute rate | 42.02 ± 0.13 % | 37.65 ± 0.13 % | **+4.4pp** |
| capot réalisé | 7.43 ± 0.07 % | 8.63 ± 0.07 % | **−1.2pp** |
| passed-out deals | 0.12 ± 0.01 % | 0.22 ± 0.01 % | −0.10pp |
| mean \|NS−EW\| gap per deal | 314.4 | 298.3 | +16 pts |
| deals per 2000-pt match | 9.88 | 10.41 | −0.53 |

The traditional game is sharper, not bigger: contract *levels* barely move, but
with wild hands on both sides opponents believe in their contre far more often
(+33% relative coinche rate) and are right more often (+4.4pp chutes). The
capot drop is the counter-intuitive one: long suits and voids cut both ways —
the defense holds voids too, so it trumps the declarer's rush more often.
Passed-out deals nearly halve: someone always has a suit worth talking about.

## Convergence: the bias installs in one single deal

Void-suit % by deal index within a match (deal 0 freshly shuffled in both
modes):

| idx | 0 | 1 | 2 | 3 | 4 | … | 12 |
|-----------|------|-------|-------|-------|-------|---|-------|
| tradition | 7.00 | 10.83 | 10.02 | 10.21 | 10.20 | … | 10.32 |
| shuffle   | 7.04 | 6.99  | 6.96  | 6.89  | 6.99  | … | 7.16  |

One played deal is enough to reach the stationary clumped regime (deal 1 even
slightly overshoots it). The bias then neither grows nor decays for the rest of
the match. So "we just shuffled at the start of the game" buys exactly one
clean deal.

## Caveats

- Trick composition depends on the playing policy; DouDou50's play is strong
  but not human. Human tables (more erratic discards) would change the *within
  trick* structure slightly, not the mechanism.
- Within-trick gather order and pile-stacking conventions are modeled simply
  (play order, coin flip). `--trick-order play|reverse|random` and `--min-cut`
  are available for sensitivity checks; the clumping mechanism (tricks =
  same-suit blocks + 3-3-2 packets + single cut) dominates these details.
- Bid v6 was trained on uniformly dealt pools. Facing a clumpier deal
  distribution, its calibration (e.g. contre thresholds) is slightly out of
  distribution; direction of the reported effects should be robust, exact
  percentages less so.

## Files

- Binary: `colver-core/src/bin/deal_bias.rs` (no extra features needed, plain
  `--release`)
- Per-match CSV: `--csv path` (mode, deals, voids, contracts, coinches, chutes,
  capots… one row per match) for custom analysis.
