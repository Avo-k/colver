# Auction-conditioned bid labels — negative result (2026-07-25)

**Outcome: the distilled bidder is ~2–3 pp weaker than v6. Do not ship it.**
The pipeline is sound and reusable; the *target definition* is wrong. This documents
what was built, what was measured, and the one change that would make it worth
retrying.

## The idea

A bid model is graded on the deal that was actually dealt, so its target carries the
noise of the 24 cards the bidder cannot see. Measured on 150 real deals
([bench_label_variance.rs](../../../colver-core/src/bin/bench_label_variance.rs)):

```
Total sd of the raw label (dd_pts):          50 pts
  explained by the visible hand:             29 pts
  noise from the 24 unseen cards:            45 pts   → 70% of the variance
```

Playgen can sample deals conditioned on the auction prefix, so the label can be the
conditional expectation instead of one draw from it. Measured on 240 real bid
positions from a held-out corpus, 40 worlds each
([bench_bid_label_cond.rs](../../../colver-core/src/bin/bench_bid_label_cond.rs)):

```
                          uniform    playgen
posterior spread (sd)        44.6       34.9   pts   → -21.6%
RMSE vs true dd_pts          46.7       36.7   pts   → -21.5%
```

Both fall by the same amount, so playgen narrows the posterior *around the right
place* rather than becoming overconfident. This looked like a clean win.

## What was built

| Binary | Role |
|---|---|
| [gen_bid_labels.rs](../../../colver-core/src/bin/gen_bid_labels.rs) | Playgen worlds given the auction prefix → DD-solve 4 trumps → COLVQL02 |
| [train_bid_distill.rs](../../../colver-core/src/bin/train_bid_distill.rs) | Rebuilds the 117-dim obs, derives Δ-winprob targets, fine-tunes v6 |
| [gen_dd_calibration.rs](../../../colver-core/src/bin/gen_dd_calibration.rs) | `E[isdd \| dd]` table from `base_5M` + `scores_isdd_5M` |

**COLVQL02 stores raw per-world points, not a mean**, because contract scoring is a
threshold in card points — the expected reward is `E[f(pts)]`, not `f(E[pts])`.
Belote is stored per world too: v6's reward credits Q+K of trump (+20) and it lands
on some team in ~23% of (world, suit) pairs, and it is a property of the sampled
hands so it cannot be recovered from the points.

**Data:** 107,990 bid positions (40k deals local + 14k on moxxi, 8 worlds each,
~2h45 wall across both machines). Corpus: 120k fresh v6/DouDou50 games, seed 77001.

## Results

Arena vs `v6_isdd_75M`, play side byte-identical so the delta is purely the bidder.
Control: v6 against itself scores 49.2% over 800 matches.

| Target treatment | argmax=PASS | Arena vs v6 |
|---|---|---|
| raw | 32.1% (v6: 57.0%) | **23–26%** |
| global re-centre | 58.3% | 40–47% |
| per-level re-centre | 44.6% | **47.3%** (3000 matches) |
| DD→IS-DD calibrated | 46.8% | 39–41% |
| calibrated + per-level | 54.5% | 34–41% |

Every variant loses. The best is per-level re-centring at 47.3% ± 0.9.

## Why it fails

**The label is not the quantity the bidder needs.** The target was built as "my team
takes this contract and it is played out". The Q-value of *bidding* 110♥ is not the
value of *playing* 110♥ — it is the value of the auction continuing from there, which
includes partner raising and opponents overcalling. A cheap bad bid gets overcalled in
reality; my target charges the bidder for playing it. This inflates bids relative to
PASS, and the raw model bids where v6 passes on a quarter of all positions.

The bias is **level-dependent** — badly wrong at 80, nearly right at 160 — which is
why a constant per-position shift cannot remove it and only per-level re-centring got
close to parity.

A second, smaller bias: DD assumes perfect play while v6 was trained on IS-DD rollout
points (`--reward real`). The calibration table quantifies it — at dd=120 IS-DD only
takes 112, at dd=140 only 125, enough to flip réussi/chute.

**The calibration attempt made things worse, and that is instructive.** Replacing each
world's points by a conditional *mean* collapses the between-world variance, so the
réussi/chute threshold becomes deterministic — reintroducing exactly the Jensen error
that per-world storage exists to avoid. Calibrating this way needs sampling from
`P(isdd | dd)`, not substituting its mean.

## Attempt 2: auction-continuation targets — also fails

The obvious fix was implemented and measured. Labels were regenerated storing the
**world hands** (COLVQL03), and [train_bid_cont.rs](../../../colver-core/src/bin/train_bid_cont.rs)
evaluates every candidate action identically: force the action, let v6 finish the
auction on all four seats inside the sampled world, score whatever contract results.
PASS goes through that same path, so no anchor and no re-centring are needed. Final
contracts are scored by *sampling* from `P(isdd | dd)` (64-quantile table from
[gen_dd_calibration.rs](../../../colver-core/src/bin/gen_dd_calibration.rs)) with common
random numbers per world, so candidate actions are compared on a paired basis.

| Variant | argmax=PASS | Arena vs v6 |
|---|---|---|
| continuation + sampled calibration, uniform match scores | 40.7% (v6: 57.0%) | 43.1–44.0% |
| same, realistic correlated match scores | 41.3% (v6: 56.4%) | 41.6–44.6% |

Worse than attempt 1's best (47.3%). **The bid/pass gap survived every fix**, and the
arena result tracks that gap across all seven variants tried: the closer a target's
bid/pass balance sits to v6's, the better it scores, and nothing beats simply leaving
v6 alone.

### Two false alarms worth recording

**"Playgen worlds are richer."** Continuation auctions settle at mean contract 128.5
against 116.1 for the real corpus, which looked like a decisive world-sampling bias.
It is not. Measured directly
([bench_world_richness.rs](../../../colver-core/src/bin/bench_world_richness.rs), 4000
positions, fresh DD solves of the true deal): playgen worlds match the true deal at
**+0.0 points on the mean over trumps and +0.7 on the best trump**. The posterior is
unbiased where it matters.

**"It is the match-score distribution."** The corpus is generated at 0-0 while the
targets used random match states, so the comparison was not like-for-like. Forcing 0-0
on both sides leaves the gap intact (127.8 vs 116.1). The real explanation is that the
comparison itself was meaningless: the continuation deliberately *forces* one of the 12
highest-Q candidate actions, many of which are bids v6 would not have made, so a higher
mean contract is the definition of the quantity, not evidence of a defect.

Both are recorded because each looked conclusive and neither was.

## Where this actually stands

Playgen's auction-conditioned posterior is good (21.5% better RMSE than uniform, and
unbiased in the max). The continuation target is, as far as it can be verified, computed
correctly. Yet every distilled model loses by 3–7 pp, and the residual is always the same:
the targets want to bid where v6 passes, on roughly a sixth of all positions.

What has *not* been ruled out is the scoring of the final contract. Both attempts settle
it from DD points — attempt 2 via a sampled `P(isdd | dd)` map, but that map was built
from `scores_isdd_5M.sc`, which is stale (pre-quick_tricks-fix) and came from a
*uniform-world* IS-DD, a weaker player than the one the arena actually uses. Scoring the
final contract with a real rollout of the current play policy is the one substantive
piece that has never been tried, and it is the expensive one
(see the cost analysis: ~74 GPU-days at pool scale).

Until that is done, the honest summary is: the measured 21.5% RMSE gain from auction
conditioning is a fact about estimating `dd_pts`, and it has not transferred to bidding
strength through any target definition tried so far.

## Reproduce

```bash
cargo run --bin gen_bid_labels --release --features parallel -- \
  --games data/training/labelcorpus_120k.bin --output data/bid_labels/shard.ql \
  --offset 0 --deals 40000 --per-deal 2 --worlds 8 --seed 4242

cargo run --bin train_bid_distill --features dmc_train --release -- \
  --labels data/bid_labels/shard_local.ql --labels data/bid_labels/shard_moxxi.ql \
  --games data/training/labelcorpus_120k.bin \
  --out-dir models/bid_v7_pl --epochs 6 --lr 5e-5 --recenter-per-level

cargo run --bin arena --release -- h2h v7pl_ep6 v6_isdd_75M --matches 1500 --no-save
```


## Aside: playgen's own bid head as a bidder (2026-07-25)

Playgen v2 carries a 43-way auction head, trained only as a by-product of next-token
prediction on game records. Wiring it in as a bidder directly (`strategy = "playgen"`
in a bot TOML, [PlaygenBidPolicy](../../../colver-core/src/agent/bid.rs)) costs nothing
and answers a question the distillation work kept raising: how much of v6 is reachable
without RL at all?

| Match-up | Result |
|---|---|
| playgen_bid vs v6_isdd_75M | 48.2% (3000 matches) |
| playgen_bid vs nn_v2_dmc50 | 61.6% (1000 matches) |
| v6_isdd_75M vs nn_v2_dmc50 | 65.0% (1000 matches) |

Against a common weaker opponent both land close together, so the near-parity is real
competence rather than a cloning artifact. **A behaviour clone that is not even
score-aware** — the playgen corpus was generated on standalone deals at 0-0, so nothing
in its prefix carries the match score — **lands within ~1-3 pp of a bidder trained for
75M RL steps with score-aware observations and match simulation.**

Two things follow. Next-token prediction on game records captures almost all of v6's
auction competence, and v6 looks close to a plateau for this observation and
architecture — which is the simplest explanation for why no distilled target managed
to improve it. The head is also usable as a component in its own right: a prior, an
ensemble member, or a cheap diverse opponent for training.

Untried and cheap: ensembling the two (softmax both, average the probabilities, take
the argmax over legal actions). Two models this close in strength and this different in
origin are the classic case where an ensemble beats both.
