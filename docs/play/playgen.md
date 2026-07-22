# Playgen: Autoregressive World Sampler for IS-DD

## Motivation

IS-DD spends exact DD solves on determinized worlds sampled uniformly over hard
constraints — including worlds no credible player could hold given the observed
auction and plays. Playgen is a causal transformer that models
p(next play | observer-visible prefix) on self-play games. **Sampling a
continuation to the end of the deal reveals a complete hidden-hand assignment**
(all 32 cards get played), i.e. a determinized world drawn from the approximate
posterior p(hands | public history) under the training policy.

Key design insight (vs the belief nets): the belief MLPs predict 24 card
locations *jointly in one forward pass* but only output independent per-card
marginals — they cannot represent correlations ("if West has the trump J, he
likely has the 9 too"). The autoregressive factorization dilutes the task over
many tokens (CoT-style) and captures joint structure for free.

Intended use: generate K candidate worlds (+ a share of uniform random legal
worlds to avoid blind spots), DD-solve them, aggregate. A likelihood-scorer mode
(same net, hands revealed in the prompt) can later provide importance weights.

## Tokenization (`playgen/tokens.rs`)

```
[BOS] [OBSPOS_d] [h1..h8] [bid tokens ×≤24] ([ACT_a] [CARD]) ×≤32
```

- 4 embedding channels per token: primary (31 ids), suit (5), actor (5,
  observer-relative), segment (header/bid/play) + learned absolute positions.
  Max 98 tokens.
- **Suit canonicalization**: suits permuted so trump = suit 0
  (`perm[trump] == 0`). The 3 non-trump suits are randomly permuted per sample
  (6 variants) as free augmentation. Combined with 4 observer perspectives:
  24× effective augmentation per game.
- **ACT query tokens**: the actor of the next play is deterministic from the
  state machine, so it is *given* as a query token; card logits (32-way) are
  read at the ACT position. The model never wastes capacity computing trick
  winners.
- **Observer-visible masks** (per prediction):
  - actor == observer → true legal mask (observer knows their hand);
  - hidden actor → unseen cards minus hard-constraint exclusions (engine voids,
    deduced trump voids, trump ceilings) via `TrumpCeilingTracker`.
  The mask does the set arithmetic (what transformers do badly); the model
  learns only strategy/inference. Softmax is normalized over this same mask at
  train and inference time (no train/test mismatch).
- Loss on all 32 plays (own plays teach self-simulation; hidden plays teach
  belief formation). Void deals are skipped.

Validation: `COLVER_GAMES=<abs path> cargo test -p colver-core
validate_games_file -- --ignored --nocapture` asserts the played card is always
in the observer-visible mask (0 false exclusions on 63.7M preds of
games_500k.bin and on fresh v6/DouDou50 games). ~13.5% of plays are forced
(mask = 1 card); avg hidden-actor mask ≈ 12 cards.

## TrumpCeilingTracker bug fixes (2026-07-21)

Tokenizer validation surfaced two false-exclusion bugs in
`game_replay.rs::TrumpCeilingTracker` (NOT in `CardBeliefs`, which had both
cases right — IS-DD runtime was never affected):

1. **"Ne pisse pas" discard**: opponent cut, player can't overtrump and
   discards → was deduced *void in trump*; correct deduction is only a trump
   ceiling (discarding while holding lower trumps is legal).
2. **Voluntary undertrump with partner master**: partner master → any card is
   legal, so playing a low trump while holding higher ones is possible → ceiling
   deduction removed in that case.

Impact: belief v2/v3 training obs (hard-constraint channels) and `eval_beliefs`
metrics were occasionally wrong. `belief_net_v2.bin` / `belief_v3.bin` were
trained on slightly corrupted constraint features — worth a retrain on fixed
extraction (+ v6 auctions). `bid_belief_v4` unaffected (bid phase only).

## Model (`playgen/model.rs`, feature `dmc_train`)

Decoder-only transformer: pre-norm RMSNorm, GeGLU FFN (2/3·4·d), multi-head
attention with causal mask, 32-way card head at every position. Default
d=256 L=4 H=8 ≈ 3.2M params. Padding needs no attention mask: pads sit at the
sequence end, causality already blocks them.

## Training (`train_playgen` binary)

Offline supervised (teacher forcing) — none of the RL instabilities that hurt
Bumblebid (no PER, no coinche reward dominance, no env in the loop).

```bash
# Data: fresh self-play with current champions (auction distribution matters —
# a playgen trained on bid-v1 auctions would misread v6 auctions, same lesson
# as belief_v3 becoming unusable after the bidder changed)
./target/release/generate_game_data \
  --dmc-model models/play_v2/play_final.bin \
  --bid-model models/bid_v6_isdd_resume/bid_nn_final.bin \
  --games 1000000 --output data/training/playgen_games_1M.bin --seed 7
# ~230 games/s on 32 threads (~72 min for 1M)

# Training (RTX 4090: ~2.4 steps/s at batch 512; batch 1024 OOMs)
./target/release/train_playgen \
  --games data/training/playgen_games_1M.bin \
  --steps 60000 --batch-size 512 --d-model 256 --layers 4 --heads 8 \
  --save-dir models/playgen
```

Per step: 512 random (game, observer, suit-perm) samples tokenized on CPU.
Eval prints overall CE/acc, hidden-only CE, and per-trick CE — expect a
decreasing per-trick profile (high entropy early, ~0 on the forced last trick).

`generate_game_data` auto-detects score-aware bid models (v4/v5/v6 obs dims,
scores fed as 0-0) and canonical DMC models (obs 411 → canonical mask/action
conversion + residual forward).

## Training run (2026-07-21, 60K steps ≈ 5.6h on 4090)

Eval (2000 held-out games): loss 0.944, acc 0.632, hidden-loss 1.166 (naive
uniform-over-mask floor ≈ ln 12 ≈ 2.48), per-trick CE monotone
[1.22 1.22 1.23 1.17 1.07 0.91 0.63 0.11]. No overfitting (train ≈ eval).
Checkpoints: `models/playgen/playgen_*.safetensors`, exported
`models/playgen/playgen_final.bin` (COLVPG01).

## Inference (`playgen/infer.rs`, pure Rust)

`PlaygenModel` (flat-f32 load) + `PlaygenSampler`: incremental per-deal KV
cache (fed via `IsDdSearch::record_action`), sequential world generation with
observer-visible masks, dead-end restart (≤4) then `determinize_greedy`
fallback. Verified: teacher-forcing acc 0.632 (matches candle eval), 600/600
valid worlds, 0 dead-ends, ~38 ms/world (~830 tokens/s single-thread).
**10.5% of sampled worlds reproduce the exact true hidden hands** (temp 1.0,
mixed game stages) — constraint-uniform sampling essentially never does.

Verification tests (ignored, env-gated):
`COLVER_PLAYGEN_BIN=... COLVER_GAMES=... cargo test --release
playgen_forward_accuracy|playgen_generate_worlds -- --ignored --nocapture`

## IS-DD integration & arena A/B (2026-07-21)

`IsDdConfig { playgen_frac, playgen_temp }` + `IsDdSearch::set_playgen_model`.
Arena TOML (`method = "smart_is_dd"`): `playgen_model`, `playgen_frac`,
`playgen_temp`, and `time_ms = 0` → no time limit (fixed determinizations).
Bots: `pg0_d10` / `pg50_d10` / `pg100_d10` / `pg0_d5` / `pg100_d5`
(bid v6, no belief net, only the world source differs).

| Matchup (fixed dets, no time limit) | Result | Margin |
|---|---|---|
| pg100_d10 vs pg0_d10 (seed 42) | **60–40** | +278 |
| pg100_d10 vs pg0_d10 (seed 1337) | **71–29** | +463 |
| pg50_d10 vs pg0_d10 (seed 42) | **64–36** | +233 |
| pg50_d10 vs pg0_d10 (seed 1337) | **59–41** | +126 |
| pg100_d5 vs pg0_d5 (seed 42) | **64–36** | +414 |
| pg100_t08_d10 vs pg100_d10 (seed 42) | **57–43** | +173 |

Playgen worlds beat constraint-uniform sampling **65.5% over 200 matches at
d10** (p ≪ 0.01); the 50% mix scores 61.5% over 200 — keeping random worlds
against blind spots costs little. Advantage holds at d5 (64%). Temperature 0.8
edges out 1.0 (57%, 100 matches — suggestive, not conclusive). Caveat: these
are equal-*determinization* comparisons; playgen adds ~38 ms/world, so
equal-*time* comparisons (vs `time_ms = 20` champions) need the
batched-lockstep forward first.

### Confirmation & sweep (overnight 2026-07-21)

**Confirmation, 400 matches:** pg100_t08_d10 vs pg0_d10 = **58.8%** (235–165,
+229/match, 95% CI [54%, 64%]). Honest central estimate of the playgen gain at
d10: ~59–62%.

**Sweep frac × temp @ d5** (100 matches/cell vs pg0_d5 — cell-level CI ±10pp,
read trends only):

| frac \ temp | 0.6 | 0.8 | 1.0 |
|---|---|---|---|
| 25% | 62% | 51% | 64% |
| 50% | 65% | 57% | 62% |
| 75% | 63% | 60% | 58% |
| 100% | 52% | **65%** | 61% |

All 12 cells beat baseline (mean 60%); no resolvable frac/temp trend at this n.
Default recommendation: **frac 0.5–1.0, temp 0.8–1.0**.

### One-pass belief nets vs playgen (2026-07-22)

Clean belief-net retrains (fixed `TrumpCeilingTracker`, v6 auctions, 400K
games, V2 and V3 obs, 15 epochs ≈ 24 min each) — **and** a fix so the arena
actually consults them (`use_nn_beliefs` was never set → `[belief]` bots had
been running without their net since the soft-beliefs refactor; this also
explains the historical "belief adds 0pp on v6" result). IS-DD now supports V3
obs at runtime (`derive_v3_temporal`).

| Matchup (d10, no time limit, 100 matches) | Result | Margin |
|---|---|---|
| bel4_v2 vs pg0 | 56–44 | +181 |
| bel4_v3 vs pg0 | 53–47 | +27 |
| bel4_v2 vs pg100 | **43–57** | −57 |
| bel4_v3 vs pg100 | **37–63** | −269 |

One-pass marginals give a small edge over uniform; the autoregressive sampler
clearly beats them (60% combined over 200 matches) — joint structure matters.
But belief weighting is ~free per world while playgen costs ~38 ms/world, so
under `time_ms` budgets belief remains the only usable option until the
batched-lockstep forward lands. Models: `models/belief_v4_fix.bin` (V3),
`models/belief_v4_fix_v2.bin` (V2).

### Time-budget test — 1s/move (2026-07-22)

Inference optimization first: the batched-lockstep forward gave **no gain**
(model is L3-resident, weight streaming was never the bottleneck); the real
win was **manual dot-product vectorization** (`dot8`, multiple accumulators to
let LLVM vectorize float reductions): **43 → 18 ms/world (2.4×)**.

At `time_ms = 1000` (scaled by cards left; ~30-50 playgen worlds vs ~70
cheap worlds per move), 100 matches each:

| Duel | Result | Margin |
|---|---|---|
| playgen vs uniform | 57–43 | +159 |
| belief V2 vs uniform | 60–40 | +269 |
| playgen vs **belief V2** | **39–61** | −139 |

**The verdict flips with the budget**: at fixed determinizations playgen ≫
belief; at fixed time belief > playgen (quality × quantity beats quality
alone). Production time-budget mode → belief_v4_fix_v2; playgen → offline /
oracle analysis / data generation, and the hybrid (belief-weighted worlds +
playgen_frac) is the natural combination since non-playgen draws already use
belief weighting when a net is loaded.

Hybrid (belief weighting + playgen_frac 0.3) vs pure belief at 1s:
**52–48 (+126)** — statistically tied at n=100, at best a small plus. The
time-budget podium stands: belief ≥ hybrid > playgen > uniform.

## Next steps
- [ ] Playgen v2: 10M-game corpus (generating on moxxi), bigger model
      (d=384 L=6?), COLVGM01 merge tool for chunked corpora — better per-world
      quality could flip the time-budget verdict back
- [ ] Faster inference if needed: explicit SIMD kernels, or distill to a
      smaller/shallower playgen
- [ ] Re-run historical belief-net h2h now that the net is actually consulted
- [ ] Batched-lockstep world generation (all K worlds advance together →
      gemm-shaped matvecs) for wall-clock parity with time-budgeted IS-DD
- [ ] Offline: implied marginals vs bid_belief_v4 via eval_beliefs
- [ ] Champion integration: playgen + belief-weighted fallback + coinche/
      surcoinche handling audit, then full leaderboard run
- [ ] Later: scorer mode (hands revealed in prompt, conditioning dropout) for
      importance-weighted DD aggregation
