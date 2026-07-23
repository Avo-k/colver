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

---

## Playgen v2 (2026-07-23): physical suits + auction as prediction target

Design changes vs v1 (decided with the user):

- **No trump canonicalization.** Suits stay physical; augmentation is a full
  random suit permutation (all 24 variants) applied per sample at load time.
  The model must learn suit equivariance and route "which suit is trump" from
  the contract tokens — same regime as the bid NN's 24× augmentation. This is
  what unlocks the auction: the v1 canonical frame (`perm[trump]=0`) needed a
  known contract and therefore could not tokenize mid-auction.
- **Auction actions are prediction targets.** Sequence
  `[BOS][OBSPOS][h1..h8]([ACT][BIDTOK])×B≤24([ACT][CARD])×P≤32` (max 122
  tokens). A 43-way bid head reads at bid ACT positions, masked to the
  *public* legal bid set (no belief machinery needed — bid legality is
  public). Corpus max auction length is 21; cap 24 (`MAX_BID_ENTRIES_V2`).
- **Void deals kept** as auction-only samples (all-pass probability matters
  for future mid-auction generation).
- Loss = mean CE over all predictions (play + bid, λ=1; bids ≈ 18% of preds).

**Code:** `tokenize_replay_v2`/`PlaygenSampleV2`/`random_suit_perm`/
`permute_bid_action`/`permute_bid_mask` (tokens.rs), `PlaygenConfig` + bid
head (model.rs), `train_playgen --v2`, `export_playgen --v2` → **COLVPG02**
(bid head appended after card head), infer.rs auto-detects magic; the sampler
emits the header at `init_deal` and ACT+BID pairs during the auction
(identity perm — all v1 canon↔phys code paths are identity-safe). Arena/IS-DD
wiring unchanged: point `playgen_model` at a COLVPG02 file.

**Validation:** 1M corpus × 4 observers: 127.87M play preds + 32.6M bid
preds, 0 skipped, 0 false exclusions; perm-equivariance test (tokenizing
under perm == permuting tokens).

**Training (running):** moxxi RTX 3090 (CUDA 13.3 via
`CUDARC_CUDA_VERSION=13010` override — cudarc 0.19.2 caps at 13.1), corpus
9M games (8M moxxi seeds 101-108 + 1M local seed 7). d=384 L=6 H=8 =
10.74M params, batch 192 (256 OOMs at 24GB — manual-softmax activations ×
L=122²), lr 2e-4, warmup 2000, 160K steps ≈ 30.7M game-samples (same budget
as v1's 60K×512). ~2.2 steps/s ≈ 20h. Checkpoints:
`moxxi:~/playgen/models/playgen_v2/`, log `~/playgen/logs/train_v2.log`.
Local 4090 smoke (300 steps): loss 2.20→1.49, play-acc 0.40, bid-acc 0.61.

**After training:** rsync checkpoint → `export_playgen --v2 --d-model 384
--layers 6 --heads 8` → `playgen_forward_accuracy_v2` (teacher-forcing parity,
play + bid heads) → `playgen_generate_worlds`/`playgen_batch_worlds` (work
as-is on v2 models) → arena pgNN bots with the v2 .bin. Note ~3.3× v1 FLOPs:
expect ~55-60 ms/world; the fixed-dets comparison is the fair first test,
then the 1s/move re-run to see if per-world quality flips the time-budget
verdict.

**New capability unlocked (not yet wired):** mid-auction worlds (sample
auction continuation + full play → hidden hands during bidding, for
BisDd/dd_bid) and sampled auction rollouts for bid EV — the bid head +
`PlaygenModel::bid_logits` exist; needs a public bid-state machine in the
sampler's generate path.

## V2 @60K checkpoint: validation, deploy, mid-auction worlds (2026-07-23)

Mid-training checkpoint (60K/160K steps) exported as `playgen_v2_half.bin`
(COLVPG02, **10.74M params**, 43.0 MB — v1: 3.20M, 12.8 MB) to validate the
whole pipeline before the final model.

**Validation**: pure-Rust teacher-forcing parity exact vs candle eval (play
nll 0.954 / acc 0.6275; bid acc 0.695); 600/600 valid worlds, 0 dead ends,
11.7% exact-true (v1: 10.5%). **~93 ms/world** (v1: 18 ms): at 43 MB the
model is no longer L3-resident — inference is now memory-bound, and the
lockstep batch path helps again (+20%, it was useless for v1).

**Arena** (fixed 10 dets, 200 matches): v2@60K 52.0% vs v1 (+34/match) —
statistical tie at 40% of training on a harder task. The real v2 win is the
new capability, not mid-play world quality.

**Mid-auction deal sampling** (`PlaygenSampler::generate_deals_from_auction`):
the bid head completes the auction (masked to the *public* legal bid set via
a cloned GameState — bid legality never reads hands), then the play head
plays the deal out; the assignment reveals full hidden hands. 100/100 valid,
~420 ms/deal mid-auction. Plus `bid_policy` (43-way logits at the current
auction point). Exposed in PyO3 (`playgen_sample_auction_deals`,
`get_playgen_bid_policy`) and deployed on colver.net: the annonces page
Oracle/DouDou sims run on **auction-conditioned worlds** (chunked generation
overlapped with DD solves, uniform fallback, per-deal provenance chips), and
the playgen bid policy is shown under the Bid V6 Q-values.

## World-credibility benchmark (`bench_world_cred`, 2026-07-23)

Self-supervised sampler eval (user's idea): the observed actions are the
oracle. For each seeded position, sample K worlds per sampler and ask the
reference policy whether it would replay each observed hidden action holding
that world's hand. Rust binary, both phases, positions fully seeded:

```bash
cargo run -p colver-core --bin bench_world_cred --release -- \
  --bid-positions 30 --play-positions 30 --worlds 12 --seed 42
```

Reference results (playgen v2 @60K, seed 42, argmax/top3):

| Phase | playgen | belief NN | uniform |
|---|---|---|---|
| Auctions (judge bid v6) | **60% / 92%** | 34% / 64% (bid_belief_v4) | 12% / 32% |
| Play (judge DouDou50)   | **85% / 98%** | 78% / 96% (belief_v4_fix_v2) | 70% / 94% |

The hierarchy holds in both phases but tightens sharply in play: hard
constraints already carry most of the play-phase signal (uniform-constrained
reaches 70% argmax) — auctions are where samplers really differ, which is
also why bid-phase conditioning (BisDd, annonces) is the biggest payoff.

## Ensemble world pool + credibility weighting (2026-07-23)

User's design: a portfolio of world sources — transformer (high quality,
slow), belief NN (decent, faster), uniform (volume/coverage) — with per-world
retroactive validation. Implemented in `IsDdConfig`:

- `belief_frac` (default 1.0): among non-playgen worlds, fraction sampled
  with belief weights; the rest constraint-uniform (coverage floor).
- `cred_alpha` (default 0.0 = off): per-world importance weight = product of
  per-bid rank factors (would the credibility bid net — default: the bot's
  own bid model — replay each observed hidden bid with this world's hand?
  argmax 1.0 / top-3 0.7 / else 0.35), flattened as `w^alpha`, applied to the
  DD score aggregation (weighted mean). Judge cost ~µs/world (auction only).

Arena TOML keys: `belief_frac`, `cred_alpha`, `cred_bid_model`. Bots:
`ens_1s` (pg30 t0.8 + belief 80% + uniform floor + cred 0.5) and `cred_1s`
(ablation: belief + cred only) vs champion `bel4v2_1s` — results pending.

Caveat (BeliefState lesson): against humans the judge must stay a soft
weight, never a hard reject — bid v6 only judges bid-v6-like auctions well.
