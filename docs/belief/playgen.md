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

### Ce que le masque ne dit pas encore : la belote (à corriger en v3)

Le masque porte les coupes et les plafonds d'atout, **pas l'annonce de belote**.
Elle n'est nulle part dans le flux : le modèle voit un Roi d'atout tomber, jamais
le fait que le siège a annoncé en le posant. Conséquence mesurée le 2026-08-03
(`bench_belote_facts --sidecar`, deux tirages de ~6 100 mondes) : **15 à 16 % des
mondes rendus aux positions concernées sont impossibles**, contre 40,1 % pour un
tirage uniforme aveugle — il en a donc appris une bonne part par corrélation, mais
un monde sur six reste faux. `worlds::retain_valid` les rejette désormais côté
client, ce qui coûte un aller-retour de plus au sidecar au lieu d'une erreur
silencieuse.

**Ce que ça vaut, honnêtement : de la correction, pas de la force.** Côté IS-DD, la
même déduction change la carte jouée 8,5 % du temps sans jouer mieux (−0,008 ±
0,031 pt DD/décision) — [is_dd.md](../play/is_dd.md#ça-change-la-décision-et-ça-ne-la-rend-pas-meilleure).
L'intérêt pour playgen est donc de ne plus fabriquer de mondes impossibles, et de
ne plus les payer en aller-retours ; pas d'attendre un gain d'arène.

**Ça se répare dans le masque, pas dans une entrée supplémentaire** — et c'est
exactement le principe déjà posé plus haut (« le masque fait l'arithmétique
d'ensembles, le modèle n'apprend que la stratégie et l'inférence ») :

- *silence* — un Roi ou une Dame d'atout tombé sans annonce : son poseur ne jouera jamais
  l'autre → exclusion permanente dans **son** masque ;
- *annonce* — `belote[t] == 1` : les **trois autres** sièges ne peuvent pas jouer
  l'autre carte → exclusion dans leurs masques, ce qui place la carte par
  élimination sans avoir à exprimer « il l'a » dans un vocabulaire de coups.

Les deux tiennent donc dans `TrumpCeilingTracker` / `compute_hard_constraints`,
sans changer le format ni le vocabulaire, mais **c'est une rupture de corpus** :
le masque d'entraînement changerait, donc COLVPG02 ne serait plus lisible avec le
nouveau masque sans écart train/test. À faire en v3, avec le score de partie.
Prédicat prêt à l'emploi : `play::belote_facts`.

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

> **Historical.** The knobs named below (`IsDdConfig::playgen_frac` /
> `playgen_temp`, `IsDdSearch::set_playgen_model`, and the matching arena TOML
> keys) were removed by the 2026-07-24 agent refactor. Worlds now come from a
> [`WorldSource`](../agents.md) that the agent owns: `[worlds] source =
> "sidecar" | "playgen" | "uniform"`. The findings below still stand — only the
> spelling changed.

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

## GPU sidecar (`playgen_gpu_server`, feature `gpu_server`)

Sampling worlds on CPU costs ~50× what it costs on a GPU, so production runs the
model as a **sidecar**: a small HTTP server on a machine with CUDA, which the
agents call over the LAN. That is what makes 100%-playgen worlds affordable
inside a per-move budget.

```bash
CUDARC_CUDA_VERSION=13010 cargo build --release --bin playgen_gpu_server --features gpu_server
./target/release/playgen_gpu_server --playgen models/playgen/playgen_v2_final.bin --port 8003
curl -s http://localhost:8003/health
```

| endpoint | returns | used by |
|----------|---------|---------|
| `POST /play_worlds` | remaining hands per seat | **`worlds::SidecarWorldSource`** — the IS-DD agent |
| `POST /auction_deals` | full 8-card deals, conditioned on the auction | web analysis pages |
| `POST /beliefs` | card marginals `[4][32]` | web analysis pages |
| `GET /health` | status + device + `surface` | agent construction check, **contrôle de fraîcheur** |

The request carries the *replayable deal* (dealer, initial hands, action prefix,
observer); the server rebuilds the sampler by replay before sampling a batch. So
the client holds no session and a restarted sidecar needs no coordination.

**Clients.** The IS-DD agent talks to it directly from Rust
([`worlds.rs`](../../colver-core/src/worlds.rs), ~60 lines of hand-rolled HTTP —
`colver-core` deliberately has no HTTP dependency). Configure with
`$COLVER_PLAYGEN_GPU_URL` or a `[worlds] url` in the bot spec. The Python client
[`web/playgen_gpu.py`](../../python/colver/web/playgen_gpu.py) now only serves
the **analysis** pages; it no longer feeds IS-DD.

### Fraîcheur du sidecar : `surface` (2026-08-03)

**Joignable ne veut pas dire à jour.** Le sidecar se déploie à la main, donc
décider s'il est à jour se faisait en lisant les titres de commits — et un
commit titré « feat(elo) » (`989d1e7`) a livré la contrainte belote **dans le
sampler** sans le mentionner. La prod a tourné **21 h** sur un sidecar périmé :
il fabriquait des mondes que `worlds::retain_valid` rejetait ensuite (~15,4 % aux
positions à belote), donc Dédé cherchait sur moins de mondes qu'il n'en
demandait. `/health` le voyait joignable, et il l'était.

`build.rs` calcule à la compilation une empreinte des sources qui décident du
comportement du sidecar — **`src/playgen/` + `src/engine/` en entier** — exposée
en `playgen::SURFACE`. Le sidecar la publie sur `GET /health` ; le conteneur web
porte la sienne (`colver._colver.PLAYGEN_SURFACE`, même build) et les compare
dans `playgen_gpu.probe()`.

**Pourquoi une empreinte de sources et pas le SHA git** : le SHA change à chaque
commit, y compris web-only. Une alerte qui s'allume à chaque déploiement est du
bruit, et une alerte qui est du bruit ne se lit plus. `worlds.rs` est
volontairement **hors** surface — il tourne côté web, pas dans le sidecar.
Vérifié : éditer `engine/play.rs` change l'empreinte, éditer `worlds.rs` ne la
change pas.

Trois états, et **« inconnu » n'est pas « périmé »** : un sidecar antérieur à
cette fonctionnalité ne publie pas `surface`, et le crier périmé apprendrait à
ignorer le champ. `/health` ne dégrade que sur `fresh: false`, et seulement là où
le sidecar est attendu (`COLVER_REQUIRE_SIDECAR`) — même arbitrage que la
joignabilité, pour qu'une machine de dev ne crie pas pour rien.

**Ne couvre pas** les poids du checkpoint, les drapeaux de compilation ni la
version de CUDA. Le checkpoint reste à vérifier par sha256 — voir juste en
dessous ; l'automatiser demanderait une dépendance `sha2`, absente du lock.

> **Keep the served model aligned with the released one.** Prod ran
> `playgen_v2_half.bin` (an intermediate checkpoint) from 2026-07-23 to
> 2026-07-24 while every benchmark used `playgen_v2_final.bin` — the site was
> playing with a different world sampler than the one being measured. Both are
> now on `playgen_v2_final.bin` (md5 `ebffd896…`, the v0.8.0 release asset).
> Deployment details live in the deployment's own private runbook, not here.

### Batching across positions, and the KV cache (2026-08-02)

The sidecar served **one request at a time** and a request cost ~220 ms whether
it returned 1 world or 256:

| `n_worlds` | 1 | 20 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| ms/request | 216 | 261 | 225 | 268 | 348 |

The cost was almost entirely fixed — ~100 sequential decode steps on a 10.7M
model — so it was neither VRAM (5.5 GB of 24) nor arithmetic, but **latency and
occupancy**. IS-DD asks for ~20 worlds per decision from a different position
each time, so it ran the GPU at ~8% of what it can do, and no amount of client
concurrency helped against a serial server.

Two independent fixes, both verified to leave behaviour **bit-identical**.

**1. Batch across positions** (`generate_worlds_multi`). Batching *within* one
position already existed; the missing axis was *between* positions. Prefixes of
different lengths are right-aligned and padded, dummy tokens excluded by the
additive mask, logical positions passed per lane — the "lockstep paddé" idiom
`auction_round` already used for desynchronized auctions. Between prefill and
decode the K prefix lanes are fanned out to `sum(n_worlds)` with an
`index_select` on the batch axis. The server became: handler threads that parse
and replay (taking replay off the GPU's critical path), a queue, and a single
owner of the device draining it up to `--lane-budget`. No artificial wait
window, so a lone request still leaves immediately.

**2. Fixed-capacity KV cache.** Profiling (`COLVER_PLAYGEN_PROFILE=1`) put 97%
of decode inside `forward_step`, and inside it:

| | before | after |
|---|---|---|
| `cat` of the KV cache | 36-43% | 10% |
| attention | 36-37% | 33% |
| FFN | 11-15% | 29% |
| qkv | 6-8% | 17% |

So **~75% of the time was memory traffic**: the cache was copied whole *twice
per step* to append one token — once by `Tensor::cat` reallocating, once by the
`transpose(2,3).contiguous()` in attention. At 640 lanes that cache is ~1.3 GB.
It is now allocated once at `prefix + 2 × cards left` and written in place with
`slice_set`, with unwritten slots masked to -1e9 (whose exponential is exactly 0
in f32, so the softmax is unchanged — that is what allows attending over the
full capacity and keeping every tensor contiguous). K is stored **pre-transposed**
`[B, H, hd, CAP]`, removing the second copy. The CPU path (`KvCacheBatch`) never
had this problem; the GPU one had drifted from it.

Measured, 32 positions × 20 worlds: one at a time **682-770 → 201-213 ms**
(3.4×), batched **~78-208 → 15.3-18.0 ms** (~6×).

**Client concurrency is part of the answer.** The GPU win did *not* show up
end-to-end at first: with 32 threads the client sits blocked in HTTP (~180% CPU
out of 3200%), so too few requests are in flight and batches average 12.8
instead of 26. Labelling throughput by client threads:

| threads | deals/s |
|---|---|
| 32 (default) | 0.237 |
| 96 | 0.396 |
| **192** | **0.496** |
| 384 | 0.492 (plateau) |

Cumulative on the labelling workload: **0.052 → 0.496 deals/s, 9.5×**.

**Verification** — `bench_playgen_batch`, in order of how much it would hurt to
get wrong: one item through the multi path is bit-identical to the single path
at the same seed; a position's card marginals shift 0.0392 when batched with 7
others against 0.0387 of sampling noise (no leak between lanes); 0 invalid
worlds. And a **fingerprint of all three paths** — auction, single play, batched
play — pinned before the KV refactor and unchanged after. The auction path feeds
the Annonces page in prod and had no equivalence check at all before this.

Two consequences worth carrying:

- `--lane-budget` now pre-allocates. At 1024 lanes the cache is ~2 GB, fine on a
  4090 but **not on a shared prod GPU** — lower it there.
- Remaining headroom is fp16/bf16 on the tensor cores (~2×), but that would break
  the CPU/GPU bit-identity the web's CPU fallback rests on. A product decision,
  not an optimization.

## Perplexité pli par pli (`bench_playgen_ppl`, 2026-08-03)

Perplexité teacher-forcing sur un corpus **retenu** (`heldout_20k_s90210.bin`),
restreinte aux coups des sièges **cachés** — les seuls qui parlent du monde. La CE
par pli qu'affiche `train_playgen` existait déjà ; ce binaire ajoute un point
d'entrée autonome (n'importe quel checkpoint, n'importe quel corpus), la
restriction aux acteurs cachés, et le **plancher de contrainte** : la même
quantité sous une loi uniforme sur le masque.

```bash
cargo run -p colver-core --bin bench_playgen_ppl --release --features parallel -- \
  --model models/playgen/playgen_v2_final.bin \
  --games data/training/heldout_20k_s90210.bin --n 500
```

`playgen_v2_final`, 500 donnes × 4 observateurs (5 988 prédictions cachées/pli) :

| pli | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| branchement modèle | 4,76 | 4,47 | 4,32 | 3,94 | 3,35 | 2,63 | 1,88 | 1,13 |
| branchement uniforme | 22,96 | 19,57 | 16,27 | 13,04 | 9,89 | 6,90 | 4,10 | 1,56 |
| **gain** | **4,82×** | 4,38× | 3,77× | 3,31× | 2,95× | 2,63× | 2,18× | **1,38×** |

**Le gain du modèle sur l'arithmétique d'ensembles décroît de bout en bout,
4,8× au pli 1 contre 1,4× au pli 8.** C'est la version résolue en temps du
constat de `bench_world_cred` (« un tirage uniforme contraint atteint déjà 70 %
d'argmax en jeu ») : les contraintes dures ne portent pas l'essentiel du signal
*partout*, elles le portent **en fin de donne**, là où elles déterminent presque
la position. Conséquence pour IS-DD : la valeur d'un monde playgen sur un monde
uniforme est concentrée dans les premiers plis.

**Piège d'unités, corrigé après coup.** Le cumul `exp(Σ nll restantes)` compte des
**continuations**, pas des mondes : plusieurs ordres de jeu réalisent la même
distribution des mains. Le premier tirage annonçait « 2,59e11 mondes » au pli 1,
alors qu'il n'existe que `24!/(8!)³ = 9,47e9` distributions — cent fois moins.
La table imprime donc ce plafond combinatoire à côté, et **seul le rapport
modèle/uniforme est lisible**, le facteur d'ordres s'y simplifiant. Le compte de
mondes proprement dit demande de l'échantillonnage (taux de monde exact) :
c'est une autre mesure, pas une autre présentation de celle-ci.

**Ce que ça ne mesure pas** : le corpus est joué par bid v6 + DouDou50, donc la
question posée est « sait-il prédire *ces bots-là* ». Comparer deux checkpoints
sur le même corpus est légitime ; en tirer un chiffre absolu ne l'est pas.
Aucun échantillonnage n'intervient, donc ce bench est immunisé par construction
contre le piège maison des questions tirées du flux mesuré.

### Les enchères, elles, sont saturées — sur les deux axes à la fois

Le bench couvre aussi la tête d'enchère, par **tour** (les quatre sièges parlent
une fois). La longueur est variable, donc `n` est imprimé : la plupart des
enchères meurent au 1er ou 2e tour, et au-delà du 3e il n'y a plus d'échantillon
(n = 150 au tour 4, n = 6 au tour 5 — du bruit, pas une mesure).

Branchement effectif, sièges cachés, 500 donnes retenues × 4 observateurs :

| modèle | params | éch. | tour 1 (n=5997) | tour 2 (n=4533) | tour 3 (n=1461) |
|---|---|---|---|---|---|
| v3-small @10K | 3,22M | 2,56M | 5,75 | 1,71 | 1,47 |
| v3-small @50K | 3,22M | 12,8M | 5,68 | 1,68 | 1,42 |
| v2 @60K | 10,74M | 11,5M | 5,69 | 1,68 | 1,42 |
| v2 @160K | 10,74M | 30,7M | **5,66** | **1,66** | **1,41** |
| *uniforme sur le masque légal* | | | *31,93* | *17,60* | *9,59* |

**Un modèle 3,3× plus petit, à 8 % du budget de données, est à 1,6 % du meilleur.**
Douze fois plus de données et trois fois plus de paramètres achètent 1,6 % au
tour 1 et 4 % au tour 3. La tête d'enchère est donc saturée en capacité **et** en
données — cohérent avec la précision d'enchère qui plafonnait à ~0,687 dès 50K
pendant l'entraînement de v2.

**Conséquence pour v3-B (le score de partie en entrée).** Si ni les paramètres ni
les données ne déplacent cette tête, le seul levier restant est de **l'information
nouvelle**. C'est exactement ce qu'est le score cumulé : v6 annonce autrement à
1500-300 qu'à 0-0, et playgen n'a jamais rien vu d'autre que 0-0. Cette mesure ne
prouve pas que le score aidera, mais elle **élimine l'alternative** « il suffirait
d'entraîner plus gros ou plus longtemps ».

C'est aussi là que le modèle sert le plus : **5,6× à 10,6× sur l'uniforme au tour
1-2, contre 4,8× au mieux en jeu** — la version chiffrée du constat de
`bench_world_cred` (« auctions are where samplers really differ »).

### Correction : v2 n'était pas saturé en jeu, seul le pli 1 l'est

Un premier dépouillement de cette échelle a conclu « v2 était saturé bien avant la
fin — 2,7× d'échantillons pour 2,7 % » (commit 321547a). **C'est faux, et l'erreur
était de lire la seule colonne du pli 1.** De 11,5M à 30,7M d'échantillons :

| pli | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| gain v2 | −2,7 % | −4,7 % | −5,9 % | **−8,2 %** | −8,5 % | −10,2 % | **−11,7 %** | −2,6 % |

v2 a progressé de 8 à 12 % en milieu et fin de donne pendant les deux tiers
restants de son entraînement. Ce qui sature au pli 1 est un **plancher
d'entropie**, pas le modèle : sans carte tombée il n'y a rien à déduire, et ce qui
reste est de la politique, pas de la croyance. Le pli 1 est donc la colonne la
moins informative de la table, et c'est précisément celle sur laquelle la
conclusion avait été tirée.

## Next steps
- [ ] **v3 : la belote dans le masque observable** (voir « Ce que le masque ne dit
      pas encore »). 15,4 % de mondes impossibles aux positions concernées, filtrés
      côté client depuis le 2026-08-03 — donc payés en aller-retours au lieu d'être
      évités. Rupture de corpus : le masque d'entraînement change.
- [ ] Playgen v2: 10M-game corpus (generating on the GPU host), bigger model
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

**Training (running):** remote RTX 3090 (CUDA 13.3 via
`CUDARC_CUDA_VERSION=13010` override — cudarc 0.19.2 caps at 13.1), corpus
9M games (8M remote seeds 101-108 + 1M local seed 7). d=384 L=6 H=8 =
10.74M params, batch 192 (256 OOMs at 24GB — manual-softmax activations ×
L=122²), lr 2e-4, warmup 2000, 160K steps ≈ 30.7M game-samples (same budget
as v1's 60K×512). ~2.2 steps/s ≈ 20h. Checkpoints:
`~/playgen/models/playgen_v2/` on the GPU host, log `~/playgen/logs/train_v2.log`.
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
DD-based bidders) and sampled auction rollouts for bid EV — the bid head +
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
auction point). Exposed in PyO3 — since 2026-07-24 through the read-only
`colver.Analyst` class (`auction_deals`, `bid_policy`, `marginals`; previously
`Env.playgen_sample_auction_deals` / `get_playgen_bid_policy`) — and deployed
on colver.net: the annonces page
Oracle/DouDou sims run on **auction-conditioned worlds** (chunked generation
overlapped with DD solves, uniform fallback, per-deal provenance chips), and
the playgen bid policy is shown under the Bid V6 Q-values.

## World-credibility benchmark (`bench_world_cred`, 2026-07-23)

Self-supervised sampler eval (user's idea): the observed actions are the
oracle. For each seeded position, sample K worlds per sampler and ask the
reference policy whether it would replay each observed hidden action holding
that world's hand.

```bash
cargo run -p colver-core --bin bench_world_cred --release \
  --features "rand,nn,parallel,dmc_train" -- \
  --bid-positions 100 --play-positions 100 --worlds 32 --seed 42
```

### Generate every question first, then answer (fixed 2026-07-23)

**The bug.** The first version ran one loop that drew a position, sampled
worlds from all three samplers, judged, and looped — all off a single `rng`.
World sampling consumes a *variable* number of draws depending on the model's
distribution, so swapping the playgen checkpoint desynced the stream and
silently re-drew every position after the first. Demonstration on the old
binary, counting judgements of the `uniform` sampler (which never touches the
playgen model, so its numbers must not move):

| positions | 60K ckpt | 120K ckpt |
|---|---|---|
| 1 | 12 | 12 |
| 2 | 24 | **48** |
| 5 | 72 | **96** |

Identical at one position, divergent from the second. In a 30+30 run this
moved the untouched belief/uniform baselines by up to 5 pp — the same order
as the checkpoint effect being measured. **Any pre-2026-07-23 cross-checkpoint
comparison from this bench is void.** Within a single run the three samplers
always shared their positions, so the sampler *hierarchy* below was never
affected; only comparisons across runs were.

**The fix** (user's framing: "generate all the questions, then answer them").
Phase 0 draws every position up front — `generate_bid_positions` /
`generate_play_positions`, which depend on the bid net and DMC net only, never
on a sampler. Then each (phase, position, sampler) triple gets its own stream
from `sub_rng`, a splitmix64 mix of the seed. Consequences:

- A sampler cannot perturb another sampler, nor the positions.
- The two phases use separate streams too, so `--bid-positions 0` leaves the
  play positions untouched.
- **Built-in control**: any baseline that does not depend on the varied
  component must come out bit-identical across runs. If `belief` or `uniform`
  moves while only the playgen checkpoint changed, a coupling is back.

**The same single-loop pattern still exists in `bench_logp_cred.rs` and
`bench_world_compress.rs`.** Their published numbers are single-configuration
and intra-run, so they stand; but do not use either to compare two
checkpoints until they get the same two-phase treatment.

The general rule: a benchmark's questions must not be drawn from a stream that
the thing under test also consumes.

### GPU inference (both phases)

`--gpu` (default when built with `dmc_train`; `--cpu` to force) runs playgen
generation on CUDA. The play path was added 2026-07-23 — and it turned out the
GPU code was *already there*: `auction_round`'s `PLAYING` branch is a complete
play generator, since auction lanes transition to play and deal out all 32
cards. Only an entry point starting lanes mid-play was missing.

To keep the two backends from drifting, the constraint bookkeeping (starting
`GenState`: voids, trump ceilings, remaining counts, current trick) is built
once in `PlaygenSampler::play_gen_spec()` and consumed by both. The forward is
the only thing that differs. Two things the port had to preserve: the suit
permutation (`permute_mask` + canon↔physical, a no-op in v2 but applied
generically), and using the observer's **initial** hand for the `unseen` mask.

The play path is simpler than the auction one: every lane replays the same
number of cards, so all lanes stay at the same logical position and the
per-lane `lens` machinery is unnecessary. Dead lanes keep appending dummy
tokens and are masked out of future attention.

Validated bit-identical to CPU (30 positions: 4043 judgements, 348 worlds, 12
missing, both backends). Speedups on a 4090 at B=12: auctions **5.8×**
(52.8 s → 9.1 s), play **3.2×** (29.7 s → 9.2 s). Play gains less because the
DMC judge and the belief/uniform samplers stayed on CPU. A 100+100 × 32-world
run takes ~1 min 30 — 8.9× the sample size of the old 30+30 × 12 default for
roughly the same wall-clock, which is why the default scale was raised.

### Reference results (100+100 positions × 32 worlds, argmax/top3)

Playgen v2 @120K, seed 42 (baselines identical on every seed pair — the
control described above):

| Phase | playgen | belief NN | uniform |
|---|---|---|---|
| Auctions (judge bid v6) | **67% / 95%** | 38% / 68% (bid_belief_v4) | 15% / 34% |
| Play (judge DouDou50)   | **86% / 98%** | 77% / 95% (belief_v4_fix_v2) | 70% / 93% |

The hierarchy holds in both phases but tightens sharply in play: hard
constraints already carry most of the play-phase signal (uniform-constrained
reaches 70% argmax) — auctions are where samplers really differ, which is
also why bid-phase conditioning (the annonces page, DD-based bidding) is the
biggest payoff.

Note the small-sample bias: at the old 30 × 12 scale the belief baseline read
34% in auctions, versus 38% here. Sample sizes below ~1000 worlds per sampler
move by several points; use 100 × 32 or larger for anything decisive.

### Checkpoint A/B: v2 @60K vs @120K (2026-07-23)

First comparison run on the fixed benchmark, three independent position sets:

| seed | auctions 60K | auctions 120K | play 60K | play 120K |
|---|---|---|---|---|
| 42 | 63% / 93% | **67% / 95%** | 86% / 97% | 86% / 98% |
| 43 | 62% / 93% | **66% / 95%** | 86% / 98% | 87% / 98% |
| 44 | 63% / 92% | **67% / 94%** | 85% / 97% | 87% / 98% |

**Auctions: +4 pp argmax and +2 pp top3, on 3/3 seeds** — the extra 60K
training steps buy real auction-conditioning quality. **Play: +1 pp**, at the
edge of resolvability. Consistent with the training curve, where bid-head
accuracy plateaued at ~0.687 by 50K while the play loss kept falling: the
auction gain here comes from better *hand* posteriors, not a better bid head.

The 120K checkpoint also produces fewer inconsistent worlds in play (rejected
by the initial-hand reconstruction): 80 vs 160 missing out of 3200 at seed 42.

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
