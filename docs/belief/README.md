# Belief Documentation

Hidden card inference: estimating opponents' hands from observable game state.

Implementation: [colver-core/src/belief/](../../colver-core/src/belief/).

## Models

- **`CardBeliefs`** ([belief/card_beliefs.rs](../../colver-core/src/belief/card_beliefs.rs)) — heuristic, deprecated. Bidirectional soft inference from bids and play. 0% false exclusion rate on hard constraints.
- **`BeliefState`** (used by [is_dd.rs](../../colver-core/src/search/is_dd.rs)) — soft weights for IS-DD determinization sampling.
- **Belief NN v3** (`models/belief_v3.bin`) — 32×3 hidden card prediction, used by `smart_ismcts` and by IS-DD when a `[belief] model` is configured.
- **Bid Belief NN v4** (`models/bid_belief_v4.bin`) — 108→256²→96, replaces heuristic bid soft weights in BeliefState.

Details worth knowing:

- `CardBeliefs` correctly handles "ne pisse pas" (discarding when you cannot
  overtrump an opponent's cut implies a **trump ceiling**, not a void).
- `BeliefState` had its hard bid constraints removed: they rejected reality 72%
  of the time against NN bidders. Soft weights only.
- `bid_belief_v4` reaches play log(p) = **-0.9565**, vs -1.0209 heuristic and
  -1.099 uniform.
- The old `belief_v3.bin` is **not usable** with NN bots. Current play belief
  net: `belief_v4_fix_v2.bin` (retrained after the TrumpCeilingTracker fix,
  val_loss 0.8797).

## Docs

- [bis_dd.md](bis_dd.md) — Bis-DD, a DD-only bid+play agent (**removed 2026-07-24**; kept for its negative result on heuristic bid inference)
- [playgen.md](playgen.md) — **Playgen world sampler**: a causal transformer
  that generates whole worlds autoregressively instead of predicting marginals.
  Same job as the belief nets (posterior over hidden hands), different method —
  which is why the two are benchmarked head to head there.

## Eval

- **`eval_beliefs`** — belief quality against ground truth, per bid step and
  per trick: log-probability, placement accuracy, false exclusion rate,
  entropy, constraint tightness, ground truth reachability. Plays deals with NN
  bots. `--nn` for the play belief net, `--bid-belief` for the bid belief net.

  ```bash
  cargo run --bin eval_beliefs --features "parallel,nn" --release -- \
    --deals 500 [--bid-belief models/bid_belief_v4.bin]
  ```

- **`bench_world_cred`** — compares belief nets against playgen and uniform as
  *world samplers*, judged by whether the reference policy would replay the
  observed hidden actions. See [playgen.md](playgen.md).
