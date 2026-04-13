# Belief Documentation

Hidden card inference: estimating opponents' hands from observable game state.

Implementation: [colver-core/src/belief/](../../colver-core/src/belief/).

## Models

- **`CardBeliefs`** ([belief/card_beliefs.rs](../../colver-core/src/belief/card_beliefs.rs)) — heuristic, deprecated. Bidirectional soft inference from bids and play. 0% false exclusion rate on hard constraints.
- **`BeliefState`** (used by [is_dd.rs](../../colver-core/src/search/is_dd.rs)) — soft weights for IS-DD determinization sampling.
- **Belief NN v3** (`models/belief_v3.bin`) — 32×3 hidden card prediction, used by `smart_ismcts` + `smart_is_dd` with belief enabled.
- **Bid Belief NN v4** (`models/bid_belief_v4.bin`) — 108→256²→96, replaces heuristic bid soft weights in BeliefState.

## Docs

- [bis_dd.md](bis_dd.md) — IS-DD with belief net integration

## Eval

Run `cargo run --bin eval_beliefs --features "parallel,nn" --release` to measure belief quality (log-prob, placement accuracy, false exclusion rate) against ground truth.
