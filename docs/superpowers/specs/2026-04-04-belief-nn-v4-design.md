# Belief NN v4 — Bid Inference Network

## Problem

The current belief system for BisDdAgent has two layers:
1. **Soft bid weights** (heuristic: J×5, 9×3, A×2 for bidders) — barely better than uniform (log(p) = -1.075 vs -1.099)
2. **Soft play weights** (heuristic: lead ace → boost 10/K, discard → reduce A/10, etc.) — improves steadily trick by trick

The previous belief_v3.bin NN was trained on heuristic bot data and is catastrophically bad with NN bots (log(p) = -2.1, worse than uniform). We need a new NN trained on bid_v2 (Bid a Dede) auction data that replaces the weak heuristic bid weights.

## Design

### Consumer

`BeliefState` in BisDdAgent. The NN output replaces the heuristic bid soft weights. Play heuristic weights continue to multiply on top during the play phase:

```
Before: uniform(1.0) × bid_heuristic × play_heuristic
After:  NN_bid_prior  × play_heuristic
```

The NN can also be called mid-auction by BisDdAgent when evaluating bid EV via DD — providing better determinization priors than uniform.

### Observation Format (108 floats)

Reuses the exact same layout as `bid_obs.rs` (`BID_OBS_DIM = 108`), parameterized by an explicit `observer` instead of `current_player`:

| Block | Content | Size | Encoding |
|-------|---------|------|----------|
| Hand | Observer's cards | 32 | Binary 0/1 |
| Bid history | 12 slots × 6 floats | 72 | Player-relative to observer. Pass=0.2, coinche=0.8, surcoinche=1.0, capot=0.6+val, bid=0.4+val+suit_onehot |
| Position | Dealer-relative seat | 4 | One-hot |
| **Total** | | **108** | |

A new function `write_bid_belief_obs(buf, offset, state, bid_history, observer)` in `belief_obs.rs` wraps the existing `encode_bid_history` logic but uses the explicit observer for the hand and player-relative encoding.

### Output Format

- **96 logits** = 32 cards × 3 classes (left / partner / right, relative to observer)
- Per-card softmax → P(player | card) probabilities
- **Loss**: Masked cross-entropy on unknown cards (mask = all cards not in observer's hand = 24 cards, constant during bidding)
- Optional count regularization (weight 0.1): penalizes unrealistic per-player card counts vs ground truth

### Architecture

Standard MLP, small since input is only 108:

```
108 → 256 → 256 → 96
     LN+ReLU  LN+ReLU
```

~93K parameters. Layer norm + ReLU at each hidden layer. Same architecture as `BeliefQNet` in `belief_candle.rs` but with different dims.

### Training Data Generation

New binary: `gen_bid_belief_data`

**Pipeline:**
1. Deal random hands (using `rand`)
2. Run bid_v2 auction (all 4 players use the same NN bidder)
3. At **each bid step** (before each player acts), record one sample per player (4 observers)
4. Skip void deals (4 passes with no bid) — no useful signal
5. Write to binary file

**Samples per deal:** ~5-8 bid steps × 4 observers = ~20-32 samples/deal

**Volume:** 500K deals → ~10-15M samples. Generation is fast (no play model, no DD) — seconds on multi-core.

**Binary format (COLVBB01):**
```
Header:
  [8 bytes]  Magic: "COLVBB01"
  [4 bytes]  obs_dim: u32 (108)
  [8 bytes]  num_samples: u64

Per sample (436 + 32 + 4 = 472 bytes):
  [432 bytes]  obs: f32 × 108
  [32 bytes]   target: u8 × 32 (player-relative: 0=observer, 1=left, 2=partner, 3=right)
  [4 bytes]    mask: u32 (unknown cards = ~observer_hand, always 24 bits set)
```

### Integration into BeliefState

```rust
// In BeliefState, after each bid action (or at end of auction):
pub fn apply_nn_bid_weights(&mut self, net: &BidBeliefNet, state: &GameState, bid_history: &[(u8, u8)]) {
    for observer in 0..4 {
        let obs = write_bid_belief_obs(state, bid_history, observer);
        let logits = net.evaluate(&obs);
        let probs = softmax_per_card(logits, 3); // 32 × 3
        // Set weights[observer][player][card] = probs[card][relative_player]
        // This replaces the heuristic bid soft weights entirely
    }
}
```

During play, existing play heuristic weights (lead ace signals, discard signals, etc.) multiply on top of these NN priors. Hard constraints (voids, trump ceiling) remain unchanged.

### Inference Module

New struct `BidBeliefNet` in `belief_net.rs` (or a new `bid_belief_net.rs`):
- Same weight format as `BeliefNet` (raw f32 little-endian)
- `load(path)` → auto-detect from file size (108 input, 256 hidden, 96 output)
- `evaluate(&obs) -> [f32; 96]` → forward pass, ~0.1ms

### Training

Reuse existing `belief_candle.rs` training infrastructure:
- `BeliefQNet` with `obs_dim=108, hidden=256, num_classes=3`
- Masked cross-entropy + optional count regularization
- AdamW, cosine LR with warmup
- 24× suit augmentation via `suit_perm.rs` (need `permute_bid_belief_obs` function)

### Data Augmentation

24 suit permutations applied to each sample:
- Permute the 32-card hand bits
- Permute suit references in bid history (suit one-hot positions)
- Permute target array (card indices change under suit permutation)
- Permute mask

This is the same augmentation used for bid NN training. `permute_bid_obs()` already exists in `suit_perm.rs` and works directly since the obs layout is identical.

### Evaluation

Extend `eval_beliefs` binary to include a 4th belief source:
- **CB** (CardBeliefs — deprecated but kept for comparison)
- **BS** (BeliefState with heuristic bid weights)
- **BS+NN** (BeliefState with NN bid weights replacing heuristic)
- Optionally **NN-only** (raw NN output without play heuristic)

Key metric: log(p) at trick 0 (before any play signals) — this is where the NN should shine vs the heuristic's near-uniform performance.

### File Inventory

| File | Action | Description |
|------|--------|-------------|
| `belief_obs.rs` | Modify | Add `write_bid_belief_obs()` |
| `belief_net.rs` | Modify | Add `BidBeliefNet` struct (or reuse `BeliefNet` with obs_dim=108) |
| `belief_candle.rs` | Minor | Ensure `BeliefQNet` works with obs_dim=108 (likely already does) |
| `belief_state.rs` | Modify | Add `apply_nn_bid_weights()`, integrate with existing flow |
| `suit_perm.rs` | None | Existing `permute_bid_obs()` reusable as-is |
| `src/bin/gen_bid_belief_data.rs` | Create | Data generator binary |
| `src/bin/train_belief_net.rs` | Minor | Ensure it handles COLVBB01 format / obs_dim=108 |
| `src/bin/eval_beliefs.rs` | Modify | Add BS+NN evaluation path |

### Success Criteria

- log(p) at trick 0 significantly better than heuristic (-1.075) — target: < -0.95
- No regression on play-phase log(p) (play heuristics still multiply on top)
- Arena: BS+NN bot ≥ current nn_v2_isdd champion in h2h
- Generation + training completes in < 30 minutes total
