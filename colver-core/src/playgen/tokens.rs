//! Tokenization of a full game (auction + plays) from one observer's perspective,
//! with per-play observer-visible legality masks.
//!
//! Sequence layout (max 98 tokens):
//! ```text
//! [BOS] [OBSPOS_d] [h1..h8] [bid tokens ×B≤24] ([ACT_a] [CARD])×P≤32
//! ```
//! Each token = primary + suit + actor + segment embeddings (+ absolute position).
//!
//! - `OBSPOS_d`: observer seat relative to dealer.
//! - Hand tokens: the observer's 8 initial cards (rank primary + suit embedding),
//!   sorted by canonical (suit, rank).
//! - Bid tokens: one per auction action, actor embedding = bidder relative to observer.
//! - Play tokens come in pairs: an `ACT_a` query token (actor of the next play,
//!   known from the state machine) followed by the played `CARD` token. The model
//!   predicts the card at the `ACT` position — 32-way logits masked to the
//!   observer-visible legal set.
//!
//! Suits are canonicalized via a suit permutation with `perm[trump] = 0` (trump
//! always lands on suit 0). The 3 non-trump suits can be permuted freely for
//! data augmentation (6 variants).
//!
//! Masks (observer-visible, canonical suit space):
//! - actor == observer: the true legal mask (observer knows their own hand).
//! - hidden actor: unseen cards minus hard-constraint exclusions (voids, deduced
//!   trump voids, trump ceilings) via `TrumpCeilingTracker` — the same machinery
//!   as belief training, with 0% false exclusion by construction.

use crate::card;
use crate::game_replay::{GameReplay, TrumpCeilingTracker};
use crate::state::{GameState, Phase};
use crate::suit_perm::permute_mask;

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

pub const P_PAD: u8 = 0;
pub const P_BOS: u8 = 1;
pub const P_OBSPOS0: u8 = 2; // +0..3
pub const P_RANK0: u8 = 6; // +0..7 (card tokens: hand + plays)
pub const P_PASS: u8 = 14;
pub const P_VAL0: u8 = 15; // +0..8 (bids 80..160)
pub const P_CAPOT: u8 = 24;
pub const P_COINCHE: u8 = 25;
pub const P_SURCOINCHE: u8 = 26;
pub const P_ACT0: u8 = 27; // +0..3 (actor query tokens, observer-relative)
pub const NUM_PRIMARY: usize = 31;

pub const S_NULL: u8 = 4;
pub const NUM_SUIT: usize = 5;

pub const A_NULL: u8 = 4;
pub const NUM_ACTOR: usize = 5;

pub const SEG_HEADER: u8 = 0;
pub const SEG_BID: u8 = 1;
pub const SEG_PLAY: u8 = 2;
pub const NUM_SEG: usize = 3;

pub const MAX_BID_TOKENS: usize = 24;
pub const MAX_SEQ_LEN: usize = 2 + 8 + MAX_BID_TOKENS + 64; // 98
pub const NUM_CARD_ACTIONS: usize = 32;

// ---------------------------------------------------------------------------
// Sample
// ---------------------------------------------------------------------------

/// One tokenized game from one observer's perspective.
pub struct PlaygenSample {
    pub primary: Vec<u8>,
    pub suit: Vec<u8>,
    pub actor: Vec<u8>,
    pub segment: Vec<u8>,
    /// Positions of the ACT query tokens (where card logits are read).
    pub pred_pos: Vec<u16>,
    /// Canonical card id (0-31) actually played at each prediction.
    pub targets: Vec<u8>,
    /// Observer-visible legal mask (canonical space) at each prediction.
    pub masks: Vec<u32>,
    /// Whether the actor at each prediction is the observer (0) or hidden (1).
    pub hidden_actor: Vec<bool>,
    /// Trick index (0-7) at each prediction, for per-stage metrics.
    pub trick_idx: Vec<u8>,
}

impl PlaygenSample {
    pub fn len(&self) -> usize {
        self.primary.len()
    }

    pub fn is_empty(&self) -> bool {
        self.primary.is_empty()
    }
}

/// Canonical suit permutation that maps `trump` to suit 0 by swapping.
/// Deterministic — use for eval; for training augmentation pick any of the
/// 6 permutations with `perm[trump] == 0` (see `random_trump_perm`).
pub fn canonical_trump_perm(trump: u8) -> [u8; 4] {
    let mut perm = [0u8, 1, 2, 3];
    perm.swap(0, trump as usize);
    perm
}

/// Random suit permutation with `perm[trump] == 0` (6 variants).
#[cfg(feature = "rand")]
pub fn random_trump_perm(trump: u8, rng: &mut impl rand::Rng) -> [u8; 4] {
    let others: [u8; 3] = match trump {
        0 => [1, 2, 3],
        1 => [0, 2, 3],
        2 => [0, 1, 3],
        _ => [0, 1, 2],
    };
    // Random assignment of the 3 non-trump suits onto canonical slots 1..3
    const ORDERS: [[usize; 3]; 6] = [
        [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0],
    ];
    let ord = ORDERS[rng.gen_range(0..6)];
    let mut perm = [0u8; 4];
    perm[trump as usize] = 0;
    for (slot, &o) in ord.iter().enumerate() {
        perm[others[o] as usize] = slot as u8 + 1;
    }
    perm
}

fn permute_card(c: u8, perm: &[u8; 4]) -> u8 {
    perm[(c / 8) as usize] * 8 + (c % 8)
}

fn bid_action_token(action: u8) -> (u8, u8) {
    // Returns (primary, physical_suit_or_255)
    match action {
        0 => (P_PASS, 255),
        1..=36 => {
            let value_idx = (action - 1) / 4;
            let suit = (action - 1) % 4;
            (P_VAL0 + value_idx, suit)
        }
        37..=40 => (P_CAPOT, action - 37),
        41 => (P_COINCHE, 255),
        _ => (P_SURCOINCHE, 255),
    }
}

/// Observer-visible mask for a hidden actor (physical suit space):
/// unseen cards minus hard-constraint exclusions.
///
/// Les coupes et plafonds d'atout viennent de `TrumpCeilingTracker` ; la belote
/// vient de `play::belote_facts` et s'ajoute **ici seulement**, pas dans
/// `compute_hard_constraints` — la même fonction alimente l'obs du réseau de
/// croyances à l'exécution d'IS-DD (`is_dd.rs`), et changer ses 96 flottants
/// périmerait `belief_v4_fix_v2` sans rapport avec playgen.
fn hidden_actor_mask(
    tracker: &TrumpCeilingTracker,
    state: &GameState,
    observer: u8,
    actor: u8,
) -> u32 {
    let hc = tracker.compute_hard_constraints(state, observer);
    let rel = (actor + 4 - observer) % 4; // 1..3
    let base = (rel as usize - 1) * 32;
    let mut mask = 0u32;
    for c in 0..32usize {
        if hc[base + c] == 0.0 {
            mask |= 1 << c;
        }
    }
    // Belote : l'annonce est publique, donc ses deux déductions le sont aussi.
    // `banned` porte déjà l'implication de `held` (une carte forcée chez un
    // siège est interdite aux trois autres), donc un seul champ suffit.
    mask & !crate::play::belote_facts(state).banned[actor as usize]
}

/// Tokenize a full game replay from `observer`'s perspective.
///
/// `perm` must satisfy `perm[trump] == 0` for the deal's final contract
/// (checked by debug_assert). Returns `None` for void deals (no play phase).
pub fn tokenize_replay(
    replay: &GameReplay,
    observer: u8,
    perm: &[u8; 4],
) -> Option<PlaygenSample> {
    let mut state = GameState::new(replay.dealer, replay.hands);

    // First pass over actions is not needed: bid tokens are emitted as we walk.
    let est_len = 2 + 8 + replay.actions.len() + 32;
    let mut primary = Vec::with_capacity(est_len);
    let mut suit = Vec::with_capacity(est_len);
    let mut actor = Vec::with_capacity(est_len);
    let mut segment = Vec::with_capacity(est_len);
    let mut pred_pos = Vec::with_capacity(32);
    let mut targets = Vec::with_capacity(32);
    let mut masks = Vec::with_capacity(32);
    let mut hidden_actor_v = Vec::with_capacity(32);
    let mut trick_idx_v = Vec::with_capacity(32);

    // Header: BOS + observer position relative to dealer
    primary.push(P_BOS);
    suit.push(S_NULL);
    actor.push(A_NULL);
    segment.push(SEG_HEADER);

    let obs_pos = (observer + 4 - replay.dealer) % 4;
    primary.push(P_OBSPOS0 + obs_pos);
    suit.push(S_NULL);
    actor.push(A_NULL);
    segment.push(SEG_HEADER);

    // Hand: observer's 8 initial cards, canonical order
    let mut hand_cards: Vec<u8> = (0..32u8)
        .filter(|&c| replay.hands[observer as usize] & (1 << c) != 0)
        .map(|c| permute_card(c, perm))
        .collect();
    hand_cards.sort_unstable();
    for &c in &hand_cards {
        primary.push(P_RANK0 + c % 8);
        suit.push(c / 8);
        actor.push(0); // observer's own cards
        segment.push(SEG_HEADER);
    }

    // Bid tokens buffered so we can truncate to the last MAX_BID_TOKENS
    let mut bid_toks: Vec<(u8, u8, u8)> = Vec::new(); // (primary, suit, actor)

    let mut tracker = TrumpCeilingTracker::new();
    let mut plays_flushed = false;
    let mut num_plays = 0usize;

    for &action in &replay.actions {
        match state.phase {
            Phase::Bidding => {
                let bidder = state.current_player();
                let (p_tok, phys_suit) = bid_action_token(action);
                let s_tok = if phys_suit == 255 { S_NULL } else { perm[phys_suit as usize] };
                bid_toks.push((p_tok, s_tok, (bidder + 4 - observer) % 4));
                state.step(action);
            }
            Phase::Playing => {
                if !plays_flushed {
                    // Auction is over: trump is known, flush bid tokens.
                    debug_assert_eq!(
                        perm[state.contract.trump as usize], 0,
                        "perm must map trump to canonical suit 0"
                    );
                    let skip = bid_toks.len().saturating_sub(MAX_BID_TOKENS);
                    for &(p, s, a) in &bid_toks[skip..] {
                        primary.push(p);
                        suit.push(s);
                        actor.push(a);
                        segment.push(SEG_BID);
                    }
                    plays_flushed = true;
                }

                let cur = state.current_player();
                let rel_actor = (cur + 4 - observer) % 4;
                let is_hidden = cur != observer;

                let mask_phys = if is_hidden {
                    hidden_actor_mask(&tracker, &state, observer, cur)
                } else {
                    state.legal_actions() as u32
                };
                let mask = permute_mask(mask_phys, perm);
                let target = permute_card(action, perm);
                debug_assert!(
                    mask & (1 << target) != 0,
                    "played card must be in observer-visible mask"
                );

                // ACT query token — logits are read here.
                pred_pos.push(primary.len() as u16);
                primary.push(P_ACT0 + rel_actor);
                suit.push(S_NULL);
                actor.push(rel_actor);
                segment.push(SEG_PLAY);

                targets.push(target);
                masks.push(mask);
                hidden_actor_v.push(is_hidden);
                trick_idx_v.push((num_plays / 4) as u8);

                // CARD token
                primary.push(P_RANK0 + target % 8);
                suit.push(target / 8);
                actor.push(rel_actor);
                segment.push(SEG_PLAY);

                tracker.record_play(&state, cur, action);
                state.step(action);
                num_plays += 1;
            }
            Phase::Done => break,
        }
    }

    if targets.is_empty() {
        return None; // void deal
    }
    debug_assert!(primary.len() <= MAX_SEQ_LEN);

    Some(PlaygenSample {
        primary,
        suit,
        actor,
        segment,
        pred_pos,
        targets,
        masks,
        hidden_actor: hidden_actor_v,
        trick_idx: trick_idx_v,
    })
}

// ---------------------------------------------------------------------------
// V2: physical suits (full 24-perm augmentation) + auction as prediction target
// ---------------------------------------------------------------------------
//
// Sequence layout (max 122 tokens):
// ```text
// [BOS] [OBSPOS_d] [h1..h8] ([ACT_a] [BIDTOK])×B≤24 ([ACT_a] [CARD])×P≤32
// ```
// Differences from v1:
// - No trump canonicalization: suits stay physical, permuted by an arbitrary
//   suit permutation (all 24 variants) for augmentation. The model must learn
//   suit equivariance and route "which suit is trump" from the contract tokens.
// - Every auction action is a prediction target: an `ACT_a` query token
//   precedes each bid token, and a 43-way head (masked to the public legal bid
//   set) is read there. Bid legality is public — no belief machinery needed.
// - Void deals (4 passes) are kept: they are auction-only samples.

pub const NUM_BID_ACTIONS: usize = 43;
pub const MAX_BID_ENTRIES_V2: usize = 24;
pub const MAX_SEQ_LEN_V2: usize = 2 + 8 + 2 * MAX_BID_ENTRIES_V2 + 64; // 122

/// One tokenized game from one observer's perspective (v2).
pub struct PlaygenSampleV2 {
    pub primary: Vec<u8>,
    pub suit: Vec<u8>,
    pub actor: Vec<u8>,
    pub segment: Vec<u8>,
    /// Positions of the play ACT query tokens (32-way card logits).
    pub pred_pos: Vec<u16>,
    /// Permuted card id (0-31) actually played at each prediction.
    pub targets: Vec<u8>,
    /// Observer-visible legal mask (permuted suit space) at each prediction.
    pub masks: Vec<u32>,
    pub hidden_actor: Vec<bool>,
    pub trick_idx: Vec<u8>,
    /// Positions of the bid ACT query tokens (43-way bid logits).
    pub bid_pred_pos: Vec<u16>,
    /// Permuted bid action (0-42) actually taken at each prediction.
    pub bid_targets: Vec<u8>,
    /// Public legal bid mask (permuted suit space) at each prediction.
    pub bid_masks: Vec<u64>,
}

impl PlaygenSampleV2 {
    pub fn len(&self) -> usize {
        self.primary.len()
    }

    pub fn is_empty(&self) -> bool {
        self.primary.is_empty()
    }
}

pub fn identity_perm() -> [u8; 4] {
    [0, 1, 2, 3]
}

/// Uniform random suit permutation (all 24 variants).
#[cfg(feature = "rand")]
pub fn random_suit_perm(rng: &mut impl rand::Rng) -> [u8; 4] {
    let mut perm = [0u8, 1, 2, 3];
    for i in (1..4usize).rev() {
        let j = rng.gen_range(0..=i);
        perm.swap(i, j);
    }
    perm
}

/// Apply a suit permutation to a bid action (0-42).
pub fn permute_bid_action(action: u8, perm: &[u8; 4]) -> u8 {
    match action {
        1..=36 => {
            let value_idx = (action - 1) / 4;
            let suit = (action - 1) % 4;
            value_idx * 4 + perm[suit as usize] + 1
        }
        37..=40 => 37 + perm[(action - 37) as usize],
        _ => action, // pass, coinche, surcoinche
    }
}

/// Apply a suit permutation to a 43-bit legal bid mask.
pub fn permute_bid_mask(mask: u64, perm: &[u8; 4]) -> u64 {
    let mut out = mask & ((1 << 0) | (1 << 41) | (1 << 42));
    let mut suited = mask & !((1u64 << 0) | (1 << 41) | (1 << 42));
    while suited != 0 {
        let a = suited.trailing_zeros() as u8;
        out |= 1 << permute_bid_action(a, perm);
        suited &= suited - 1;
    }
    out
}

/// Tokenize a full game replay from `observer`'s perspective (v2).
///
/// `perm` is any suit permutation (no trump constraint). Returns `None` only
/// for pathological auctions longer than `MAX_BID_ENTRIES_V2` (absent from
/// generated corpora; engine-legal worst case is barely above). Void deals
/// are kept as auction-only samples (empty play targets).
pub fn tokenize_replay_v2(
    replay: &GameReplay,
    observer: u8,
    perm: &[u8; 4],
) -> Option<PlaygenSampleV2> {
    let mut state = GameState::new(replay.dealer, replay.hands);

    let est_len = 2 + 8 + 2 * replay.actions.len();
    let mut primary = Vec::with_capacity(est_len);
    let mut suit = Vec::with_capacity(est_len);
    let mut actor = Vec::with_capacity(est_len);
    let mut segment = Vec::with_capacity(est_len);
    let mut pred_pos = Vec::with_capacity(32);
    let mut targets = Vec::with_capacity(32);
    let mut masks = Vec::with_capacity(32);
    let mut hidden_actor_v = Vec::with_capacity(32);
    let mut trick_idx_v = Vec::with_capacity(32);
    let mut bid_pred_pos = Vec::with_capacity(8);
    let mut bid_targets = Vec::with_capacity(8);
    let mut bid_masks = Vec::with_capacity(8);

    // Header: BOS + observer position relative to dealer
    primary.push(P_BOS);
    suit.push(S_NULL);
    actor.push(A_NULL);
    segment.push(SEG_HEADER);

    let obs_pos = (observer + 4 - replay.dealer) % 4;
    primary.push(P_OBSPOS0 + obs_pos);
    suit.push(S_NULL);
    actor.push(A_NULL);
    segment.push(SEG_HEADER);

    // Hand: observer's 8 initial cards, sorted in permuted (suit, rank) order
    let mut hand_cards: Vec<u8> = (0..32u8)
        .filter(|&c| replay.hands[observer as usize] & (1 << c) != 0)
        .map(|c| permute_card(c, perm))
        .collect();
    hand_cards.sort_unstable();
    for &c in &hand_cards {
        primary.push(P_RANK0 + c % 8);
        suit.push(c / 8);
        actor.push(0); // observer's own cards
        segment.push(SEG_HEADER);
    }

    let mut tracker = TrumpCeilingTracker::new();
    let mut num_plays = 0usize;

    for &action in &replay.actions {
        match state.phase {
            Phase::Bidding => {
                if bid_targets.len() >= MAX_BID_ENTRIES_V2 {
                    return None; // pathological over-long auction
                }
                let bidder = state.current_player();
                let rel_actor = (bidder + 4 - observer) % 4;

                let mask = permute_bid_mask(state.legal_actions(), perm);
                let target = permute_bid_action(action, perm);
                debug_assert!(
                    mask & (1u64 << target) != 0,
                    "bid action must be in public legal mask"
                );

                // ACT query token — bid logits are read here.
                bid_pred_pos.push(primary.len() as u16);
                primary.push(P_ACT0 + rel_actor);
                suit.push(S_NULL);
                actor.push(rel_actor);
                segment.push(SEG_BID);

                bid_targets.push(target);
                bid_masks.push(mask);

                // BID token (suit already permuted via the permuted action)
                let (p_tok, perm_suit) = bid_action_token(target);
                primary.push(p_tok);
                suit.push(if perm_suit == 255 { S_NULL } else { perm_suit });
                actor.push(rel_actor);
                segment.push(SEG_BID);

                state.step(action);
            }
            Phase::Playing => {
                let cur = state.current_player();
                let rel_actor = (cur + 4 - observer) % 4;
                let is_hidden = cur != observer;

                let mask_phys = if is_hidden {
                    hidden_actor_mask(&tracker, &state, observer, cur)
                } else {
                    state.legal_actions() as u32
                };
                let mask = permute_mask(mask_phys, perm);
                let target = permute_card(action, perm);
                debug_assert!(
                    mask & (1 << target) != 0,
                    "played card must be in observer-visible mask"
                );

                // ACT query token — card logits are read here.
                pred_pos.push(primary.len() as u16);
                primary.push(P_ACT0 + rel_actor);
                suit.push(S_NULL);
                actor.push(rel_actor);
                segment.push(SEG_PLAY);

                targets.push(target);
                masks.push(mask);
                hidden_actor_v.push(is_hidden);
                trick_idx_v.push((num_plays / 4) as u8);

                // CARD token
                primary.push(P_RANK0 + target % 8);
                suit.push(target / 8);
                actor.push(rel_actor);
                segment.push(SEG_PLAY);

                tracker.record_play(&state, cur, action);
                state.step(action);
                num_plays += 1;
            }
            Phase::Done => break,
        }
    }

    debug_assert!(primary.len() <= MAX_SEQ_LEN_V2);

    Some(PlaygenSampleV2 {
        primary,
        suit,
        actor,
        segment,
        pred_pos,
        targets,
        masks,
        hidden_actor: hidden_actor_v,
        trick_idx: trick_idx_v,
        bid_pred_pos,
        bid_targets,
        bid_masks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rollout::select_nth_bit;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_replay(rng: &mut StdRng) -> GameReplay {
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, rng);
        let hands = state.hands;
        let mut actions = Vec::new();
        while !state.is_terminal() {
            let legal = state.legal_actions();
            let idx = rng.gen_range(0..legal.count_ones());
            let action = select_nth_bit(legal, idx);
            actions.push(action);
            state.step(action);
        }
        GameReplay { dealer, hands, actions }
    }

    fn final_trump(replay: &GameReplay) -> Option<u8> {
        let mut state = GameState::new(replay.dealer, replay.hands);
        for &a in &replay.actions {
            if state.phase == Phase::Playing {
                return Some(state.contract.trump);
            }
            state.step(a);
        }
        None
    }

    #[test]
    fn test_tokenize_invariants() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut tokenized = 0;

        for _ in 0..200 {
            let replay = random_replay(&mut rng);
            let Some(trump) = final_trump(&replay) else { continue };

            for observer in 0..4u8 {
                let perm = if rng.gen_bool(0.5) {
                    canonical_trump_perm(trump)
                } else {
                    random_trump_perm(trump, &mut rng)
                };
                assert_eq!(perm[trump as usize], 0);

                let Some(s) = tokenize_replay(&replay, observer, &perm) else {
                    panic!("non-void deal must tokenize")
                };
                tokenized += 1;

                assert!(s.len() <= MAX_SEQ_LEN);
                assert_eq!(s.primary.len(), s.suit.len());
                assert_eq!(s.primary.len(), s.actor.len());
                assert_eq!(s.primary.len(), s.segment.len());
                let p = s.pred_pos.len();
                assert_eq!(p, s.targets.len());
                assert_eq!(p, s.masks.len());
                assert!(p >= 4 && p <= 32, "plays: {}", p);

                for i in 0..p {
                    // The played card is always in the observer-visible mask.
                    assert!(
                        s.masks[i] & (1u32 << s.targets[i]) != 0,
                        "target {} not in mask {:032b}",
                        s.targets[i],
                        s.masks[i]
                    );
                    // ACT token followed by its CARD token.
                    let pos = s.pred_pos[i] as usize;
                    assert!(s.primary[pos] >= P_ACT0 && s.primary[pos] < P_ACT0 + 4);
                    assert_eq!(s.primary[pos + 1], P_RANK0 + s.targets[i] % 8);
                    assert_eq!(s.suit[pos + 1], s.targets[i] / 8);
                    // Trump canonicalization: any trump-suit target sits in suit 0.
                    assert!(s.masks[i] != 0);
                }

                // Vocabulary bounds
                for (&pr, (&su, &ac)) in
                    s.primary.iter().zip(s.suit.iter().zip(s.actor.iter()))
                {
                    assert!((pr as usize) < NUM_PRIMARY);
                    assert!((su as usize) < NUM_SUIT);
                    assert!((ac as usize) < NUM_ACTOR);
                }

                // Exactly 8 hand tokens
                let hand_toks = s.segment.iter().filter(|&&g| g == SEG_HEADER).count();
                assert_eq!(hand_toks, 10); // BOS + OBSPOS + 8 cards
            }
        }
        assert!(tokenized > 100, "too few tokenized games: {}", tokenized);
    }

    /// Full-file validation on real NN games (run manually):
    /// `COLVER_GAMES=data/training/playgen_smoke_5k.bin cargo test -p colver-core \
    ///  validate_games_file -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn validate_games_file() {
        let path = std::env::var("COLVER_GAMES").expect("set COLVER_GAMES");
        let replays = GameReplay::load_all(&path).expect("load failed");
        let mut rng = StdRng::seed_from_u64(0);
        let mut games = 0u64;
        let mut preds = 0u64;
        let mut forced = 0u64; // single-card masks
        let mut hidden_mask_bits = 0u64;
        let mut hidden_preds = 0u64;

        for replay in &replays {
            let Some(trump) = final_trump(replay) else { continue };
            for observer in 0..4u8 {
                let perm = random_trump_perm(trump, &mut rng);
                let Some(s) = tokenize_replay(replay, observer, &perm) else { continue };
                for i in 0..s.targets.len() {
                    assert!(
                        s.masks[i] & (1u32 << s.targets[i]) != 0,
                        "false exclusion: game with dealer {} obs {} pred {}",
                        replay.dealer, observer, i
                    );
                    preds += 1;
                    if s.masks[i].count_ones() == 1 {
                        forced += 1;
                    }
                    if s.hidden_actor[i] {
                        hidden_mask_bits += s.masks[i].count_ones() as u64;
                        hidden_preds += 1;
                    }
                }
            }
            games += 1;
        }
        println!(
            "validated {} games, {} preds, {:.1}% forced, avg hidden-mask size {:.1}",
            games,
            preds,
            forced as f64 / preds as f64 * 100.0,
            hidden_mask_bits as f64 / hidden_preds as f64,
        );
    }

    #[test]
    fn test_observer_mask_is_true_legal() {
        // For the observer's own plays, the mask must equal the legal mask.
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..50 {
            let replay = random_replay(&mut rng);
            let Some(trump) = final_trump(&replay) else { continue };
            let perm = canonical_trump_perm(trump);
            let observer = rng.gen_range(0..4u8);
            let Some(s) = tokenize_replay(&replay, observer, &perm) else { continue };

            // Re-walk the game and compare masks at observer decision points.
            let mut state = GameState::new(replay.dealer, replay.hands);
            let mut i = 0;
            for &a in &replay.actions {
                if state.phase == Phase::Playing {
                    if state.current_player() == observer {
                        let expected = permute_mask(state.legal_actions() as u32, &perm);
                        assert_eq!(s.masks[i], expected);
                        assert!(!s.hidden_actor[i]);
                    }
                    i += 1;
                }
                state.step(a);
            }
        }
    }

    #[test]
    fn test_tokenize_v2_invariants() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut tokenized = 0;
        let mut with_plays = 0;

        for _ in 0..200 {
            let replay = random_replay(&mut rng);
            for observer in 0..4u8 {
                let perm = if rng.gen_bool(0.25) {
                    identity_perm()
                } else {
                    random_suit_perm(&mut rng)
                };

                let Some(s) = tokenize_replay_v2(&replay, observer, &perm) else {
                    panic!("auction within bounds must tokenize")
                };
                tokenized += 1;

                assert!(s.len() <= MAX_SEQ_LEN_V2);
                assert_eq!(s.primary.len(), s.suit.len());
                assert_eq!(s.primary.len(), s.actor.len());
                assert_eq!(s.primary.len(), s.segment.len());
                assert_eq!(s.pred_pos.len(), s.targets.len());
                assert_eq!(s.pred_pos.len(), s.masks.len());
                assert_eq!(s.bid_pred_pos.len(), s.bid_targets.len());
                assert_eq!(s.bid_pred_pos.len(), s.bid_masks.len());
                // Min auction: bid+coinche+surcoinche (surcoinche ends bidding)
                assert!(s.bid_targets.len() >= 3, "at least 3 auction actions");

                for i in 0..s.bid_targets.len() {
                    assert!(
                        s.bid_masks[i] & (1u64 << s.bid_targets[i]) != 0,
                        "bid target {} not in mask {:043b}",
                        s.bid_targets[i],
                        s.bid_masks[i]
                    );
                    assert!((s.bid_targets[i] as usize) < NUM_BID_ACTIONS);
                    let pos = s.bid_pred_pos[i] as usize;
                    assert!(s.primary[pos] >= P_ACT0 && s.primary[pos] < P_ACT0 + 4);
                    assert_eq!(s.segment[pos], SEG_BID);
                }
                for i in 0..s.targets.len() {
                    assert!(s.masks[i] & (1u32 << s.targets[i]) != 0);
                    let pos = s.pred_pos[i] as usize;
                    assert!(s.primary[pos] >= P_ACT0 && s.primary[pos] < P_ACT0 + 4);
                    assert_eq!(s.segment[pos], SEG_PLAY);
                    assert_eq!(s.primary[pos + 1], P_RANK0 + s.targets[i] % 8);
                    assert_eq!(s.suit[pos + 1], s.targets[i] / 8);
                }
                if !s.targets.is_empty() {
                    assert_eq!(s.targets.len(), 32, "non-void deal has 32 plays");
                    with_plays += 1;
                }

                for (&pr, (&su, &ac)) in
                    s.primary.iter().zip(s.suit.iter().zip(s.actor.iter()))
                {
                    assert!((pr as usize) < NUM_PRIMARY);
                    assert!((su as usize) < NUM_SUIT);
                    assert!((ac as usize) < NUM_ACTOR);
                }
            }
        }
        assert!(tokenized >= 800 - 100, "too few tokenized: {}", tokenized);
        assert!(with_plays > 100);
    }

    #[test]
    fn test_v2_perm_equivariance() {
        // Tokenizing under a permutation must equal permuting the tokens:
        // masks/targets under perm == permuted masks/targets under identity.
        let mut rng = StdRng::seed_from_u64(1234);
        for _ in 0..50 {
            let replay = random_replay(&mut rng);
            let observer = rng.gen_range(0..4u8);
            let perm = random_suit_perm(&mut rng);
            let a = tokenize_replay_v2(&replay, observer, &identity_perm()).unwrap();
            let b = tokenize_replay_v2(&replay, observer, &perm).unwrap();

            assert_eq!(a.len(), b.len());
            for i in 0..a.targets.len() {
                assert_eq!(permute_card(a.targets[i], &perm), b.targets[i]);
                assert_eq!(permute_mask(a.masks[i], &perm), b.masks[i]);
            }
            for i in 0..a.bid_targets.len() {
                assert_eq!(permute_bid_action(a.bid_targets[i], &perm), b.bid_targets[i]);
                assert_eq!(permute_bid_mask(a.bid_masks[i], &perm), b.bid_masks[i]);
            }
        }
    }

    /// Full-file v2 validation on real NN games (run manually):
    /// `COLVER_GAMES=data/training/playgen_smoke_5k.bin cargo test -p colver-core \
    ///  validate_games_file_v2 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn validate_games_file_v2() {
        let path = std::env::var("COLVER_GAMES").expect("set COLVER_GAMES");
        let replays = GameReplay::load_all(&path).expect("load failed");
        let mut rng = StdRng::seed_from_u64(0);
        let mut games = 0u64;
        let mut skipped = 0u64;
        let mut play_preds = 0u64;
        let mut bid_preds = 0u64;

        for replay in &replays {
            let perm = random_suit_perm(&mut rng);
            for observer in 0..4u8 {
                let Some(s) = tokenize_replay_v2(replay, observer, &perm) else {
                    skipped += 1;
                    continue;
                };
                for i in 0..s.targets.len() {
                    assert!(
                        s.masks[i] & (1u32 << s.targets[i]) != 0,
                        "false exclusion: dealer {} obs {} pred {}",
                        replay.dealer, observer, i
                    );
                    play_preds += 1;
                }
                for i in 0..s.bid_targets.len() {
                    assert!(s.bid_masks[i] & (1u64 << s.bid_targets[i]) != 0);
                    bid_preds += 1;
                }
            }
            games += 1;
        }
        println!(
            "validated {} games ({} skipped): {} play preds, {} bid preds",
            games, skipped, play_preds, bid_preds
        );
    }

    #[test]
    fn test_hidden_mask_excludes_observer_hand() {
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..50 {
            let replay = random_replay(&mut rng);
            let Some(trump) = final_trump(&replay) else { continue };
            let perm = canonical_trump_perm(trump);
            let observer = rng.gen_range(0..4u8);
            let Some(s) = tokenize_replay(&replay, observer, &perm) else { continue };

            let obs_hand_canon =
                permute_mask(replay.hands[observer as usize], &perm);
            for i in 0..s.targets.len() {
                if s.hidden_actor[i] {
                    assert_eq!(
                        s.masks[i] & obs_hand_canon,
                        0,
                        "hidden-actor mask must exclude observer's cards"
                    );
                }
            }
        }
    }
}
