/// Alpha-beta double-dummy solver for Belote Contrée.
///
/// Techniques (inspired by bridge DD solvers like DDS/GIB):
/// - Alpha-beta with fail-soft
/// - Transposition table with relative (future) scores and hash move
/// - Card equivalence pruning — only one representative per equivalence class
/// - Quick tricks bounds — guaranteed future points from top cards
/// - Upper/lower bound pre-pruning
/// - Principal Variation Search (null-window for non-PV moves)
/// - Forced-move skip for single legal cards
/// - Move ordering: hash move → killer moves → history heuristic + static score
/// - Killer move heuristic (2 per ply)
/// - History heuristic (depth² bonus on cutoff)
///
/// No external dependencies — compiled unconditionally.
use crate::card::*;
use crate::play;
use crate::state::*;

const TT_EXACT: u8 = 0;
const TT_LOWER: u8 = 1;
const TT_UPPER: u8 = 2;
const TT_SIZE: usize = 1 << 18; // 256K entries = 2MB (L2 cache friendly)

/// TT entry layout (u64):
/// bits 63-40: key (24 bits)
/// bits 39-24: future_score as u16 (16 bits)
/// bits 23-21: flag (3 bits)
/// bits 20-16: best_move card index (5 bits)
/// bits 15-1:  unused
/// bit 0:      occupied sentinel

/// Solve a playing-phase state: returns [team0_points, team1_points] with perfect play.
pub fn solve(state: &GameState) -> [u8; 2] {
    debug_assert_eq!(state.phase, Phase::Playing);

    let mut tt = vec![0u64; TT_SIZE];
    let mut history = [[0u32; 32]; 2]; // [team][card] — cutoff history
    let mut killers = [[EMPTY; 2]; 32]; // [ply][0..2] — killer moves per ply
    let ns_pts = alphabeta(state, 0, 252, &mut tt, &mut history, &mut killers);

    let ew_pts = if ns_pts == 252 || ns_pts == 0 {
        252 - ns_pts
    } else {
        162 - ns_pts
    };

    [ns_pts as u8, ew_pts as u8]
}

/// Solve a playing-phase state using an external TT buffer (avoids repeated 2MB allocations).
/// The TT is cleared at the start of each call. History and killers are stack-allocated.
pub fn solve_reuse_tt(state: &GameState, tt_buf: &mut [u64]) -> [u8; 2] {
    debug_assert_eq!(state.phase, Phase::Playing);

    // Clear TT for this solve
    tt_buf.iter_mut().for_each(|x| *x = 0);

    let mut history = [[0u32; 32]; 2];
    let mut killers = [[EMPTY; 2]; 32];
    let ns_pts = alphabeta(state, 0, 252, tt_buf, &mut history, &mut killers);

    let ew_pts = if ns_pts == 252 || ns_pts == 0 {
        252 - ns_pts
    } else {
        162 - ns_pts
    };

    [ns_pts as u8, ew_pts as u8]
}

/// Convenience: create DD state and solve for a specific trump suit.
pub fn solve_for_trump(hands: [CardSet; 4], dealer: u8, trump: u8) -> [u8; 2] {
    let state = GameState::setup_dd(dealer, hands, trump);
    solve(&state)
}

/// Convenience: solve for trump using an external TT buffer.
pub fn solve_for_trump_reuse_tt(
    hands: [CardSet; 4],
    dealer: u8,
    trump: u8,
    tt_buf: &mut [u64],
) -> [u8; 2] {
    let state = GameState::setup_dd(dealer, hands, trump);
    solve_reuse_tt(&state, tt_buf)
}

/// Allocate a fresh TT buffer for reuse across multiple `solve_with_scores` calls.
/// Returns a 2MB Vec suitable for passing to `solve_with_scores`.
pub fn new_tt_buffer() -> Vec<u64> {
    vec![0u64; TT_SIZE]
}

/// Per-card DD scores returned by `solve_with_scores`.
/// Fixed-size array (max 8 legal moves in Belote), no heap allocation.
pub struct SolveScores {
    /// (card, ns_points) for each legal move, sorted by score descending for NS.
    pub scores: [(u8, i16); 8],
    /// Number of valid entries in `scores`.
    pub count: usize,
    /// Best card for the current player's team.
    pub best_card: u8,
}

/// Solve and return DD scores for every legal root move.
///
/// Like `solve_best_card` but collects `(card, ns_score)` for every root move.
/// An optional external TT buffer can be passed to avoid repeated 2MB allocations
/// across determinizations. The TT is cleared at the start of each call.
pub fn solve_with_scores(state: &GameState, tt_buf: Option<&mut Vec<u64>>) -> SolveScores {
    debug_assert_eq!(state.phase, Phase::Playing);
    debug_assert!(!state.is_terminal());

    let mut owned_tt;
    let tt: &mut Vec<u64> = match tt_buf {
        Some(buf) => {
            // Clear TT for this solve (different determinized worlds = different trees)
            buf.iter_mut().for_each(|x| *x = 0);
            buf
        }
        None => {
            owned_tt = vec![0u64; TT_SIZE];
            &mut owned_tt
        }
    };

    let mut history = [[0u32; 32]; 2];
    let mut killers = [[EMPTY; 2]; 32];

    let legal = play::legal_plays(state);
    let team = GameState::player_team(state.current_player);
    let maximizing = team == 0;

    let ordered = order_moves(state, legal, EMPTY, &history, [EMPTY; 2]);

    let mut result = SolveScores {
        scores: [(0, 0); 8],
        count: ordered.1,
        best_card: ordered.0[0],
    };

    let mut best_score = if maximizing { i16::MIN } else { i16::MAX };

    for i in 0..ordered.1 {
        let card = ordered.0[i];
        let mut child = *state;
        play::apply_play(&mut child, card);

        let score = if child.is_terminal() {
            child.points[0] as i16
        } else {
            alphabeta(&child, 0, 252, tt, &mut history, &mut killers)
        };

        result.scores[i] = (card, score);

        if maximizing {
            if score > best_score {
                best_score = score;
                result.best_card = card;
            }
        } else if score < best_score {
            best_score = score;
            result.best_card = card;
        }
    }

    result
}

/// Returns the optimal card for the current player to play (DD best move).
pub fn solve_best_card(state: &GameState) -> u8 {
    debug_assert_eq!(state.phase, Phase::Playing);
    debug_assert!(!state.is_terminal());

    let mut tt = vec![0u64; TT_SIZE];
    let mut history = [[0u32; 32]; 2];

    let legal = play::legal_plays(state);
    let team = GameState::player_team(state.current_player);
    let maximizing = team == 0;

    let mut best_card = legal.trailing_zeros() as u8;
    let mut best_score = if maximizing { i16::MIN } else { i16::MAX };

    let mut killers = [[EMPTY; 2]; 32];
    let ordered = order_moves(state, legal, EMPTY, &history, [EMPTY; 2]);
    for i in 0..ordered.1 {
        let card = ordered.0[i];
        let mut child = *state;
        play::apply_play(&mut child, card);

        let score = if child.is_terminal() {
            child.points[0] as i16
        } else {
            alphabeta(&child, 0, 252, &mut tt, &mut history, &mut killers)
        };

        if maximizing {
            if score > best_score {
                best_score = score;
                best_card = card;
            }
        } else if score < best_score {
            best_score = score;
            best_card = card;
        }
    }

    best_card
}

// ---- Core alpha-beta ----

#[inline(always)]
fn tt_pack(key: u32, future_score: i16, flag: u8, best_move: u8) -> u64 {
    let k = (key & 0x00FF_FFFF) as u64;
    (k << 40)
        | (((future_score as u16) as u64) << 24)
        | (((flag & 0x7) as u64) << 21)
        | (((best_move & 0x1F) as u64) << 16)
        | 1
}

#[inline(always)]
fn tt_unpack(packed: u64) -> (u32, i16, u8, u8) {
    let key = ((packed >> 40) & 0x00FF_FFFF) as u32;
    let future_score = ((packed >> 24) & 0xFFFF) as i16;
    let flag = ((packed >> 21) & 0x7) as u8;
    let best_move = ((packed >> 16) & 0x1F) as u8;
    (key, future_score, flag, best_move)
}

fn alphabeta(
    state: &GameState,
    mut alpha: i16,
    mut beta: i16,
    tt: &mut [u64],
    history: &mut [[u32; 32]; 2],
    killers: &mut [[u8; 2]; 32],
) -> i16 {
    if state.is_terminal() {
        return state.points[0] as i16;
    }

    // ---- Simple bounds pruning ----
    let remaining = 152 - state.points[0] as i16 - state.points[1] as i16;
    let dix_max = if state.tricks_won[1] == 0 { 100 } else { 10 };
    let ns_upper = state.points[0] as i16 + remaining + dix_max;
    if ns_upper <= alpha {
        return ns_upper;
    }
    let ns_lower = state.points[0] as i16;
    if ns_lower >= beta {
        return ns_lower;
    }

    let legal = play::legal_plays(state);

    // Forced move: single legal card
    if legal & (legal - 1) == 0 {
        let card = legal.trailing_zeros() as u8;
        let mut child = *state;
        play::apply_play(&mut child, card);
        return if child.is_terminal() {
            child.points[0] as i16
        } else {
            alphabeta(&child, alpha, beta, tt, history, killers)
        };
    }

    // ---- Quick tricks bounds (at trick start only) ----
    if state.trick_count == 0 {
        let (ns_quick, ew_quick) = quick_tricks(state);
        let ns_min = state.points[0] as i16 + ns_quick;
        if ns_min > alpha {
            alpha = ns_min;
        }
        let ns_max = state.points[0] as i16 + remaining - ew_quick + dix_max;
        if ns_max < beta {
            beta = ns_max;
        }
        if alpha >= beta {
            return if ns_min >= beta { ns_min } else { ns_max };
        }
    }

    let ns_base = state.points[0] as i16;

    let hash = position_hash(state);
    let tt_idx = (hash as usize) & (tt.len() - 1);
    let tt_key = (hash >> 40) as u32 & 0x00FF_FFFF;

    // TT probe
    let mut hash_move = EMPTY;
    let packed = tt[tt_idx];
    if packed != 0 {
        let (stored_key, stored_future, stored_flag, stored_move) = tt_unpack(packed);
        if stored_key == tt_key {
            let stored_abs = stored_future + ns_base;
            match stored_flag {
                TT_EXACT => return stored_abs,
                TT_LOWER => {
                    if stored_abs > alpha {
                        alpha = stored_abs;
                    }
                }
                TT_UPPER => {
                    if stored_abs < beta {
                        beta = stored_abs;
                    }
                }
                _ => {}
            }
            if alpha >= beta {
                return stored_abs;
            }
            if (legal & card_to_bit(stored_move)) != 0 {
                hash_move = stored_move;
            }
        }
    }

    let team = GameState::player_team(state.current_player);
    let maximizing = team == 0;
    let ply = (state.tricks_won[0] + state.tricks_won[1]) as usize * 4
        + state.trick_count as usize;

    // Apply card equivalence + order with hash move first, then killers, then by history
    let reduced = reduce_equivalent(legal, state);
    let killer_pair = if ply < 32 { killers[ply] } else { [EMPTY; 2] };
    let ordered = order_moves(state, reduced, hash_move, history, killer_pair);

    let orig_alpha = alpha;
    let orig_beta = beta;
    let mut best_score = if maximizing { i16::MIN } else { i16::MAX };
    let mut best_move = ordered.0[0];

    for i in 0..ordered.1 {
        let card = ordered.0[i];
        let mut child = *state;
        play::apply_play(&mut child, card);

        let score = if child.is_terminal() {
            child.points[0] as i16
        } else if i == 0 {
            alphabeta(&child, alpha, beta, tt, history, killers)
        } else {
            // PVS: null window search after first move
            let scout = if maximizing {
                alphabeta(&child, alpha, alpha + 1, tt, history, killers)
            } else {
                alphabeta(&child, beta - 1, beta, tt, history, killers)
            };
            let needs_research = if maximizing {
                scout > alpha && scout < orig_beta
            } else {
                scout < beta && scout > orig_alpha
            };
            if needs_research {
                alphabeta(&child, alpha, beta, tt, history, killers)
            } else {
                scout
            }
        };

        if maximizing {
            if score > best_score {
                best_score = score;
                best_move = card;
            }
            if score > alpha {
                alpha = score;
            }
        } else {
            if score < best_score {
                best_score = score;
                best_move = card;
            }
            if score < beta {
                beta = score;
            }
        }

        if alpha >= beta {
            // Killer heuristic: remember cutoff-causing card at this ply
            if ply < 32 && card != killers[ply][0] {
                killers[ply][1] = killers[ply][0];
                killers[ply][0] = card;
            }
            // History heuristic: reward the cutoff-causing card
            let depth = 8 - (state.tricks_won[0] + state.tricks_won[1]);
            history[team as usize][card as usize] += (depth as u32) * (depth as u32);
            break;
        }
    }

    let future_score = best_score - ns_base;
    let flag = if best_score <= orig_alpha {
        TT_UPPER
    } else if best_score >= orig_beta {
        TT_LOWER
    } else {
        TT_EXACT
    };

    tt[tt_idx] = tt_pack(tt_key, future_score, flag, best_move);
    best_score
}

// ---- Quick tricks ----

/// Ranks ordered by trump strength descending: J(3), 9(2), A(7), 10(6), K(5), Q(4), 8(1), 7(0).
const TRUMP_RANK_ORDER: [u8; 8] = [3, 2, 7, 6, 5, 4, 1, 0];

/// Compute guaranteed future card points for each team from "sure" tricks.
/// Returns (ns_quick_points, ew_quick_points).
/// Only valid at trick start (trick_count == 0).
fn quick_tricks(state: &GameState) -> (i16, i16) {
    let trump = state.contract.trump as usize;
    let trump_shift = SUIT_SHIFT[trump];

    let ns_hand = state.hands[0] | state.hands[2];
    let ew_hand = state.hands[1] | state.hands[3];

    let mut ns_pts = 0i16;
    let mut ew_pts = 0i16;

    // Trump suit: consecutive top trumps are always guaranteed
    let ns_trump = ((ns_hand >> trump_shift) & 0xFF) as u8;
    let ew_trump = ((ew_hand >> trump_shift) & 0xFF) as u8;

    // NS top consecutive trumps
    for &rank in &TRUMP_RANK_ORDER {
        let bit = 1u8 << rank;
        if (ns_trump | ew_trump) & bit == 0 {
            continue; // already played
        }
        if ns_trump & bit != 0 {
            ns_pts += TRUMP_POINTS[rank as usize] as i16;
        } else {
            break;
        }
    }

    // EW top consecutive trumps
    for &rank in &TRUMP_RANK_ORDER {
        let bit = 1u8 << rank;
        if (ns_trump | ew_trump) & bit == 0 {
            continue;
        }
        if ew_trump & bit != 0 {
            ew_pts += TRUMP_POINTS[rank as usize] as i16;
        } else {
            break;
        }
    }

    // Plain suits: master cards where neither opponent can ruff
    for suit in 0..4u8 {
        if suit as usize == trump {
            continue;
        }
        let shift = SUIT_SHIFT[suit as usize];
        let ns_suit = ((ns_hand >> shift) & 0xFF) as u8;
        let ew_suit = ((ew_hand >> shift) & 0xFF) as u8;

        if ns_suit == 0 || ew_suit == 0 {
            continue; // one team void → no guaranteed plain tricks for either
        }

        // NS masters: check if either EW player can ruff
        let e_void = ((state.hands[1] >> shift) & 0xFF) == 0;
        let w_void = ((state.hands[3] >> shift) & 0xFF) == 0;
        let e_has_trump = ((state.hands[1] >> trump_shift) & 0xFF) != 0;
        let w_has_trump = ((state.hands[3] >> trump_shift) & 0xFF) != 0;
        let ew_can_ruff = (e_void && e_has_trump) || (w_void && w_has_trump);

        if !ew_can_ruff {
            // Count consecutive top plain cards held by NS
            for rank in (0..8u8).rev() {
                let bit = 1u8 << rank;
                if (ns_suit | ew_suit) & bit == 0 {
                    continue;
                }
                if ns_suit & bit != 0 {
                    ns_pts += PLAIN_POINTS[rank as usize] as i16;
                } else {
                    break;
                }
            }
        }

        // EW masters: check if either NS player can ruff
        let n_void = ((state.hands[0] >> shift) & 0xFF) == 0;
        let s_void = ((state.hands[2] >> shift) & 0xFF) == 0;
        let n_has_trump = ((state.hands[0] >> trump_shift) & 0xFF) != 0;
        let s_has_trump = ((state.hands[2] >> trump_shift) & 0xFF) != 0;
        let ns_can_ruff = (n_void && n_has_trump) || (s_void && s_has_trump);

        if !ns_can_ruff {
            for rank in (0..8u8).rev() {
                let bit = 1u8 << rank;
                if (ns_suit | ew_suit) & bit == 0 {
                    continue;
                }
                if ew_suit & bit != 0 {
                    ew_pts += PLAIN_POINTS[rank as usize] as i16;
                } else {
                    break;
                }
            }
        }
    }

    (ns_pts, ew_pts)
}

// ---- Card equivalence ----

/// Reduce legal moves by removing equivalent cards.
/// Two cards in the same suit are equivalent if:
/// 1. They are adjacent in the relevant ordering (no unplayed card between them)
/// 2. They have the same point value
/// We keep only one representative (the highest) from each equivalence class.
pub fn reduce_equivalent(legal: CardSet, state: &GameState) -> CardSet {
    let trump = state.contract.trump;
    let played = state.played_cards;
    let player = state.current_player as usize;
    let hand = state.hands[player];
    // "Outstanding" cards: in other players' hands (not played, not in our hand)
    let outstanding = ALL_CARDS & !played & !hand;

    let mut result = legal;

    for suit_idx in 0..4u8 {
        let shift = SUIT_SHIFT[suit_idx as usize];
        let legal_suit = ((legal >> shift) & 0xFF) as u8;
        if legal_suit == 0 || legal_suit & (legal_suit - 1) == 0 {
            continue; // 0 or 1 card in this suit — nothing to reduce
        }

        let outstanding_suit = ((outstanding >> shift) & 0xFF) as u8;

        if suit_idx == trump {
            result = reduce_trump_equiv(result, legal_suit, outstanding_suit, shift);
        } else {
            result = reduce_plain_equiv(result, legal_suit, outstanding_suit, shift);
        }
    }

    result
}

/// Reduce equivalent cards in a plain suit.
fn reduce_plain_equiv(
    mut result: CardSet,
    legal_bits: u8,
    outstanding_bits: u8,
    shift: u8,
) -> CardSet {
    let mut prev_rank: i8 = -1;
    let mut prev_pts: u8 = 0;
    let mut bits = legal_bits;

    while bits != 0 {
        let rank = bits.trailing_zeros() as u8;
        bits &= bits - 1;

        let pts = PLAIN_POINTS[rank as usize];

        if prev_rank >= 0 {
            let between_mask = between_bits(prev_rank as u8, rank);
            if between_mask & outstanding_bits == 0 && pts == prev_pts {
                result &= !(1u32 << (prev_rank as u32 + shift as u32));
            }
        }

        prev_rank = rank as i8;
        prev_pts = pts;
    }

    result
}

/// Reduce equivalent cards in a trump suit using trump strength ordering.
fn reduce_trump_equiv(
    mut result: CardSet,
    legal_bits: u8,
    outstanding_bits: u8,
    shift: u8,
) -> CardSet {
    let mut cards: [(u8, u8, u8); 8] = [(0, 0, 0); 8]; // (strength, rank, points)
    let mut count = 0;

    let mut bits = legal_bits;
    while bits != 0 {
        let rank = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        let strength = TRUMP_STRENGTH[rank as usize];
        let pts = TRUMP_POINTS[rank as usize];
        cards[count] = (strength, rank, pts);
        count += 1;
    }

    cards[..count].sort_unstable_by_key(|&(s, _, _)| s);

    let mut out_strengths: [u8; 8] = [0; 8];
    let mut out_count = 0;
    let mut obits = outstanding_bits;
    while obits != 0 {
        let rank = obits.trailing_zeros() as u8;
        obits &= obits - 1;
        out_strengths[out_count] = TRUMP_STRENGTH[rank as usize];
        out_count += 1;
    }
    out_strengths[..out_count].sort_unstable();

    for i in 1..count {
        let (prev_str, prev_rank, prev_pts) = cards[i - 1];
        let (curr_str, _curr_rank, curr_pts) = cards[i];

        if prev_pts != curr_pts {
            continue;
        }

        let has_between = out_strengths[..out_count]
            .iter()
            .any(|&s| s > prev_str && s < curr_str);

        if !has_between {
            result &= !(1u32 << (prev_rank as u32 + shift as u32));
        }
    }

    result
}

/// Bitmask of ranks strictly between lo and hi (exclusive).
#[inline]
fn between_bits(lo: u8, hi: u8) -> u8 {
    if hi <= lo + 1 {
        return 0;
    }
    let width = hi - lo - 1;
    ((1u8 << width) - 1) << (lo + 1)
}

// ---- Hashing ----

fn position_hash(state: &GameState) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;

    h ^= state.played_cards as u64;
    h = h.wrapping_mul(0x100000001b3);

    let trick_packed: u64 = (state.current_trick[0] as u64)
        | ((state.current_trick[1] as u64) << 8)
        | ((state.current_trick[2] as u64) << 16)
        | ((state.current_trick[3] as u64) << 24)
        | ((state.trick_lead as u64) << 32)
        | ((state.trick_count as u64) << 40);
    h ^= trick_packed;
    h = h.wrapping_mul(0x100000001b3);

    h ^= state.tricks_won[0] as u64;
    h = h.wrapping_mul(0x100000001b3);

    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;

    h
}

// ---- Move ordering ----

fn order_moves(
    state: &GameState,
    legal: CardSet,
    hash_move: u8,
    history: &[[u32; 32]; 2],
    killer_pair: [u8; 2],
) -> ([u8; 8], usize) {
    let trump = state.contract.trump;
    let ct = state.contract.contract_type();
    let team = GameState::player_team(state.current_player) as usize;

    let mut result = [0u8; 8];
    let mut count = 0usize;
    let mut used = 0u32; // bitmask of cards already placed in result

    // 1. Hash move first (if valid and legal)
    if hash_move < 32 && (legal & card_to_bit(hash_move)) != 0 {
        result[count] = hash_move;
        count += 1;
        used |= card_to_bit(hash_move);
    }

    // 2. Killer moves (if legal and not already placed)
    for &km in &killer_pair {
        if km < 32 && (legal & card_to_bit(km)) != 0 && (used & card_to_bit(km)) == 0 {
            result[count] = km;
            count += 1;
            used |= card_to_bit(km);
        }
    }

    // 3. Score and sort remaining moves (static heuristic + history bonus)
    let remaining = legal & !used;

    let mut scored: [(i32, u8); 8] = [(0, 0); 8];
    let mut scount = 0usize;

    let mut mask = remaining;
    while mask != 0 {
        let card = mask.trailing_zeros() as u8;
        mask &= mask - 1;
        let static_score = move_order_score(state, card, trump, ct) as i32;
        let hist_bonus = history[team][card as usize] as i32;
        scored[scount] = (static_score + hist_bonus, card);
        scount += 1;
    }

    scored[..scount].sort_unstable_by(|a, b| b.0.cmp(&a.0));

    for i in 0..scount {
        result[count] = scored[i].1;
        count += 1;
    }

    (result, count)
}

fn move_order_score(state: &GameState, card: u8, trump: u8, ct: ContractType) -> i16 {
    let suit = card_suit_u8(card);
    let is_trump = suit == trump;
    let rank = card_rank(card);
    let pts = card_points(card, ct) as i16;

    if state.trick_count == 0 {
        if is_trump {
            100 + TRUMP_STRENGTH[rank as usize] as i16
        } else {
            pts
        }
    } else {
        let lead_suit = card_suit_u8(state.current_trick[state.trick_lead as usize]);
        let can_win = is_trump || suit == lead_suit;

        if can_win {
            50 + pts
        } else {
            -pts
        }
    }
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    fn suit_hands() -> [CardSet; 4] {
        [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000]
    }

    #[test]
    fn test_solve_point_total_invariant() {
        let hands = suit_hands();
        let result = solve_for_trump(hands, 0, 0);
        let total = result[0] as u16 + result[1] as u16;
        assert!(total == 162 || total == 252, "got {}", total);
    }

    #[test]
    fn test_solve_all_trumps_capot() {
        let hands = suit_hands();
        let result = solve_for_trump(hands, 0, 0);
        assert_eq!(result[0] as u16 + result[1] as u16, 252);
        assert_eq!(result[0], 252);
    }

    #[test]
    fn test_solve_symmetry() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let r0 = solve_for_trump(hands, 0, 1);
        let hands_rot = [0xFF00_0000, 0xFF, 0xFF00, 0xFF_0000];
        let r1 = solve_for_trump(hands_rot, 3, 1);
        assert_eq!(r0[0], r1[1]);
        assert_eq!(r0[1], r1[0]);
    }

    #[test]
    fn test_solve_known_deal() {
        let p0 = card_to_bit(make_card(Suit::Hearts, 3))
            | card_to_bit(make_card(Suit::Hearts, 2))
            | card_to_bit(make_card(Suit::Spades, 0))
            | card_to_bit(make_card(Suit::Spades, 1))
            | card_to_bit(make_card(Suit::Spades, 2))
            | card_to_bit(make_card(Suit::Spades, 3))
            | card_to_bit(make_card(Suit::Diamonds, 0))
            | card_to_bit(make_card(Suit::Diamonds, 1));

        let p2 = card_to_bit(make_card(Suit::Hearts, 7))
            | card_to_bit(make_card(Suit::Hearts, 6))
            | card_to_bit(make_card(Suit::Spades, 4))
            | card_to_bit(make_card(Suit::Spades, 5))
            | card_to_bit(make_card(Suit::Spades, 6))
            | card_to_bit(make_card(Suit::Spades, 7))
            | card_to_bit(make_card(Suit::Diamonds, 2))
            | card_to_bit(make_card(Suit::Diamonds, 3));

        let ns = p0 | p2;
        let remaining = ALL_CARDS & !ns;
        let ew_cards: Vec<u8> = CardIter(remaining).collect();
        let mut p1: CardSet = 0;
        let mut p3: CardSet = 0;
        for (i, &c) in ew_cards.iter().enumerate() {
            if i < 8 {
                p1 |= card_to_bit(c);
            } else {
                p3 |= card_to_bit(c);
            }
        }

        let result = solve_for_trump([p0, p1, p2, p3], 0, 1);
        let total = result[0] as u16 + result[1] as u16;
        assert!(total == 162 || total == 252, "got {}", total);
        assert!(result[0] >= 55, "NS got {}", result[0]);
    }

    #[test]
    fn test_solve_best_card() {
        let state = GameState::setup_dd(0, suit_hands(), 0);
        let best = solve_best_card(&state);
        assert_eq!(card_suit_u8(best), 1, "P1 should lead a heart");
    }

    #[test]
    fn test_solve_mid_game() {
        let mut state = GameState::setup_dd(0, suit_hands(), 1);
        play::apply_play(&mut state, make_card(Suit::Hearts, 7));
        play::apply_play(&mut state, make_card(Suit::Diamonds, 0));
        play::apply_play(&mut state, make_card(Suit::Clubs, 0));
        play::apply_play(&mut state, make_card(Suit::Spades, 0));

        let result = solve(&state);
        let total = result[0] as u16 + result[1] as u16;
        assert!(total == 162 || total == 252, "got {}", total);
    }

    #[test]
    fn test_card_equivalence_plain() {
        let legal: CardSet = 0b11; // 7S, 8S
        let mut state = GameState::setup_dd(0, [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000], 1);
        state.current_player = 0;
        let reduced = reduce_equivalent(legal, &state);
        assert_eq!(reduced.count_ones(), 1);
    }

    #[test]
    fn test_card_equivalence_different_points() {
        let legal: CardSet = 0b1100; // 9S, JS
        let mut state = GameState::setup_dd(0, [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000], 1);
        state.current_player = 0;
        let reduced = reduce_equivalent(legal, &state);
        assert_eq!(reduced.count_ones(), 2);
    }

    #[test]
    fn test_between_bits() {
        assert_eq!(between_bits(0, 1), 0); // adjacent
        assert_eq!(between_bits(0, 2), 0b10); // rank 1 between 0 and 2
        assert_eq!(between_bits(1, 5), 0b11100); // ranks 2,3,4 between 1 and 5
        assert_eq!(between_bits(3, 3), 0); // same
        assert_eq!(between_bits(0, 7), 0b0111_1110); // ranks 1-6
    }

    #[test]
    fn test_quick_tricks_top_trumps() {
        // NS has J+9 of trump (spades) → sure 34 pts (20+14)
        let p0 = card_to_bit(make_card(Suit::Spades, 3)) // J
            | card_to_bit(make_card(Suit::Spades, 2)) // 9
            | card_to_bit(make_card(Suit::Hearts, 0))
            | card_to_bit(make_card(Suit::Hearts, 1))
            | card_to_bit(make_card(Suit::Hearts, 2))
            | card_to_bit(make_card(Suit::Hearts, 3))
            | card_to_bit(make_card(Suit::Hearts, 4))
            | card_to_bit(make_card(Suit::Hearts, 5));
        let p2 = card_to_bit(make_card(Suit::Diamonds, 0))
            | card_to_bit(make_card(Suit::Diamonds, 1))
            | card_to_bit(make_card(Suit::Diamonds, 2))
            | card_to_bit(make_card(Suit::Diamonds, 3))
            | card_to_bit(make_card(Suit::Diamonds, 4))
            | card_to_bit(make_card(Suit::Diamonds, 5))
            | card_to_bit(make_card(Suit::Diamonds, 6))
            | card_to_bit(make_card(Suit::Diamonds, 7));
        let ns = p0 | p2;
        let remaining = ALL_CARDS & !ns;
        let ew_cards: Vec<u8> = CardIter(remaining).collect();
        let mut p1: CardSet = 0;
        let mut p3: CardSet = 0;
        for (i, &c) in ew_cards.iter().enumerate() {
            if i < 8 {
                p1 |= card_to_bit(c);
            } else {
                p3 |= card_to_bit(c);
            }
        }

        let state = GameState::setup_dd(0, [p0, p1, p2, p3], 0); // trump = spades
        let (ns_quick, _ew_quick) = quick_tricks(&state);
        assert!(ns_quick >= 34, "NS should have at least 34 quick pts from J+9, got {}", ns_quick);
    }

    #[test]
    fn test_tt_pack_unpack() {
        let key = 0x123456u32;
        let future_score = 42i16;
        let flag = TT_EXACT;
        let best_move = 17u8;

        let packed = tt_pack(key, future_score, flag, best_move);
        let (k, fs, f, bm) = tt_unpack(packed);
        assert_eq!(k, key & 0x00FF_FFFF);
        assert_eq!(fs, future_score);
        assert_eq!(f, flag);
        assert_eq!(bm, best_move);
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_solve_random_deals_point_total() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..20 {
            let state = GameState::deal_random(0, &mut rng);
            let hands = state.hands;
            for trump in 0..4u8 {
                let result = solve_for_trump(hands, 0, trump);
                let total = result[0] as u16 + result[1] as u16;
                assert!(
                    total == 162 || total == 252,
                    "Total must be 162 or 252, got {} (trump={})",
                    total, trump
                );
            }
        }
    }

    #[test]
    fn test_solve_with_scores_consistent_with_best_card() {
        let state = GameState::setup_dd(0, suit_hands(), 0);
        let best = solve_best_card(&state);
        let scores = solve_with_scores(&state, None);
        assert_eq!(scores.best_card, best);
        assert!(scores.count > 0);
    }

    #[test]
    fn test_solve_with_scores_tt_reuse() {
        let state = GameState::setup_dd(0, suit_hands(), 0);
        let mut tt = new_tt_buffer();

        let s1 = solve_with_scores(&state, Some(&mut tt));
        let s2 = solve_with_scores(&state, Some(&mut tt));

        // Same state → same results
        assert_eq!(s1.best_card, s2.best_card);
        assert_eq!(s1.count, s2.count);
        for i in 0..s1.count {
            assert_eq!(s1.scores[i], s2.scores[i]);
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_solve_reuse_tt_matches_solve() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut tt = new_tt_buffer();

        for _ in 0..10 {
            let state = GameState::deal_random(0, &mut rng);
            let hands = state.hands;
            for trump in 0..4u8 {
                let expected = solve_for_trump(hands, 0, trump);
                let actual = solve_for_trump_reuse_tt(hands, 0, trump, &mut tt);
                assert_eq!(expected, actual, "Mismatch for trump={}", trump);
            }
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_solve_with_scores_random_deals() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let mut tt = new_tt_buffer();

        for _ in 0..10 {
            let state = GameState::deal_random(0, &mut rng);
            let hands = state.hands;
            for trump in 0..4u8 {
                let dd_state = GameState::setup_dd(0, hands, trump);
                let scores = solve_with_scores(&dd_state, Some(&mut tt));
                let best_card_result = solve_best_card(&dd_state);

                assert!(scores.count > 0);
                assert_eq!(scores.best_card, best_card_result);

                // All returned cards must be legal
                let legal = play::legal_plays(&dd_state);
                for i in 0..scores.count {
                    let (card, ns_score) = scores.scores[i];
                    assert!(
                        legal & card_to_bit(card) != 0,
                        "Card {} not legal",
                        card
                    );
                    assert!(ns_score >= 0 && ns_score <= 252);
                }
            }
        }
    }
}
