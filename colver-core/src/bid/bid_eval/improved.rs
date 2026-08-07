use crate::bidding::{self, BID_COINCHE, BID_PASS};
use crate::card::*;
use crate::state::*;
use super::eval_helpers::{
    evaluate_for_trump, evaluate_suit, count_side_aces, quality_ok, has_lead,
    count_total_aces, bidding_position,
};

// ---------------------------------------------------------------------------
// improved_bid: balanced deterministic bidder
// ---------------------------------------------------------------------------

/// Balanced score→bid-value mapping (tournament-tuned). Returns encoded value (value/10), or 0 for PASS.
fn balanced_bid_value(score: u16) -> u8 {
    if score < 10 {
        0 // PASS
    } else if score < 13 {
        8 // 80
    } else if score < 17 {
        9 // 90
    } else if score < 20 {
        10 // 100
    } else if score < 25 {
        11 // 110
    } else {
        12 // 120
    }
}

/// Tournament-tuned balanced bidder. Quality gate + score→value mapping (10→80, 13→90,
/// 17→100, 20→110, 25→120). Caps: opening 120, overcall 120, response 130.
/// Won round-robin tournament with 62% overall win rate vs 5 other strategies,
/// then fine-tuned (110-threshold 21→20) to 52.6% in 12-strategy fine-tune tournament.
pub fn improved_bid(state: &GameState) -> u8 {
    debug_assert_eq!(state.phase, Phase::Bidding);

    let player = state.current_player;
    let hand = state.hands[player as usize];
    let legal = state.legal_actions();
    let partner = GameState::partner(player);

    // After coinche: always PASS (never surcoinche in deterministic bidder)
    if state.coinche_state > 0 {
        return BID_PASS;
    }

    // Coinche check (opponent bid, not yet coinched)
    if state.last_bid_value > 0 && state.coinche_state == 0 {
        let bidder_team = GameState::player_team(state.last_bidder);
        let my_team = GameState::player_team(player);

        if bidder_team != my_team {
            let their_suit = Suit::from_u8(state.last_bid_suit);
            let my_their_suit = evaluate_suit(hand, their_suit);

            // J+9 in opponent's suit → COINCHE
            if my_their_suit.has_jack && my_their_suit.has_nine {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
            // 4+ trumps in their suit + 1+ side ace → COINCHE
            if my_their_suit.trump_count >= 4 && count_side_aces(hand, their_suit) >= 1 {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
        }
    }

    // Opening: no bid yet
    if state.last_bid_value == 0 {
        return improved_opening(hand, &legal);
    }

    // Partner response
    if state.last_bidder == partner {
        return improved_respond(state, hand, &legal);
    }

    // Overcall (opponent bid)
    improved_overcall(state, hand, &legal)
}

fn improved_opening(hand: CardSet, legal: &u64) -> u8 {
    // Evaluate all 4 suits
    let mut scores = [0u16; 4];
    for suit_idx in 0..4u8 {
        scores[suit_idx as usize] = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
    }

    // Find best suit
    let mut best_suit = 0u8;
    let mut best_score = 0u16;
    for i in 0..4u8 {
        if scores[i as usize] > best_score {
            best_score = scores[i as usize];
            best_suit = i;
        }
    }

    // Quality gate
    if !quality_ok(hand, Suit::from_u8(best_suit)) {
        return BID_PASS;
    }

    let mut bid_value = balanced_bid_value(best_score);
    if bid_value == 0 {
        return BID_PASS;
    }
    // Cap opening at 120
    if bid_value > 12 {
        bid_value = 12;
    }

    let action = bidding::encode_bid(bid_value, best_suit);
    if legal & (1u64 << action) != 0 {
        action
    } else {
        BID_PASS
    }
}

pub(super) fn improved_respond(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let partner_suit = Suit::from_u8(state.last_bid_suit);
    let partner_value = state.last_bid_value;
    let my_score = evaluate_for_trump(hand, partner_suit);

    // Partner bid 130+: don't push higher
    if partner_value >= 13 {
        return BID_PASS;
    }

    // Support raise in partner's suit using balanced mapping
    let target_value = balanced_bid_value(my_score);
    // Cap at 130
    let target_value = if target_value > 13 { 13 } else { target_value };

    if target_value > partner_value {
        let action = bidding::encode_bid(target_value, state.last_bid_suit);
        if legal & (1u64 << action) != 0 {
            return action;
        }
    }

    // Alternative suit bid: if I can't support partner but have a strong suit of my own
    let mut alt_best_suit = 0u8;
    let mut alt_best_score = 0u16;
    for suit_idx in 0..4u8 {
        if suit_idx == state.last_bid_suit {
            continue;
        }
        let score = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
        if score > alt_best_score {
            alt_best_score = score;
            alt_best_suit = suit_idx;
        }
    }

    if alt_best_score >= 16 && quality_ok(hand, Suit::from_u8(alt_best_suit)) {
        let mut alt_value = balanced_bid_value(alt_best_score);
        // Cap at 120
        if alt_value > 12 {
            alt_value = 12;
        }
        if alt_value > partner_value {
            let action = bidding::encode_bid(alt_value, alt_best_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    BID_PASS
}

pub(super) fn improved_overcall(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    // Don't compete above 120
    if state.last_bid_value >= 12 {
        return BID_PASS;
    }

    // Find best non-opponent suit
    let mut best_suit = 0u8;
    let mut best_score = 0u16;
    for suit_idx in 0..4u8 {
        if suit_idx == state.last_bid_suit {
            continue;
        }
        let score = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
        if score > best_score {
            best_score = score;
            best_suit = suit_idx;
        }
    }

    if best_score >= 13 && quality_ok(hand, Suit::from_u8(best_suit)) {
        let mut bid_value = balanced_bid_value(best_score);
        // Cap at 120
        if bid_value > 12 {
            bid_value = 12;
        }
        // Must overbid
        if bid_value > state.last_bid_value {
            let action = bidding::encode_bid(bid_value, best_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    BID_PASS
}

// ---------------------------------------------------------------------------
// improved_v2_bid: Improved bidding with configurable enhancements.
// ---------------------------------------------------------------------------

/// Configuration for ImprovedV2 bidder.
#[derive(Clone, Copy, Debug)]
pub struct V2Config {
    pub name: &'static str,
    /// Lead bonus added to opening score when team has first trick lead.
    pub lead_bonus: u16,
    /// 4th position gate: minimum score to open in 4th seat.
    /// 0 = disabled (use normal quality gate). Requires J or 9 when > 0.
    pub fourth_pos_min: u16,
    /// Use structured partner response with complement bonuses.
    pub partner_response: bool,
    /// Jack complement bonus in partner response.
    pub resp_jack_bonus: i16,
    /// Nine complement bonus (with 2+ trumps) in partner response.
    pub resp_nine_bonus: i16,
    /// Per-side-ace bonus in partner response.
    pub resp_ace_bonus: i16,
    /// 3+ trump support bonus in partner response.
    pub resp_support_bonus: i16,
    /// 0-trump misfit penalty in partner response.
    pub resp_misfit_penalty: i16,
    /// Théorème 3 coinche: 0 trumps in opponent suit + N aces.
    pub theoreme3_aces: u32, // 0 = disabled, 3 = standard, 4 = conservative
}

impl V2Config {
    /// Full V2 as originally designed.
    pub fn full() -> Self {
        V2Config {
            name: "v2_full",
            lead_bonus: 2,
            fourth_pos_min: 15,
            partner_response: true,
            resp_jack_bonus: 3,
            resp_nine_bonus: 2,
            resp_ace_bonus: 2,
            resp_support_bonus: 1,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// Only Théorème 3 coinche, everything else = Improved.
    pub fn coinche_only() -> Self {
        V2Config {
            name: "v2_coinche",
            lead_bonus: 0,
            fourth_pos_min: 0,
            partner_response: false,
            resp_jack_bonus: 0,
            resp_nine_bonus: 0,
            resp_ace_bonus: 0,
            resp_support_bonus: 0,
            resp_misfit_penalty: 0,
            theoreme3_aces: 3,
        }
    }

    /// Coinche + partner response only (no lead, no 4th gate).
    pub fn coinche_resp() -> Self {
        V2Config {
            name: "v2_co+resp",
            lead_bonus: 0,
            fourth_pos_min: 0,
            partner_response: true,
            resp_jack_bonus: 3,
            resp_nine_bonus: 2,
            resp_ace_bonus: 2,
            resp_support_bonus: 1,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// Coinche + 4th position gate only.
    pub fn coinche_4th() -> Self {
        V2Config {
            name: "v2_co+4th",
            lead_bonus: 0,
            fourth_pos_min: 15,
            partner_response: false,
            resp_jack_bonus: 0,
            resp_nine_bonus: 0,
            resp_ace_bonus: 0,
            resp_support_bonus: 0,
            resp_misfit_penalty: 0,
            theoreme3_aces: 3,
        }
    }

    /// All except lead bonus.
    pub fn no_lead() -> Self {
        V2Config {
            name: "v2_nolead",
            lead_bonus: 0,
            fourth_pos_min: 15,
            partner_response: true,
            resp_jack_bonus: 3,
            resp_nine_bonus: 2,
            resp_ace_bonus: 2,
            resp_support_bonus: 1,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// Lead bonus +1 (half).
    pub fn lead1() -> Self {
        V2Config {
            name: "v2_lead1",
            lead_bonus: 1,
            fourth_pos_min: 15,
            partner_response: true,
            resp_jack_bonus: 3,
            resp_nine_bonus: 2,
            resp_ace_bonus: 2,
            resp_support_bonus: 1,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// Partner response with reduced bonuses (all halved).
    pub fn resp_light() -> Self {
        V2Config {
            name: "v2_rlight",
            lead_bonus: 0,
            fourth_pos_min: 0,
            partner_response: true,
            resp_jack_bonus: 2,
            resp_nine_bonus: 1,
            resp_ace_bonus: 1,
            resp_support_bonus: 0,
            resp_misfit_penalty: -2,
            theoreme3_aces: 3,
        }
    }

    /// Partner response: only misfit penalty (no positive bonuses).
    pub fn resp_misfit() -> Self {
        V2Config {
            name: "v2_misfit",
            lead_bonus: 0,
            fourth_pos_min: 0,
            partner_response: true,
            resp_jack_bonus: 0,
            resp_nine_bonus: 0,
            resp_ace_bonus: 0,
            resp_support_bonus: 0,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// Théorème 3 with 4 aces required (more conservative).
    pub fn coinche_4aces() -> Self {
        V2Config {
            name: "v2_co4ace",
            lead_bonus: 0,
            fourth_pos_min: 0,
            partner_response: false,
            resp_jack_bonus: 0,
            resp_nine_bonus: 0,
            resp_ace_bonus: 0,
            resp_support_bonus: 0,
            resp_misfit_penalty: 0,
            theoreme3_aces: 4,
        }
    }

    /// 4th position gate at 13 (looser than 15).
    pub fn fourth_loose() -> Self {
        V2Config {
            name: "v2_4th13",
            lead_bonus: 0,
            fourth_pos_min: 13,
            partner_response: false,
            resp_jack_bonus: 0,
            resp_nine_bonus: 0,
            resp_ace_bonus: 0,
            resp_support_bonus: 0,
            resp_misfit_penalty: 0,
            theoreme3_aces: 3,
        }
    }

    /// Best guess: coinche + light response + no lead + loose 4th.
    pub fn balanced() -> Self {
        V2Config {
            name: "v2_bal",
            lead_bonus: 0,
            fourth_pos_min: 13,
            partner_response: true,
            resp_jack_bonus: 2,
            resp_nine_bonus: 1,
            resp_ace_bonus: 1,
            resp_support_bonus: 0,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// Misfit-only response + coinche + 4th gate.
    pub fn defensive() -> Self {
        V2Config {
            name: "v2_def",
            lead_bonus: 0,
            fourth_pos_min: 15,
            partner_response: true,
            resp_jack_bonus: 0,
            resp_nine_bonus: 0,
            resp_ace_bonus: 0,
            resp_support_bonus: 0,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// Tournament winner: coinche + misfit penalty + 4th@15 + lead +1.
    pub fn defensive_lead1() -> Self {
        V2Config {
            name: "v2_def_l1",
            lead_bonus: 1,
            fourth_pos_min: 15,
            partner_response: true,
            resp_jack_bonus: 0,
            resp_nine_bonus: 0,
            resp_ace_bonus: 0,
            resp_support_bonus: 0,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// All presets for tournament.
    pub fn all_presets() -> Vec<V2Config> {
        vec![
            Self::full(),
            Self::coinche_only(),
            Self::coinche_resp(),
            Self::coinche_4th(),
            Self::no_lead(),
            Self::lead1(),
            Self::resp_light(),
            Self::resp_misfit(),
            Self::coinche_4aces(),
            Self::fourth_loose(),
            Self::balanced(),
            Self::defensive(),
        ]
    }
}

/// Configurable Improved V2 bidder.
pub fn improved_v2_configurable_bid(state: &GameState, cfg: &V2Config) -> u8 {
    debug_assert_eq!(state.phase, Phase::Bidding);

    let player = state.current_player;
    let hand = state.hands[player as usize];
    let legal = state.legal_actions();
    let partner = GameState::partner(player);

    // After coinche: always PASS
    if state.coinche_state > 0 {
        return BID_PASS;
    }

    // Coinche check (opponent bid, not yet coinched)
    if state.last_bid_value > 0 && state.coinche_state == 0 {
        let bidder_team = GameState::player_team(state.last_bidder);
        let my_team = GameState::player_team(player);

        if bidder_team != my_team {
            let their_suit = Suit::from_u8(state.last_bid_suit);
            let my_their_suit = evaluate_suit(hand, their_suit);

            // J+9 in opponent's suit → COINCHE
            if my_their_suit.has_jack && my_their_suit.has_nine {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
            // 4+ trumps in their suit + 1+ side ace → COINCHE
            if my_their_suit.trump_count >= 4 && count_side_aces(hand, their_suit) >= 1 {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
            // Théorème 3: 0 trumps in their suit + N total aces → COINCHE
            if cfg.theoreme3_aces > 0
                && my_their_suit.trump_count == 0
                && count_total_aces(hand) >= cfg.theoreme3_aces
            {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
        }
    }

    // Opening: no bid yet
    if state.last_bid_value == 0 {
        return v2_cfg_opening(state, hand, &legal, cfg);
    }

    // Partner response
    if state.last_bidder == partner {
        if cfg.partner_response {
            return v2_cfg_respond(state, hand, &legal, cfg);
        } else {
            return improved_respond(state, hand, &legal);
        }
    }

    // Overcall: reuse improved_overcall (already well-tuned)
    improved_overcall(state, hand, &legal)
}

fn v2_cfg_opening(state: &GameState, hand: CardSet, legal: &u64, cfg: &V2Config) -> u8 {
    // Evaluate all 4 suits
    let mut scores = [0u16; 4];
    for suit_idx in 0..4u8 {
        scores[suit_idx as usize] = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
    }

    // Find best suit
    let mut best_suit = 0u8;
    let mut best_score = 0u16;
    for i in 0..4u8 {
        if scores[i as usize] > best_score {
            best_score = scores[i as usize];
            best_suit = i;
        }
    }

    // 4th position gate
    if cfg.fourth_pos_min > 0 && bidding_position(state) == 3 {
        if best_score < cfg.fourth_pos_min {
            return BID_PASS;
        }
        let bits = suit_bits(hand, Suit::from_u8(best_suit));
        if bits & (1 << 3) == 0 && bits & (1 << 2) == 0 {
            return BID_PASS; // require J or 9
        }
    }

    // Quality gate
    if !quality_ok(hand, Suit::from_u8(best_suit)) {
        return BID_PASS;
    }

    // Lead bonus
    if cfg.lead_bonus > 0 && has_lead(state) {
        best_score += cfg.lead_bonus;
    }

    let mut bid_value = balanced_bid_value(best_score);
    if bid_value == 0 {
        return BID_PASS;
    }
    if bid_value > 12 {
        bid_value = 12; // cap at 120
    }

    let action = bidding::encode_bid(bid_value, best_suit);
    if legal & (1u64 << action) != 0 {
        action
    } else {
        BID_PASS
    }
}

fn v2_cfg_respond(state: &GameState, hand: CardSet, legal: &u64, cfg: &V2Config) -> u8 {
    let partner_suit = Suit::from_u8(state.last_bid_suit);
    let partner_value = state.last_bid_value;

    if partner_value >= 13 {
        return BID_PASS;
    }

    let base_score = evaluate_for_trump(hand, partner_suit);
    let trump_bits = suit_bits(hand, partner_suit);
    let trump_count = trump_bits.count_ones();
    let has_jack = trump_bits & (1 << 3) != 0;
    let has_nine = trump_bits & (1 << 2) != 0;

    let mut bonus: i16 = 0;
    if has_jack {
        bonus += cfg.resp_jack_bonus;
    }
    if has_nine && trump_count >= 2 {
        bonus += cfg.resp_nine_bonus;
    }
    bonus += count_side_aces(hand, partner_suit) as i16 * cfg.resp_ace_bonus;
    if trump_count >= 3 {
        bonus += cfg.resp_support_bonus;
    }
    if trump_count == 0 {
        bonus += cfg.resp_misfit_penalty; // negative
    }

    let adjusted_score = (base_score as i16 + bonus).max(0) as u16;

    let target_value = balanced_bid_value(adjusted_score);
    let target_value = if target_value > 13 { 13 } else { target_value };

    if target_value > partner_value {
        let action = bidding::encode_bid(target_value, state.last_bid_suit);
        if legal & (1u64 << action) != 0 {
            return action;
        }
    }

    // Alternative suit
    let mut alt_best_suit = 0u8;
    let mut alt_best_score = 0u16;
    for suit_idx in 0..4u8 {
        if suit_idx == state.last_bid_suit {
            continue;
        }
        let score = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
        if score > alt_best_score {
            alt_best_score = score;
            alt_best_suit = suit_idx;
        }
    }

    if alt_best_score >= 16 && quality_ok(hand, Suit::from_u8(alt_best_suit)) {
        let mut alt_value = balanced_bid_value(alt_best_score);
        if alt_value > 12 {
            alt_value = 12;
        }
        if alt_value > partner_value {
            let action = bidding::encode_bid(alt_value, alt_best_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    BID_PASS
}

/// Default improved_v2_bid uses V2Config::defensive_lead1() (tournament winner).
pub fn improved_v2_bid(state: &GameState) -> u8 {
    improved_v2_configurable_bid(state, &V2Config::defensive_lead1())
}

// ---------------------------------------------------------------------------
// ImprovedV3: conservative bidding tuned via NN comparison analysis
// ---------------------------------------------------------------------------
//
// Key insight from arena traces vs nn_dmc35:
//   - V2 takes 76% of available contracts but wins only 45% of them
//   - NN takes 53% but wins 75% — much more selective
//   - 60% of big losses come from wrong suit or overbidding
//
// Changes vs V2:
//   1. Stricter quality: require J or 9 (not A/10/3+cards)
//   2. Require 2+ trump cards for any bid
//   3. Devalue K/Q of trump (1→0 each)
//   4. Higher score thresholds: 12→80, 15→90, 19→100, 23→110, 27→120
//   5. Conservative partner response: only with J, 9, or 3+ trumps
//   6. Tighter overcall: require 15+ and J or 9

/// V3 suit selection score: like evaluate_for_trump but with higher J/9 weight.
/// Used ONLY for choosing between suits, not for bid level.
/// J→10, 9→8 (vs V2's J→8, 9→6) to make trump control dominate suit choice.
const TRUMP_EVAL_V3: [u16; 8] = [0, 0, 8, 10, 1, 1, 3, 4];

fn v3_suit_score(hand: CardSet, suit: Suit) -> u16 {
    let mut score: u16 = 0;
    let trump_bits = suit_bits(hand, suit);
    let trump_count = trump_bits.count_ones() as u16;

    let mut b = trump_bits;
    while b != 0 {
        let rank = b.trailing_zeros() as usize;
        score += TRUMP_EVAL_V3[rank];
        b &= b - 1;
    }
    if trump_count > 2 {
        score += (trump_count - 2) * 2;
    }

    for suit_idx in 0..4u8 {
        if suit_idx == suit as u8 { continue; }
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        let count = bits.count_ones();
        if bits & (1 << 7) != 0 { score += 3; }
        if count == 0 { score += 3; }
        else if count == 1 { score += 1; }
    }
    score
}

pub fn improved_v3_bid(state: &GameState) -> u8 {
    debug_assert_eq!(state.phase, Phase::Bidding);

    let player = state.current_player;
    let hand = state.hands[player as usize];
    let legal = state.legal_actions();
    let partner = GameState::partner(player);

    // After coinche: always PASS
    if state.coinche_state > 0 {
        return BID_PASS;
    }

    // Coinche check (reuse V2 logic — it's already good)
    if state.last_bid_value > 0 && state.coinche_state == 0 {
        let bidder_team = GameState::player_team(state.last_bidder);
        let my_team = GameState::player_team(player);

        if bidder_team != my_team {
            let their_suit = Suit::from_u8(state.last_bid_suit);
            let my_their = evaluate_suit(hand, their_suit);

            // J+9 in opponent's suit → COINCHE
            if my_their.has_jack && my_their.has_nine {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
            // 4+ trumps in their suit + 1+ side ace → COINCHE
            if my_their.trump_count >= 4 && count_side_aces(hand, their_suit) >= 1 {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
            // Théorème 3: 0 trumps in their suit + 3 total aces → COINCHE
            if my_their.trump_count == 0 && count_total_aces(hand) >= 3 {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
        }
    }

    // Opening: no bid yet
    if state.last_bid_value == 0 {
        return v3_opening(state, hand, &legal);
    }

    // Partner response
    if state.last_bidder == partner {
        return v3_respond(state, hand, &legal);
    }

    // Overcall
    v3_overcall(state, hand, &legal)
}

fn v3_opening(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    // Use V3 scoring (higher J/9 weight) for suit SELECTION
    let mut suit_scores = [0u16; 4];
    for suit_idx in 0..4u8 {
        suit_scores[suit_idx as usize] = v3_suit_score(hand, Suit::from_u8(suit_idx));
    }

    let mut best_suit = 0u8;
    let mut best_suit_score = 0u16;
    for i in 0..4u8 {
        if suit_scores[i as usize] > best_suit_score {
            best_suit_score = suit_scores[i as usize];
            best_suit = i;
        }
    }

    // Use standard V2 scoring for bid LEVEL (thresholds are calibrated for it)
    let bid_score = evaluate_for_trump(hand, Suit::from_u8(best_suit));

    // Quality gate (same as V2)
    if !quality_ok(hand, Suit::from_u8(best_suit)) {
        return BID_PASS;
    }

    // 4th position gate: require score ≥ 15 and (J or 9)
    if bidding_position(state) == 3 {
        if bid_score < 15 {
            return BID_PASS;
        }
        let bits = suit_bits(hand, Suit::from_u8(best_suit));
        if bits & (1 << 3) == 0 && bits & (1 << 2) == 0 {
            return BID_PASS;
        }
    }

    // Lead bonus
    let mut score = bid_score;
    if has_lead(state) {
        score += 1;
    }

    let mut bid_value = balanced_bid_value(score);
    if bid_value == 0 {
        return BID_PASS;
    }
    // Cap opening at 120
    if bid_value > 12 {
        bid_value = 12;
    }

    let action = bidding::encode_bid(bid_value, best_suit);
    if legal & (1u64 << action) != 0 {
        action
    } else {
        BID_PASS
    }
}

fn v3_respond(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    // Reuse V2 improved_respond (already well-tuned for partner response)
    improved_respond(state, hand, legal)
}

fn v3_overcall(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    // Don't compete above 120
    if state.last_bid_value >= 12 {
        return BID_PASS;
    }

    // Use V3 scoring for suit selection, V2 scoring for bid level
    let mut best_suit = 0u8;
    let mut best_suit_score = 0u16;
    for suit_idx in 0..4u8 {
        if suit_idx == state.last_bid_suit { continue; }
        let score = v3_suit_score(hand, Suit::from_u8(suit_idx));
        if score > best_suit_score {
            best_suit_score = score;
            best_suit = suit_idx;
        }
    }

    let bid_score = evaluate_for_trump(hand, Suit::from_u8(best_suit));
    if bid_score >= 13 && quality_ok(hand, Suit::from_u8(best_suit)) {
        let mut bid_value = balanced_bid_value(bid_score);
        if bid_value > 12 { bid_value = 12; }
        if bid_value > state.last_bid_value {
            let action = bidding::encode_bid(bid_value, best_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    BID_PASS
}
