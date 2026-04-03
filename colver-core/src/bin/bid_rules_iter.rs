/// V3: History-aware rule-based bidder.
///
/// V1/V2 insight: static rules hit a ceiling ~65 pts below NN.
/// The NN's edge comes from reading bid history (72 floats).
///
/// V3 parses the full auction to extract signals:
///   - Partner bid? → complement-based raises
///   - Opponent bid light (after passes)? → aggressive coinche
///   - Opponent overbid our suit? → support coinche
///   - Nobody bid? → position-scaled aggression
///   - They coinched us? → surcoinche with strong hand
///   - Competitive auction? → defensive raises
///
/// Usage:
///   cargo run -p colver-core --bin bid_rules_iter --release -- [--matches N] [--bid-model PATH]

use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use colver_core::bid_eval;
use colver_core::bid_eval::evaluate_for_trump;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::bidding::{self, BID_COINCHE, BID_PASS, BID_SURCOINCHE};
use colver_core::card::*;
use colver_core::is_dd::{IsDdConfig, IsDdSearch};
use colver_core::scoring::compute_deal_score;
use colver_core::solver;
use colver_core::state::{GameState, Phase};

// =====================================================================
//  Auction Context — parsed from bid history
// =====================================================================

struct AuctionCtx {
    /// My bidding position (0 = first, 3 = last).
    position: u8,
    /// Number of passes before any bid was made.
    initial_passes: u8,
    /// Did my partner make a non-pass bid?
    partner_bid: bool,
    /// Partner's bid suit and level (if any).
    partner_suit: u8,
    partner_level: u8, // value/10 encoded
    /// Did an opponent make a non-pass bid?
    opponent_bid: bool,
    /// Opponent's latest bid suit and level.
    opponent_suit: u8,
    opponent_level: u8,
    /// Did the opponent overbid a bid from our team?
    opponent_overbid_us: bool,
    /// Did partner pass AFTER seeing an opponent bid? (partner is weak)
    partner_passed_over_opp: bool,
    /// Is this a competitive auction (both teams have bid)?
    competitive: bool,
    /// Total non-pass actions in the auction.
    total_bids: u8,
}

fn parse_auction(history: &[(u8, u8)], me: u8, partner: u8, my_team: u8) -> AuctionCtx {
    let mut ctx = AuctionCtx {
        position: 0,
        initial_passes: 0,
        partner_bid: false,
        partner_suit: 0,
        partner_level: 0,
        opponent_bid: false,
        opponent_suit: 0,
        opponent_level: 0,
        opponent_overbid_us: false,
        partner_passed_over_opp: false,
        competitive: false,
        total_bids: 0,
    };

    // Count initial passes
    let mut found_bid = false;
    for &(_, action) in history {
        if action == 0 && !found_bid {
            ctx.initial_passes += 1;
        } else if action >= 1 && action <= 40 {
            found_bid = true;
        }
    }

    ctx.position = ctx.initial_passes; // simplified: position ≈ passes before first bid or me

    let mut my_team_has_bid = false;
    let mut opp_team_has_bid = false;
    let mut last_my_team_bid = false; // did my team make the last real bid?

    for &(seat, action) in history {
        let seat_team = GameState::player_team(seat);
        let is_my_team = seat_team == my_team;

        match action {
            0 => {
                // PASS
                if seat == partner && opp_team_has_bid {
                    ctx.partner_passed_over_opp = true;
                }
            }
            1..=40 => {
                let (val, suit) = bidding::decode_bid(action);
                ctx.total_bids += 1;

                if is_my_team {
                    my_team_has_bid = true;
                    last_my_team_bid = true;
                    if seat == partner {
                        ctx.partner_bid = true;
                        ctx.partner_suit = suit;
                        ctx.partner_level = val;
                    }
                } else {
                    opp_team_has_bid = true;
                    ctx.opponent_bid = true;
                    ctx.opponent_suit = suit;
                    ctx.opponent_level = val;
                    // Did they overbid our team?
                    if last_my_team_bid {
                        ctx.opponent_overbid_us = true;
                    }
                    last_my_team_bid = false;
                }
            }
            41 | 42 => {
                ctx.total_bids += 1;
            }
            _ => {}
        }
    }

    ctx.competitive = my_team_has_bid && opp_team_has_bid;
    ctx
}

// =====================================================================
//  V3 Rule Set: History-Aware
// =====================================================================

/// Quality gate (same as V1: J or 9, or 5+ cards, or 4+ with ace).
fn quality_ok(hand: CardSet, suit: Suit) -> bool {
    let bits = suit_bits(hand, suit);
    let count = bits.count_ones();
    let has_j = bits & (1 << 3) != 0;
    let has_9 = bits & (1 << 2) != 0;
    let has_a = bits & (1 << 7) != 0;
    has_j || has_9 || count >= 5 || (count >= 4 && has_a)
}

fn side_aces(hand: CardSet, trump: Suit) -> u32 {
    (0..4u8)
        .filter(|&s| s != trump as u8 && hand & (1u32 << (s * 8 + 7)) != 0)
        .count() as u32
}

fn total_aces(hand: CardSet) -> u32 {
    (0..4u8)
        .filter(|&s| hand & (1u32 << (s * 8 + 7)) != 0)
        .count() as u32
}

/// Score → bid level. NN-calibrated: conservative opening, save high bids for
/// genuinely strong hands. NN bids 80 for 61%, 100 for 29%.
fn bid_level(score: u16) -> u8 {
    if score < 10 {
        0
    } else if score < 16 {
        8 // 80  (wider band — most hands open at 80)
    } else if score < 22 {
        10 // 100
    } else if score < 28 {
        11 // 110
    } else {
        12 // 120
    }
}

// --- Opening ---

fn v3_opening(state: &GameState, hand: CardSet, legal: u64, ctx: &AuctionCtx) -> u8 {
    let mut best_suit = 0u8;
    let mut best_score = 0u16;
    for s in 0..4u8 {
        let sc = evaluate_for_trump(hand, Suit::from_u8(s));
        if sc > best_score {
            best_score = sc;
            best_suit = s;
        }
    }

    if !quality_ok(hand, Suit::from_u8(best_suit)) {
        return BID_PASS;
    }

    // Position bonus: after 2+ passes, others are weak → be more aggressive
    if ctx.initial_passes >= 2 {
        best_score += 3;
    } else if ctx.initial_passes == 1 {
        best_score += 1;
    }

    let mut level = bid_level(best_score);
    if level == 0 {
        return BID_PASS;
    }
    if level > 12 {
        level = 12;
    }

    let action = bidding::encode_bid(level, best_suit);
    if legal & (1u64 << action) != 0 {
        action
    } else {
        BID_PASS
    }
}

// --- Partner Response (history-aware) ---

fn v3_respond(state: &GameState, hand: CardSet, legal: u64, ctx: &AuctionCtx) -> u8 {
    let partner_suit = Suit::from_u8(state.last_bid_suit);
    let partner_value = state.last_bid_value;

    if partner_value >= 13 {
        return BID_PASS;
    }

    let base = evaluate_for_trump(hand, partner_suit) as i16;
    let bits = suit_bits(hand, partner_suit);
    let count = bits.count_ones();
    let has_j = bits & (1 << 3) != 0;
    let has_9 = bits & (1 << 2) != 0;

    // Complement bonuses: partner already has strength, we add what they lack
    let mut bonus: i16 = 0;
    if has_j {
        bonus += 5; // J is huge complement
    }
    if has_9 && count >= 2 {
        bonus += 3;
    }
    bonus += side_aces(hand, partner_suit) as i16 * 2;
    if count >= 3 {
        bonus += 2; // trump support
    }
    if count == 0 {
        bonus -= 5; // misfit — don't support
    }

    // History-aware adjustments:
    // If opponent overbid partner → competitive, be more aggressive to defend
    if ctx.opponent_overbid_us {
        bonus += 3; // fight back
    }
    // If nobody else competed (just passes), partner's contract is likely fine
    // → only raise with real complement, not just marginal hands
    if !ctx.opponent_bid {
        bonus -= 2; // partner's contract stands, don't overbid it
    }

    let adjusted = (base + bonus).max(0) as u16;
    let mut target = bid_level(adjusted);
    if target > 13 {
        target = 13; // cap response at 130
    }

    if target > partner_value {
        let action = bidding::encode_bid(target, state.last_bid_suit);
        if legal & (1u64 << action) != 0 {
            return action;
        }
    }

    // Alternative suit (strong independent hand)
    let mut alt_suit = 0u8;
    let mut alt_score = 0u16;
    for s in 0..4u8 {
        if s == state.last_bid_suit {
            continue;
        }
        let sc = evaluate_for_trump(hand, Suit::from_u8(s));
        if sc > alt_score {
            alt_score = sc;
            alt_suit = s;
        }
    }
    if alt_score >= 16 && quality_ok(hand, Suit::from_u8(alt_suit)) {
        let mut alt_level = bid_level(alt_score);
        if alt_level > 12 {
            alt_level = 12;
        }
        if alt_level > partner_value {
            let action = bidding::encode_bid(alt_level, alt_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    BID_PASS
}

// --- Overcall / Rebid (history-aware) ---
//
// Key insight from diagnostics: NN competes aggressively in bidding wars.
// V3 was giving up too early. Now we:
// - Rebid partner's suit if we have support and opponent overbid
// - Compete in our own suit up to 130
// - Only give up if we truly have nothing

fn v3_overcall(state: &GameState, hand: CardSet, legal: u64, ctx: &AuctionCtx) -> u8 {
    let current_bid = state.last_bid_value; // encoded as value/10

    // Never compete above 130
    if current_bid >= 13 {
        return BID_PASS;
    }

    // CASE 1: Opponent overbid OUR team's suit → rebid with support
    // This is the "competitive raise" pattern the NN does heavily.
    if ctx.opponent_overbid_us && ctx.partner_bid {
        let our_suit = Suit::from_u8(ctx.partner_suit);
        let our_bits = suit_bits(hand, our_suit);
        let our_count = our_bits.count_ones();
        let our_has_j = our_bits & (1 << 3) != 0;
        let our_has_9 = our_bits & (1 << 2) != 0;
        let our_score = evaluate_for_trump(hand, our_suit);

        // With trump support (J, 9, or 3+ cards), rebid partner's suit
        let support = our_has_j || our_has_9 || our_count >= 3;
        if support && our_score >= 8 {
            // Bid just above opponent's level
            let target = current_bid + 1;
            if target <= 13 {
                let action = bidding::encode_bid(target, ctx.partner_suit);
                if legal & (1u64 << action) != 0 {
                    return action;
                }
            }
        }
    }

    // CASE 2: Standard overcall with own strong suit
    let mut best_suit = 0u8;
    let mut best_score = 0u16;
    for s in 0..4u8 {
        let sc = evaluate_for_trump(hand, Suit::from_u8(s));
        if sc > best_score {
            best_score = sc;
            best_suit = s;
        }
    }

    // Threshold depends on auction context
    let threshold: u16 = if ctx.partner_passed_over_opp {
        15 // partner weak, I'm alone
    } else if ctx.competitive {
        12 // competitive — fight with decent hands
    } else {
        13
    };

    if best_score >= threshold && quality_ok(hand, Suit::from_u8(best_suit)) {
        let mut level = bid_level(best_score);
        // In competitive auction, willing to go up to 130
        let cap = if ctx.competitive { 13 } else { 12 };
        if level > cap {
            level = cap;
        }
        if level > current_bid {
            let action = bidding::encode_bid(level, best_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
        // If we can't overbid at our level, try minimum overbid
        if ctx.competitive && best_score >= 12 {
            let min_bid = current_bid + 1;
            if min_bid <= 13 {
                let action = bidding::encode_bid(min_bid, best_suit);
                if legal & (1u64 << action) != 0 {
                    return action;
                }
            }
        }
    }

    BID_PASS
}

// --- Smart Coinche (history-aware) ---

fn v3_coinche(state: &GameState, hand: CardSet, legal: u64, ctx: &AuctionCtx) -> u8 {
    if legal & (1u64 << BID_COINCHE) == 0 {
        return BID_PASS;
    }

    let their_suit = Suit::from_u8(state.last_bid_suit);
    let bits = suit_bits(hand, their_suit);
    let count = bits.count_ones();
    let has_j = bits & (1 << 3) != 0;
    let has_9 = bits & (1 << 2) != 0;
    let has_a = bits & (1 << 7) != 0;
    let has_10 = bits & (1 << 6) != 0;
    let has_k = bits & (1 << 5) != 0;
    let has_q = bits & (1 << 4) != 0;

    // Classic coinche rules (from NN probe):
    // Rule 1: J+9 in their suit → always coinche
    if has_j && has_9 {
        return BID_COINCHE;
    }
    // Rule 2: 4+ trump in their suit + side ace → coinche
    if count >= 4 && side_aces(hand, their_suit) >= 1 {
        return BID_COINCHE;
    }
    // Rule 3: AKQT in their suit → coinche (from NN probe!)
    if has_a && has_k && has_q && has_10 {
        return BID_COINCHE;
    }
    // Rule 4: Théorème 3 — 0 in their suit + 3+ aces
    if count == 0 && total_aces(hand) >= 3 {
        return BID_COINCHE;
    }

    // NEW: Before applying speculative coinche rules, check if we'd rather
    // OVERBID instead. Diagnostics showed V3 coinching when it should compete.
    // Only use aggressive coinches when we can't overbid (no strong suit of our own).
    let my_best_score = (0..4u8)
        .map(|s| evaluate_for_trump(hand, Suit::from_u8(s)))
        .max()
        .unwrap_or(0);
    let can_overbid = my_best_score >= 12 && state.last_bid_value < 13;

    // Rule 5: J alone in their suit + 2+ side aces + can't overbid → coinche
    if !can_overbid && has_j && side_aces(hand, their_suit) >= 2 {
        return BID_COINCHE;
    }

    // Rule 6: Opponent overbid our partner's suit in the SAME suit, and
    // we have J or 9 in that suit. This means between partner + us we likely
    // have the top trumps → coinche.
    if ctx.opponent_overbid_us && ctx.partner_bid && state.last_bid_suit == ctx.partner_suit {
        if (has_j || has_9) && count >= 2 {
            return BID_COINCHE;
        }
    }

    BID_PASS
}

// --- Surcoinche (NEW in V3) ---

fn v3_surcoinche(_state: &GameState, _hand: CardSet, _legal: u64) -> u8 {
    // Diagnostics showed surcoinche is almost always a disaster (4× stakes).
    // NN rarely surcoinches either. Just pass.
    BID_PASS
}

// --- Main V3 bidding function ---

fn v3_bid(state: &GameState, history: &[(u8, u8)]) -> u8 {
    debug_assert_eq!(state.phase, Phase::Bidding);

    let me = state.current_player;
    let hand = state.hands[me as usize];
    let legal = state.legal_actions();
    let partner = GameState::partner(me);
    let my_team = GameState::player_team(me);

    let ctx = parse_auction(history, me, partner, my_team);

    // After coinche: consider surcoinche
    if state.coinche_state == 1 {
        // Our team was coinched → consider surcoinche
        let bidder_team = GameState::player_team(state.last_bidder);
        if bidder_team == my_team {
            // WE were coinched (opponent coinched our bid)
            // Actually coinche_state=1 means someone coinched. If the coincher is opponent
            // and the original bidder is our team, we can surcoinche.
            return v3_surcoinche(state, hand, legal);
        }
        return BID_PASS;
    }
    if state.coinche_state >= 2 {
        return BID_PASS; // after surcoinche: done
    }

    // Coinche opportunity (opponent just bid, not yet coinched)
    if state.last_bid_value > 0 && state.coinche_state == 0 {
        let bidder_team = GameState::player_team(state.last_bidder);
        if bidder_team != my_team {
            let c = v3_coinche(state, hand, legal, &ctx);
            if c != BID_PASS {
                return c;
            }
        }
    }

    // Opening
    if state.last_bid_value == 0 {
        return v3_opening(state, hand, legal, &ctx);
    }

    // Partner response
    if state.last_bidder == partner {
        return v3_respond(state, hand, legal, &ctx);
    }

    // Overcall
    v3_overcall(state, hand, legal, &ctx)
}

// =====================================================================
//  Tournament Engine
// =====================================================================

#[derive(Clone, Copy)]
enum Bidder {
    V3Rules,
    ImprovedV2,
    Nn,
}

struct DealResult {
    void: bool,
    ns_score: i16,
    ew_score: i16,
}

/// Play a deal with DD oracle (fast: ~60ms/deal, uses pre-computed DD points).
fn play_deal_dd(
    state: &mut GameState,
    ns_bidder: Bidder,
    ew_bidder: Bidder,
    nn: &mut Option<BidNet>,
    dd_pts: &[u8; 4],
) -> DealResult {
    let mut bid_history: Vec<(u8, u8)> = Vec::new();

    while state.phase == Phase::Bidding {
        let player = state.current_player();
        let team = GameState::player_team(player);
        let bidder = if team == 0 { ns_bidder } else { ew_bidder };

        let action = match bidder {
            Bidder::V3Rules => v3_bid(state, &bid_history),
            Bidder::ImprovedV2 => bid_eval::improved_v2_bid(state),
            Bidder::Nn => {
                let net = nn.as_mut().unwrap();
                let obs = bid_obs::make_bid_observation(state, &bid_history);
                let legal = state.legal_actions();
                net.best_action_fast(&obs, legal)
            }
        };

        bid_history.push((player, action));
        state.step(action);
    }

    if state.contract.value == 0 {
        return DealResult { void: true, ns_score: 0, ew_score: 0 };
    }

    let contract_trump = state.contract.trump;
    let ns_pts = dd_pts[contract_trump as usize];
    dd_score(state, ns_pts)
}

/// Play a deal with IS-DD (realistic: ~1-2s/deal, actual card play with beliefs).
fn play_deal_isdd(
    state: &mut GameState,
    ns_bidder: Bidder,
    ew_bidder: Bidder,
    nn: &mut Option<BidNet>,
    isdd_time_ms: u32,
    rng: &mut StdRng,
) -> DealResult {
    let mut bid_history: Vec<(u8, u8)> = Vec::new();

    while state.phase == Phase::Bidding {
        let player = state.current_player();
        let team = GameState::player_team(player);
        let bidder = if team == 0 { ns_bidder } else { ew_bidder };

        let action = match bidder {
            Bidder::V3Rules => v3_bid(state, &bid_history),
            Bidder::ImprovedV2 => bid_eval::improved_v2_bid(state),
            Bidder::Nn => {
                let net = nn.as_mut().unwrap();
                let obs = bid_obs::make_bid_observation(state, &bid_history);
                let legal = state.legal_actions();
                net.best_action_fast(&obs, legal)
            }
        };

        bid_history.push((player, action));
        state.step(action);
    }

    if state.contract.value == 0 {
        return DealResult { void: true, ns_score: 0, ew_score: 0 };
    }

    // IS-DD play: each player uses belief-weighted determinization + DD solver
    let isdd_config = IsDdConfig {
        determinizations: 20,
        time_limit_ms: Some(isdd_time_ms),
        ..Default::default()
    };

    let mut searches: Vec<IsDdSearch> = (0..4).map(|_| IsDdSearch::new()).collect();
    for p in 0..4u8 {
        searches[p as usize].init_deal(state, p, true);
    }

    while state.phase == Phase::Playing && !state.is_terminal() {
        let player = state.current_player();
        let state_before = *state;
        let action = searches[player as usize].search(state, &isdd_config, rng);
        for s in &mut searches {
            s.record_action(&state_before, player, action);
        }
        state.step(action);
    }

    let score = compute_deal_score(state);
    DealResult {
        void: false,
        ns_score: score.scores[0],
        ew_score: score.scores[1],
    }
}

/// Compute score from DD points (for DD oracle mode).
fn dd_score(state: &GameState, ns_pts: u8) -> DealResult {
    let contract_team = state.contract.team;
    let contract_value = state.contract.point_value();
    let contract_trump = state.contract.trump;
    let contract_coinche = state.contract.coinche;

    let taker = contract_team as usize;
    let taker_pts = if taker == 0 { ns_pts } else { 162 - ns_pts };

    let mut belote = [0u8; 2];
    for p in 0..4u8 {
        let team = GameState::player_team(p) as usize;
        let tbits = suit_bits(state.hands[p as usize], Suit::from_u8(contract_trump));
        if tbits & (1 << 4) != 0 && tbits & (1 << 5) != 0 {
            belote[team] = 2;
        }
    }

    let belote_bonus = if belote[taker] == 2 { 20 } else { 0 };
    let reussi = if state.contract.is_capot() {
        taker_pts >= 162
    } else {
        (taker_pts as u16 + belote_bonus) >= contract_value
    };

    let coinche_mult = match contract_coinche { 1 => 2u16, 2 => 4, _ => 1 };

    let (ns_score, ew_score) = if reussi {
        let scored = (contract_value + taker_pts as u16) * coinche_mult;
        if contract_team == 0 {
            (scored as i16 + if belote[0] == 2 { 20 } else { 0 }, if belote[1] == 2 { 20 } else { 0 })
        } else {
            (if belote[0] == 2 { 20 } else { 0 }, scored as i16 + if belote[1] == 2 { 20 } else { 0 })
        }
    } else {
        let penalty = (160 + contract_value) * coinche_mult;
        if contract_team == 0 {
            (if belote[0] == 2 { 20 } else { 0 }, penalty as i16 + if belote[1] == 2 { 20 } else { 0 })
        } else {
            (penalty as i16 + if belote[0] == 2 { 20 } else { 0 }, if belote[1] == 2 { 20 } else { 0 })
        }
    };

    DealResult { void: false, ns_score, ew_score }
}

fn run_matchup(
    label: &str,
    ns: Bidder,
    ew: Bidder,
    n_deals: usize,
    nn: &mut Option<BidNet>,
    rng: &mut StdRng,
    use_isdd: bool,
    isdd_time_ms: u32,
) {
    let start = Instant::now();
    let mut tt_buf = solver::new_tt_buffer();

    let mut ns_total = 0i64;
    let mut ew_total = 0i64;
    let mut voids = 0u32;
    let mut ns_contracts = 0u32;
    let mut ns_made = 0u32;
    let mut ew_contracts = 0u32;
    let mut ew_made = 0u32;
    let mut coinches = 0u32;
    let mut ns_wins = 0u32;

    for i in 0..n_deals {
        let dealer = (i % 4) as u8;
        let mut state = GameState::deal_random(dealer, rng);

        let result = if use_isdd {
            play_deal_isdd(&mut state, ns, ew, nn, isdd_time_ms, rng)
        } else {
            let mut dd_pts = [0u8; 4];
            for s in 0..4u8 {
                let [ns_p, _] = solver::solve_for_trump_reuse_tt(state.hands, dealer, s, &mut tt_buf);
                dd_pts[s as usize] = ns_p;
            }
            play_deal_dd(&mut state, ns, ew, nn, &dd_pts)
        };

        if result.void {
            voids += 1;
            continue;
        }

        ns_total += result.ns_score as i64;
        ew_total += result.ew_score as i64;

        if state.contract.coinche > 0 {
            coinches += 1;
        }
        if state.contract.team == 0 {
            ns_contracts += 1;
            if result.ns_score > result.ew_score {
                ns_made += 1;
            }
        } else {
            ew_contracts += 1;
            if result.ew_score > result.ns_score {
                ew_made += 1;
            }
        }
        if result.ns_score > result.ew_score {
            ns_wins += 1;
        }

        // Progress for slow IS-DD mode
        if use_isdd && (i + 1) % 10 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            eprint!("\r  [{}/{}] {:.1} deals/sec", i + 1, n_deals, rate);
        }
    }
    if use_isdd { eprintln!(); }

    let played = n_deals as u32 - voids;
    let elapsed = start.elapsed().as_secs_f64();

    let mode = if use_isdd { format!("IS-DD {}ms", isdd_time_ms) } else { "DD oracle".into() };
    println!("  ┌────────────────────────────────────────────────────┐");
    println!("  │  {:<50} │", label);
    println!("  └────────────────────────────────────────────────────┘");
    println!("  {} deals ({} void), {:.1}s [{}]\n", n_deals, voids, elapsed, mode);
    println!(
        "  NS win rate: {:.1}% ({}/{})",
        ns_wins as f64 / played as f64 * 100.0, ns_wins, played
    );
    println!(
        "  Avg score:   NS {:+.0}  EW {:+.0}  (margin {:+.1})",
        ns_total as f64 / played as f64,
        ew_total as f64 / played as f64,
        (ns_total - ew_total) as f64 / played as f64,
    );
    println!(
        "  Contracts:   NS took {} ({:.0}%), made {} ({:.0}%)",
        ns_contracts, ns_contracts as f64 / played as f64 * 100.0,
        ns_made, if ns_contracts > 0 { ns_made as f64 / ns_contracts as f64 * 100.0 } else { 0.0 }
    );
    println!(
        "               EW took {} ({:.0}%), made {} ({:.0}%)",
        ew_contracts, ew_contracts as f64 / played as f64 * 100.0,
        ew_made, if ew_contracts > 0 { ew_made as f64 / ew_contracts as f64 * 100.0 } else { 0.0 }
    );
    println!("  Coinches:    {}\n", coinches);
}

// =====================================================================
//  Diagnostic: show examples where V3 and NN disagree
// =====================================================================

const SUIT_SYM: [&str; 4] = ["♠", "♥", "♦", "♣"];
const RANK_STR: [&str; 8] = ["7", "8", "9", "J", "Q", "K", "10", "A"];
const PLAYER_NAME: [&str; 4] = ["N", "E", "S", "W"];

fn pretty_hand(hand: CardSet) -> String {
    let mut parts = Vec::new();
    for s in 0..4u8 {
        let bits = suit_bits(hand, Suit::from_u8(s));
        if bits == 0 { continue; }
        let mut ranks = Vec::new();
        for r in (0..8).rev() {
            if bits & (1 << r) != 0 { ranks.push(RANK_STR[r]); }
        }
        parts.push(format!("{}{}", SUIT_SYM[s as usize], ranks.join("")));
    }
    parts.join(" ")
}

fn act_str(action: u8) -> String {
    match action {
        0 => "Pass".into(),
        41 => "Coinche".into(),
        42 => "Surcoinche".into(),
        1..=40 => {
            let (val, suit) = bidding::decode_bid(action);
            if val == 25 { format!("Capot{}", SUIT_SYM[suit as usize]) }
            else { format!("{}{}", val as u16 * 10, SUIT_SYM[suit as usize]) }
        }
        _ => "?".into(),
    }
}

/// Run both V3 and NN on the same deal, compare the full auction.
/// Returns true if they produced different contracts.
fn diagnose_deal(
    hands: [u32; 4],
    dealer: u8,
    nn: &mut BidNet,
    dd_pts: &[u8; 4],
    deal_idx: usize,
) -> bool {
    // --- Run with V3 for all 4 players ---
    let mut st_v3 = GameState::new(dealer, hands);
    let mut hist_v3: Vec<(u8, u8)> = Vec::new();
    while st_v3.phase == Phase::Bidding {
        let action = v3_bid(&st_v3, &hist_v3);
        let p = st_v3.current_player();
        hist_v3.push((p, action));
        st_v3.step(action);
    }

    // --- Run with NN for all 4 players ---
    let mut st_nn = GameState::new(dealer, hands);
    let mut hist_nn: Vec<(u8, u8)> = Vec::new();
    while st_nn.phase == Phase::Bidding {
        let obs = bid_obs::make_bid_observation(&st_nn, &hist_nn);
        let legal = st_nn.legal_actions();
        let action = nn.best_action_fast(&obs, legal);
        let p = st_nn.current_player();
        hist_nn.push((p, action));
        st_nn.step(action);
    }

    // Compare contracts
    let v3_void = st_v3.contract.value == 0;
    let nn_void = st_nn.contract.value == 0;
    let v3_suit = st_v3.contract.trump;
    let v3_val = st_v3.contract.point_value();
    let v3_team = st_v3.contract.team;
    let v3_co = st_v3.contract.coinche;
    let nn_suit = st_nn.contract.trump;
    let nn_val = st_nn.contract.point_value();
    let nn_team = st_nn.contract.team;
    let nn_co = st_nn.contract.coinche;

    let same = v3_void == nn_void
        && v3_suit == nn_suit
        && v3_val == nn_val
        && v3_team == nn_team
        && v3_co == nn_co;

    if same { return false; }

    // Print this deal
    println!("  ─── Deal {} (dealer={}) ───", deal_idx, PLAYER_NAME[dealer as usize]);
    for p in 0..4u8 {
        let team = if GameState::player_team(p) == 0 { "NS" } else { "EW" };
        println!("    {} ({}): {}", PLAYER_NAME[p as usize], team, pretty_hand(hands[p as usize]));
    }
    println!("    DD: ♠={} ♥={} ♦={} ♣={}", dd_pts[0], dd_pts[1], dd_pts[2], dd_pts[3]);

    // V3 auction
    print!("    V3 auction: ");
    for (p, a) in &hist_v3 {
        print!("{}={} ", PLAYER_NAME[*p as usize], act_str(*a));
    }
    if v3_void {
        println!("→ VOID");
    } else {
        let co = match v3_co { 1 => "×", 2 => "××", _ => "" };
        let team = if v3_team == 0 { "NS" } else { "EW" };
        println!("→ {}{}{} by {}", v3_val, SUIT_SYM[v3_suit as usize], co, team);
    }

    // NN auction
    print!("    NN auction: ");
    for (p, a) in &hist_nn {
        print!("{}={} ", PLAYER_NAME[*p as usize], act_str(*a));
    }
    if nn_void {
        println!("→ VOID");
    } else {
        let co = match nn_co { 1 => "×", 2 => "××", _ => "" };
        let team = if nn_team == 0 { "NS" } else { "EW" };
        println!("→ {}{}{} by {}", nn_val, SUIT_SYM[nn_suit as usize], co, team);
    }

    // Who was right? (using DD)
    if !v3_void && !nn_void {
        let v3_taker = v3_team as usize;
        let v3_pts = if v3_taker == 0 { dd_pts[v3_suit as usize] } else { 162 - dd_pts[v3_suit as usize] };
        let v3_ok = v3_pts as u16 >= v3_val;
        let nn_taker = nn_team as usize;
        let nn_pts = if nn_taker == 0 { dd_pts[nn_suit as usize] } else { 162 - dd_pts[nn_suit as usize] };
        let nn_ok = nn_pts as u16 >= nn_val;
        println!("    DD verdict: V3={} ({}pts vs {}), NN={} ({}pts vs {})",
            if v3_ok { "✓" } else { "✗" }, v3_pts, v3_val,
            if nn_ok { "✓" } else { "✗" }, nn_pts, nn_val);
    }
    println!();
    true
}

fn run_diagnostics(nn: &mut BidNet, n_deals: usize) {
    println!("\n{}", "=".repeat(80));
    println!("  DIAGNOSTICS: V3 vs NN side-by-side ({} deals)", n_deals);
    println!("{}\n", "=".repeat(80));
    println!("  Showing deals where V3 and NN reach DIFFERENT contracts.\n");

    let mut rng = StdRng::seed_from_u64(99);
    let mut tt_buf = solver::new_tt_buffer();
    let mut diff_count = 0u32;
    let mut shown = 0u32;
    let max_show = 30;

    for i in 0..n_deals {
        let dealer = (i % 4) as u8;
        let state = GameState::deal_random(dealer, &mut rng);

        let mut dd_pts = [0u8; 4];
        for s in 0..4u8 {
            let [ns, _] = solver::solve_for_trump_reuse_tt(state.hands, dealer, s, &mut tt_buf);
            dd_pts[s as usize] = ns;
        }

        let differs = diagnose_deal(state.hands, dealer, nn, &dd_pts, i);
        if differs {
            diff_count += 1;
            shown += 1;
            if shown >= max_show {
                println!("  ... (showing first {} differences)\n", max_show);
                // Keep counting but stop printing
            }
        }
    }

    println!("  Total: {} / {} deals differ ({:.0}%)\n",
        diff_count, n_deals, diff_count as f64 / n_deals as f64 * 100.0);
}

// =====================================================================
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut n_matches = 500usize;
    let mut bid_model: Option<String> = None;
    let mut diag_only = false;
    let mut use_isdd = false;
    let mut isdd_time_ms = 20u32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--matches" => { n_matches = args[i + 1].parse().unwrap(); i += 2; }
            "--bid-model" => { bid_model = Some(args[i + 1].clone()); i += 2; }
            "--diag" => { diag_only = true; i += 1; }
            "--isdd" => { use_isdd = true; i += 1; }
            "--isdd-ms" => { isdd_time_ms = args[i + 1].parse().unwrap(); use_isdd = true; i += 2; }
            _ => i += 1,
        }
    }

    let model_path = bid_model.as_deref().unwrap_or("models/bid_nn_final.bin");
    let mut nn: Option<BidNet> = BidNet::load_with_hidden(model_path, 256).ok().map(|net| {
        println!("NN: {} (obs={}, dueling={})", model_path, net.obs_dim(), net.is_dueling());
        net
    });

    if nn.is_none() {
        println!("NN not loaded — need model for diagnostics/comparison");
        return;
    }

    let mode_str = if use_isdd { format!("IS-DD {}ms/move", isdd_time_ms) } else { "DD oracle".into() };
    println!("Play mode: {}\n", mode_str);

    if !use_isdd && !diag_only {
        // Quick diagnostics in DD mode
        run_diagnostics(nn.as_mut().unwrap(), 200);
    }

    if diag_only { return; }

    // Calibration: run 5 deals first to estimate time
    if use_isdd {
        println!("  Calibrating IS-DD speed (5 deals)...");
        let cal_start = Instant::now();
        let mut cal_rng = StdRng::seed_from_u64(999);
        for ci in 0..5 {
            let mut st = GameState::deal_random((ci % 4) as u8, &mut cal_rng);
            let _ = play_deal_isdd(&mut st, Bidder::V3Rules, Bidder::ImprovedV2, &mut nn, isdd_time_ms, &mut cal_rng);
        }
        let cal_elapsed = cal_start.elapsed().as_secs_f64();
        let per_deal = cal_elapsed / 5.0;
        println!("  → {:.2}s/deal, {} deals ≈ {:.0}s per matchup\n",
            per_deal, n_matches, per_deal * n_matches as f64);
    }

    println!("=== Tournament (V3) — {} deals [{}] ===\n", n_matches, mode_str);
    let mut rng = StdRng::seed_from_u64(42);

    run_matchup("V3 (NS) vs ImprovedV2 (EW)", Bidder::V3Rules, Bidder::ImprovedV2,
        n_matches, &mut nn, &mut rng, use_isdd, isdd_time_ms);
    run_matchup("ImprovedV2 (NS) vs V3 (EW)", Bidder::ImprovedV2, Bidder::V3Rules,
        n_matches, &mut nn, &mut rng, use_isdd, isdd_time_ms);
    run_matchup("V3 (NS) vs NN (EW)", Bidder::V3Rules, Bidder::Nn,
        n_matches, &mut nn, &mut rng, use_isdd, isdd_time_ms);
    run_matchup("NN (NS) vs V3 (EW)", Bidder::Nn, Bidder::V3Rules,
        n_matches, &mut nn, &mut rng, use_isdd, isdd_time_ms);
    run_matchup("NN (NS) vs ImprovedV2 (EW) [ref]", Bidder::Nn, Bidder::ImprovedV2,
        n_matches, &mut nn, &mut rng, use_isdd, isdd_time_ms);

    println!("Done.");
}
