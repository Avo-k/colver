/// Debug binary: find all false hard constraint exclusions.
///
/// Plays deals with NN bots, tracks CardBeliefs, and reports every case where
/// a hard constraint (weight == 0.0 for true holder) is wrong.
///
/// Usage:
///   cargo run --bin debug_hard_constraints --features "parallel,nn" --release -- [--deals 2000]

use rand::rngs::StdRng;
use rand::SeedableRng;

use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::card::*;
use colver_core::card_beliefs::CardBeliefs;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking};
use colver_core::state::{GameState, Phase};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_deals = 2000u32;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--deals" | "-n" => { i += 1; num_deals = args[i].parse().unwrap_or(2000); }
            _ => {}
        }
        i += 1;
    }

    let mut bid_net = BidNet::load_with_hidden("models/bid_v2/bid_nn_final.bin", 512).expect("bid model");
    let mut play_net = DmcNet::load("models/play_v2/play_final.bin").expect("play model");
    play_net.set_residual(true);
    let obs_dim = play_net.obs_dim();
    let canonical = obs_dim == dmc_obs::OBS_DIM_TR;

    let mut rng = StdRng::seed_from_u64(42);
    let mut bid_obs_buf = vec![0.0f32; bid_obs::BID_OBS_DIM];
    let mut obs_buf = vec![0.0f32; obs_dim];

    let mut total_hidden = 0u64;
    let mut total_excl = 0u64;
    let mut total_decisions = 0u64;

    println!("Scanning {} deals for false hard constraint exclusions...\n", num_deals);

    for deal_idx in 0..num_deals {
        let dealer = (deal_idx % 4) as u8;
        let mut state = GameState::deal_random(dealer, &mut rng);
        let mut tracking = EnvTracking::new();
        tracking.reset(dealer);

        let mut cb = [
            CardBeliefs::new(&state, 0), CardBeliefs::new(&state, 1),
            CardBeliefs::new(&state, 2), CardBeliefs::new(&state, 3),
        ];

        // Bidding
        while state.phase == Phase::Bidding && !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;
            bid_obs::write_bid_observation(&mut bid_obs_buf, 0, &state, &tracking.bid_history);
            let legal = state.legal_actions();
            let action = bid_net.best_action_fast(&bid_obs_buf, legal);
            for p in 0..4u8 { cb[p as usize].record_action(&state_before, player, action); }
            tracking.track_action(&state_before, action);
            state.step(action);
        }

        if state.is_terminal() { continue; }

        // Play
        let mut trick_actions: Vec<(u8, u8, u8)> = Vec::new(); // (player, action, trick_count_before)
        while !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;
            let observer = player;
            let observer_hand = state.hands[observer as usize];
            let trick_num = state.tricks_won[0] + state.tricks_won[1];
            let trick_pos = state.trick_count;

            total_decisions += 1;

            // Check raw weights (NOT normalized) for false exclusions
            let raw = cb[observer as usize].raw_weights();
            for card in 0..32u8 {
                let bit = card_to_bit(card);
                if observer_hand & bit != 0 || state.played_cards & bit != 0 { continue; }
                if (0..4).any(|i| state.current_trick[i] == card) { continue; }

                let mut true_p = 255u8;
                for p in 0..4u8 {
                    if state.hands[p as usize] & bit != 0 { true_p = p; break; }
                }
                if true_p == 255 { continue; }

                total_hidden += 1;

                if raw[true_p as usize][card as usize] == 0.0 {
                    total_excl += 1;
                    let card_name = card_name(card);
                    let suit = card_suit_u8(card);
                    let trump = state.contract.trump;

                    // Check WHY the weight is zero
                    let is_void = state.voids[true_p as usize] & (1 << suit) != 0;
                    let trump_ceiling = suit == trump && {
                        // Check if any higher trump was zeroed
                        let strength = TRUMP_STRENGTH[card_rank(card) as usize];
                        strength > 0 && raw[true_p as usize][card as usize] == 0.0
                    };

                    println!("  FALSE EXCL deal={} trick={} pos={} observer=P{} card={} true_holder=P{}",
                        deal_idx, trick_num, trick_pos, observer, card_name, true_p);
                    println!("    voids[P{}] = {:04b} (suit {}), is_void_match={}",
                        true_p, state.voids[true_p as usize], suit, is_void);
                    println!("    trump={}, card_suit={}, trump_ceiling_candidate={}",
                        trump, suit, trump_ceiling);

                    // Show recent trick history for context
                    let start = if trick_actions.len() > 8 { trick_actions.len() - 8 } else { 0 };
                    print!("    recent actions:");
                    for &(ap, aa, atc) in &trick_actions[start..] {
                        print!(" P{}:{}", ap, colver_core::card::card_name(aa));
                        if atc == 0 { print!("(lead)"); }
                    }
                    println!();

                    // Show all raw weights for this card
                    print!("    weights:");
                    for p in 0..4u8 {
                        print!(" P{}={:.2}", p, raw[p as usize][card as usize]);
                    }
                    println!();
                    println!();
                }
            }

            // Play action
            let action = if canonical {
                dmc_obs::write_observation_tr(&mut obs_buf, 0, &state, &tracking);
                let order = dmc_obs::current_player_order(&state, &tracking);
                let cm = dmc_obs::cardset_to_canonical(state.legal_actions() as u32, &order);
                let (best, _) = play_net.best_action(&obs_buf, cm as u32);
                dmc_obs::card_to_physical(best, &order)
            } else {
                dmc_obs::write_observation(&mut obs_buf, 0, &state, &tracking);
                let (a, _) = play_net.best_action(&obs_buf, state.legal_actions() as u32);
                a
            };

            trick_actions.push((player, action, state.trick_count));
            for p in 0..4u8 { cb[p as usize].record_action(&state_before, player, action); }
            tracking.track_action(&state_before, action);
            state.step(action);
        }
    }

    println!("\n══════════════════════════════════════════════════");
    println!("  Total decisions: {}", total_decisions);
    println!("  Total hidden cards checked: {}", total_hidden);
    println!("  False exclusions: {} ({:.4}%)", total_excl,
        if total_hidden > 0 { 100.0 * total_excl as f64 / total_hidden as f64 } else { 0.0 });
    println!("══════════════════════════════════════════════════");
}
