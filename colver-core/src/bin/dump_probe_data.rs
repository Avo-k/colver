/// Dump obs + NN action + minimal hand features for hidden-layer probing.
///
/// Mirrors distill_bid.rs scenarios but writes:
///   header: u32 obs_dim, u32 n_samples, u32 n_features
///   per sample: scenario_id(u8), position(u8), nn_action(u8), padding(u8),
///               obs (obs_dim f32), hand_features (n_features f32)
///
/// hand_features: 17 base features from distill_bid for the NN's chosen suit:
///   trump_count, has_jack, has_nine, has_ace, has_ten, has_king, has_queen,
///   trump_points, trump_score, has_belote, side_aces, side_tens, side_voids,
///   side_singletons, side_doubletons, total_aces, best_side_length

use colver_core::bid_eval::evaluate_for_trump;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::bidding;
use colver_core::card::*;
use colver_core::state::GameState;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::Write;

struct Scenario {
    name: &'static str,
    id: u8,
    dealer: u8,
    seat: u8,
    position: u8,
    prior: Vec<(u8, u8)>,
    varies_by_suit: bool,
}

fn scenarios() -> Vec<Scenario> {
    let pass = 0u8;
    let bid80s = bidding::encode_bid(8, 0);
    vec![
        Scenario { name: "pos1_open", id: 0, dealer: 3, seat: 0, position: 1, prior: vec![], varies_by_suit: false },
        Scenario { name: "pos2_after_pass", id: 1, dealer: 2, seat: 0, position: 2, prior: vec![(3, pass)], varies_by_suit: false },
        Scenario { name: "pos3_after_2p", id: 2, dealer: 1, seat: 0, position: 3, prior: vec![(2, pass), (3, pass)], varies_by_suit: false },
        Scenario { name: "pos4_after_3p", id: 3, dealer: 0, seat: 0, position: 4, prior: vec![(1, pass), (2, pass), (3, pass)], varies_by_suit: false },
        Scenario { name: "pos3_partner80", id: 4, dealer: 1, seat: 0, position: 3, prior: vec![(2, bid80s), (3, pass)], varies_by_suit: true },
        Scenario { name: "pos4_partner80", id: 5, dealer: 0, seat: 0, position: 4, prior: vec![(1, pass), (2, bid80s), (3, pass)], varies_by_suit: true },
        Scenario { name: "pos2_opp80", id: 6, dealer: 2, seat: 0, position: 2, prior: vec![(3, bid80s)], varies_by_suit: true },
        Scenario { name: "pos3_opp80", id: 7, dealer: 1, seat: 0, position: 3, prior: vec![(2, pass), (3, bid80s)], varies_by_suit: true },
        Scenario { name: "pos4_opp80", id: 8, dealer: 0, seat: 0, position: 4, prior: vec![(1, pass), (2, pass), (3, bid80s)], varies_by_suit: true },
    ]
}

fn compute_hand_features(hand: CardSet, suit_idx: u8) -> [f32; 17] {
    let suit = Suit::from_u8(suit_idx);
    let bits = suit_bits(hand, suit);
    let count = bits.count_ones();
    let has_jack = ((bits >> 3) & 1) as f32;
    let has_nine = ((bits >> 2) & 1) as f32;
    let has_ace = ((bits >> 7) & 1) as f32;
    let has_ten = ((bits >> 6) & 1) as f32;
    let has_king = ((bits >> 5) & 1) as f32;
    let has_queen = ((bits >> 4) & 1) as f32;
    let has_belote = (has_king > 0.5 && has_queen > 0.5) as u8 as f32;

    let mut trump_pts = 0u16;
    let mut b = bits;
    while b != 0 {
        let rank = b.trailing_zeros() as usize;
        trump_pts += TRUMP_POINTS[rank] as u16;
        b &= b - 1;
    }
    let trump_score = evaluate_for_trump(hand, suit);

    let mut side_aces = 0u32;
    let mut side_tens = 0u32;
    let mut side_voids = 0u32;
    let mut side_singletons = 0u32;
    let mut side_doubletons = 0u32;
    let mut best_side_length = 0u32;
    for s in 0..4u8 {
        if s == suit_idx { continue; }
        let sb = suit_bits(hand, Suit::from_u8(s));
        let sc = sb.count_ones();
        if sb & (1 << 7) != 0 { side_aces += 1; }
        if sb & (1 << 6) != 0 { side_tens += 1; }
        if sc == 0 { side_voids += 1; }
        else if sc == 1 { side_singletons += 1; }
        else if sc == 2 { side_doubletons += 1; }
        if sc > best_side_length { best_side_length = sc; }
    }
    let total_aces = (0..4u8)
        .filter(|&s| suit_bits(hand, Suit::from_u8(s)) & (1 << 7) != 0)
        .count() as f32;

    [
        count as f32, has_jack, has_nine, has_ace, has_ten, has_king, has_queen,
        trump_pts as f32, trump_score as f32, has_belote,
        side_aces as f32, side_tens as f32, side_voids as f32,
        side_singletons as f32, side_doubletons as f32,
        total_aces, best_side_length as f32,
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("models/bid_v5_isdd/bid_nn_final.bin");
    let n_deals: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let out_path = args.get(3).map(|s| s.as_str()).unwrap_or("/tmp/probe_data.bin");

    let mut net = BidNet::load_with_hidden(model_path, 512).unwrap();
    let obs_dim = net.obs_dim();
    let n_features = 17u32;

    // Estimate total samples
    let all = scenarios();
    let total_samples: usize = all.iter().map(|s| if s.varies_by_suit { n_deals } else { n_deals }).sum();
    eprintln!("Generating ~{} samples, obs_dim={}, n_features={}", total_samples, obs_dim, n_features);

    let mut f = std::io::BufWriter::new(std::fs::File::create(out_path).unwrap());
    f.write_all(&(obs_dim as u32).to_le_bytes()).unwrap();
    f.write_all(&(0u32).to_le_bytes()).unwrap(); // placeholder for n_samples (overwritten later)
    f.write_all(&n_features.to_le_bytes()).unwrap();

    let mut rng = StdRng::seed_from_u64(7);
    let mut count = 0u32;

    for scen in &all {
        eprintln!("  scenario {} (position {})", scen.name, scen.position);
        let suit_variants: Vec<u8> = if scen.varies_by_suit { vec![0, 1, 2, 3] } else { vec![255] };
        let deals_per_variant = if scen.varies_by_suit { n_deals / 4 } else { n_deals };

        for &prior_suit in &suit_variants {
            for _ in 0..deals_per_variant {
                let state_init = GameState::deal_random(scen.dealer, &mut rng);
                let hand = state_init.hands[scen.seat as usize];
                let mut history: Vec<(u8, u8)> = Vec::new();
                let mut state = state_init;
                for &(seat, action_template) in &scen.prior {
                    let action = if scen.varies_by_suit && action_template > 0 && action_template <= 40 {
                        let (val, _) = bidding::decode_bid(action_template);
                        bidding::encode_bid(val, prior_suit)
                    } else {
                        action_template
                    };
                    history.push((seat, action));
                    state.step(action);
                }
                assert_eq!(state.current_player(), scen.seat);

                let mut obs = vec![0.0f32; obs_dim];
                match obs_dim {
                    108 => bid_obs::write_bid_observation(&mut obs, 0, &state, &history),
                    110 => bid_obs::write_bid_observation_score_aware(&mut obs, 0, &state, &history, 0, 0),
                    113 => bid_obs::write_bid_observation_score_aware_v2(&mut obs, 0, &state, &history, 0, 0),
                    _ => panic!("unsupported obs_dim"),
                }
                let legal = state.legal_actions();
                let (best_action, _) = net.best_action(&obs, legal);

                // Pick the "relevant" suit for hand_features:
                //   If NN bid in a specific suit, use that suit.
                //   Else if scenario has a prior bid suit (partner/opp), use that.
                //   Else suit 0 (arbitrary).
                let ref_suit = if (1..=40).contains(&best_action) {
                    let (_, s) = bidding::decode_bid(best_action);
                    s
                } else if prior_suit < 4 {
                    prior_suit
                } else {
                    0
                };
                let hf = compute_hand_features(hand, ref_suit);

                // Write sample
                f.write_all(&[scen.id, scen.position, best_action, 0u8]).unwrap();
                for v in &obs {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
                for v in &hf {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
                count += 1;
            }
        }
    }

    // Rewrite n_samples in header
    f.flush().unwrap();
    drop(f);
    use std::io::{Seek, SeekFrom};
    let mut f = std::fs::OpenOptions::new().write(true).open(out_path).unwrap();
    f.seek(SeekFrom::Start(4)).unwrap();
    f.write_all(&count.to_le_bytes()).unwrap();

    eprintln!("Wrote {} samples to {}", count, out_path);
}
