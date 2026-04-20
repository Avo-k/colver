use wasm_bindgen::prelude::*;

use colver_core::bid_net::BidNet;
use colver_core::bid_obs::{
    self, write_bid_observation, write_bid_observation_score_aware,
    write_bid_observation_score_aware_v2, BID_OBS_DIM, BID_OBS_DIM_SCORE_AWARE,
    BID_OBS_DIM_SCORE_AWARE_V2,
};
use colver_core::card::*;
use colver_core::solver;
use colver_core::state::GameState;

/// BidNet wrapper for WASM.  Constructed once from weight bytes, reused across calls.
///
/// Auto-detects architecture: tries hidden sizes 256, 512, 1024 and matches the
/// weight-file layout. Supports obs_dim 108 / 110 / 113 via build_bid_obs().
#[wasm_bindgen]
pub struct WasmBidNet {
    net: BidNet,
}

#[wasm_bindgen]
impl WasmBidNet {
    /// Construct from raw weight bytes (little-endian f32).
    #[wasm_bindgen(constructor)]
    pub fn new(weight_bytes: &[u8]) -> Result<WasmBidNet, JsValue> {
        if weight_bytes.len() % 4 != 0 {
            return Err(JsValue::from_str("weight bytes not a multiple of 4"));
        }

        // Write bytes to a temp in-memory "file" conceptually: BidNet::load_with_hidden
        // reads from a path, so reimplement the hidden-size sweep by parsing floats once.
        let floats: Vec<f32> = weight_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Sweep hidden sizes and try to match architecture (standard + dueling × 2-4 layers).
        for &hidden in &[256usize, 512, 1024] {
            let total = floats.len();
            let dueling_tail = hidden + 1 + hidden * 43 + 43;
            let standard_tail = hidden * 43 + 43;
            let known_obs_dims = [108usize, 110, 113, 114];

            for layers in 2..=4 {
                let trunk_fixed = (layers - 1) * (hidden * hidden + 3 * hidden) + 3 * hidden;
                for &(tail, dueling) in &[(dueling_tail, true), (standard_tail, false)] {
                    let fixed = trunk_fixed + tail;
                    if total > fixed && (total - fixed) % hidden == 0 {
                        let obs_dim = (total - fixed) / hidden;
                        if !known_obs_dims.contains(&obs_dim) {
                            continue;
                        }
                        if let Ok(net) = BidNet::from_floats_with_layers(
                            &floats, hidden, obs_dim, dueling, layers,
                        ) {
                            return Ok(WasmBidNet { net });
                        }
                    }
                }
            }
        }

        Err(JsValue::from_str(
            "cannot infer BidNet architecture — weight file does not match any known shape (hidden 256/512/1024, layers 2-4, obs_dim 108/110/113/114)",
        ))
    }

    /// Evaluate a hand with prior bid actions.
    /// `hand`: Uint8Array of 8 card indices (0-31)
    /// `prior_actions`: Uint8Array of prior bid action indices
    /// Returns JSON string: {"q_values":[[action,q],...], "best_action":N}
    pub fn evaluate(&mut self, hand: &[u8], prior_actions: &[u8]) -> Result<String, JsValue> {
        if hand.len() != 8 {
            return Err(JsValue::from_str("hand must have exactly 8 cards"));
        }

        // Build hands: seat 2 (Sud) gets user's hand, others get random remaining
        let mut my_cards: CardSet = 0;
        for &c in hand {
            if c >= 32 {
                return Err(JsValue::from_str("card index out of range"));
            }
            my_cards |= 1u32 << c;
        }

        let remaining_set = ALL_CARDS ^ my_cards;
        let mut remaining: Vec<u8> = Vec::with_capacity(24);
        for i in 0..32u8 {
            if remaining_set & (1u32 << i) != 0 {
                remaining.push(i);
            }
        }

        // Shuffle remaining cards
        shuffle_u8(&mut remaining);

        let seat = 2u8;
        let mut hands = [0u32; 4];
        hands[seat as usize] = my_cards;
        let others = [0u8, 1, 3];
        for (i, &p) in others.iter().enumerate() {
            for j in 0..8 {
                hands[p as usize] |= 1u32 << remaining[i * 8 + j];
            }
        }

        let n_prior = prior_actions.len();
        let dealer = ((seat as i32 - 1 - n_prior as i32).rem_euclid(4)) as u8;
        let mut state = GameState::new(dealer, hands);

        // Build bid history and step through prior actions
        let mut bid_history: Vec<(u8, u8)> = Vec::with_capacity(n_prior);
        for &action in prior_actions {
            let player = state.current_player();
            bid_history.push((player, action));
            state.step(action);
        }

        // Build observation matching the loaded NN's obs_dim (108 / 110 / 113).
        // Score-aware models receive a match-neutral 0/0 score (the context the
        // web page runs in — single-deal analysis with no match history).
        let obs_dim = self.net.obs_dim();
        let mut obs = match obs_dim {
            BID_OBS_DIM => {
                let mut buf = vec![0.0f32; BID_OBS_DIM];
                write_bid_observation(&mut buf, 0, &state, &bid_history);
                buf
            }
            BID_OBS_DIM_SCORE_AWARE => {
                let mut buf = vec![0.0f32; BID_OBS_DIM_SCORE_AWARE];
                write_bid_observation_score_aware(&mut buf, 0, &state, &bid_history, 0, 0);
                buf
            }
            BID_OBS_DIM_SCORE_AWARE_V2 => {
                let mut buf = vec![0.0f32; BID_OBS_DIM_SCORE_AWARE_V2];
                write_bid_observation_score_aware_v2(&mut buf, 0, &state, &bid_history, 0, 0);
                buf
            }
            other => {
                return Err(JsValue::from_str(&format!(
                    "unsupported BidNet obs_dim={other} (expected 108/110/113)"
                )));
            }
        };
        let _ = &mut obs; // silence if unused

        let legal_mask = state.legal_actions();
        let (best_action, legal_q) = self.net.best_action(&obs, legal_mask);

        // Sort by q-value descending
        let mut sorted_q = legal_q;
        sorted_q.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Build JSON manually (no serde needed)
        let mut json = String::with_capacity(512);
        json.push_str("{\"q_values\":[");
        for (i, (a, q)) in sorted_q.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push('[');
            json.push_str(&a.to_string());
            json.push(',');
            // Round to 3 decimal places
            json.push_str(&format!("{:.3}", q));
            json.push(']');
        }
        json.push_str("],\"best_action\":");
        json.push_str(&best_action.to_string());
        json.push('}');

        Ok(json)
    }
}

/// Oracle DD solver wrapper for WASM.  Owns a reusable TT buffer (2MB).
#[wasm_bindgen]
pub struct WasmOracle {
    tt_buf: Vec<u64>,
}

#[wasm_bindgen]
impl WasmOracle {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmOracle {
        WasmOracle {
            tt_buf: solver::new_tt_buffer(),
        }
    }

    /// Run one oracle simulation.
    /// `hand`: Uint8Array of 8 card indices
    /// Returns JSON string: {"suits":[[ns,ew],...], "hands":{"0":[...],"1":[...],"3":[...]}}
    pub fn single_sim(&mut self, hand: &[u8]) -> Result<String, JsValue> {
        if hand.len() != 8 {
            return Err(JsValue::from_str("hand must have exactly 8 cards"));
        }

        let mut my_cards: CardSet = 0;
        for &c in hand {
            if c >= 32 {
                return Err(JsValue::from_str("card index out of range"));
            }
            my_cards |= 1u32 << c;
        }

        let remaining_set = ALL_CARDS ^ my_cards;
        let mut remaining: Vec<u8> = Vec::with_capacity(24);
        for i in 0..32u8 {
            if remaining_set & (1u32 << i) != 0 {
                remaining.push(i);
            }
        }

        shuffle_u8(&mut remaining);

        let seat = 2usize;
        let mut hands = [0u32; 4];
        hands[seat] = my_cards;
        let others = [0usize, 1, 3];
        for (i, &p) in others.iter().enumerate() {
            for j in 0..8 {
                hands[p] |= 1u32 << remaining[i * 8 + j];
            }
        }

        // Solve all 4 suits using the reusable TT buffer
        let mut suits = [[0u8; 2]; 4];
        for trump in 0..4u8 {
            suits[trump as usize] =
                solver::solve_for_trump_reuse_tt(hands, 0, trump, &mut self.tt_buf);
        }

        // Build JSON manually
        let mut json = String::with_capacity(256);
        json.push_str("{\"suits\":[");
        for (i, [ns, ew]) in suits.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push('[');
            json.push_str(&ns.to_string());
            json.push(',');
            json.push_str(&ew.to_string());
            json.push(']');
        }
        json.push_str("],\"hands\":{");
        for (idx, &p) in others.iter().enumerate() {
            if idx > 0 {
                json.push(',');
            }
            json.push('"');
            json.push_str(&p.to_string());
            json.push_str("\":[");
            let mut first = true;
            for c in 0..32u8 {
                if hands[p] & (1u32 << c) != 0 {
                    if !first {
                        json.push(',');
                    }
                    json.push_str(&c.to_string());
                    first = false;
                }
            }
            json.push(']');
        }
        json.push_str("}}");

        Ok(json)
    }
}

/// Simple Fisher-Yates shuffle using getrandom (crypto.getRandomValues in browser).
fn shuffle_u8(arr: &mut [u8]) {
    let len = arr.len();
    if len <= 1 {
        return;
    }
    // Get random bytes — we need one u32 per swap
    let mut rand_buf = vec![0u8; len * 4];
    getrandom::getrandom(&mut rand_buf).unwrap();

    for i in (1..len).rev() {
        let rand_val =
            u32::from_le_bytes([rand_buf[i * 4], rand_buf[i * 4 + 1], rand_buf[i * 4 + 2], rand_buf[i * 4 + 3]]);
        let j = (rand_val as usize) % (i + 1);
        arr.swap(i, j);
    }
}
