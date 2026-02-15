use crate::card::{Card, EMPTY};
use crate::state::GameState;

/// Feature vector dimension for the neural network value function.
///
/// Layout (278 floats total):
///   [0..128)    4 hands × 32 one-hot bitmask expansion
///   [128..256)  current trick: 4 seats × 32 one-hot (zeros if not played)
///   [256..260)  trump suit one-hot
///   [260]       bid value / 160.0 (capot = 1.0)
///   [261..264)  coinche state one-hot (none / coinche / surcoinche)
///   [264..266)  taker team one-hot (NS / EW)
///   [266..268)  points per team / 162.0
///   [268..270)  tricks won per team / 8.0
///   [270..274)  current player one-hot
///   [274..278)  trick lead one-hot
pub const FEATURE_DIM: usize = 278;

/// Extract features from a game state into a pre-allocated buffer.
///
/// The state should be in the Playing or Done phase (has a contract set).
/// All features are written relative to the actual teams (team 0 = NS, team 1 = EW).
/// No allocations.
#[inline]
pub fn extract_features(state: &GameState, buf: &mut [f32; FEATURE_DIM]) {
    // Zero the buffer
    *buf = [0.0; FEATURE_DIM];

    // [0..128) 4 hands × 32 one-hot bitmask expansion
    for p in 0..4u32 {
        let hand = state.hands[p as usize];
        let base = (p * 32) as usize;
        expand_cardset(hand, &mut buf[base..base + 32]);
    }

    // [128..256) current trick: 4 seats × 32 one-hot
    for seat in 0..4usize {
        let card: Card = state.current_trick[seat];
        if card != EMPTY {
            let idx = 128 + seat * 32 + card as usize;
            buf[idx] = 1.0;
        }
    }

    // [256..260) trump suit one-hot
    let trump = state.contract.trump as usize;
    if trump < 4 {
        buf[256 + trump] = 1.0;
    }

    // [260] bid value normalized
    let point_value = state.contract.point_value(); // 80-160 or 250
    buf[260] = if state.contract.is_capot() {
        1.0
    } else {
        point_value as f32 / 160.0
    };

    // [261..264) coinche state one-hot
    let coinche = state.contract.coinche as usize;
    if coinche < 3 {
        buf[261 + coinche] = 1.0;
    }

    // [264..266) taker team one-hot
    let taker = state.contract.team as usize;
    if taker < 2 {
        buf[264 + taker] = 1.0;
    }

    // [266..268) points per team / 162.0
    buf[266] = state.points[0] as f32 / 162.0;
    buf[267] = state.points[1] as f32 / 162.0;

    // [268..270) tricks won per team / 8.0
    buf[268] = state.tricks_won[0] as f32 / 8.0;
    buf[269] = state.tricks_won[1] as f32 / 8.0;

    // [270..274) current player one-hot
    let cp = state.current_player as usize;
    if cp < 4 {
        buf[270 + cp] = 1.0;
    }

    // [274..278) trick lead one-hot
    let lead = state.trick_lead as usize;
    if lead < 4 {
        buf[274 + lead] = 1.0;
    }
}

/// Expand a CardSet bitmask into 32 floats (0.0 or 1.0).
#[inline]
fn expand_cardset(cs: u32, out: &mut [f32]) {
    debug_assert!(out.len() >= 32);
    let mut mask = cs;
    while mask != 0 {
        let bit = mask.trailing_zeros() as usize;
        out[bit] = 1.0;
        mask &= mask - 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Contract, Phase};

    #[test]
    fn test_feature_dim() {
        assert_eq!(FEATURE_DIM, 278);
    }

    #[test]
    fn test_expand_cardset() {
        let mut out = [0.0f32; 32];
        expand_cardset(0b1010_0101, &mut out);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 1.0);
        assert_eq!(out[3], 0.0);
        assert_eq!(out[4], 0.0);
        assert_eq!(out[5], 1.0);
        assert_eq!(out[6], 0.0);
        assert_eq!(out[7], 1.0);
        for i in 8..32 {
            assert_eq!(out[i], 0.0);
        }
    }

    #[test]
    fn test_extract_features_basic() {
        // Create a playing-phase state with known values
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let mut state = GameState::new(0, hands);
        state.phase = Phase::Playing;
        state.contract = Contract {
            trump: 1,    // Hearts
            value: 8,    // 80
            team: 0,     // NS
            coinche: 0,  // normal
        };
        state.trick_lead = 1;
        state.current_player = 1;

        let mut buf = [0.0f32; FEATURE_DIM];
        extract_features(&state, &mut buf);

        // Check hand 0 (Spades: bits 0-7 set)
        for i in 0..8 {
            assert_eq!(buf[i], 1.0, "hand 0, bit {}", i);
        }
        for i in 8..32 {
            assert_eq!(buf[i], 0.0, "hand 0, bit {}", i);
        }

        // Check hand 1 (Hearts: bits 8-15 set)
        for i in 32..40 {
            assert_eq!(buf[i], 0.0, "hand 1, bit {}", i - 32);
        }
        for i in 40..48 {
            assert_eq!(buf[i], 1.0, "hand 1, bit {}", i - 32);
        }

        // Trump suit: Hearts = index 1
        assert_eq!(buf[256], 0.0); // Spades
        assert_eq!(buf[257], 1.0); // Hearts
        assert_eq!(buf[258], 0.0); // Diamonds
        assert_eq!(buf[259], 0.0); // Clubs

        // Bid value: 80/160 = 0.5
        assert!((buf[260] - 0.5).abs() < 1e-6);

        // Coinche: none = index 0
        assert_eq!(buf[261], 1.0); // none
        assert_eq!(buf[262], 0.0); // coinche
        assert_eq!(buf[263], 0.0); // surcoinche

        // Taker team: NS = index 0
        assert_eq!(buf[264], 1.0); // NS
        assert_eq!(buf[265], 0.0); // EW

        // Points: both 0
        assert_eq!(buf[266], 0.0);
        assert_eq!(buf[267], 0.0);

        // Tricks: both 0
        assert_eq!(buf[268], 0.0);
        assert_eq!(buf[269], 0.0);

        // Current player: 1 (East)
        assert_eq!(buf[270], 0.0);
        assert_eq!(buf[271], 1.0);
        assert_eq!(buf[272], 0.0);
        assert_eq!(buf[273], 0.0);

        // Trick lead: 1 (East)
        assert_eq!(buf[274], 0.0);
        assert_eq!(buf[275], 1.0);
        assert_eq!(buf[276], 0.0);
        assert_eq!(buf[277], 0.0);
    }

    #[test]
    fn test_extract_features_capot() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let mut state = GameState::new(0, hands);
        state.phase = Phase::Playing;
        state.contract = Contract {
            trump: 3,    // Clubs
            value: 25,   // capot
            team: 1,     // EW
            coinche: 2,  // surcontré
        };
        state.current_player = 3;
        state.trick_lead = 0;

        let mut buf = [0.0f32; FEATURE_DIM];
        extract_features(&state, &mut buf);

        // Trump: Clubs = index 3
        assert_eq!(buf[259], 1.0);

        // Bid value: capot → 1.0
        assert!((buf[260] - 1.0).abs() < 1e-6);

        // Coinche: surcontré = index 2
        assert_eq!(buf[261], 0.0);
        assert_eq!(buf[262], 0.0);
        assert_eq!(buf[263], 1.0);

        // Taker: EW = index 1
        assert_eq!(buf[264], 0.0);
        assert_eq!(buf[265], 1.0);

        // Current player: 3 (West)
        assert_eq!(buf[273], 1.0);

        // Trick lead: 0 (North)
        assert_eq!(buf[274], 1.0);
    }

    #[test]
    fn test_extract_features_with_trick_cards() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let mut state = GameState::new(0, hands);
        state.phase = Phase::Playing;
        state.contract = Contract {
            trump: 0,
            value: 8,
            team: 0,
            coinche: 0,
        };
        // Set some trick cards
        state.current_trick[0] = 5;    // Card 5 played by seat 0
        state.current_trick[1] = 10;   // Card 10 played by seat 1
        state.current_trick[2] = EMPTY; // Seat 2 hasn't played
        state.current_trick[3] = EMPTY; // Seat 3 hasn't played

        let mut buf = [0.0f32; FEATURE_DIM];
        extract_features(&state, &mut buf);

        // Seat 0 trick card: card 5 → buf[128 + 0*32 + 5] = buf[133]
        assert_eq!(buf[133], 1.0);

        // Seat 1 trick card: card 10 → buf[128 + 1*32 + 10] = buf[170]
        assert_eq!(buf[170], 1.0);

        // Seat 2: no card → all zeros in [128+64..128+96)
        for i in 192..224 {
            assert_eq!(buf[i], 0.0);
        }
    }
}
