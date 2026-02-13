use crate::card::*;
use crate::state::*;

/// Bidding action encoding:
///   Action 0:      PASS
///   Action 1-36:   BID = (value_idx * 4) + suit_idx + 1
///                  value_idx: 0..8 → 80,90,100,110,120,130,140,150,160
///                  suit_idx: 0..3  → Spades,Hearts,Diamonds,Clubs
///   Action 37-40:  CAPOT × 4 suits
///   Action 41:     COINCHE
///   Action 42:     SURCOINCHE
/// Total: 43 bidding actions (fits in u64 bitmask)

pub const BID_PASS: u8 = 0;
pub const BID_COINCHE: u8 = 41;
pub const BID_SURCOINCHE: u8 = 42;
pub const NUM_BID_ACTIONS: usize = 43;

/// Value indices map to actual bid values (×10).
pub const BID_VALUES: [u8; 10] = [8, 9, 10, 11, 12, 13, 14, 15, 16, 25];
// Index 0=80, 1=90, ..., 8=160, 9=capot(250)

/// Decode a bid action (1-40) into (value_encoded, suit_idx).
/// For actions 1-36: value_idx = (action-1)/4, suit_idx = (action-1)%4
/// For actions 37-40 (capot): value_idx = 9 (capot), suit_idx = action-37
#[inline]
pub fn decode_bid(action: u8) -> (u8, u8) {
    debug_assert!(action >= 1 && action <= 40);
    if action <= 36 {
        let idx = action - 1;
        let value_idx = idx / 4;
        let suit_idx = idx % 4;
        (BID_VALUES[value_idx as usize], suit_idx)
    } else {
        // 37-40: capot
        let suit_idx = action - 37;
        (25, suit_idx) // 25 = capot
    }
}

/// Encode a bid from (value_encoded, suit_idx) into action.
#[inline]
pub fn encode_bid(value: u8, suit_idx: u8) -> u8 {
    debug_assert!(suit_idx < 4);
    if value == 25 {
        37 + suit_idx
    } else {
        let value_idx = value - 8; // 8→0, 9→1, ..., 16→8
        value_idx * 4 + suit_idx + 1
    }
}

/// Compute legal bidding actions as a u64 bitmask.
pub fn legal_bids(state: &GameState) -> u64 {
    debug_assert_eq!(state.phase, Phase::Bidding);
    let mut mask: u64 = 0;

    // Pass is always legal
    mask |= 1 << BID_PASS;

    let player = state.current_player;
    let player_team = GameState::player_team(player);

    // If no bid has been made yet, or there is a bid:
    if state.last_bid_value > 0 {
        // There is an existing bid.
        let bidder_team = GameState::player_team(state.last_bidder);

        // Coinche: only if opponent made last bid and not already coinched
        if bidder_team != player_team && state.coinche_state == 0 {
            mask |= 1 << BID_COINCHE;
        }

        // Surcoinche: only if opponents coinched our team's bid
        if bidder_team == player_team && state.coinche_state == 1 {
            mask |= 1 << BID_SURCOINCHE;
        }

        // After a coinche, the contract is frozen: no more bids allowed.
        // Only surcoinche (by the coinched team) or pass.
        if state.coinche_state == 0 {
            // Can raise: must bid strictly higher than current bid.
            let current_val = state.last_bid_value;

            // Normal bids: all (value, suit) where value > current_val
            for vi in 0..9u8 {
                let val = BID_VALUES[vi as usize];
                if val > current_val {
                    for si in 0..4u8 {
                        let action = vi * 4 + si + 1;
                        mask |= 1 << action;
                    }
                }
            }
            // Capot: always available if not already bid capot
            if current_val < 25 {
                for si in 0..4u8 {
                    mask |= 1 << (37 + si);
                }
            }
        }
    } else {
        // No bid yet: any bid is legal
        for action in 1..=40u8 {
            mask |= 1 << action;
        }
    }

    mask
}

/// Process a bidding action and update the state.
/// Returns true if bidding is over.
pub fn apply_bid(state: &mut GameState, action: u8) {
    debug_assert_eq!(state.phase, Phase::Bidding);
    debug_assert!(action < NUM_BID_ACTIONS as u8);

    match action {
        BID_PASS => {
            state.consecutive_passes += 1;

            if state.last_bid_value == 0 {
                // No bid made yet
                if state.consecutive_passes >= 4 {
                    // 4 passes with no bid → deal is void
                    state.phase = Phase::Done;
                    return;
                }
            } else {
                // A bid exists
                if state.consecutive_passes >= 3 {
                    // 3 passes after a bid → bidding ends
                    finalize_contract(state);
                    return;
                }
            }
        }
        BID_COINCHE => {
            debug_assert_eq!(state.coinche_state, 0);
            state.coinche_state = 1;
            state.consecutive_passes = 0;
        }
        BID_SURCOINCHE => {
            debug_assert_eq!(state.coinche_state, 1);
            state.coinche_state = 2;
            // Surcoinche ends bidding immediately
            finalize_contract(state);
            return;
        }
        _ => {
            // Regular bid or capot
            let (value, suit_idx) = decode_bid(action);
            state.last_bid_value = value;
            state.last_bid_suit = suit_idx;
            state.last_bidder = state.current_player;
            state.coinche_state = 0;
            state.consecutive_passes = 0;
        }
    }

    // Advance to next player
    state.current_player = (state.current_player + 1) % 4;
}

/// Finalize bidding: set contract and transition to playing phase.
fn finalize_contract(state: &mut GameState) {
    state.contract = Contract {
        trump: state.last_bid_suit,
        value: state.last_bid_value,
        team: GameState::player_team(state.last_bidder),
        coinche: state.coinche_state,
    };
    state.phase = Phase::Playing;
    // First player to lead is the one after the dealer
    state.trick_lead = (state.dealer + 1) % 4;
    state.current_player = state.trick_lead;
    state.trick_count = 0;
    state.current_trick = [EMPTY; 4];
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_state() -> GameState {
        // Simple state with all hands dealt
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        GameState::new(0, hands)
    }

    #[test]
    fn test_encode_decode_bid() {
        // 80 Spades
        let action = encode_bid(8, 0);
        assert_eq!(action, 1);
        let (v, s) = decode_bid(action);
        assert_eq!((v, s), (8, 0));

        // 100 Hearts
        let action = encode_bid(10, 1);
        let (v, s) = decode_bid(action);
        assert_eq!((v, s), (10, 1));

        // 160 Clubs
        let action = encode_bid(16, 3);
        assert_eq!(action, 36);
        let (v, s) = decode_bid(action);
        assert_eq!((v, s), (16, 3));

        // Capot Diamonds
        let action = encode_bid(25, 2);
        assert_eq!(action, 39);
        let (v, s) = decode_bid(action);
        assert_eq!((v, s), (25, 2));
    }

    #[test]
    fn test_legal_bids_initial() {
        let state = make_test_state();
        let legal = legal_bids(&state);

        // Pass is legal
        assert!(legal & 1 != 0);
        // All normal bids legal (1-40)
        for i in 1..=40u8 {
            assert!(legal & (1u64 << i) != 0, "Action {} should be legal", i);
        }
        // Coinche and surcoinche not legal (no bid yet)
        assert!(legal & (1u64 << BID_COINCHE) == 0);
        assert!(legal & (1u64 << BID_SURCOINCHE) == 0);
    }

    #[test]
    fn test_four_passes_void() {
        let mut state = make_test_state();
        // 4 passes → done
        for _ in 0..4 {
            assert_eq!(state.phase, Phase::Bidding);
            apply_bid(&mut state, BID_PASS);
        }
        assert_eq!(state.phase, Phase::Done);
    }

    #[test]
    fn test_bid_then_three_passes() {
        let mut state = make_test_state();
        // Player 1 bids 80 Spades
        apply_bid(&mut state, encode_bid(8, 0));
        assert_eq!(state.last_bid_value, 8);
        assert_eq!(state.last_bidder, 1);

        // 3 passes → contract set
        apply_bid(&mut state, BID_PASS);
        apply_bid(&mut state, BID_PASS);
        apply_bid(&mut state, BID_PASS);

        assert_eq!(state.phase, Phase::Playing);
        assert_eq!(state.contract.trump, 0); // Spades
        assert_eq!(state.contract.value, 8); // 80
        assert_eq!(state.contract.team, 1); // Player 1 = EW
    }

    #[test]
    fn test_coinche() {
        let mut state = make_test_state();
        // Player 1 (EW) bids 80 Spades
        apply_bid(&mut state, encode_bid(8, 0));
        // Player 2 (NS) can coinche
        let legal = legal_bids(&state);
        assert!(legal & (1u64 << BID_COINCHE) != 0);
        apply_bid(&mut state, BID_COINCHE);
        assert_eq!(state.coinche_state, 1);

        // Player 3 (EW, same team as bidder) can surcoinche
        let legal = legal_bids(&state);
        assert!(legal & (1u64 << BID_SURCOINCHE) != 0);
        assert!(legal & (1u64 << BID_COINCHE) == 0); // can't coinche again
    }

    #[test]
    fn test_surcoinche_ends_bidding() {
        let mut state = make_test_state();
        // Player 1 bids, Player 2 coinches, Player 3 surcoinches
        apply_bid(&mut state, encode_bid(8, 0)); // P1: 80 Spades
        apply_bid(&mut state, BID_COINCHE); // P2: coinche
        apply_bid(&mut state, BID_SURCOINCHE); // P3: surcoinche

        assert_eq!(state.phase, Phase::Playing);
        assert_eq!(state.contract.coinche, 2);
    }

    #[test]
    fn test_overbid_required() {
        let mut state = make_test_state();
        // Player 1 bids 100 Spades
        apply_bid(&mut state, encode_bid(10, 0));

        let legal = legal_bids(&state);
        // Bids at 80 and 90 should be illegal
        for vi in 0..2u8 {
            for si in 0..4u8 {
                let action = vi * 4 + si + 1;
                assert!(
                    legal & (1u64 << action) == 0,
                    "Action {} (val={}) should be illegal",
                    action,
                    BID_VALUES[vi as usize] * 10
                );
            }
        }
        // 100 should also be illegal (not strictly higher)
        for si in 0..4u8 {
            let action = 2 * 4 + si + 1; // value_idx=2 → value=100
            assert!(
                legal & (1u64 << action) == 0,
                "100 bid should be illegal"
            );
        }
        // 110+ should be legal
        for vi in 3..9u8 {
            for si in 0..4u8 {
                let action = vi * 4 + si + 1;
                assert!(
                    legal & (1u64 << action) != 0,
                    "Action {} (val={}) should be legal",
                    action,
                    BID_VALUES[vi as usize] * 10
                );
            }
        }
        // Capot should be legal
        for si in 0..4u8 {
            assert!(legal & (1u64 << (37 + si)) != 0);
        }
    }

    #[test]
    fn test_after_coinche_no_more_bids() {
        let mut state = make_test_state();
        apply_bid(&mut state, encode_bid(8, 0)); // P1 (EW): 80S
        apply_bid(&mut state, BID_COINCHE); // P2 (NS): coinche

        // P3 (EW, same team as bidder) can surcoinche or pass.
        // No more bids allowed (coinche freezes the contract).
        let legal = legal_bids(&state);
        assert!(legal & (1u64 << BID_PASS) != 0);
        assert!(legal & (1u64 << BID_SURCOINCHE) != 0);
        // Cannot overbid
        let action_90s = encode_bid(9, 0);
        assert!(legal & (1u64 << action_90s) == 0);
    }

    #[test]
    fn test_coinche_then_three_passes_ends() {
        let mut state = make_test_state();
        apply_bid(&mut state, encode_bid(8, 0)); // P1: 80S
        apply_bid(&mut state, BID_COINCHE); // P2: coinche
        // Now 3 passes end the bidding
        apply_bid(&mut state, BID_PASS); // P3
        apply_bid(&mut state, BID_PASS); // P0
        apply_bid(&mut state, BID_PASS); // P1

        assert_eq!(state.phase, Phase::Playing);
        assert_eq!(state.contract.coinche, 1);
        assert_eq!(state.contract.value, 8);
    }
}
