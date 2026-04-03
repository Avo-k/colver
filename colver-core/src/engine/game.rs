use crate::bidding;
use crate::play;
use crate::scoring;
use crate::state::*;

/// Action encoding:
/// During bidding: 0-42 (see bidding.rs)
/// During playing: 0-31 (card index)
///
/// `legal_actions()` returns a u64 bitmask. The interpretation depends on the phase.

impl GameState {
    /// Get legal actions as a u64 bitmask.
    /// Bidding phase: bits 0-42 represent bid actions.
    /// Playing phase: bits 0-31 represent card indices.
    pub fn legal_actions(&self) -> u64 {
        match self.phase {
            Phase::Bidding => bidding::legal_bids(self),
            Phase::Playing => play::legal_plays(self) as u64,
            Phase::Done => 0,
        }
    }

    /// Apply an action. Dispatches based on phase.
    pub fn step(&mut self, action: u8) {
        match self.phase {
            Phase::Bidding => bidding::apply_bid(self, action),
            Phase::Playing => play::apply_play(self, action),
            Phase::Done => panic!("Cannot step a terminal state"),
        }
    }

    /// Get rewards for both teams. Only meaningful when terminal.
    pub fn rewards(&self) -> [f32; 2] {
        debug_assert!(self.is_terminal());
        scoring::deal_rewards(self)
    }

    /// Get the deal score breakdown.
    pub fn deal_score(&self) -> scoring::DealScore {
        scoring::compute_deal_score(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::ALL_CARDS;

    #[test]
    fn test_random_game_to_completion() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            let mut state = GameState::deal_random(0, &mut rng);

            while !state.is_terminal() {
                let legal = state.legal_actions();
                assert!(legal != 0, "No legal actions but not terminal: {:?}", state);

                // Pick a random legal action
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                let action = select_nth_bit(legal, idx);
                state.step(action);
            }

            // Verify invariants at terminal state
            if state.contract.value > 0 {
                // A contract was played
                let total_tricks = state.tricks_won[0] + state.tricks_won[1];
                assert_eq!(total_tricks, 8, "Total tricks should be 8, got {}", total_tricks);

                // All cards should be played
                assert_eq!(
                    state.played_cards, ALL_CARDS,
                    "Not all cards were played"
                );

                // Each hand should be empty
                for i in 0..4 {
                    assert_eq!(state.hands[i], 0, "Player {} still has cards", i);
                }

                // Total trick points (before dix de der) + dix de der = 162 or 252
                let total_pts = state.points[0] as u16 + state.points[1] as u16;
                let is_capot = state.tricks_won[0] == 8 || state.tricks_won[1] == 8;
                if is_capot {
                    assert_eq!(total_pts, 252, "Capot total should be 252, got {}", total_pts);
                } else {
                    assert_eq!(total_pts, 162, "Total points should be 162, got {}", total_pts);
                }
            }
        }
    }

    #[test]
    fn test_void_deal() {
        // 4 passes → void deal
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let mut state = GameState::new(0, hands);

        for _ in 0..4 {
            state.step(0); // PASS
        }

        assert!(state.is_terminal());
        let rewards = state.rewards();
        assert_eq!(rewards, [0.0, 0.0]);
    }

    /// Select the nth set bit from a u64.
    fn select_nth_bit(mask: u64, mut n: u32) -> u8 {
        let mut remaining = mask;
        loop {
            let bit = remaining.trailing_zeros() as u8;
            if n == 0 {
                return bit;
            }
            n -= 1;
            remaining &= remaining - 1;
        }
    }

    #[test]
    fn test_scripted_game() {
        // Play a fully scripted game to verify everything ties together.
        let hands = [
            0xFF,           // P0: 7S 8S 9S JS QS KS 10S AS
            0xFF00,         // P1: 7H 8H 9H JH QH KH 10H AH
            0xFF_0000,      // P2: 7D 8D 9D JD QD KD 10D AD
            0xFF00_0000,    // P3: 7C 8C 9C JC QC KC 10C AC
        ];
        let mut state = GameState::new(0, hands);
        // Dealer = P0, first bidder = P1

        // P1 bids 80 Hearts (Hearts = trump)
        state.step(bidding::encode_bid(8, 1)); // P1: 80H
        state.step(0); // P2: pass
        state.step(0); // P3: pass
        state.step(0); // P0: pass → contract set

        assert_eq!(state.phase, Phase::Playing);
        assert_eq!(state.contract.trump, 1); // Hearts
        assert_eq!(state.contract.value, 8); // 80
        assert_eq!(state.contract.team, 1); // EW

        // Play phase: P1 leads (right of dealer P0)
        // Each player plays from their suit in order.
        // Since no one has the lead suit except the leader, others must cut or discard.

        // Trick 1: P1 leads AH. Others have no hearts → discard.
        // Actually, trump is hearts. P1 leads AH (trump). Others must follow trump but have none.
        // So they discard.
        // P1 leads
        assert_eq!(state.current_player, 1);

        // Let's play each suit against each other. P1 leads hearts (trump).
        // P1: AH(15), P2: 7D(16), P3: 7C(24), P0: 7S(0)
        state.step(15); // P1: AH
        state.step(16); // P2: 7D
        state.step(24); // P3: 7C
        state.step(0);  // P0: 7S

        // AH wins (only trump). P1 leads again.
        // Points: AH=11 (trump) + 7D=0 + 7C=0 + 7S=0 = 11
        assert_eq!(state.trick_lead, 1);
        assert_eq!(state.tricks_won[1], 1); // EW
    }
}
