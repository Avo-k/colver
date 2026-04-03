use crate::card::*;
use crate::state::*;

/// Determine the winner of a completed trick.
/// Returns the seat index (0-3) of the winning player.
pub fn trick_winner(trick: &[Card; 4], lead: u8, contract: &Contract) -> u8 {
    let trump_suit = contract.trump_suit();
    let lead_card = trick[lead as usize];
    let lead_suit = card_suit(lead_card);

    // First check if any trump was played
    let mut best_trump_seat: Option<u8> = None;
    let mut best_trump_strength: u8 = 0;

    let mut best_lead_seat = lead;
    let mut best_lead_rank = card_rank(lead_card);

    // Check lead card
    if lead_suit == trump_suit {
        best_trump_seat = Some(lead);
        best_trump_strength = TRUMP_STRENGTH[card_rank(lead_card) as usize];
    }

    for i in 1..4u8 {
        let seat = (lead + i) % 4;
        let card = trick[seat as usize];
        let suit = card_suit(card);

        if suit == trump_suit {
            let strength = TRUMP_STRENGTH[card_rank(card) as usize];
            if best_trump_seat.is_none() || strength > best_trump_strength {
                best_trump_strength = strength;
                best_trump_seat = Some(seat);
            }
        } else if suit == lead_suit {
            let rank = card_rank(card);
            if rank > best_lead_rank {
                best_lead_rank = rank;
                best_lead_seat = seat;
            }
        }
    }

    // Trump beats plain; highest trump wins; else highest in lead suit wins
    best_trump_seat.unwrap_or(best_lead_seat)
}

/// Sum points of the 4 cards in a trick.
pub fn trick_points(trick: &[Card; 4], contract: &Contract) -> u8 {
    let ct = contract.contract_type();
    let mut total: u8 = 0;
    for &card in trick.iter() {
        total += card_points(card, ct);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trick_winner_plain() {
        // Hearts trump, all play Spades (plain suit) — highest plain rank wins
        let contract = Contract {
            trump: 1, // Hearts
            value: 8,
            team: 0,
            coinche: 0,
        };

        // P0 leads 7S(0), P1 plays AS(7), P2 plays QS(4), P3 plays KS(5)
        let trick = [
            make_card(Suit::Spades, 0), // 7S
            make_card(Suit::Spades, 7), // AS
            make_card(Suit::Spades, 4), // QS
            make_card(Suit::Spades, 5), // KS
        ];
        assert_eq!(trick_winner(&trick, 0, &contract), 1); // AS wins
    }

    #[test]
    fn test_trick_winner_trump_cuts() {
        // Hearts trump, Spades led
        let contract = Contract {
            trump: 1, // Hearts
            value: 8,
            team: 0,
            coinche: 0,
        };

        // P0 leads AS(7), P1 plays 7H(8), P2 plays KS(5), P3 plays 10S(6)
        let trick = [
            make_card(Suit::Spades, 7),  // AS
            make_card(Suit::Hearts, 0),  // 7H (trump)
            make_card(Suit::Spades, 5),  // KS
            make_card(Suit::Spades, 6),  // 10S
        ];
        assert_eq!(trick_winner(&trick, 0, &contract), 1); // 7H (trump) beats AS
    }

    #[test]
    fn test_trick_winner_highest_trump() {
        // Hearts trump, Spades led
        let contract = Contract {
            trump: 1, // Hearts
            value: 8,
            team: 0,
            coinche: 0,
        };

        // P0 leads 7S, P1 plays 7H (trump), P2 plays JH (trump), P3 plays AH (trump)
        let trick = [
            make_card(Suit::Spades, 0),  // 7S
            make_card(Suit::Hearts, 0),  // 7H
            make_card(Suit::Hearts, 3),  // JH (strongest trump!)
            make_card(Suit::Hearts, 7),  // AH
        ];
        assert_eq!(trick_winner(&trick, 0, &contract), 2); // JH wins (J > A in trump)
    }

    #[test]
    fn test_trick_winner_non_lead_suit_loses() {
        // Hearts trump, Spades led — non-lead/non-trump cards lose
        let contract = Contract {
            trump: 1, // Hearts
            value: 8,
            team: 0,
            coinche: 0,
        };

        // P0 leads 7S, P1 plays AD (wrong suit, not trump), P2 plays 8S, P3 plays AC (wrong suit, not trump)
        let trick = [
            make_card(Suit::Spades, 0),   // 7S
            make_card(Suit::Diamonds, 7), // AD
            make_card(Suit::Spades, 1),   // 8S
            make_card(Suit::Clubs, 7),    // AC
        ];
        assert_eq!(trick_winner(&trick, 0, &contract), 2); // 8S wins (only lead suit counts, no trump played)
    }

    #[test]
    fn test_trick_points_color() {
        let contract = Contract {
            trump: 1, // Hearts
            value: 8,
            team: 0,
            coinche: 0,
        };

        // JH(20) + 9H(14) + AS(11) + 10S(10)
        let trick = [
            make_card(Suit::Hearts, 3),  // JH: 20 (trump)
            make_card(Suit::Hearts, 2),  // 9H: 14 (trump)
            make_card(Suit::Spades, 7),  // AS: 11 (plain)
            make_card(Suit::Spades, 6),  // 10S: 10 (plain)
        ];
        assert_eq!(trick_points(&trick, &contract), 55);
    }

    #[test]
    fn test_trick_winner_trump_led() {
        // Hearts trump, Hearts led
        let contract = Contract {
            trump: 1,
            value: 8,
            team: 0,
            coinche: 0,
        };

        // P2 leads 8H, P3 plays AH, P0 plays 9H, P1 plays 7H
        let trick = [
            make_card(Suit::Hearts, 2), // 9H (played by P0)
            make_card(Suit::Hearts, 0), // 7H (played by P1)
            make_card(Suit::Hearts, 1), // 8H (played by P2, lead)
            make_card(Suit::Hearts, 7), // AH (played by P3)
        ];
        // Trump order: J>9>A>10>K>Q>8>7
        // 9H (strength 6) > AH (strength 5) > 8H (strength 1) > 7H (strength 0)
        assert_eq!(trick_winner(&trick, 2, &contract), 0); // 9H (P0) wins
    }
}
