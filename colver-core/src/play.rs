use crate::card::*;
use crate::state::*;
use crate::trick::trick_winner;

/// Compute legal card plays as a CardSet bitmask.
///
/// Rules:
/// 1. Must follow lead suit if possible
/// 2. If can't follow:
///    a. If partner is winning ("master") → play anything
///    b. Else must trump if possible; must overtrump if possible
///    c. "Ne pisse pas": if can't overtrump, may discard instead of undertrumping
/// 3. When playing trump (following or cutting): must overtrump highest trump on table if possible
/// 4. Exception to 3: partner is master with a trump cut AND you only have trumps → can undertrump
pub fn legal_plays(state: &GameState) -> CardSet {
    let hand = state.hands[state.current_player as usize];
    debug_assert!(hand != 0, "Player has no cards");

    // Leader plays anything
    if state.trick_count == 0 {
        return hand;
    }

    let lead_card = state.current_trick[state.trick_lead as usize];
    let lead_suit = card_suit(lead_card);
    let trump_suit = state.contract.trump_suit();

    legal_plays_color(hand, lead_suit, trump_suit, state)
}

/// Color contract play logic.
fn legal_plays_color(
    hand: CardSet,
    lead_suit: Suit,
    trump_suit: Suit,
    state: &GameState,
) -> CardSet {
    let in_lead = cards_in_suit(hand, lead_suit);

    if lead_suit == trump_suit {
        // Trump was led
        if in_lead != 0 {
            // Must follow with trump; must overtrump
            let best_rank = best_trump_rank_on_trick(state, trump_suit);
            if let Some(br) = best_rank {
                let higher = overtrump_in_suit(in_lead, trump_suit, br);
                if higher != 0 {
                    return higher;
                }
            }
            // Can't overtrump → play any trump
            in_lead
        } else {
            // No trump in hand → discard anything
            hand
        }
    } else {
        // Non-trump suit was led
        if in_lead != 0 {
            // Must follow suit (no overtrump requirement for non-trump suits)
            return in_lead;
        }

        // Can't follow suit
        let in_trump = cards_in_suit(hand, trump_suit);

        if partner_is_master(state) {
            // Partner is currently winning → can play anything
            return hand;
        }

        if in_trump != 0 {
            // Must trump (must cut)
            // Check if we need to overtrump
            let best_trump_rank = best_trump_rank_on_trick(state, trump_suit);
            if let Some(br) = best_trump_rank {
                let higher = overtrump_in_suit(in_trump, trump_suit, br);
                if higher != 0 {
                    return higher;
                }
                // "Ne pisse pas": can't overtrump opponent's trump
                // → can discard (non-trump) instead of undertrumping
                let non_trump = hand & !SUIT_MASK[trump_suit as usize];
                if non_trump != 0 {
                    return in_trump | non_trump;
                }
                // Only have trump → must undertrump
                return in_trump;
            }
            // No trump on table yet → must cut with any trump
            in_trump
        } else {
            // No trump in hand → discard anything
            hand
        }
    }
}

/// Find the highest trump strength rank currently on the trick for a given suit.
/// Returns None if no cards of that suit are on the trick.
fn best_trump_rank_on_trick(state: &GameState, suit: Suit) -> Option<u8> {
    let mut best: Option<u8> = None;
    let mut best_strength = 0u8;

    for i in 0..state.trick_count {
        let seat = (state.trick_lead + i) % 4;
        let card = state.current_trick[seat as usize];
        if card == EMPTY {
            continue;
        }
        if card_suit(card) == suit {
            let rank = card_rank(card);
            let strength = TRUMP_STRENGTH[rank as usize];
            if best.is_none() || strength > best_strength {
                best_strength = strength;
                best = Some(rank);
            }
        }
    }

    best
}

/// Check if the current player's partner is currently winning the trick.
fn partner_is_master(state: &GameState) -> bool {
    if state.trick_count < 2 {
        return false; // Partner hasn't played yet (or only lead played)
    }

    let player = state.current_player;
    let partner = GameState::partner(player);

    // Build partial trick for winner computation
    // We need to determine who's winning among the cards played so far.
    let lead = state.trick_lead;
    let lead_card = state.current_trick[lead as usize];
    let lead_suit = card_suit(lead_card);
    let trump_suit = state.contract.trump_suit();

    let mut best_seat = lead;
    let mut has_trump = false;
    let mut best_trump_strength = 0u8;
    let mut best_lead_rank = 0u8;
    let mut best_lead_seat = lead;

    // Check lead
    if lead_suit == trump_suit {
        has_trump = true;
        best_trump_strength = TRUMP_STRENGTH[card_rank(lead_card) as usize];
        best_seat = lead;
    } else {
        best_lead_rank = card_rank(lead_card);
        best_lead_seat = lead;
    }

    for i in 1..state.trick_count {
        let seat = (lead + i) % 4;
        let card = state.current_trick[seat as usize];
        let suit = card_suit(card);

        if suit == trump_suit {
            let s = TRUMP_STRENGTH[card_rank(card) as usize];
            if !has_trump || s > best_trump_strength {
                best_trump_strength = s;
                best_seat = seat;
                has_trump = true;
            }
        } else if suit == lead_suit && !has_trump {
            let r = card_rank(card);
            if r > best_lead_rank {
                best_lead_rank = r;
                best_lead_seat = seat;
            }
        }
    }

    if !has_trump {
        best_seat = best_lead_seat;
    }

    best_seat == partner
}

/// Get the subset of cards in `hand_in_suit` (which are all in `suit`) that
/// overtrump the given rank (using trump strength ordering).
#[inline]
fn overtrump_in_suit(hand_in_suit: CardSet, suit: Suit, best_rank: u8) -> CardSet {
    let higher_ranks = HIGHER_TRUMP_MASK[best_rank as usize];
    let shift = SUIT_SHIFT[suit as usize];
    let higher_mask = (higher_ranks as u32) << shift;
    hand_in_suit & higher_mask
}

/// Apply a card play action. The action is a card index (0-31).
pub fn apply_play(state: &mut GameState, card: Card) {
    let player = state.current_player;
    let bit = card_to_bit(card);

    debug_assert!(
        state.hands[player as usize] & bit != 0,
        "Player {} doesn't have card {}",
        player,
        card_name(card)
    );

    // Remove card from hand
    state.hands[player as usize] &= !bit;
    state.played_cards |= bit;

    // Place card in trick
    state.current_trick[player as usize] = card;
    state.trick_count += 1;

    // Track voids: if player didn't follow lead suit, mark void
    if state.trick_count > 1 {
        let lead_card = state.current_trick[state.trick_lead as usize];
        let lead_suit = card_suit(lead_card);
        if card_suit(card) != lead_suit {
            state.voids[player as usize] |= 1 << (lead_suit as u8);
        }
    }

    // Check for belote/rebelote (Q+K of trump)
    check_belote(state, player, card);

    if state.trick_count == 4 {
        // Trick complete - resolve
        resolve_trick(state);
    } else {
        // Next player
        state.current_player = (player + 1) % 4;
    }
}

/// Resolve a completed trick.
fn resolve_trick(state: &mut GameState) {
    let winner = trick_winner(&state.current_trick, state.trick_lead, &state.contract);
    let team = GameState::player_team(winner) as usize;

    let pts = crate::trick::trick_points(&state.current_trick, &state.contract);
    state.points[team] += pts;
    state.tricks_won[team] += 1;

    // Check if this is the last trick (8 tricks total)
    let total_tricks = state.tricks_won[0] + state.tricks_won[1];
    if total_tricks == 8 {
        // "Dix de der": last trick bonus.
        // Normal: 10 points. Capot (8 tricks by one team): 100 points.
        if state.tricks_won[team] == 8 {
            state.points[team] += 100; // capot dix de der
        } else {
            state.points[team] += 10; // normal dix de der
        }
        state.phase = Phase::Done;
    } else {
        // Start new trick
        state.trick_lead = winner;
        state.current_player = winner;
        state.trick_count = 0;
        state.current_trick = [EMPTY; 4];
    }
}

/// Check and track belote (Q+K of trump suit).
fn check_belote(state: &mut GameState, player: u8, card: Card) {
    let trump_suit = state.contract.trump_suit();
    if card_suit(card) == trump_suit {
        let rank = card_rank(card);
        if rank == 4 || rank == 5 {
            // Queen or King of trump
            let team = GameState::player_team(player) as usize;
            if state.belote[team] == 0 {
                state.belote[team] = 1; // belote
            } else if state.belote[team] == 1 {
                state.belote[team] = 2; // rebelote
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_playing_state(trump: u8, hands: [CardSet; 4]) -> GameState {
        let mut state = GameState::new(0, hands);
        state.phase = Phase::Playing;
        state.contract = Contract {
            trump,
            value: 8,
            team: 0,
            coinche: 0,
        };
        state.trick_lead = 1;
        state.current_player = 1;
        state
    }

    #[test]
    fn test_leader_plays_anything() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = make_playing_state(0, hands); // Spades trump
        let legal = legal_plays(&state);
        assert_eq!(legal, 0xFF00); // P1 can play any of their 8 cards
    }

    #[test]
    fn test_must_follow_suit() {
        let mut state = make_playing_state(1, [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000]);
        // P1 leads 7H (card 8)
        state.current_trick[1] = make_card(Suit::Hearts, 0); // 7H
        state.trick_count = 1;
        state.current_player = 2;

        // P2 has all diamonds - can they follow hearts? No → play anything
        let legal = legal_plays(&state);
        assert_eq!(legal, 0xFF_0000); // all diamonds (can't follow, partner hasn't played, must trump... but no trump)

        // Actually P2 has diamonds, trump is hearts.
        // P2 can't follow hearts AND has no trump (hearts) → discard anything
        assert_eq!(legal, 0xFF_0000);
    }

    #[test]
    fn test_must_follow_lead_suit() {
        // P1 leads Spade, P2 has some spades → must play a spade
        let mut state = make_playing_state(1, [
            0xFF,           // P0: all spades
            0xFF00,         // P1: all hearts
            0x0F | 0xF0_0000, // P2: 7S,8S,9S,JS + some diamonds
            0xFF00_0000,    // P3: all clubs
        ]);
        // P1 leads 7H
        state.current_trick[1] = make_card(Suit::Hearts, 0);
        state.trick_count = 1;
        state.current_player = 2;

        // P2 has no hearts. Trump is hearts. P2 has no trump either.
        // Spades in P2's hand: 0x0F. No hearts (trump).
        // Partner (P0) hasn't played yet. No trump → discard anything.
        let legal = legal_plays(&state);
        assert_eq!(legal, state.hands[2]); // can play anything
    }

    #[test]
    fn test_must_cut_with_trump() {
        // Trump: Spades (0). Lead: Hearts.
        // P2 has no hearts but has spades (trump) → must cut
        let mut state = make_playing_state(0, [
            0xFF,                       // P0: all spades
            0xFF00,                     // P1: all hearts
            0x0300_0000 | 0x03_0000,    // P2: 7C,8C,7D,8D
            0xFC00_0000,                // P3: rest of clubs
        ]);

        // Wait, P2 has no trump (spades) in this setup. Let me fix.
        // Trump = Spades (0). P2 needs some spades.
        state.hands[0] = 0xF0;           // P0: Q,K,10,A of spades
        state.hands[2] = 0x0F | 0x0F_0000; // P2: 7,8,9,J of spades + 7,8,9,J of diamonds

        state.current_trick[1] = make_card(Suit::Hearts, 0); // P1 leads 7H
        state.trick_count = 1;
        state.current_player = 2;

        // P2 can't follow hearts. Has trump (spades: 0x0F). Partner (P0) hasn't played.
        // Must cut with trump. No trump on table yet → any trump.
        let legal = legal_plays(&state);
        assert_eq!(legal, 0x0F); // only spades (trump)
    }

    #[test]
    fn test_partner_master_can_discard() {
        // Trump: Spades (0). Lead: Hearts by P1. P2 plays AH (wins). P3 must play.
        // P3 has no hearts but has trump. Partner (P1) is NOT master (P2 is master).
        // So P3 must cut.

        // Now test when partner IS master:
        // P1 leads AH. P2 plays 7D (discard). P3: partner is P1, P1 has AH (winning).
        let mut state = make_playing_state(0, [
            0xF0,           // P0
            0xFF00,         // P1: all hearts
            0xFF_0000,      // P2: all diamonds
            0x0F | 0xF000_0000, // P3: 7,8,9,J spades + some clubs
        ]);

        state.trick_lead = 1;
        state.current_trick[1] = make_card(Suit::Hearts, 7); // P1 leads AH
        state.current_trick[2] = make_card(Suit::Diamonds, 0); // P2 plays 7D (discard)
        state.trick_count = 2;
        state.current_player = 3;

        // P3's partner is P1. P1 played AH which is currently winning. Partner IS master.
        // P3 can play anything.
        let legal = legal_plays(&state);
        assert_eq!(legal, state.hands[3]);
    }

    #[test]
    fn test_ne_pisse_pas() {
        // Trump: Spades (0). Lead: Hearts. Opponent plays trump (overcut).
        // Player has only lower trumps → "ne pisse pas" → can discard.
        let _state = make_playing_state(0, [
            0xFF_0000,      // P0: all diamonds
            0xFF00,         // P1: all hearts
            0x01 | 0xFF00_0000, // P2: 7S (weakest trump) + all clubs
            0xFE,           // P3: 8S-AS (all other spades)
        ]);

        // Trump: Clubs (3). Lead: Hearts.
        let mut state = make_playing_state(3, [
            0xFF_0000,      // P0: all diamonds
            0xFF00,         // P1: all hearts
            0x03 | 0x0300_0000, // P2: 7S,8S + 7C,8C
            0xFC00_0000,    // P3: 9C-AC (strong clubs)
        ]);

        // P1 leads AH. P3 (opponent of P2) cuts with JC (strong trump).
        // But trick order: P1 leads, P2 next, P3 next, P0 next.
        state.trick_lead = 1;
        state.current_trick[1] = make_card(Suit::Hearts, 7); // P1: AH
        // Actually I need P3 to have played before P2. Let me use a different lead.
        // Lead = P3, so: P3 leads, P0, P1, P2.
        state.trick_lead = 3;
        state.current_trick[3] = make_card(Suit::Hearts, 7); // P3: AH (lead)
        state.current_trick[0] = make_card(Suit::Diamonds, 0); // P0: 7D (discard)
        state.current_trick[1] = make_card(Suit::Hearts, 6); // P1: 10H (follows suit but lower)
        // Hmm, P3 leads AH. Trump is clubs. AH is winning (no trump played).
        // P2 is next. P2 can't follow hearts. P2's partner is P0 (who discarded, not winning).
        // Opponent P3 is winning with AH. P2 must cut.
        // P2 has 7C, 8C (trump). No opponent trump on table → must cut with any trump.
        state.trick_count = 3;
        state.current_player = 2;
        let legal = legal_plays(&state);
        // Should be just the two club trumps (7C and 8C)
        assert_eq!(legal, 0x0300_0000);

        // Now test "ne pisse pas": opponent already trumped with a strong trump
        // P3 leads 7D. P0 plays AD (follows). P1 cuts with JC (trump). P2 next.
        let mut state2 = make_playing_state(3, [
            0xFF_0000,       // P0: all diamonds
            0x08_0000 | 0x0800_0000, // P1: JD + JC (trump Jack)
            0x03 | 0x0300_0000, // P2: 7S,8S + 7C,8C (weak trump)
            0xFF00,          // P3: all hearts -- wait needs diamonds to lead
        ]);
        // Fix: give P3 some diamonds
        state2.hands[3] = 0xF0_0000 | 0xF000; // P3: Q,K,10,A diamonds + some hearts
        state2.hands[0] = 0x0F_0000; // P0: 7,8,9,J diamonds
        state2.hands[1] = 0xF000 | 0x0800_0000; // P1: Q,K,10,A hearts + JC (trump)

        state2.trick_lead = 3;
        state2.current_trick[3] = make_card(Suit::Diamonds, 4); // P3: QD
        state2.current_trick[0] = make_card(Suit::Diamonds, 0); // P0: 7D
        state2.current_trick[1] = make_card(Suit::Clubs, 3); // P1: JC (trump - strongest!)
        state2.trick_count = 3;
        state2.current_player = 2;

        // P2 can't follow diamonds, has trump (7C, 8C) but both are weaker than JC.
        // "Ne pisse pas": can't overtrump → can discard non-trump OR undertrump.
        let legal2 = legal_plays(&state2);
        // Should include all P2's cards: 7S, 8S (non-trump discard) + 7C, 8C (undertrump)
        assert_eq!(legal2, state2.hands[2]);
    }

    #[test]
    fn test_apply_play_basic() {
        let mut state = make_playing_state(2, [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000]);
        // Diamonds trump. P1 leads.
        state.current_player = 1;
        state.trick_lead = 1;

        apply_play(&mut state, make_card(Suit::Hearts, 7)); // P1: AH
        assert_eq!(state.trick_count, 1);
        assert_eq!(state.current_player, 2);
        assert!(state.hands[1] & card_to_bit(make_card(Suit::Hearts, 7)) == 0);

        apply_play(&mut state, make_card(Suit::Diamonds, 0)); // P2: 7D (can't follow hearts)
        assert_eq!(state.trick_count, 2);
        assert_eq!(state.current_player, 3);
        // P2 should be marked void in hearts
        assert!(state.voids[2] & (1 << Suit::Hearts as u8) != 0);
    }

    #[test]
    fn test_full_trick_resolution() {
        let mut state = make_playing_state(2, [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000]);
        // Diamonds trump. P1 leads AH.
        state.current_player = 1;
        state.trick_lead = 1;

        apply_play(&mut state, make_card(Suit::Hearts, 7)); // P1: AH (11 pts plain)
        apply_play(&mut state, make_card(Suit::Diamonds, 0)); // P2: 7D (trump, 0 pts)
        apply_play(&mut state, make_card(Suit::Clubs, 0)); // P3: 7C (0 pts)
        apply_play(&mut state, make_card(Suit::Spades, 0)); // P0: 7S (0 pts)

        // 7D (trump) beats AH, P2 (team NS=0) wins
        assert_eq!(state.tricks_won[0], 1);
        assert_eq!(state.points[0], 11); // AH(11) + 7D(0) + 7C(0) + 7S(0) = 11
        assert_eq!(state.trick_lead, 2); // P2 leads next
        assert_eq!(state.trick_count, 0);
    }
}
