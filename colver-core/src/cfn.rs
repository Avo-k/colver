//! CFN: Contrée FEN Notation
//!
//! A compact string notation for Belote Contrée game states, analogous to FEN in chess.
//! Three space-separated sections: `<dealer>:<hands> <tricks> <contract>`
//!
//! Key design: trick history is stored, and points/tricks_won/voids/belote/current_player
//! are all derived by replaying the trick sequence.

use crate::bidding;
use crate::card::*;
use crate::state::*;
use crate::trick;

use core::fmt;
use core::str::FromStr;

/// Errors when parsing a CFN string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CfnError {
    InvalidFormat(String),
    InvalidDealer(String),
    InvalidHand(String),
    InvalidCard(String),
    InvalidContract(String),
    InvalidTrick(String),
    OverlappingCards,
    WrongCardCount(String),
}

impl fmt::Display for CfnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CfnError::InvalidFormat(s) => write!(f, "invalid CFN format: {}", s),
            CfnError::InvalidDealer(s) => write!(f, "invalid dealer: {}", s),
            CfnError::InvalidHand(s) => write!(f, "invalid hand: {}", s),
            CfnError::InvalidCard(s) => write!(f, "invalid card: {}", s),
            CfnError::InvalidContract(s) => write!(f, "invalid contract: {}", s),
            CfnError::InvalidTrick(s) => write!(f, "invalid trick: {}", s),
            CfnError::OverlappingCards => write!(f, "overlapping cards between hands/tricks"),
            CfnError::WrongCardCount(s) => write!(f, "wrong card count: {}", s),
        }
    }
}

// ---- Character <-> index mappings ----

const RANK_CHARS: [char; 8] = ['7', '8', '9', 'J', 'Q', 'K', 'T', 'A'];
const SUIT_CHARS: [char; 4] = ['s', 'h', 'd', 'c'];
const PLAYER_CHARS: [char; 4] = ['N', 'E', 'S', 'W'];

fn rank_to_char(rank: u8) -> char {
    RANK_CHARS[rank as usize]
}

fn char_to_rank(c: char) -> Result<u8, CfnError> {
    match c {
        '7' => Ok(0),
        '8' => Ok(1),
        '9' => Ok(2),
        'J' => Ok(3),
        'Q' => Ok(4),
        'K' => Ok(5),
        'T' => Ok(6),
        'A' => Ok(7),
        _ => Err(CfnError::InvalidCard(format!("unknown rank '{}'", c))),
    }
}

fn suit_to_char(suit: u8) -> char {
    SUIT_CHARS[suit as usize]
}

fn char_to_suit(c: char) -> Result<u8, CfnError> {
    match c {
        's' => Ok(0),
        'h' => Ok(1),
        'd' => Ok(2),
        'c' => Ok(3),
        _ => Err(CfnError::InvalidCard(format!("unknown suit '{}'", c))),
    }
}

fn player_to_char(player: u8) -> char {
    PLAYER_CHARS[player as usize]
}

fn char_to_player(c: char) -> Result<u8, CfnError> {
    match c {
        'N' => Ok(0),
        'E' => Ok(1),
        'S' => Ok(2),
        'W' => Ok(3),
        _ => Err(CfnError::InvalidDealer(format!("unknown player '{}'", c))),
    }
}

// ---- Card formatting ----

/// Format a card as 2 chars: rank + lowercase suit (e.g. "Ah", "Js", "Td")
fn format_card(card: Card) -> String {
    let rank = card_rank(card);
    let suit = card_suit_u8(card);
    format!("{}{}", rank_to_char(rank), suit_to_char(suit))
}

/// Parse a 2-char card string into a Card index.
fn parse_card(s: &str) -> Result<Card, CfnError> {
    let mut chars = s.chars();
    let rc = chars.next().ok_or_else(|| CfnError::InvalidCard("empty card".into()))?;
    let sc = chars.next().ok_or_else(|| CfnError::InvalidCard("incomplete card".into()))?;
    let rank = char_to_rank(rc)?;
    let suit = char_to_suit(sc)?;
    Ok(suit * 8 + rank)
}

// ---- Hand formatting ----

/// Format a CardSet as bridge-style: suits in S.H.D.C order, ranks high-to-low.
/// Empty hand = "-".
fn format_hand(hand: CardSet) -> String {
    if hand == 0 {
        return "-".into();
    }
    let mut parts = Vec::with_capacity(4);
    for suit in 0..4u8 {
        let bits = suit_bits(hand, Suit::from_u8(suit));
        let mut suit_str = String::new();
        // Ranks high-to-low: A(7), T(6), K(5), Q(4), J(3), 9(2), 8(1), 7(0)
        for rank in (0..8u8).rev() {
            if bits & (1 << rank) != 0 {
                suit_str.push(rank_to_char(rank));
            }
        }
        parts.push(suit_str);
    }
    parts.join(".")
}

/// Parse a bridge-style hand string into a CardSet.
fn parse_hand(s: &str) -> Result<CardSet, CfnError> {
    if s == "-" {
        return Ok(0);
    }
    let suits: Vec<&str> = s.split('.').collect();
    if suits.len() != 4 {
        return Err(CfnError::InvalidHand(format!(
            "expected 4 suit groups, got {} in '{}'",
            suits.len(),
            s
        )));
    }
    let mut hand: CardSet = 0;
    for (suit_idx, suit_str) in suits.iter().enumerate() {
        for c in suit_str.chars() {
            let rank = char_to_rank(c).map_err(|_| {
                CfnError::InvalidHand(format!("unknown rank '{}' in '{}'", c, s))
            })?;
            let card = (suit_idx as u8) * 8 + rank;
            hand |= card_to_bit(card);
        }
    }
    Ok(hand)
}

// ---- Trick formatting ----

/// Format the trick section from game state.
/// Uses trick_history for completed tricks, plus current_trick if partial.
fn format_tricks(state: &GameState) -> String {
    let completed = (state.tricks_won[0] + state.tricks_won[1]) as usize;
    if completed == 0 && state.trick_count == 0 {
        return "-".into();
    }

    let mut result = String::new();

    // Reconstruct trick leads for ordering
    let first_lead = (state.dealer + 1) % 4;
    let mut current_lead = first_lead;

    for t in 0..completed {
        if t > 0 {
            result.push('/');
        }
        let trick = state.trick_history[t];

        // Write cards in play order: lead first, then lead+1, lead+2, lead+3
        for i in 0..4u8 {
            let seat = (current_lead + i) % 4;
            let card = trick[seat as usize];
            result.push_str(&format_card(card));
        }

        // Compute winner to determine next lead
        let winner = trick::trick_winner(&trick, current_lead, &state.contract);
        current_lead = winner;
    }

    // Current partial trick (if any cards played and game not over)
    if state.trick_count > 0 && state.phase != Phase::Done {
        if completed > 0 {
            result.push('/');
        }
        for i in 0..state.trick_count {
            let seat = (state.trick_lead + i) % 4;
            let card = state.current_trick[seat as usize];
            result.push_str(&format_card(card));
        }
    }

    result
}

// ---- Contract formatting ----

fn format_contract(state: &GameState) -> String {
    match state.phase {
        Phase::Done => {
            if state.contract.value == 0 {
                // Void deal (4 passes)
                "0".into()
            } else {
                format_resolved_contract(&state.contract)
            }
        }
        Phase::Playing => format_resolved_contract(&state.contract),
        Phase::Bidding => format_bidding_state(state),
    }
}

fn format_resolved_contract(contract: &Contract) -> String {
    let value = if contract.is_capot() {
        250
    } else {
        contract.point_value()
    };
    let suit = suit_to_char(contract.trump);
    let team = if contract.team == 0 { "NS" } else { "EW" };
    let coinche = match contract.coinche {
        1 => "x",
        2 => "xx",
        _ => "",
    };
    format!("{}{}{}{}", value, suit, team, coinche)
}

fn format_bidding_state(state: &GameState) -> String {
    if state.last_bid_value == 0 {
        // No bid yet
        return format!("bid:0/p{}/{}", state.consecutive_passes, player_to_char(state.current_player));
    }
    let value = if state.last_bid_value == 25 {
        250
    } else {
        state.last_bid_value as u16 * 10
    };
    let suit = suit_to_char(state.last_bid_suit);
    let bidder = player_to_char(state.last_bidder);
    format!(
        "bid:{}{}{}/p{}/c{}/{}",
        value,
        suit,
        bidder,
        state.consecutive_passes,
        state.coinche_state,
        player_to_char(state.current_player)
    )
}

// ---- Contract parsing ----

fn parse_contract_section(s: &str, state: &mut GameState) -> Result<(), CfnError> {
    if s == "0" {
        // Void deal
        state.phase = Phase::Done;
        state.contract = Contract::default();
        return Ok(());
    }

    if let Some(rest) = s.strip_prefix("bid:") {
        return parse_bidding_contract(rest, state);
    }

    // Resolved contract: <value><suit><team>[x|xx]
    parse_resolved_contract(s, state)
}

fn parse_resolved_contract(s: &str, state: &mut GameState) -> Result<(), CfnError> {
    let err = || CfnError::InvalidContract(s.into());

    // Extract coinche suffix
    let (main, coinche) = if let Some(m) = s.strip_suffix("xx") {
        (m, 2u8)
    } else if let Some(m) = s.strip_suffix('x') {
        (m, 1u8)
    } else {
        (s, 0u8)
    };

    // Extract team suffix
    let (main, team) = if let Some(m) = main.strip_suffix("NS") {
        (m, 0u8)
    } else if let Some(m) = main.strip_suffix("EW") {
        (m, 1u8)
    } else {
        return Err(err());
    };

    // Extract suit (last char of main)
    if main.is_empty() {
        return Err(err());
    }
    let suit_char = main.chars().last().ok_or_else(err)?;
    let suit = char_to_suit(suit_char).map_err(|_| err())?;
    let value_str = &main[..main.len() - 1];
    let value: u16 = value_str.parse().map_err(|_| err())?;

    let value_enc = if value == 250 {
        25u8
    } else if value >= 80 && value <= 160 && value % 10 == 0 {
        (value / 10) as u8
    } else {
        return Err(err());
    };

    state.contract = Contract {
        trump: suit,
        value: value_enc,
        team,
        coinche,
    };

    // Resolved contract means we're at least in Playing phase
    // (may be upgraded to Done based on trick count later)
    state.phase = Phase::Playing;

    Ok(())
}

fn parse_bidding_contract(s: &str, state: &mut GameState) -> Result<(), CfnError> {
    let err = || CfnError::InvalidContract(format!("bid:{}", s));

    // Format: <value><suit><bidder>/p<passes>/c<coinche>/<current>
    // or:     0/p<passes>/<current>  (no bid yet)
    let parts: Vec<&str> = s.split('/').collect();

    if parts.len() < 3 {
        return Err(err());
    }

    state.phase = Phase::Bidding;

    let bid_part = parts[0];
    if bid_part == "0" {
        // No bid yet
        state.last_bid_value = 0;
        state.last_bid_suit = 0;
        state.last_bidder = 0;
    } else {
        // Parse bid: value + suit char + bidder char
        if bid_part.len() < 3 {
            return Err(err());
        }
        let bidder_char = bid_part.chars().last().ok_or_else(err)?;
        let suit_char = bid_part.chars().nth(bid_part.len() - 2).ok_or_else(err)?;
        let value_str = &bid_part[..bid_part.len() - 2];
        let value: u16 = value_str.parse().map_err(|_| err())?;

        state.last_bid_value = if value == 250 {
            25
        } else if value >= 80 && value <= 160 && value % 10 == 0 {
            (value / 10) as u8
        } else {
            return Err(err());
        };
        state.last_bid_suit = char_to_suit(suit_char).map_err(|_| err())?;
        state.last_bidder = char_to_player(bidder_char).map_err(|_| err())?;
    }

    // Parse passes
    let pass_part = parts[1];
    if !pass_part.starts_with('p') {
        return Err(err());
    }
    state.consecutive_passes = pass_part[1..].parse().map_err(|_| err())?;

    // Parse coinche (if present) and current player
    if parts.len() == 4 {
        // Has coinche field
        let coinche_part = parts[2];
        if !coinche_part.starts_with('c') {
            return Err(err());
        }
        state.coinche_state = coinche_part[1..].parse().map_err(|_| err())?;

        let current_part = parts[3];
        if current_part.len() != 1 {
            return Err(err());
        }
        state.current_player = char_to_player(current_part.chars().next().unwrap()).map_err(|_| err())?;
    } else {
        // No coinche field (legacy: 0/p<passes>/<current>)
        state.coinche_state = 0;

        let current_part = parts[2];
        if current_part.len() != 1 {
            return Err(err());
        }
        state.current_player = char_to_player(current_part.chars().next().unwrap()).map_err(|_| err())?;
    }

    Ok(())
}

// ---- Main to_cfn / from_cfn ----

impl GameState {
    /// Convert game state to CFN string.
    pub fn to_cfn(&self) -> String {
        let dealer_char = player_to_char(self.dealer);
        let hands: Vec<String> = self.hands.iter().map(|&h| format_hand(h)).collect();
        let hands_str = hands.join("/");
        let tricks_str = format_tricks(self);
        let contract_str = format_contract(self);
        format!("{}:{} {} {}", dealer_char, hands_str, tricks_str, contract_str)
    }

    /// Parse a CFN string into a GameState.
    pub fn from_cfn(s: &str) -> Result<GameState, CfnError> {
        let sections: Vec<&str> = s.split(' ').collect();
        if sections.len() != 3 {
            return Err(CfnError::InvalidFormat(format!(
                "expected 3 space-separated sections, got {}",
                sections.len()
            )));
        }

        let hands_section = sections[0];
        let tricks_section = sections[1];
        let contract_section = sections[2];

        // Parse dealer and current hands
        let (dealer, current_hands) = parse_dealer_hands(hands_section)?;

        // Parse trick cards (without replaying yet) to reconstruct original hands
        let trick_cards = if tricks_section != "-" {
            parse_trick_cards(tricks_section)?
        } else {
            Vec::new()
        };

        // Reconstruct original hands: current hands + all cards from tricks
        let mut original_hands = current_hands;
        // We need to know which seat played each card. Parse tricks with lead tracking.
        // First, parse contract to know trump (needed for trick winners).
        let mut temp_state = GameState::new(dealer, current_hands);
        parse_contract_section(contract_section, &mut temp_state)?;

        if !trick_cards.is_empty() && temp_state.phase == Phase::Bidding {
            return Err(CfnError::InvalidTrick(
                "tricks present but contract is bidding".into(),
            ));
        }

        // Assign trick cards to seats (lead + play order) and add back to hands
        if !trick_cards.is_empty() {
            let first_lead = (dealer + 1) % 4;
            let mut current_lead = first_lead;

            for chunk in trick_cards.chunks(4) {
                for (i, &card) in chunk.iter().enumerate() {
                    let seat = (current_lead + i as u8) % 4;
                    original_hands[seat as usize] |= card_to_bit(card);
                }
                if chunk.len() == 4 {
                    // Complete trick — compute winner for next lead
                    let mut trick = [EMPTY; 4];
                    for (i, &card) in chunk.iter().enumerate() {
                        let seat = (current_lead + i as u8) % 4;
                        trick[seat as usize] = card;
                    }
                    let winner = trick::trick_winner(&trick, current_lead, &temp_state.contract);
                    current_lead = winner;
                }
            }
        }

        // Create state with original hands, then replay tricks
        let mut state = GameState::new(dealer, original_hands);
        parse_contract_section(contract_section, &mut state)?;

        if !trick_cards.is_empty() {
            replay_tricks(&mut state, &trick_cards)?;
        } else if state.phase == Phase::Playing {
            // No tricks yet but contract is resolved — set up play phase
            state.trick_lead = (dealer + 1) % 4;
            state.current_player = state.trick_lead;
        }

        // Determine final phase based on trick count
        if state.phase != Phase::Bidding {
            let completed = (state.tricks_won[0] + state.tricks_won[1]) as usize;
            if completed == 8 {
                state.phase = Phase::Done;
            }
        }

        Ok(state)
    }
}

fn parse_dealer_hands(s: &str) -> Result<(u8, [CardSet; 4]), CfnError> {
    // Format: <dealer>:<hand0>/<hand1>/<hand2>/<hand3>
    let colon_pos = s
        .find(':')
        .ok_or_else(|| CfnError::InvalidFormat("missing ':' in hands section".into()))?;
    let dealer_str = &s[..colon_pos];
    let hands_str = &s[colon_pos + 1..];

    if dealer_str.len() != 1 {
        return Err(CfnError::InvalidDealer(dealer_str.into()));
    }
    let dealer = char_to_player(dealer_str.chars().next().unwrap())?;

    let hand_strs: Vec<&str> = hands_str.split('/').collect();
    if hand_strs.len() != 4 {
        return Err(CfnError::InvalidHand(format!(
            "expected 4 hands, got {}",
            hand_strs.len()
        )));
    }

    let mut hands = [0u32; 4];
    let mut all = 0u32;
    for (i, hs) in hand_strs.iter().enumerate() {
        hands[i] = parse_hand(hs)?;
        if all & hands[i] != 0 {
            return Err(CfnError::OverlappingCards);
        }
        all |= hands[i];
    }

    Ok((dealer, hands))
}

/// Parse trick cards from the trick section string into a flat Vec<Card>.
fn parse_trick_cards(tricks_str: &str) -> Result<Vec<Card>, CfnError> {
    let trick_groups: Vec<&str> = tricks_str.split('/').collect();
    let mut cards = Vec::new();

    for group in &trick_groups {
        if group.is_empty() {
            continue;
        }
        if group.len() % 2 != 0 {
            return Err(CfnError::InvalidTrick(format!(
                "odd-length trick group '{}'",
                group
            )));
        }
        let num_cards = group.len() / 2;
        if num_cards > 4 {
            return Err(CfnError::InvalidTrick(format!(
                "trick has {} cards (max 4)",
                num_cards
            )));
        }

        for i in 0..num_cards {
            let card_str = &group[i * 2..i * 2 + 2];
            let card = parse_card(card_str)?;
            cards.push(card);
        }
    }

    Ok(cards)
}

/// Replay trick cards to reconstruct derived state.
/// State must have original hands (current hands + trick cards added back).
fn replay_tricks(state: &mut GameState, trick_cards: &[Card]) -> Result<(), CfnError> {
    // Set up for playing
    state.phase = Phase::Playing;
    state.trick_lead = (state.dealer + 1) % 4;
    state.current_player = state.trick_lead;
    state.trick_count = 0;
    state.current_trick = [EMPTY; 4];
    state.played_cards = 0;
    state.points = [0; 2];
    state.tricks_won = [0; 2];
    state.voids = [0; 4];
    state.belote = [0; 2];
    state.belote_player = [0; 2];
    state.trick_history = [[EMPTY; 4]; 8];

    for &card in trick_cards {
        let player = state.current_player as usize;
        let bit = card_to_bit(card);
        if state.hands[player] & bit == 0 {
            return Err(CfnError::InvalidTrick(format!(
                "player {} doesn't have card {} ({})",
                player,
                format_card(card),
                cardset_str(state.hands[player])
            )));
        }

        // Apply the play (handles everything: remove from hand, void tracking,
        // belote detection, trick resolution, history saving)
        crate::play::apply_play(state, card);
    }

    Ok(())
}

impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_cfn())
    }
}

impl FromStr for GameState {
    type Err = CfnError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        GameState::from_cfn(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidding;
    use rand::Rng;

    #[test]
    fn test_format_card_roundtrip() {
        for card in 0..32u8 {
            let s = format_card(card);
            let parsed = parse_card(&s).unwrap();
            assert_eq!(parsed, card, "card {} -> '{}' -> {}", card, s, parsed);
        }
    }

    #[test]
    fn test_format_hand_roundtrip() {
        // All spades
        let hand: CardSet = 0xFF;
        let s = format_hand(hand);
        assert_eq!(s, "ATKQJ987...");
        assert_eq!(parse_hand(&s).unwrap(), hand);

        // All hearts
        let hand: CardSet = 0xFF00;
        let s = format_hand(hand);
        assert_eq!(s, ".ATKQJ987..");
        assert_eq!(parse_hand(&s).unwrap(), hand);

        // Empty hand
        assert_eq!(format_hand(0), "-");
        assert_eq!(parse_hand("-").unwrap(), 0);

        // Mixed hand
        let hand = card_to_bit(make_card(Suit::Spades, 7))  // AS
                 | card_to_bit(make_card(Suit::Spades, 3))   // JS
                 | card_to_bit(make_card(Suit::Hearts, 5))   // KH
                 | card_to_bit(make_card(Suit::Diamonds, 6)) // TD
                 | card_to_bit(make_card(Suit::Clubs, 0));   // 7C
        let s = format_hand(hand);
        let parsed = parse_hand(&s).unwrap();
        assert_eq!(parsed, hand);
    }

    #[test]
    fn test_cfn_start_of_game() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let cfn = state.to_cfn();
        assert!(cfn.starts_with("N:ATKQJ987.../"));
        assert!(cfn.contains(" - "));
        assert!(cfn.ends_with("/E"));

        // Round-trip
        let parsed = GameState::from_cfn(&cfn).unwrap();
        assert_eq!(parsed.dealer, state.dealer);
        assert_eq!(parsed.hands, state.hands);
        assert_eq!(parsed.phase, Phase::Bidding);
    }

    #[test]
    fn test_cfn_void_deal() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let mut state = GameState::new(0, hands);
        // 4 passes
        for _ in 0..4 {
            state.step(0);
        }
        assert_eq!(state.phase, Phase::Done);
        let cfn = state.to_cfn();
        assert!(cfn.ends_with(" 0"), "CFN should end with '0' for void deal, got: {}", cfn);

        let parsed = GameState::from_cfn(&cfn).unwrap();
        assert_eq!(parsed.phase, Phase::Done);
        assert_eq!(parsed.contract.value, 0);
    }

    #[test]
    fn test_cfn_mid_bidding() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let mut state = GameState::new(0, hands);
        // P1 bids 80 Spades
        state.step(bidding::encode_bid(8, 0));
        // P2 passes
        state.step(0);

        let cfn = state.to_cfn();
        assert!(cfn.contains("bid:80sE/p1/c0/W"), "got: {}", cfn);

        let parsed = GameState::from_cfn(&cfn).unwrap();
        assert_eq!(parsed.phase, Phase::Bidding);
        assert_eq!(parsed.last_bid_value, 8);
        assert_eq!(parsed.last_bid_suit, 0); // Spades
        assert_eq!(parsed.last_bidder, 1); // East
        assert_eq!(parsed.consecutive_passes, 1);
        assert_eq!(parsed.current_player, 3); // West
    }

    #[test]
    fn test_cfn_playing_start() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let mut state = GameState::new(0, hands);
        // P1 bids 80 Hearts, 3 passes
        state.step(bidding::encode_bid(8, 1));
        state.step(0);
        state.step(0);
        state.step(0);

        assert_eq!(state.phase, Phase::Playing);
        let cfn = state.to_cfn();
        assert!(cfn.contains(" - "), "no tricks yet: {}", cfn);
        assert!(cfn.ends_with("80hEW"), "contract: {}", cfn);

        let parsed = GameState::from_cfn(&cfn).unwrap();
        assert_eq!(parsed.phase, Phase::Playing);
        assert_eq!(parsed.contract.trump, 1); // Hearts
        assert_eq!(parsed.contract.value, 8);
        assert_eq!(parsed.contract.team, 1); // EW
        assert_eq!(parsed.trick_lead, 1); // P1 leads
        assert_eq!(parsed.current_player, 1);
    }

    #[test]
    fn test_cfn_mid_play() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let mut state = GameState::new(0, hands);
        // Bid: P1 bids 80 Hearts, 3 passes
        state.step(bidding::encode_bid(8, 1));
        state.step(0);
        state.step(0);
        state.step(0);

        // Play trick 1: P1 leads AH
        state.step(make_card(Suit::Hearts, 7)); // P1: AH
        state.step(make_card(Suit::Diamonds, 0)); // P2: 7D (discard)
        state.step(make_card(Suit::Clubs, 0)); // P3: 7C (discard)
        state.step(make_card(Suit::Spades, 0)); // P0: 7S (discard)
        // AH wins, P1 leads again

        // Play 1 card of trick 2
        state.step(make_card(Suit::Hearts, 6)); // P1: 10H

        let cfn = state.to_cfn();
        // Should have 1 completed trick + 1 partial
        assert!(cfn.contains('/'), "should have trick separator: {}", cfn);

        let parsed = GameState::from_cfn(&cfn).unwrap();
        assert_eq!(parsed.phase, Phase::Playing);
        assert_eq!(parsed.tricks_won[1], 1); // EW won 1
        assert_eq!(parsed.trick_count, 1); // 1 card in current trick
        assert_eq!(parsed.current_player, 2); // P2 next
        assert_eq!(parsed.points, state.points);
    }

    #[test]
    fn test_cfn_complete_game_roundtrip() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let mut state = GameState::deal_random(0, &mut rng);

            // Play random game
            while !state.is_terminal() {
                let legal = state.legal_actions();
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                let mut remaining = legal;
                for _ in 0..idx {
                    remaining &= remaining - 1;
                }
                let action = remaining.trailing_zeros() as u8;
                state.step(action);
            }

            // Check that void deals round-trip
            if state.contract.value == 0 {
                let cfn = state.to_cfn();
                let parsed = GameState::from_cfn(&cfn).unwrap();
                assert_eq!(parsed.phase, Phase::Done);
                assert_eq!(parsed.contract.value, 0);
                continue;
            }

            let cfn = state.to_cfn();
            let parsed = GameState::from_cfn(&cfn).unwrap();

            assert_eq!(parsed.phase, state.phase, "phase mismatch for CFN: {}", cfn);
            assert_eq!(parsed.tricks_won, state.tricks_won, "tricks_won mismatch for CFN: {}", cfn);
            assert_eq!(parsed.points, state.points, "points mismatch for CFN: {}", cfn);
            assert_eq!(parsed.contract, state.contract, "contract mismatch for CFN: {}", cfn);
            assert_eq!(parsed.hands, state.hands, "hands mismatch for CFN: {}", cfn);
            assert_eq!(parsed.belote, state.belote, "belote mismatch for CFN: {}", cfn);
            assert_eq!(parsed.voids, state.voids, "voids mismatch for CFN: {}", cfn);
        }
    }

    #[test]
    fn test_cfn_mid_game_roundtrip() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);

        for _ in 0..100 {
            let mut state = GameState::deal_random(0, &mut rng);

            // Play random number of steps
            let steps = rng.gen_range(0..40);
            for _ in 0..steps {
                if state.is_terminal() {
                    break;
                }
                let legal = state.legal_actions();
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                let mut remaining = legal;
                for _ in 0..idx {
                    remaining &= remaining - 1;
                }
                let action = remaining.trailing_zeros() as u8;
                state.step(action);
            }

            if state.phase == Phase::Bidding {
                // Bidding CFN roundtrip
                let cfn = state.to_cfn();
                let parsed = GameState::from_cfn(&cfn).unwrap();
                assert_eq!(parsed.phase, Phase::Bidding);
                assert_eq!(parsed.hands, state.hands);
                assert_eq!(parsed.dealer, state.dealer);
                assert_eq!(parsed.last_bid_value, state.last_bid_value);
                assert_eq!(parsed.last_bid_suit, state.last_bid_suit);
                assert_eq!(parsed.last_bidder, state.last_bidder);
                assert_eq!(parsed.consecutive_passes, state.consecutive_passes);
                assert_eq!(parsed.coinche_state, state.coinche_state);
                assert_eq!(parsed.current_player, state.current_player);
                continue;
            }

            if state.is_terminal() && state.contract.value == 0 {
                continue; // void deal already tested
            }

            let cfn = state.to_cfn();
            let parsed = GameState::from_cfn(&cfn).unwrap();

            assert_eq!(parsed.phase, state.phase, "phase mismatch for CFN: {}", cfn);
            assert_eq!(parsed.hands, state.hands, "hands mismatch for CFN: {}", cfn);
            assert_eq!(parsed.contract, state.contract, "contract mismatch for CFN: {}", cfn);
            assert_eq!(parsed.tricks_won, state.tricks_won, "tricks_won mismatch");
            assert_eq!(parsed.points, state.points, "points mismatch");
            assert_eq!(parsed.current_player, state.current_player, "current_player mismatch");
            assert_eq!(parsed.trick_lead, state.trick_lead, "trick_lead mismatch");
            assert_eq!(parsed.trick_count, state.trick_count, "trick_count mismatch");
            assert_eq!(parsed.voids, state.voids, "voids mismatch");
            assert_eq!(parsed.belote, state.belote, "belote mismatch");
            assert_eq!(parsed.played_cards, state.played_cards, "played_cards mismatch");
        }
    }

    #[test]
    fn test_cfn_coinche_contract() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let mut state = GameState::new(0, hands);
        // P1 bids 80S, P2 coinches, P3 surcoinches
        state.step(bidding::encode_bid(8, 0)); // P1: 80S
        state.step(41); // P2: coinche
        state.step(42); // P3: surcoinche → ends bidding

        assert_eq!(state.phase, Phase::Playing);
        let cfn = state.to_cfn();
        assert!(cfn.ends_with("80sEWxx"), "expected surcoinche, got: {}", cfn);

        let parsed = GameState::from_cfn(&cfn).unwrap();
        assert_eq!(parsed.contract.coinche, 2);
        assert_eq!(parsed.contract.trump, 0);
        assert_eq!(parsed.contract.team, 1);
    }

    #[test]
    fn test_cfn_capot_contract() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let mut state = GameState::new(0, hands);
        // P1 bids capot Hearts (action 38 = 37 + suit 1)
        state.step(bidding::encode_bid(25, 1)); // P1: capot Hearts
        state.step(0); // P2 pass
        state.step(0); // P3 pass
        state.step(0); // P0 pass

        let cfn = state.to_cfn();
        assert!(cfn.contains("250hEW"), "capot contract: {}", cfn);

        let parsed = GameState::from_cfn(&cfn).unwrap();
        assert!(parsed.contract.is_capot());
        assert_eq!(parsed.contract.trump, 1);
    }

    #[test]
    fn test_cfn_display_fromstr() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let display_str = format!("{}", state);
        let parsed: GameState = display_str.parse().unwrap();
        assert_eq!(parsed.hands, state.hands);
        assert_eq!(parsed.dealer, state.dealer);
    }

    #[test]
    fn test_cfn_error_cases() {
        // Wrong section count
        assert!(GameState::from_cfn("a b").is_err());
        assert!(GameState::from_cfn("a b c d").is_err());

        // Invalid dealer
        assert!(GameState::from_cfn("X:.../.../.../.../... - 0").is_err());

        // Wrong number of hands
        assert!(GameState::from_cfn("N:AJ9/T8 - bid:0/p0/E").is_err());

        // Invalid card in trick
        assert!(GameState::from_cfn("N:ATKQJ987.../.ATKQJ987../..ATKQJ987./...ATKQJ987 Zz 80sNS").is_err());

        // Invalid contract
        assert!(GameState::from_cfn("N:ATKQJ987.../.ATKQJ987../..ATKQJ987./...ATKQJ987 - 99sNS").is_err());
    }

    #[test]
    fn test_cfn_known_position() {
        // Simple start: North deals, all spades/hearts/diamonds/clubs
        let cfn = "N:ATKQJ987.../.ATKQJ987../..ATKQJ987./...ATKQJ987 - bid:0/p0/E";
        let state = GameState::from_cfn(cfn).unwrap();
        assert_eq!(state.dealer, 0); // North
        assert_eq!(state.hands[0], 0xFF); // all spades
        assert_eq!(state.hands[1], 0xFF00); // all hearts
        assert_eq!(state.hands[2], 0xFF_0000); // all diamonds
        assert_eq!(state.hands[3], 0xFF00_0000); // all clubs
        assert_eq!(state.phase, Phase::Bidding);
        assert_eq!(state.current_player, 1); // East bids first
    }

    #[test]
    fn test_cfn_trick_history_stored() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let mut state = GameState::new(0, hands);
        // Bid
        state.step(bidding::encode_bid(8, 1)); // P1: 80H
        state.step(0); state.step(0); state.step(0); // 3 passes

        // Play trick 1
        let p1_card = make_card(Suit::Hearts, 7); // AH
        let p2_card = make_card(Suit::Diamonds, 0); // 7D
        let p3_card = make_card(Suit::Clubs, 0); // 7C
        let p0_card = make_card(Suit::Spades, 0); // 7S
        state.step(p1_card);
        state.step(p2_card);
        state.step(p3_card);
        state.step(p0_card);

        // Verify trick_history[0] is stored
        assert_eq!(state.trick_history[0][1], p1_card);
        assert_eq!(state.trick_history[0][2], p2_card);
        assert_eq!(state.trick_history[0][3], p3_card);
        assert_eq!(state.trick_history[0][0], p0_card);
    }

    #[test]
    fn test_cfn_overlapping_cards() {
        // Two hands share the same card
        let result = GameState::from_cfn("N:A.../A.../.../... - bid:0/p0/E");
        assert!(result.is_err());
        match result {
            Err(CfnError::OverlappingCards) => {},
            other => panic!("expected OverlappingCards, got {:?}", other),
        }
    }

    #[test]
    fn test_cfn_coinche_bidding_state() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let mut state = GameState::new(0, hands);
        // P1 bids 80S, P2 coinches
        state.step(bidding::encode_bid(8, 0)); // P1: 80S
        state.step(41); // P2: coinche
        // Now P3's turn (can surcoinche or pass)

        let cfn = state.to_cfn();
        assert!(cfn.contains("bid:80sE/p0/c1/W"), "got: {}", cfn);

        let parsed = GameState::from_cfn(&cfn).unwrap();
        assert_eq!(parsed.coinche_state, 1);
        assert_eq!(parsed.current_player, 3);
    }

    #[test]
    fn test_parse_card_specific() {
        assert_eq!(parse_card("As").unwrap(), make_card(Suit::Spades, 7));
        assert_eq!(parse_card("7h").unwrap(), make_card(Suit::Hearts, 0));
        assert_eq!(parse_card("Jd").unwrap(), make_card(Suit::Diamonds, 3));
        assert_eq!(parse_card("Tc").unwrap(), make_card(Suit::Clubs, 6));
        assert_eq!(parse_card("9s").unwrap(), make_card(Suit::Spades, 2));
    }

    #[test]
    fn test_format_card_specific() {
        assert_eq!(format_card(make_card(Suit::Spades, 7)), "As");
        assert_eq!(format_card(make_card(Suit::Hearts, 0)), "7h");
        assert_eq!(format_card(make_card(Suit::Diamonds, 3)), "Jd");
        assert_eq!(format_card(make_card(Suit::Clubs, 6)), "Tc");
    }
}
