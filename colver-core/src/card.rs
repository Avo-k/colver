/// Card representation using bitmasks for maximum performance.
///
/// A card is an index 0..31: (suit * 8 + rank)
/// A set of cards is a u32 bitmask where bit i means card i is present.
///
/// Bit layout of CardSet (u32):
///   Bits [31..24]: Clubs     — bit 24=7C, 25=8C, ..., 31=AC
///   Bits [23..16]: Diamonds  — bit 16=7D, 17=8D, ..., 23=AD
///   Bits [15.. 8]: Hearts    — bit  8=7H,  9=8H, ..., 15=AH
///   Bits [ 7.. 0]: Spades    — bit  0=7S,  1=8S, ...,  7=AS
///
/// Within each suit, bits go from weakest (7=bit0) to strongest (A=bit7) in plain order.

pub type Card = u8;
pub type CardSet = u32;

pub const EMPTY: Card = 0xFF;
pub const ALL_CARDS: CardSet = 0xFFFF_FFFF;
pub const NUM_CARDS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Suit {
    Spades = 0,
    Hearts = 1,
    Diamonds = 2,
    Clubs = 3,
}

impl Suit {
    #[inline(always)]
    pub fn from_u8(v: u8) -> Self {
        debug_assert!(v < 4);
        unsafe { core::mem::transmute(v) }
    }
}

pub const ALL_SUITS: [Suit; 4] = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Rank {
    Seven = 0,
    Eight = 1,
    Nine = 2,
    Jack = 3,
    Queen = 4,
    King = 5,
    Ten = 6,
    Ace = 7,
}

/// Contract type for scoring and rule dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContractType {
    Color(Suit),
}

// Suit masks: extract 8 bits for a given suit from a CardSet.
pub const SUIT_MASK: [CardSet; 4] = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
pub const SUIT_SHIFT: [u8; 4] = [0, 8, 16, 24];

// Points per rank indexed by rank (0=Seven .. 7=Ace)
// Trump:  7=0, 8=0, 9=14, J=20, Q=3, K=4, 10=10, A=11
pub const TRUMP_POINTS: [u8; 8] = [0, 0, 14, 20, 3, 4, 10, 11];
// Plain:  7=0, 8=0, 9=0,  J=2,  Q=3, K=4, 10=10, A=11
pub const PLAIN_POINTS: [u8; 8] = [0, 0, 0, 2, 3, 4, 10, 11];

/// Trump strength: maps rank index to trump strength (0=weakest, 7=strongest).
/// Trump order: 7(0) < 8(1) < Q(4) < K(5) < 10(6) < A(7) < 9(2) < J(3)
/// So: rank→strength: 7→0, 8→1, 9→6, J→7, Q→2, K→3, 10→4, A→5
pub const TRUMP_STRENGTH: [u8; 8] = [0, 1, 6, 7, 2, 3, 4, 5];

/// For each rank (as trump), bitmask of ranks that are strictly stronger.
/// The bitmask is over the 8 rank bits within a suit.
/// rank 0 (7, str=0): beaten by 8,Q,K,10,A,9,J = bits 1,2,3,4,5,6,7 = 0xFE
/// rank 1 (8, str=1): beaten by Q,K,10,A,9,J   = bits 2,3,4,5,6,7   = 0xFC
/// rank 4 (Q, str=2): beaten by K,10,A,9,J      = bits 2,3,5,6,7     = 0xEC
/// rank 5 (K, str=3): beaten by 10,A,9,J         = bits 2,3,6,7       = 0xCC
/// rank 6 (10,str=4): beaten by A,9,J            = bits 2,3,7         = 0x8C
/// rank 7 (A, str=5): beaten by 9,J              = bits 2,3           = 0x0C
/// rank 2 (9, str=6): beaten by J                = bit 3              = 0x08
/// rank 3 (J, str=7): beaten by nothing          =                      0x00
pub const HIGHER_TRUMP_MASK: [u8; 8] = [
    0xFE, // 7: all others beat it
    0xFC, // 8: all except 7
    0x08, // 9: only J beats it
    0x00, // J: nothing beats it
    0xEC, // Q: K,10,A,9,J
    0xCC, // K: 10,A,9,J
    0x8C, // 10: A,9,J
    0x0C, // A: 9,J
];

// ---- Inline helpers ----

#[inline(always)]
pub fn card_to_bit(card: Card) -> CardSet {
    1u32 << card
}

#[inline(always)]
pub fn card_suit(card: Card) -> Suit {
    Suit::from_u8(card >> 3)
}

#[inline(always)]
pub fn card_suit_u8(card: Card) -> u8 {
    card >> 3
}

#[inline(always)]
pub fn card_rank(card: Card) -> u8 {
    card & 7
}

#[inline(always)]
pub fn make_card(suit: Suit, rank: u8) -> Card {
    (suit as u8) * 8 + rank
}

#[inline(always)]
pub fn cards_in_suit(set: CardSet, suit: Suit) -> CardSet {
    set & SUIT_MASK[suit as usize]
}

#[inline(always)]
pub fn cards_in_suit_idx(set: CardSet, suit_idx: u8) -> CardSet {
    set & SUIT_MASK[suit_idx as usize]
}

#[inline(always)]
pub fn has_suit(set: CardSet, suit: Suit) -> bool {
    set & SUIT_MASK[suit as usize] != 0
}

#[inline(always)]
pub fn card_count(set: CardSet) -> u32 {
    set.count_ones()
}

/// Extract the suit-local 8-bit mask from a CardSet for a given suit.
#[inline(always)]
pub fn suit_bits(set: CardSet, suit: Suit) -> u8 {
    ((set >> SUIT_SHIFT[suit as usize]) & 0xFF) as u8
}

/// Points for a card given contract type.
#[inline]
pub fn card_points(card: Card, contract_type: ContractType) -> u8 {
    let rank = card_rank(card) as usize;
    let ContractType::Color(trump) = contract_type;
    if card_suit(card) == trump {
        TRUMP_POINTS[rank]
    } else {
        PLAIN_POINTS[rank]
    }
}

/// Sum points for all cards in a set.
pub fn set_points(set: CardSet, contract_type: ContractType) -> u8 {
    let mut total: u8 = 0;
    let mut remaining = set;
    while remaining != 0 {
        let card = remaining.trailing_zeros() as Card;
        total += card_points(card, contract_type);
        remaining &= remaining - 1; // clear lowest bit
    }
    total
}

/// Get the highest card in a suit from a CardSet, using plain strength
/// (highest bit position = strongest in plain). Returns EMPTY if none.
#[inline(always)]
pub fn highest_plain_in_suit(set: CardSet, suit: Suit) -> Card {
    let masked = cards_in_suit(set, suit);
    if masked == 0 {
        return EMPTY;
    }
    (31 - masked.leading_zeros()) as Card
}

/// Get the highest trump card in a suit from a CardSet using trump strength.
/// Returns EMPTY if none.
#[inline]
pub fn highest_trump_in_set(set: CardSet, suit: Suit) -> Card {
    let bits = suit_bits(set, suit);
    if bits == 0 {
        return EMPTY;
    }
    // Find rank with highest trump strength
    let mut best_rank = 0u8;
    let mut best_strength = 0u8;
    let mut b = bits;
    while b != 0 {
        let rank = b.trailing_zeros() as u8;
        let str_ = TRUMP_STRENGTH[rank as usize];
        if str_ >= best_strength {
            best_strength = str_;
            best_rank = rank;
        }
        b &= b - 1;
    }
    make_card(suit, best_rank)
}

/// Given a set of trump cards in a specific suit and the rank of the current
/// highest trump on the table, return only the cards that can overtrump.
#[inline]
pub fn overtrump_candidates(hand_suit_bits: u8, best_rank_on_table: u8) -> u8 {
    // HIGHER_TRUMP_MASK[best_rank_on_table] gives ranks that beat it
    // AND with hand_suit_bits to get playable overtrumps
    hand_suit_bits & HIGHER_TRUMP_MASK[best_rank_on_table as usize]
}

/// Select a random card from a CardSet (uniform among set bits).
/// `rand_val` should be uniform in [0, count).
#[inline]
pub fn select_nth_card(set: CardSet, mut n: u32) -> Card {
    debug_assert!(set != 0);
    let mut remaining = set;
    loop {
        let card = remaining.trailing_zeros() as Card;
        if n == 0 {
            return card;
        }
        n -= 1;
        remaining &= remaining - 1;
    }
}

/// Iterate over cards in a CardSet.
pub struct CardIter(pub CardSet);

impl Iterator for CardIter {
    type Item = Card;
    #[inline]
    fn next(&mut self) -> Option<Card> {
        if self.0 == 0 {
            None
        } else {
            let card = self.0.trailing_zeros() as Card;
            self.0 &= self.0 - 1;
            Some(card)
        }
    }
}

/// Pretty-print a card.
pub fn card_name(card: Card) -> String {
    let rank_names = ["7", "8", "9", "J", "Q", "K", "10", "A"];
    let suit_names = ["S", "H", "D", "C"];
    format!(
        "{}{}",
        rank_names[card_rank(card) as usize],
        suit_names[card_suit_u8(card) as usize]
    )
}

/// Pretty-print a card set.
pub fn cardset_str(set: CardSet) -> String {
    let mut cards: Vec<String> = Vec::new();
    for card in CardIter(set) {
        cards.push(card_name(card));
    }
    cards.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_basics() {
        // 7 of Spades = 0, Ace of Spades = 7
        let seven_s = make_card(Suit::Spades, 0);
        assert_eq!(seven_s, 0);
        assert_eq!(card_suit(seven_s), Suit::Spades);
        assert_eq!(card_rank(seven_s), 0);

        let ace_s = make_card(Suit::Spades, 7);
        assert_eq!(ace_s, 7);
        assert_eq!(card_suit(ace_s), Suit::Spades);
        assert_eq!(card_rank(ace_s), 7);

        // Jack of Hearts = 1*8 + 3 = 11
        let jack_h = make_card(Suit::Hearts, 3);
        assert_eq!(jack_h, 11);
        assert_eq!(card_suit(jack_h), Suit::Hearts);
        assert_eq!(card_rank(jack_h), 3);
    }

    #[test]
    fn test_card_to_bit() {
        assert_eq!(card_to_bit(0), 1);
        assert_eq!(card_to_bit(7), 128);
        assert_eq!(card_to_bit(31), 1 << 31);
    }

    #[test]
    fn test_suit_masks() {
        assert_eq!(SUIT_MASK[0], 0xFF);
        assert_eq!(SUIT_MASK[1], 0xFF00);
        assert_eq!(SUIT_MASK[2], 0xFF_0000);
        assert_eq!(SUIT_MASK[3], 0xFF00_0000);

        // Union of all suit masks = all cards
        assert_eq!(
            SUIT_MASK[0] | SUIT_MASK[1] | SUIT_MASK[2] | SUIT_MASK[3],
            ALL_CARDS
        );
    }

    #[test]
    fn test_cards_in_suit() {
        let hand: CardSet = 0b1010_0000_0000_0000_0101_0000_1100_0011;
        // Spades: bits 0-7 = 0b11000011 = 7S, 8S, KS, AS
        let spades = cards_in_suit(hand, Suit::Spades);
        assert_eq!(card_count(spades), 4);

        // Hearts: bits 8-15 = 0b01010000 = QH, 10H
        let hearts = cards_in_suit(hand, Suit::Hearts);
        assert_eq!(card_count(hearts), 2);
    }

    #[test]
    fn test_color_points_total() {
        // In Color with one trump: trump suit = 0+0+14+20+3+4+10+11 = 62, other 3 = 30*3 = 90, total = 152
        let trump_suit = Suit::Hearts;
        let mut total_color = 0u16;
        for card in 0..32u8 {
            if card_suit(card) == trump_suit {
                total_color += TRUMP_POINTS[card_rank(card) as usize] as u16;
            } else {
                total_color += PLAIN_POINTS[card_rank(card) as usize] as u16;
            }
        }
        assert_eq!(total_color, 152);
    }

    #[test]
    fn test_trump_strength_ordering() {
        // J should be strongest (7), 9 next (6), then A(5), 10(4), K(3), Q(2), 8(1), 7(0)
        assert_eq!(TRUMP_STRENGTH[3], 7); // J
        assert_eq!(TRUMP_STRENGTH[2], 6); // 9
        assert_eq!(TRUMP_STRENGTH[7], 5); // A
        assert_eq!(TRUMP_STRENGTH[6], 4); // 10
        assert_eq!(TRUMP_STRENGTH[5], 3); // K
        assert_eq!(TRUMP_STRENGTH[4], 2); // Q
        assert_eq!(TRUMP_STRENGTH[1], 1); // 8
        assert_eq!(TRUMP_STRENGTH[0], 0); // 7
    }

    #[test]
    fn test_higher_trump_mask() {
        // J (rank 3) beats everything, nothing beats it
        assert_eq!(HIGHER_TRUMP_MASK[3], 0x00);

        // 9 (rank 2) is beaten only by J (rank 3)
        assert_eq!(HIGHER_TRUMP_MASK[2], 0x08); // bit 3

        // 7 (rank 0) is beaten by everything else
        assert_eq!(HIGHER_TRUMP_MASK[0], 0xFE); // bits 1-7

        // A (rank 7) is beaten by 9 and J (ranks 2,3)
        assert_eq!(HIGHER_TRUMP_MASK[7], 0x0C); // bits 2,3
    }

    #[test]
    fn test_overtrump_candidates() {
        // Hand has 9 and J of trump (bits 2 and 3)
        let hand_bits = 0b0000_1100; // 9 + J
        // Table has A (rank 7): beaten by 9,J
        assert_eq!(overtrump_candidates(hand_bits, 7), 0b0000_1100);
        // Table has 9 (rank 2): beaten by J only
        assert_eq!(overtrump_candidates(hand_bits, 2), 0b0000_1000); // J only
        // Table has J (rank 3): nothing beats it
        assert_eq!(overtrump_candidates(hand_bits, 3), 0b0000_0000);
    }

    #[test]
    fn test_highest_trump() {
        // Hand with 7H and JH (bits 8 and 11)
        let set: CardSet = (1 << 8) | (1 << 11);
        let best = highest_trump_in_set(set, Suit::Hearts);
        // J (rank 3) has higher trump strength than 7 (rank 0)
        assert_eq!(card_rank(best), 3); // Jack
    }

    #[test]
    fn test_select_nth_card() {
        let set: CardSet = 0b1010; // cards 1 and 3
        assert_eq!(select_nth_card(set, 0), 1);
        assert_eq!(select_nth_card(set, 1), 3);
    }

    #[test]
    fn test_card_iter() {
        let set: CardSet = 0b1011; // cards 0, 1, 3
        let cards: Vec<Card> = CardIter(set).collect();
        assert_eq!(cards, vec![0, 1, 3]);
    }

    #[test]
    fn test_all_cards_count() {
        assert_eq!(card_count(ALL_CARDS), 32);
    }
}
