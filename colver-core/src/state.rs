use crate::card::*;

/// Phase of the game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Phase {
    Bidding = 0,
    Playing = 1,
    Done = 2,
}

/// Contract information.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Contract {
    /// 0-3 = suit (Spades/Hearts/Diamonds/Clubs)
    pub trump: u8,
    /// Encoded value: 0-8 → 80..160 (step 10), 25 = capot (250 points)
    /// Stored as the actual bid index for simplicity: 80,90,...,160,250
    /// We store raw value / 10 so it fits in u8: 8,9,...,16, 25
    pub value: u8,
    /// 0 = North-South, 1 = East-West
    pub team: u8,
    /// 0 = normal, 1 = contré, 2 = surcontré
    pub coinche: u8,
}

impl Default for Contract {
    fn default() -> Self {
        Contract {
            trump: 0,
            value: 0,
            team: 0,
            coinche: 0,
        }
    }
}

impl Contract {
    #[inline]
    pub fn contract_type(&self) -> ContractType {
        debug_assert!(self.trump < 4);
        ContractType::Color(Suit::from_u8(self.trump))
    }

    /// The actual point value of the contract (80-160 or 250 for capot).
    #[inline]
    pub fn point_value(&self) -> u16 {
        (self.value as u16) * 10
    }

    #[inline]
    pub fn is_capot(&self) -> bool {
        self.value == 25
    }

    #[inline]
    pub fn trump_suit(&self) -> Suit {
        debug_assert!(self.trump < 4);
        Suit::from_u8(self.trump)
    }
}

/// Full game state. Designed to be Copy and small for fast MCTS cloning.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct GameState {
    // ---- Hands: 16 bytes ----
    /// Each player's remaining cards as bitmask. Players: 0=N, 1=E, 2=S, 3=W.
    pub hands: [CardSet; 4],

    // ---- Trick state: 6 bytes ----
    /// Cards played in the current trick (card index, EMPTY=0xFF if not played yet).
    pub current_trick: [Card; 4],
    /// Seat that led the current trick.
    pub trick_lead: u8,
    /// Number of cards played in the current trick (0-4).
    pub trick_count: u8,

    // ---- Contract: 4 bytes ----
    pub contract: Contract,

    // ---- Accumulation: 4 bytes ----
    /// Trick points per team so far. Max 162, fits u8.
    pub points: [u8; 2],
    /// Number of tricks won per team (0-8).
    pub tricks_won: [u8; 2],

    // ---- Bidding state: 5 bytes ----
    /// Highest bid value so far (0 if no bid). Encoded as value/10.
    pub last_bid_value: u8,
    /// Trump of highest bid (0-3).
    pub last_bid_suit: u8,
    /// Seat that made the highest bid.
    pub last_bidder: u8,
    /// Consecutive passes since last bid or coinche/surcoinche.
    pub consecutive_passes: u8,
    /// 0=none, 1=coinched, 2=surcoinched
    pub coinche_state: u8,

    // ---- Tracking: 8 bytes (with padding) ----
    pub current_player: u8,
    pub phase: Phase,
    pub dealer: u8,
    _pad: u8,
    /// Bitmask of all cards played this deal.
    pub played_cards: CardSet,

    // ---- Void tracking: 4 bytes ----
    /// Per player: bit i = known void in suit i.
    pub voids: [u8; 4],

    // ---- Belote tracking: 4 bytes ----
    /// Per team: 0=none, 1=belote declared, 2=rebelote declared (complete).
    pub belote: [u8; 2],
    /// Per team: which player (0-3) declared belote (valid when belote[team] >= 1).
    pub belote_player: [u8; 2],

    // ---- Trick history: 32 bytes ----
    /// Completed tricks stored by seat index (like current_trick).
    /// trick_history[i] contains the cards for the (i+1)-th completed trick.
    pub trick_history: [[Card; 4]; 8],
}

// Ensure we're small enough for fast copies.
const _: () = assert!(core::mem::size_of::<GameState>() <= 96);

impl GameState {
    /// Create a new game state with dealt hands.
    pub fn new(dealer: u8, hands: [CardSet; 4]) -> Self {
        // First bidder is to the right of the dealer (dealer+1 mod 4).
        let first_bidder = (dealer + 1) % 4;
        GameState {
            hands,
            current_trick: [EMPTY; 4],
            trick_lead: 0,
            trick_count: 0,
            contract: Contract::default(),
            points: [0; 2],
            tricks_won: [0; 2],
            last_bid_value: 0,
            last_bid_suit: 0,
            last_bidder: 0,
            consecutive_passes: 0,
            coinche_state: 0,
            current_player: first_bidder,
            phase: Phase::Bidding,
            dealer,
            _pad: 0,
            played_cards: 0,
            voids: [0; 4],
            belote: [0; 2],
            belote_player: [0; 2],
            trick_history: [[EMPTY; 4]; 8],
        }
    }

    /// Deal random hands.
    #[cfg(feature = "rand")]
    pub fn deal_random(dealer: u8, rng: &mut impl rand::Rng) -> Self {
        use rand::seq::SliceRandom;
        let mut cards: [u8; 32] = core::array::from_fn(|i| i as u8);
        cards.shuffle(rng);
        let mut hands = [0u32; 4];
        for (i, &card) in cards.iter().enumerate() {
            hands[i / 8] |= card_to_bit(card);
        }
        Self::new(dealer, hands)
    }

    #[inline(always)]
    pub fn is_terminal(&self) -> bool {
        self.phase == Phase::Done
    }

    #[inline(always)]
    pub fn current_player(&self) -> u8 {
        self.current_player
    }

    /// Team index for a player: 0=NS (players 0,2), 1=EW (players 1,3).
    #[inline(always)]
    pub fn player_team(player: u8) -> u8 {
        player & 1
    }

    /// Partner seat for a player.
    #[inline(always)]
    pub fn partner(player: u8) -> u8 {
        player ^ 2
    }
}

impl core::fmt::Debug for GameState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "GameState {{ phase={:?}, player={}, dealer={}, ",
            self.phase, self.current_player, self.dealer
        )?;
        if self.phase != Phase::Bidding {
            write!(
                f,
                "contract={{trump={}, val={}, team={}, coinche={}}}, ",
                self.contract.trump,
                self.contract.point_value(),
                self.contract.team,
                self.contract.coinche
            )?;
        }
        for i in 0..4 {
            write!(f, "P{}: [{}] ", i, cardset_str(self.hands[i]))?;
        }
        write!(
            f,
            "pts=[{},{}] tricks=[{},{}]",
            self.points[0], self.points[1], self.tricks_won[0], self.tricks_won[1]
        )?;
        write!(f, " }}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_size() {
        assert!(
            core::mem::size_of::<GameState>() <= 96,
            "GameState is {} bytes, must be <= 96",
            core::mem::size_of::<GameState>()
        );
    }

    #[test]
    fn test_state_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<GameState>();
    }

    #[test]
    fn test_new_state() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        assert_eq!(state.phase, Phase::Bidding);
        assert_eq!(state.current_player, 1); // dealer+1
        assert_eq!(state.dealer, 0);
        assert_eq!(card_count(state.hands[0]), 8);
        assert_eq!(card_count(state.hands[1]), 8);
        assert_eq!(card_count(state.hands[2]), 8);
        assert_eq!(card_count(state.hands[3]), 8);
    }

    #[test]
    fn test_player_team() {
        assert_eq!(GameState::player_team(0), 0); // North = NS
        assert_eq!(GameState::player_team(1), 1); // East = EW
        assert_eq!(GameState::player_team(2), 0); // South = NS
        assert_eq!(GameState::player_team(3), 1); // West = EW
    }

    #[test]
    fn test_partner() {
        assert_eq!(GameState::partner(0), 2); // N <-> S
        assert_eq!(GameState::partner(1), 3); // E <-> W
        assert_eq!(GameState::partner(2), 0);
        assert_eq!(GameState::partner(3), 1);
    }

    #[test]
    fn test_contract_type() {
        let mut c = Contract::default();
        c.trump = 0;
        assert_eq!(c.contract_type(), ContractType::Color(Suit::Spades));
        c.trump = 1;
        assert_eq!(c.contract_type(), ContractType::Color(Suit::Hearts));
        c.trump = 2;
        assert_eq!(c.contract_type(), ContractType::Color(Suit::Diamonds));
        c.trump = 3;
        assert_eq!(c.contract_type(), ContractType::Color(Suit::Clubs));
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_deal_random() {
        let mut rng = rand::thread_rng();
        let state = GameState::deal_random(0, &mut rng);
        let all = state.hands[0] | state.hands[1] | state.hands[2] | state.hands[3];
        assert_eq!(all, ALL_CARDS);
        for i in 0..4 {
            assert_eq!(card_count(state.hands[i]), 8);
        }
        // No overlap
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_eq!(state.hands[i] & state.hands[j], 0);
            }
        }
    }
}
