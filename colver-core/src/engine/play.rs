use crate::card::*;
use crate::state::*;
use crate::trick::trick_winner;

/// Compute legal card plays as a CardSet bitmask.
///
/// Rules:
/// 1. Must follow lead suit if possible
/// 2. If can't follow:
///    a. If partner is winning ("master") → play anything, **without exception** — including a
///       trump lower than theirs when they cut and trump is all that is left (FFB contrée §2.3)
///    b. Else must trump if possible; must overtrump if possible
///    c. "Ne pisse pas": if can't overtrump, may discard instead of undertrumping
/// 3. When playing trump (following or cutting): must overtrump highest trump on table if possible
///
/// 2a used to carve out an exception forcing the overtrump; removed 2026-08-01, see `docs/RULES.md`.
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
            // Partner holds the trick: no obligation at all, play anything — including a
            // trump *lower* than theirs when they cut and trump is all we have left.
            // FFB contrée §2.3: « il n'est pas obligatoire de couper. On peut se défausser
            // de n'importe quelle carte sans exception (y compris un atout inférieur au
            // sien). » Same wording in the 2016 belote rules; the 2015 contrée rules spell
            // the trump-only case out as « le seul cas de figure où il est permis de jouer
            // un atout inférieur ».
            //
            // Until 2026-08-01 this branch forced an overtrump in the trump-only case,
            // following the one FFB edition that drops the « n'est pas » — see
            // docs/rules-survey/matrices/jeu-de-la-carte.md.
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
pub(crate) fn best_trump_rank_on_trick(state: &GameState, suit: Suit) -> Option<u8> {
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
pub(crate) fn partner_is_master(state: &GameState) -> bool {
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
pub(crate) fn overtrump_in_suit(hand_in_suit: CardSet, suit: Suit, best_rank: u8) -> CardSet {
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
    // Save trick to history before incrementing tricks_won
    let trick_idx = (state.tricks_won[0] + state.tricks_won[1]) as usize;
    state.trick_history[trick_idx] = state.current_trick;

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
/// Belote is only valid when the SAME player holds both Q and K of trump.
/// Note: the card has already been removed from the player's hand at this point.
fn check_belote(state: &mut GameState, player: u8, card: Card) {
    let trump_suit = state.contract.trump_suit();
    if card_suit(card) != trump_suit {
        return;
    }
    let rank = card_rank(card);
    if rank != 4 && rank != 5 {
        return; // Not Queen (4) or King (5)
    }
    let team = GameState::player_team(player) as usize;
    if state.belote[team] == 0 {
        // First Q or K of trump played — check if player still has the other card
        let other_rank = if rank == 4 { 5 } else { 4 }; // Q↔K
        let other_card = make_card(trump_suit, other_rank);
        let other_bit = card_to_bit(other_card);
        if state.hands[player as usize] & other_bit != 0 {
            // Player has both — declare belote
            state.belote[team] = 1;
            state.belote_player[team] = player;
        }
    } else if state.belote[team] == 1 && state.belote_player[team] == player {
        // Same player plays the second card — rebelote
        state.belote[team] = 2;
    }
}

/// Ce que l'annonce de belote apprend sur les mains cachées.
///
/// La belote **s'annonce** : on ne peut pas tenir Dame *et* Roi d'atout et poser
/// le premier des deux en silence. `check_belote` le fait pour les quatre sièges,
/// donc `state.belote` / `state.belote_player` sont de l'information **publique**,
/// au même titre qu'une coupe — et elle se lit dans les deux sens :
///
/// - **annonce** (`belote[t] == 1`) : l'annonceur détient forcément l'autre carte
///   de la paire, jusqu'à ce qu'il la joue (`belote[t] == 2`) ;
/// - **silence** (`belote[t] == 0` alors qu'un Roi ou une Dame d'atout est déjà
///   tombé) : celui qui l'a posée ne tenait pas l'autre à cet instant, et une main
///   ne fait que rétrécir — il ne l'aura donc **jamais**.
///
/// Le second cas est le plus fréquent des deux : il se déclenche à chaque Roi ou
/// Dame d'atout joué sans annonce, alors que le premier demande qu'un adversaire
/// ait la belote.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BeloteFacts {
    /// `held[p]` : cartes que `p` détient forcément (au plus une : l'autre moitié
    /// d'une belote annoncée).
    pub held: [CardSet; 4],
    /// `banned[p]` : cartes que `p` ne peut pas détenir. Contient l'implication
    /// de `held` (une carte forcée chez `p` est interdite aux trois autres), pour
    /// qu'un consommateur qui ne regarde qu'un seul des deux champs reste correct.
    pub banned: [CardSet; 4],
}

impl BeloteFacts {
    /// Aucune déduction disponible — le cas courant tant que ni le Roi ni la Dame
    /// d'atout ne sont tombés.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.held == [0; 4] && self.banned == [0; 4]
    }

    /// Les quatre mains sont-elles compatibles avec ce qui a été annoncé ?
    /// Sert à filtrer un monde produit par un échantillonneur qui ne voit pas
    /// l'annonce (playgen), pas à valider le moteur.
    #[inline]
    pub fn allows(&self, hands: &[CardSet; 4]) -> bool {
        (0..4).all(|p| hands[p] & self.held[p] == self.held[p] && hands[p] & self.banned[p] == 0)
    }
}

/// Siège qui a joué `card` dans cette donne, s'il est déjà tombé.
///
/// `current_trick` et `trick_history` sont tous deux indexés par siège, donc la
/// recherche est directe.
fn seat_that_played(state: &GameState, card: Card) -> Option<u8> {
    for seat in 0..4u8 {
        if state.current_trick[seat as usize] == card {
            return Some(seat);
        }
    }
    let done = (state.tricks_won[0] + state.tricks_won[1]) as usize;
    for trick in state.trick_history.iter().take(done) {
        for seat in 0..4u8 {
            if trick[seat as usize] == card {
                return Some(seat);
            }
        }
    }
    None
}

/// Déductions publiques tirées de la belote à la position courante.
///
/// Rend des ensembles vides hors phase de jeu et tant qu'aucune des deux cartes
/// n'est tombée — c'est-à-dire dans la grande majorité des appels, d'où la sortie
/// anticipée avant tout parcours de l'historique.
/// Interrupteur d'ablation : `COLVER_NO_BELOTE_FACTS=1` fait comme si l'annonce
/// n'existait pas. Compilé hors du binaire sans la feature `belief_ablation` —
/// il n'existe que pour qu'un A/B tienne dans un seul binaire, la déduction
/// étant une règle du jeu et non un réglage.
#[cfg(feature = "belief_ablation")]
fn facts_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("COLVER_NO_BELOTE_FACTS").as_deref() == Ok("1"))
}

/// Étiquette de configuration, pour qu'un run dise ce qu'il a réellement fait.
pub fn belote_ablation_label() -> &'static str {
    #[cfg(feature = "belief_ablation")]
    if facts_disabled() {
        return "belote_facts=off";
    }
    "belote_facts=on"
}

pub fn belote_facts(state: &GameState) -> BeloteFacts {
    let mut facts = BeloteFacts::default();
    #[cfg(feature = "belief_ablation")]
    if facts_disabled() {
        return facts;
    }
    if state.phase == Phase::Bidding {
        return facts;
    }
    let trump = state.contract.trump_suit();
    let queen = make_card(trump, 4);
    let king = make_card(trump, 5);
    let (qbit, kbit) = (card_to_bit(queen), card_to_bit(king));

    // Rien n'est annonçable tant qu'aucune des deux n'est jouée.
    if state.played_cards & (qbit | kbit) == 0 {
        return facts;
    }

    // Annonce faite, seconde carte encore en main : celle des deux qui n'est pas
    // tombée est chez l'annonceur, et nulle part ailleurs.
    for team in 0..2usize {
        if state.belote[team] == 1 {
            let holder = state.belote_player[team] as usize;
            let other = if state.played_cards & qbit != 0 { kbit } else { qbit };
            facts.held[holder] |= other;
            for p in 0..4usize {
                if p != holder {
                    facts.banned[p] |= other;
                }
            }
        }
    }

    // Silence : qui a posé une des deux cartes sans annoncer n'avait pas l'autre.
    // Le camp à interroger est celui du **poseur**, pas celui qu'on itère : quand
    // un camp a annoncé, l'autre voit lui aussi `belote == 0` alors que la carte
    // est justement chez l'annonceur.
    for (played_card, other_bit) in [(queen, kbit), (king, qbit)] {
        if state.played_cards & card_to_bit(played_card) == 0
            || state.played_cards & other_bit != 0
        {
            continue; // pas encore posée, ou l'autre est déjà tombée
        }
        if let Some(seat) = seat_that_played(state, played_card) {
            if state.belote[GameState::player_team(seat) as usize] == 0 {
                facts.banned[seat as usize] |= other_bit;
            }
        }
    }

    facts
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

    /// Atout = pique : Dame = 4, Roi = 5.
    const QS: Card = 4;
    const KS: Card = 5;

    fn cards(list: &[Card]) -> CardSet {
        list.iter().fold(0, |acc, &c| acc | card_to_bit(c))
    }

    #[test]
    fn belote_facts_empty_before_either_trump_honour_falls() {
        let state = make_playing_state(
            0,
            [
                cards(&[0, 1, 2, 3, 6, 7, 14, 15]),
                cards(&[QS, KS, 8, 9, 10, 11, 12, 13]),
                cards(&[16, 17, 18, 19, 20, 21, 22, 23]),
                cards(&[24, 25, 26, 27, 28, 29, 30, 31]),
            ],
        );
        assert!(belote_facts(&state).is_empty());
    }

    #[test]
    fn belote_facts_forces_the_second_card_of_an_announced_belote() {
        let mut state = make_playing_state(
            0,
            [
                cards(&[0, 1, 2, 3, 6, 7, 14, 15]),
                cards(&[QS, KS, 8, 9, 10, 11, 12, 13]),
                cards(&[16, 17, 18, 19, 20, 21, 22, 23]),
                cards(&[24, 25, 26, 27, 28, 29, 30, 31]),
            ],
        );
        apply_play(&mut state, KS); // siège 1 annonce « belote »
        assert_eq!(state.belote[1], 1);

        let facts = belote_facts(&state);
        assert_eq!(facts.held[1], card_to_bit(QS), "la Dame est chez l'annonceur");
        for p in [0usize, 2, 3] {
            assert_eq!(facts.banned[p], card_to_bit(QS), "et nulle part ailleurs");
        }
        // Le piège : l'annonceur voit `belote == 0` du côté du camp adverse, et
        // une lecture par camp au lieu de par siège lui interdirait sa propre Dame.
        assert_eq!(facts.banned[1], 0);

        let mut world = state.hands;
        assert!(facts.allows(&world));
        world[1] &= !card_to_bit(QS);
        world[0] |= card_to_bit(QS);
        assert!(!facts.allows(&world), "monde impossible : Dame déplacée");
    }

    #[test]
    fn belote_facts_bans_the_other_honour_when_nobody_announced() {
        let mut state = make_playing_state(
            0,
            [
                cards(&[0, 1, 2, 3, 6, 7, 15, 23]),
                cards(&[KS, 8, 9, 10, 11, 12, 13, 14]),
                cards(&[QS, 16, 17, 18, 19, 20, 21, 22]),
                cards(&[24, 25, 26, 27, 28, 29, 30, 31]),
            ],
        );
        apply_play(&mut state, KS); // Roi d'atout, sans annonce
        assert_eq!(state.belote, [0, 0]);

        let facts = belote_facts(&state);
        assert_eq!(facts.held, [0; 4], "rien n'est placé, seulement exclu");
        assert_eq!(facts.banned[1], card_to_bit(QS));
        for p in [0usize, 2, 3] {
            assert_eq!(facts.banned[p], 0, "on n'apprend rien sur les autres");
        }
    }

    #[test]
    fn belote_facts_empty_after_rebelote() {
        let mut state = make_playing_state(
            0,
            [
                cards(&[0, 1, 2, 3, 6, 7, 14, 15]),
                cards(&[QS, KS, 8, 9, 10, 11, 12, 13]),
                cards(&[16, 17, 18, 19, 20, 21, 22, 23]),
                cards(&[24, 25, 26, 27, 28, 29, 30, 31]),
            ],
        );
        apply_play(&mut state, KS); // belote
        apply_play(&mut state, 16); // siège 2, carreau
        apply_play(&mut state, 24); // siège 3, trèfle
        apply_play(&mut state, 0); // siège 0, 7 d'atout — le siège 1 remporte le pli
        assert_eq!(state.current_player, 1);
        assert!(!belote_facts(&state).is_empty(), "la Dame est encore en main");

        apply_play(&mut state, QS); // rebelote
        assert_eq!(state.belote[1], 2);
        assert!(
            belote_facts(&state).is_empty(),
            "les deux cartes sont tombées : plus rien à déduire"
        );
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
    fn test_partner_cut_only_trump_is_free_choice() {
        // Partner cut a non-trump lead and holds the trick, and trump is all we have left.
        // FFB contrée §2.3: no obligation whatsoever — an *undertrump* is explicitly allowed,
        // and 2015 §4 calls it "le seul cas de figure où il est permis de jouer un atout
        // inférieur". Before 2026-08-01 this engine forced an overtrump here, following the
        // single FFB edition that drops the negation from that sentence.
        //
        // Trump: Clubs (bits 24..31, trump strength 7<8<Q<K<10<A<9<J).
        // P1 leads 7D. P2 (P0's partner) cuts with 8C. P3 discards 7H. P0 to play.
        let mut state = make_playing_state(3, [
            0,                  // P0: set per case below
            0x0F_0000,          // P1: 7,8,9,J diamonds
            0x0200_0000,        // P2: 8C only (cuts)
            0xF0_0000 | 0xFF00, // P3: Q,K,10,A diamonds + all hearts
        ]);
        state.trick_lead = 1;
        state.current_trick[1] = make_card(Suit::Diamonds, 0); // lead 7D
        state.current_trick[2] = make_card(Suit::Clubs, 1); // partner cuts with 8C
        state.current_trick[3] = make_card(Suit::Hearts, 0); // opponent discards
        state.trick_count = 3;
        state.current_player = 0;

        // Only trumps stronger than partner's → all legal (nothing to arbitrate).
        state.hands[0] = 0x8400_0000; // 9C + AC
        assert_eq!(legal_plays(&state), state.hands[0]);

        // Only a *weaker* trump → legal, and it is the whole point of the rule.
        state.hands[0] = 0x0100_0000; // 7C
        assert_eq!(legal_plays(&state), 0x0100_0000);

        // Mixed weak + strong → BOTH legal. This is the assertion that flipped:
        // the engine used to return AC alone, forcing the overtrump.
        state.hands[0] = 0x8100_0000; // 7C + AC
        assert_eq!(legal_plays(&state), 0x8100_0000);
    }

    #[test]
    fn test_rule4_partner_cut_with_non_trump_cards_can_discard() {
        // When partner has cut and is master, but we have non-trump cards too,
        // Rule 2.1 applies: can play anything (discard).
        // Trump: Clubs (3).
        let mut state = make_playing_state(3, [
            0x0300_0000 | 0x03, // P0: 7C,8C (trump) + 7S,8S (non-trump)
            0x0F_0000,          // P1: 7,8,9,J diamonds
            0x0400_0000,        // P2: 9C (partner of P0, cuts)
            0xF0_0000 | 0xFF00, // P3: diamonds + hearts
        ]);

        state.trick_lead = 1;
        state.current_trick[1] = make_card(Suit::Diamonds, 0); // P1: 7D (lead)
        state.current_trick[2] = make_card(Suit::Clubs, 2);    // P2: 9C (partner cuts)
        state.current_trick[3] = make_card(Suit::Hearts, 0);   // P3: 7H (discard)
        state.trick_count = 3;
        state.current_player = 0;

        // P0 has both trump and non-trump → can play anything
        let legal = legal_plays(&state);
        assert_eq!(legal, state.hands[0]);
    }

    #[test]
    fn test_partner_master_no_cut_only_trump() {
        // Partner is master with non-trump (led the suit and winning), we only have trump.
        // Rule 2.1: can play anything (= any trump since that's all we have).
        // Trump: Clubs (3). Lead suit: Diamonds.
        let mut state = make_playing_state(3, [
            0x0300_0000,        // P0: 7C,8C (only trump)
            0xF0_0000,          // P1: Q,K,10,A diamonds
            0x0F_0000,          // P2: 7,8,9,J diamonds (partner of P0)
            0xFF00,             // P3: all hearts
        ]);

        // P2 leads AD (strongest diamond, non-trump). P3 plays 7H (discard). P0 to play.
        // Wait, P2 leads so order is P2, P3, P0, P1.
        state.trick_lead = 2;
        state.current_trick[2] = make_card(Suit::Diamonds, 7); // P2: AD (lead, winning)
        state.current_trick[3] = make_card(Suit::Hearts, 0);   // P3: 7H (discard)
        state.trick_count = 2;
        state.current_player = 0;

        // Partner P2 is winning with AD (non-trump, no cut). P0 only has trump.
        // No cut involved → Rule 2.1: play anything = both trumps.
        let legal = legal_plays(&state);
        assert_eq!(legal, state.hands[0]); // 7C + 8C, no overtrump required
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
