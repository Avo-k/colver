/// Bidding Exam: a curated set of obvious/extreme hands to test how bid
/// strategies behave in situations where a competent human player would know
/// the right answer.
///
/// Each scenario specifies:
///   - A hand (8 cards) and position
///   - Optional prior bid history
///   - An expectation (must pass, must bid suit X, must bid >= level, must capot, must coinche)
///
/// The exam runs every registered strategy (heuristic + NN models) and prints
/// a human-readable verdict table: ✓ correct, ✗ wrong, with the actual action.
///
/// Usage:
///   cargo run -p colver-core --bin bid_exam --release
///   cargo run -p colver-core --bin bid_exam --release -- --verbose
///   cargo run -p colver-core --bin bid_exam --release -- --category capot

use colver_core::bid_eval::{
    heuristic_bid, improved_v2_bid, moelleux_bid, petit_bide_bid, roro_bid, smart_bid,
};
use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::bidding::{self, BID_COINCHE, BID_PASS, BID_SURCOINCHE};
use colver_core::card::*;
use colver_core::state::GameState;

// ── Card helpers ──────────────────────────────────────────────────────────

const SYM: [&str; 4] = ["♠", "♥", "♦", "♣"];
const _RNK: [&str; 8] = ["7", "8", "9", "J", "Q", "K", "10", "A"];

// Rank indices
const R7: u8 = 0;
const R8: u8 = 1;
const R9: u8 = 2;
const RJ: u8 = 3;
const RQ: u8 = 4;
const RK: u8 = 5;
const R10: u8 = 6;
const RA: u8 = 7;

// Suit indices
const S: u8 = 0;
const H: u8 = 1;
const D: u8 = 2;
const C: u8 = 3;

fn c(suit: u8, rank: u8) -> u8 {
    suit * 8 + rank
}

fn hand_of(cards: &[u8]) -> u32 {
    assert_eq!(cards.len(), 8, "Hand must have exactly 8 cards");
    cards.iter().fold(0u32, |h, &c| h | (1u32 << c))
}

fn pretty(hand: u32) -> String {
    let mut parts = Vec::new();
    for s in 0..4u8 {
        let bits = suit_bits(hand, Suit::from_u8(s));
        if bits == 0 {
            continue;
        }
        let mut ranks = Vec::new();
        for r in (0..8).rev() {
            if bits & (1 << r) != 0 {
                ranks.push(_RNK[r]);
            }
        }
        parts.push(format!("{}{}", SYM[s as usize], ranks.join("")));
    }
    parts.join(" ")
}

fn act_str(action: u8) -> String {
    match action {
        0 => "PASS".into(),
        41 => "COINCHE".into(),
        42 => "SURCOINCHE".into(),
        1..=40 => {
            let (val, suit) = bidding::decode_bid(action);
            if val == 25 {
                format!("Capot{}", SYM[suit as usize])
            } else {
                format!("{}{}", val as u16 * 10, SYM[suit as usize])
            }
        }
        _ => format!("?{}", action),
    }
}

/// Distribute remaining 24 cards to other 3 players deterministically.
fn fill_hands(seat: u8, hand: u32) -> [u32; 4] {
    let remaining: Vec<u8> = (0..32).filter(|&i| hand & (1 << i) == 0).collect();
    let mut hands = [0u32; 4];
    hands[seat as usize] = hand;
    let mut idx = 0;
    for p in 0..4u8 {
        if p == seat {
            continue;
        }
        for _ in 0..8 {
            hands[p as usize] |= 1u32 << remaining[idx];
            idx += 1;
        }
    }
    hands
}

// ── Expectation types ─────────────────────────────────────────────────────

#[derive(Clone)]
enum Expect {
    /// Must pass
    Pass,
    /// Must bid (not pass), in this suit
    BidSuit(u8),
    /// Must bid >= this level (value_encoded, e.g. 8=80) in this suit
    BidAtLeast(u8, u8), // (min_value_enc, suit)
    /// Must bid capot in this suit
    Capot(u8),
    /// Must bid capot (any suit)
    #[allow(dead_code)]
    CapotAny,
    /// Must coinche
    Coinche,
    /// Must NOT pass (any bid is OK)
    NotPass,
    /// Must bid >= level in any suit
    #[allow(dead_code)]
    BidAtLeastAny(u8), // min_value_enc (e.g. 13=130)
}

impl Expect {
    fn check(&self, action: u8) -> bool {
        match self {
            Expect::Pass => action == BID_PASS,
            Expect::BidSuit(suit) => {
                if action == BID_PASS || action == BID_COINCHE || action == BID_SURCOINCHE {
                    return false;
                }
                let (_, s) = bidding::decode_bid(action);
                s == *suit
            }
            Expect::BidAtLeast(min_val, suit) => {
                if action == BID_PASS || action == BID_COINCHE || action == BID_SURCOINCHE {
                    return false;
                }
                let (val, s) = bidding::decode_bid(action);
                s == *suit && val >= *min_val
            }
            Expect::Capot(suit) => {
                if action < 37 || action > 40 {
                    return false;
                }
                let (_, s) = bidding::decode_bid(action);
                s == *suit
            }
            Expect::CapotAny => action >= 37 && action <= 40,
            Expect::Coinche => action == BID_COINCHE,
            Expect::NotPass => action != BID_PASS,
            Expect::BidAtLeastAny(min_val) => {
                if action == BID_PASS || action == BID_COINCHE || action == BID_SURCOINCHE {
                    return false;
                }
                let (val, _) = bidding::decode_bid(action);
                val >= *min_val
            }
        }
    }

    fn describe(&self) -> String {
        match self {
            Expect::Pass => "PASS".into(),
            Expect::BidSuit(s) => format!("bid {}", SYM[*s as usize]),
            Expect::BidAtLeast(v, s) => format!("≥{}{}", *v as u16 * 10, SYM[*s as usize]),
            Expect::Capot(s) => format!("Capot{}", SYM[*s as usize]),
            Expect::CapotAny => "Capot (any)".into(),
            Expect::Coinche => "COINCHE".into(),
            Expect::NotPass => "bid (any)".into(),
            Expect::BidAtLeastAny(v) => format!("≥{} (any suit)", *v as u16 * 10),
        }
    }
}

// ── Test scenario ─────────────────────────────────────────────────────────

struct Scenario {
    name: &'static str,
    category: &'static str,
    hand: u32,
    seat: u8,
    /// Bidding position: 1 = first to bid, 2 = second, etc.
    position: u8,
    /// Prior actions as (seat, action) pairs before our turn.
    prior: Vec<(u8, u8)>,
    expect: Expect,
}

// ── Bidder trait ───────────────────────────────────────────────────────────

enum Bidder {
    Heuristic(&'static str, fn(&GameState) -> u8),
    Nn(&'static str, BidNet),
}

impl Bidder {
    fn name(&self) -> &str {
        match self {
            Bidder::Heuristic(n, _) => n,
            Bidder::Nn(n, _) => n,
        }
    }

    fn bid(&mut self, state: &GameState, bid_history: &[(u8, u8)]) -> u8 {
        match self {
            Bidder::Heuristic(_, f) => f(state),
            Bidder::Nn(_, net) => {
                let obs = bid_obs::make_bid_observation(state, bid_history);
                let legal = state.legal_actions();
                net.best_action_fast(&obs, legal)
            }
        }
    }
}

// ── Scenarios ─────────────────────────────────────────────────────────────

fn build_scenarios() -> Vec<Scenario> {
    vec![
        // ════════════════════════════════════════════════════════════════
        // CATEGORY: Capot évident
        // ════════════════════════════════════════════════════════════════
        Scenario {
            name: "8 atouts: tous les piques",
            category: "capot",
            // A 10 K Q J 9 8 7 of Spades
            hand: hand_of(&[
                c(S, RA), c(S, R10), c(S, RK), c(S, RQ),
                c(S, RJ), c(S, R9),  c(S, R8), c(S, R7),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::Capot(S),
        },
        Scenario {
            name: "8 atouts: tous les cœurs",
            category: "capot",
            hand: hand_of(&[
                c(H, RA), c(H, R10), c(H, RK), c(H, RQ),
                c(H, RJ), c(H, R9),  c(H, R8), c(H, R7),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::Capot(H),
        },
        Scenario {
            name: "J9A10 atout + 4 as dehors",
            category: "capot",
            // J9A10♠ + A♥ A♦ A♣ + 10♥
            hand: hand_of(&[
                c(S, RJ), c(S, R9), c(S, RA), c(S, R10),
                c(H, RA), c(D, RA), c(C, RA), c(H, R10),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::Capot(S),
        },
        Scenario {
            name: "5 top atout (J9ATK) + A10K dehors",
            category: "capot",
            // J 9 A 10 K ♦ + A 10 K ♣
            hand: hand_of(&[
                c(D, RJ), c(D, R9), c(D, RA), c(D, R10), c(D, RK),
                c(C, RA), c(C, R10), c(C, RK),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::Capot(D),
        },
        Scenario {
            name: "6 atout top (J9ATKQ) + 2 as",
            category: "capot",
            // J 9 A 10 K Q ♣ + A♠ A♥
            hand: hand_of(&[
                c(C, RJ), c(C, R9), c(C, RA), c(C, R10), c(C, RK), c(C, RQ),
                c(S, RA), c(H, RA),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::Capot(C),
        },

        // ════════════════════════════════════════════════════════════════
        // CATEGORY: Main très forte (130-160)
        // ════════════════════════════════════════════════════════════════
        Scenario {
            name: "J9A10♥ + A♠ + vide ♦♣",
            category: "fort",
            // J 9 A 10 ♥ + K Q 8 ♥ would be 7 trump, but let's do 4 trump + sides
            // J9A10♥ + A♠ + 10♠ + 7♦ + 7♣
            hand: hand_of(&[
                c(H, RJ), c(H, R9), c(H, RA), c(H, R10),
                c(S, RA), c(S, R10), c(D, R7), c(C, R7),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::BidAtLeast(13, H), // >= 130♥
        },
        Scenario {
            name: "J9♠ + 4 atout (KQJT) + A♥ + coupe ♦",
            category: "fort",
            // J 9 K Q 10 8 ♠ + A♥ + void ♦
            // Wait, that's 6 spades + 1 heart = 7 cards. Need 8.
            // J 9 K Q 10 ♠ + A♥ + A♦ + 7♣
            hand: hand_of(&[
                c(S, RJ), c(S, R9), c(S, RK), c(S, RQ), c(S, R10),
                c(H, RA), c(D, RA), c(C, R7),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::BidAtLeast(13, S), // >= 130♠
        },
        Scenario {
            name: "J9A♦ + 2 atout + A♠ A♣ + coupe ♥",
            category: "fort",
            // J 9 A K Q ♦ + A♠ + A♣ + 7♥
            hand: hand_of(&[
                c(D, RJ), c(D, R9), c(D, RA), c(D, RK), c(D, RQ),
                c(S, RA), c(C, RA), c(H, R7),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::BidAtLeast(13, D), // >= 130♦
        },

        // ════════════════════════════════════════════════════════════════
        // CATEGORY: Annonce normale (80-100)
        // ════════════════════════════════════════════════════════════════
        Scenario {
            name: "J9A♠ + fillers → ouverture classique",
            category: "normal",
            // J 9 A ♠ + 10♥ K♥ + 8♦ + 7♣ 8♣
            hand: hand_of(&[
                c(S, RJ), c(S, R9), c(S, RA),
                c(H, R10), c(H, RK),
                c(D, R8),
                c(C, R7), c(C, R8),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::BidSuit(S),
        },
        Scenario {
            name: "J9♥ + 8♥ → 3 atout minimum",
            category: "normal",
            // J 9 8 ♥ + 7♠ 8♠ + 7♦ 8♦ + 7♣
            hand: hand_of(&[
                c(H, RJ), c(H, R9), c(H, R8),
                c(S, R7), c(S, R8),
                c(D, R7), c(D, R8),
                c(C, R7),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::BidSuit(H),
        },
        Scenario {
            name: "J9♣ + A♣K♣ → bonne couleur 4e",
            category: "normal",
            // J 9 A K ♣ + 7♠ 8♠ + 7♥ + 7♦
            hand: hand_of(&[
                c(C, RJ), c(C, R9), c(C, RA), c(C, RK),
                c(S, R7), c(S, R8),
                c(H, R7),
                c(D, R7),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::BidSuit(C),
        },

        // ════════════════════════════════════════════════════════════════
        // CATEGORY: Passe évidente
        // ════════════════════════════════════════════════════════════════
        Scenario {
            name: "Poubelle: 7 8 partout, pas de J/9/A",
            category: "passe",
            // 7♠ 8♠ 7♥ 8♥ 7♦ 8♦ 7♣ 8♣
            hand: hand_of(&[
                c(S, R7), c(S, R8),
                c(H, R7), c(H, R8),
                c(D, R7), c(D, R8),
                c(C, R7), c(C, R8),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::Pass,
        },
        Scenario {
            name: "Flat Q K partout, pas d'atout",
            category: "passe",
            // Q♠ K♠ Q♥ K♥ Q♦ K♦ Q♣ K♣
            hand: hand_of(&[
                c(S, RQ), c(S, RK),
                c(H, RQ), c(H, RK),
                c(D, RQ), c(D, RK),
                c(C, RQ), c(C, RK),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::Pass,
        },
        Scenario {
            name: "Singleton J sans 9, le reste est nul",
            category: "passe",
            // J♠ seul, rempli de 7/8 dans les autres couleurs
            // J♠ + 7♥ 8♥ + 7♦ 8♦ + 7♣ 8♣ + Q♠
            hand: hand_of(&[
                c(S, RJ), c(S, RQ),
                c(H, R7), c(H, R8),
                c(D, R7), c(D, R8),
                c(C, R7), c(C, R8),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::Pass,
        },
        Scenario {
            name: "4 as mais zéro atout (pas de J/9)",
            category: "passe",
            // A♠ A♥ A♦ A♣ + 7♠ 7♥ 7♦ 7♣
            hand: hand_of(&[
                c(S, RA), c(H, RA), c(D, RA), c(C, RA),
                c(S, R7), c(H, R7), c(D, R7), c(C, R7),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::Pass,
        },

        // ════════════════════════════════════════════════════════════════
        // CATEGORY: Coinche évidente
        // ════════════════════════════════════════════════════════════════
        Scenario {
            name: "Adversaire annonce 80♠, j'ai J9A10♠",
            category: "coinche",
            // Opponent (East, seat 1) bids 80♠. We are South (seat 2).
            // Our hand: J9A10♠ + 7♥ 8♥ 7♦ 8♦
            hand: hand_of(&[
                c(S, RJ), c(S, R9), c(S, RA), c(S, R10),
                c(H, R7), c(H, R8),
                c(D, R7), c(D, R8),
            ]),
            seat: 2,
            position: 2, // we bid 2nd (after opponent)
            prior: {
                // Seat 1 bids 80♠ (action = encode_bid(8, 0) = 1)
                vec![(1, bidding::encode_bid(8, 0))]
            },
            expect: Expect::Coinche,
        },
        Scenario {
            name: "Adversaire annonce 80♥, j'ai J9♥ + A♥ + fort dehors",
            category: "coinche",
            // Opponent bids 80♥. We have J9A♥ + A♠ + A♦ + 10♣ + 7♣ + 8♣
            hand: hand_of(&[
                c(H, RJ), c(H, R9), c(H, RA),
                c(S, RA), c(D, RA),
                c(C, R10), c(C, R7), c(C, R8),
            ]),
            seat: 2,
            position: 2,
            prior: vec![(1, bidding::encode_bid(8, 1))], // opponent bids 80♥
            expect: Expect::Coinche,
        },

        // ════════════════════════════════════════════════════════════════
        // CATEGORY: Réponse au partenaire
        // ════════════════════════════════════════════════════════════════
        Scenario {
            name: "Partenaire annonce 80♠, j'ai 9♠ + A♠ + A♥",
            category: "réponse",
            // Partner (North=0) bids 80♠. East passes. We are South (seat 2).
            // Hand: 9♠ A♠ + A♥ 10♥ + 7♦ 8♦ + 7♣ 8♣
            hand: hand_of(&[
                c(S, R9), c(S, RA),
                c(H, RA), c(H, R10),
                c(D, R7), c(D, R8),
                c(C, R7), c(C, R8),
            ]),
            seat: 2,
            position: 3, // P0 bids, P1 passes, we are 3rd
            prior: vec![
                (0, bidding::encode_bid(8, 0)), // partner bids 80♠
                (1, BID_PASS),                  // opponent passes
            ],
            expect: Expect::BidAtLeast(9, S), // should raise to >= 90♠
        },
        Scenario {
            name: "Partenaire annonce 80♥, j'ai rien → passe",
            category: "réponse",
            // Partner bids 80♥, but we have nothing useful
            // Hand: 7♠ 8♠ Q♠ + 7♦ 8♦ + 7♣ 8♣ + Q♥
            hand: hand_of(&[
                c(S, R7), c(S, R8), c(S, RQ),
                c(H, RQ),
                c(D, R7), c(D, R8),
                c(C, R7), c(C, R8),
            ]),
            seat: 2,
            position: 3,
            prior: vec![
                (0, bidding::encode_bid(8, 1)), // partner bids 80♥
                (1, BID_PASS),
            ],
            expect: Expect::Pass,
        },

        // ════════════════════════════════════════════════════════════════
        // CATEGORY: Distributions extrêmes
        // ════════════════════════════════════════════════════════════════
        Scenario {
            name: "7-1-0-0: J9ATK♠ + Q8♠ + A♥",
            category: "extrême",
            // 7 spades (J9ATKQ8) + A♥ → very strong, should bid high
            hand: hand_of(&[
                c(S, RJ), c(S, R9), c(S, RA), c(S, R10), c(S, RK), c(S, RQ), c(S, R8),
                c(H, RA),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::Capot(S),
        },
        Scenario {
            name: "6-2-0-0: J9ATK♦ + 8♦ + A♣ 10♣",
            category: "extrême",
            // 6 trump + strong side → capot or 160
            hand: hand_of(&[
                c(D, RJ), c(D, R9), c(D, RA), c(D, R10), c(D, RK), c(D, R8),
                c(C, RA), c(C, R10),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::BidAtLeast(16, D), // >= 160♦ or capot
        },
        Scenario {
            name: "5-0-0-3: J9ATK♣ + 7♠ 8♠ 7♥",
            category: "extrême",
            // 5 top trump + 3 low cards, 2 voids → very strong
            hand: hand_of(&[
                c(C, RJ), c(C, R9), c(C, RA), c(C, R10), c(C, RK),
                c(S, R7), c(S, R8), c(H, R7),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::BidAtLeast(13, C), // >= 130♣
        },
        Scenario {
            name: "4-4-0-0: J9A♠ + 8♠ + J9A♥ + 8♥ → choix de couleur",
            category: "extrême",
            // Two equally good 4-card suits — should bid one of them
            hand: hand_of(&[
                c(S, RJ), c(S, R9), c(S, RA), c(S, R8),
                c(H, RJ), c(H, R9), c(H, RA), c(H, R8),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::NotPass, // must bid something, either suit
        },
        Scenario {
            name: "8 carreaux: tout sauf les honneurs (ATKQ 8 7 — sans J9)",
            category: "extrême",
            // 8 diamonds but missing J and 9 → bad trump quality despite length
            // A 10 K Q 8 7 + need 2 more diamonds... but there are only 8 per suit
            // So: A 10 K Q 8 7 ♦ = 6 cards. Need 2 more.
            // Wait, 8 diamonds = all diamonds. Let me reconsider.
            // With all 8 diamonds including J9, that's 8 atouts → capot, already covered.
            // Instead: 6 diamonds without J9 + side cards
            // A 10 K Q 8 7 ♦ + A♠ + 7♣
            hand: hand_of(&[
                c(D, RA), c(D, R10), c(D, RK), c(D, RQ), c(D, R8), c(D, R7),
                c(S, RA), c(C, R7),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            // 6 trump with A10KQ but no J and no 9 → risky, but length makes it viable
            // A good bidder should still bid (the length is enormous)
            expect: Expect::BidSuit(D),
        },

        // ════════════════════════════════════════════════════════════════
        // CATEGORY: Piège — situations trompeuses
        // ════════════════════════════════════════════════════════════════
        Scenario {
            name: "3 as + 10 mais aucun J/9 → piège: ne pas annoncer",
            category: "piège",
            // A♠ A♥ A♦ 10♠ 10♥ + 7♣ 8♣ Q♣
            hand: hand_of(&[
                c(S, RA), c(S, R10),
                c(H, RA), c(H, R10),
                c(D, RA),
                c(C, RQ), c(C, R7), c(C, R8),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::Pass,
        },
        Scenario {
            name: "J seul dans 2 couleurs, 9 nulle part → trop dispersé",
            category: "piège",
            // J♠ J♥ + Q♠ K♥ + 7♦ 8♦ + 7♣ 8♣
            hand: hand_of(&[
                c(S, RJ), c(S, RQ),
                c(H, RJ), c(H, RK),
                c(D, R7), c(D, R8),
                c(C, R7), c(C, R8),
            ]),
            seat: 0,
            position: 1,
            prior: vec![],
            expect: Expect::Pass,
        },

        // ════════════════════════════════════════════════════════════════
        // CATEGORY: Position (dernier à parler)
        // ════════════════════════════════════════════════════════════════
        Scenario {
            name: "Main marginale en 4e position après 3 passes → tenter 80",
            category: "position",
            // J♠ 9♠ + 8♠ + A♥ + 7♥ 7♦ 8♦ 7♣
            // Marginal hand but 4th position after 3 passes — must bid to avoid void deal
            hand: hand_of(&[
                c(S, RJ), c(S, R9), c(S, R8),
                c(H, RA),
                c(H, R7), c(D, R7), c(D, R8), c(C, R7),
            ]),
            seat: 3,
            position: 4,
            prior: vec![
                (0, BID_PASS),
                (1, BID_PASS),
                (2, BID_PASS),
            ],
            expect: Expect::BidSuit(S), // should open to avoid void deal
        },
    ]
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");
    let category_filter: Option<&str> = args
        .windows(2)
        .find(|w| w[0] == "--category" || w[0] == "-c")
        .map(|w| w[1].as_str());

    // Build bidders
    let mut bidders: Vec<Bidder> = vec![
        Bidder::Heuristic("heuristic", heuristic_bid),
        Bidder::Heuristic("smart", smart_bid),
        Bidder::Heuristic("improved_v2", improved_v2_bid),
        Bidder::Heuristic("roro", roro_bid),
        Bidder::Heuristic("petit_bide", petit_bide_bid),
        Bidder::Heuristic("moelleux", moelleux_bid),
    ];

    // NN models — try to load each, skip if file not found
    let nn_models: Vec<(&str, &str)> = vec![
        ("nn_v1 (doudou)", "models/bid_nn_final.bin"),
        ("nn_v2 (dede)", "models/bid_v2/bid_nn_final.bin"),
        ("nn_v3_max", "models/bid_v3_max_20M/bid_nn_final.bin"),
    ];
    for (name, path) in &nn_models {
        match BidNet::load(path) {
            Ok(net) => {
                eprintln!("  Loaded {} from {}", name, path);
                bidders.push(Bidder::Nn(name, net));
            }
            Err(e) => {
                eprintln!("  Skip {}: {}", name, e);
            }
        }
    }

    let scenarios = build_scenarios();
    let num_bidders = bidders.len();

    // Header
    let name_col = 48;
    let expect_col = 16;
    let bidder_col = 14;

    println!();
    println!(
        "  {:<name_col$} {:<expect_col$} {}",
        "SCENARIO",
        "EXPECTED",
        bidders
            .iter()
            .map(|b| format!("{:>w$}", b.name(), w = bidder_col))
            .collect::<Vec<_>>()
            .join("")
    );
    println!("  {}", "─".repeat(name_col + expect_col + num_bidders * bidder_col));

    let mut current_category = "";
    let mut total_pass = vec![0usize; num_bidders];
    let mut total_tests = 0usize;

    for scenario in &scenarios {
        // Category filter
        if let Some(filter) = category_filter {
            if scenario.category != filter {
                continue;
            }
        }

        // Category separator
        if scenario.category != current_category {
            current_category = scenario.category;
            println!();
            println!("  ── {} {}", current_category.to_uppercase(), "─".repeat(60));
        }

        total_tests += 1;

        // Build game state
        let dealer = (scenario.seat + 4 - scenario.position) % 4;
        let hands = fill_hands(scenario.seat, scenario.hand);
        let mut state = GameState::new(dealer, hands);

        // Replay prior actions
        let mut bid_history: Vec<(u8, u8)> = Vec::new();
        for &(s, a) in &scenario.prior {
            bid_history.push((s, a));
            state.step(a);
        }

        // Verify it's the right player's turn
        assert_eq!(
            state.current_player, scenario.seat,
            "Scenario '{}': expected seat {} to act, got {}",
            scenario.name, scenario.seat, state.current_player
        );

        // Query each bidder
        let mut results = Vec::new();
        for (i, bidder) in bidders.iter_mut().enumerate() {
            let action = bidder.bid(&state, &bid_history);
            let pass = scenario.expect.check(action);
            if pass {
                total_pass[i] += 1;
            }
            results.push((action, pass));
        }

        // Print row
        let result_strs: Vec<String> = results
            .iter()
            .map(|(action, pass)| {
                let mark = if *pass { "✓" } else { "✗" };
                let astr = act_str(*action);
                format!("{:>w$}", format!("{}{}", mark, astr), w = bidder_col)
            })
            .collect();

        // Context line (hand + prior)
        let context = if scenario.prior.is_empty() {
            format!("    {} (pos {})", pretty(scenario.hand), scenario.position)
        } else {
            let hist: Vec<String> = scenario.prior.iter().map(|(_, a)| act_str(*a)).collect();
            format!(
                "    {} (pos {}, après {})",
                pretty(scenario.hand),
                scenario.position,
                hist.join(" → ")
            )
        };

        println!(
            "  {:<name_col$} {:<expect_col$} {}",
            scenario.name,
            scenario.expect.describe(),
            result_strs.join("")
        );
        println!("{}", context);

        // Verbose: show Q-values for NN bidders
        if verbose {
            for bidder in bidders.iter_mut() {
                if let Bidder::Nn(name, net) = bidder {
                    let obs = bid_obs::make_bid_observation(&state, &bid_history);
                    let legal = state.legal_actions();
                    let (_best, qvals) = net.best_action(&obs, legal);
                    // Show top-5 Q-values
                    let mut sorted = qvals.clone();
                    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let top5: Vec<String> = sorted
                        .iter()
                        .take(5)
                        .map(|(a, q)| format!("{}={:.3}", act_str(*a), q))
                        .collect();
                    println!(
                        "    {} top-5: {}",
                        name,
                        top5.join("  ")
                    );
                }
            }
        }
    }

    // Summary
    println!();
    println!("  {}", "═".repeat(name_col + expect_col + num_bidders * bidder_col));
    let summary_strs: Vec<String> = total_pass
        .iter()
        .enumerate()
        .map(|(_, &p)| {
            format!(
                "{:>w$}",
                format!("{}/{}", p, total_tests),
                w = bidder_col
            )
        })
        .collect();
    println!(
        "  {:<name_col$} {:<expect_col$} {}",
        "SCORE",
        "",
        summary_strs.join("")
    );
    let pct_strs: Vec<String> = total_pass
        .iter()
        .map(|&p| {
            let pct = if total_tests > 0 {
                (p as f64 / total_tests as f64) * 100.0
            } else {
                0.0
            };
            format!("{:>w$}", format!("{:.0}%", pct), w = bidder_col)
        })
        .collect();
    println!(
        "  {:<name_col$} {:<expect_col$} {}",
        "",
        "",
        pct_strs.join("")
    );
    println!();
}
