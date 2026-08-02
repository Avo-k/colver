//! Classification des mains de 8 cartes, insensible aux couleurs.
//!
//! Les quatre couleurs sont interchangeables tant qu'aucun atout n'est nommé :
//! une main et la même main couleurs échangées sont la même position. Ce module
//! en tire deux couches, qui ne servent pas à la même chose et qu'il ne faut pas
//! confondre.
//!
//! # 1. L'index canonique — exact, sans perte
//!
//! [`hand_class_id`] range une main dans `[0, NUM_HAND_CLASSES)` = 472 579
//! classes ; [`hand_class_id_trump`] fait de même à atout désigné (le groupe
//! tombe de S4 à S3), soit 1 820 803 classes. Deux mains équivalentes ont le
//! même index, deux mains inéquivalentes en ont deux différents. C'est une
//! bijection : [`hand_from_class_id`] la parcourt en sens inverse, ce qui rend
//! l'espace des mains **énumérable** — 472 579 entrées, là où une politique
//! d'ouverture (premier à parler, historique vide) est justement une fonction
//! pure de la main.
//!
//! # 2. Le code — avec perte, lisible
//!
//! [`HandCode`] ne retient que ce qui pèse. Le choix de ce qui pèse n'est pas
//! une opinion : il vient d'une mesure appariée sur le solveur DD (on échange
//! une carte de la main contre la plus faible de la même couleur détenue par un
//! adversaire, on re-résout, le reste de la donne est identique). Perte moyenne
//! en points DD, échelle 0-252 :
//!
//! | à l'atout | | à côté | |
//! |---|---|---|---|
//! | J  | +49,2 | A  | +26,0 |
//! | 9  | +18,9 | 10 | +6,3  |
//! | A  | +9,5  | K  | +1,5  |
//! | 10 | +5,6  | Q  | −0,1  |
//! | K  | +1,8  | 9  |  0,0  |
//! | Q  | +0,9  | 8  |  0,0  |
//! | 8  | +0,4  | J  | −0,5  |
//!
//! D'où le contenu du code : à l'atout les quatre cartes qui dépassent 5 points
//! (J, 9, A, 10) ; à côté l'As et le 10 ; partout la longueur, qui porte les
//! coupes franches et les longues. Le reste — Dame, Valet, 9, 8, 7 de côté — est
//! mesuré sous le point et n'est pas encodé. La belote (K+Q d'atout) est gardée
//! bien qu'invisible au DD : elle ne vaut aucun point carte, mais 20 points de
//! marque, et c'est une décision d'enchère.
//!
//! Le code s'emboîte du grossier au fin via [`HandCode::coarsen`]. Nombre de
//! codes distincts et concentration, comptés exactement sur les 10 518 300 mains :
//!
//! | niveau | codes | 50 % des mains | 90 % |
//! |---|---:|---:|---:|
//! | [`CodeLevel::Length`] | 9 | 2 | 4 |
//! | [`CodeLevel::Trump`]  | 80 | 8 | 28 |
//! | [`CodeLevel::Shape`]  | 339 | 28 | 122 |
//! | [`CodeLevel::Tops`]   | 5 277 | 388 | 1 927 |
//! | [`CodeLevel::Full`]   | 6 654 | 420 | 2 281 |
//!
//! Rendu : `T5.J9AT.A1/A1/x1` — cinq atouts, les quatre gros atouts détenus,
//! deux As secs et une basse à côté.

use crate::card::{CardSet, TRUMP_STRENGTH};
use core::fmt;

// ===========================================================================
// Couche 1 : index canonique exact
// ===========================================================================

/// `COUNTS[k][b][p]` = nombre de k-uplets décroissants de masques de couleur,
/// chacun ≤ `b`, dont les popcounts somment à `p`.
///
/// Récurrence en O(5·256·9) : soit le premier élément vaut `b`, soit tout le
/// uplet tient sous `b-1`.
const COUNTS: [[[u32; 9]; 256]; 5] = build_counts();

const fn build_counts() -> [[[u32; 9]; 256]; 5] {
    let mut f = [[[0u32; 9]; 256]; 5];
    // k = 0 : le uplet vide, valide si et seulement s'il ne reste rien à placer.
    let mut b = 0;
    while b < 256 {
        f[0][b][0] = 1;
        b += 1;
    }
    let mut k = 1;
    while k < 5 {
        // b = 0 : tous les éléments valent 0, donc popcount total nul.
        f[k][0][0] = 1;
        let mut b = 1;
        while b < 256 {
            let pc = (b as u32).count_ones() as usize;
            let mut p = 0;
            while p < 9 {
                let mut s = f[k][b - 1][p];
                if pc <= p {
                    s += f[k - 1][b][p - pc];
                }
                f[k][b][p] = s;
                p += 1;
            }
            b += 1;
        }
        k += 1;
    }
    f
}

/// Nombre de mains de 8 cartes distinctes à permutation de couleurs près.
pub const NUM_HAND_CLASSES: u32 = COUNTS[4][255][8];

/// Idem, mais l'atout est désigné : seules les 3 autres couleurs s'échangent.
pub const NUM_HAND_CLASSES_TRUMP: u32 = trump_class_total();

const fn trump_class_total() -> u32 {
    TRUMP_BASE[255] + COUNTS[3][255][8 - (255u32.count_ones() as usize)]
}

/// `TRUMP_BASE[t]` = nombre de classes dont le masque d'atout est `< t`.
const TRUMP_BASE: [u32; 256] = build_trump_base();

const fn build_trump_base() -> [u32; 256] {
    let mut base = [0u32; 256];
    let mut t = 1;
    while t < 256 {
        let pc = (t as u32 - 1).count_ones() as usize;
        base[t] = base[t - 1] + COUNTS[3][255][8 - pc];
        t += 1;
    }
    base
}

/// Les quatre masques de couleur d'une main, dans l'ordre physique S/H/D/C.
#[inline]
pub fn suit_masks(hand: CardSet) -> [u8; 4] {
    [
        hand as u8,
        (hand >> 8) as u8,
        (hand >> 16) as u8,
        (hand >> 24) as u8,
    ]
}

#[inline]
fn hand_from_masks(masks: [u8; 4]) -> CardSet {
    (masks[0] as CardSet)
        | (masks[1] as CardSet) << 8
        | (masks[2] as CardSet) << 16
        | (masks[3] as CardSet) << 24
}

/// Rang lexicographique d'un uplet décroissant parmi ceux de même longueur,
/// borne initiale 255 et popcount total `p`.
///
/// Un uplet est décroissant, donc « son premier élément est `< m` » équivaut à
/// « tous ses éléments sont `≤ m-1` » : les uplets qui précèdent sont comptés
/// d'un seul coup par `COUNTS[k][m-1][p]`, sans boucler sur les valeurs.
fn rank_tuple(sorted: &[u8], mut p: usize) -> u32 {
    let mut rank = 0u32;
    let k = sorted.len();
    for (i, &m) in sorted.iter().enumerate() {
        if m > 0 {
            rank += COUNTS[k - i][m as usize - 1][p];
        }
        p -= (m as u32).count_ones() as usize;
    }
    rank
}

/// Inverse de [`rank_tuple`] : reconstruit le uplet décroissant de rang donné.
fn unrank_tuple(mut rank: u32, k: usize, mut p: usize, out: &mut [u8]) {
    for i in 0..k {
        // `COUNTS[k-i][a][p]` est le nombre d'uplets dont le premier élément est
        // `≤ a`, donc croissant en `a` : on cherche le premier qui dépasse.
        let rem = k - i;
        let mut a = 0usize;
        while COUNTS[rem][a][p] <= rank {
            a += 1;
            debug_assert!(a < 256, "rang hors domaine");
        }
        if a > 0 {
            rank -= COUNTS[rem][a - 1][p];
        }
        out[i] = a as u8;
        p -= (a as u32).count_ones() as usize;
    }
}

/// Index canonique d'une main de 8 cartes, dans `[0, NUM_HAND_CLASSES)`.
///
/// Invariant : `hand_class_id` est constant sur l'orbite des 24 permutations de
/// couleurs, et deux orbites distinctes ont deux index distincts.
pub fn hand_class_id(hand: CardSet) -> u32 {
    debug_assert_eq!(hand.count_ones(), 8, "main de 8 cartes attendue");
    let mut m = suit_masks(hand);
    m.sort_unstable_by(|a, b| b.cmp(a));
    rank_tuple(&m, 8)
}

/// Une main représentative de la classe `id` (couleurs dans l'ordre physique).
pub fn hand_from_class_id(id: u32) -> CardSet {
    debug_assert!(id < NUM_HAND_CLASSES);
    let mut m = [0u8; 4];
    unrank_tuple(id, 4, 8, &mut m);
    hand_from_masks(m)
}

/// Index canonique à atout désigné, dans `[0, NUM_HAND_CLASSES_TRUMP)`.
///
/// L'atout n'est plus interchangeable : seules les 3 autres couleurs le sont.
pub fn hand_class_id_trump(hand: CardSet, trump: u8) -> u32 {
    debug_assert_eq!(hand.count_ones(), 8, "main de 8 cartes attendue");
    debug_assert!(trump < 4);
    let masks = suit_masks(hand);
    let t = masks[trump as usize];
    // Décalage : toutes les classes dont le masque d'atout est plus petit.
    let base = TRUMP_BASE[t as usize];
    let mut sides = [0u8; 3];
    let mut n = 0;
    for (s, &m) in masks.iter().enumerate() {
        if s != trump as usize {
            sides[n] = m;
            n += 1;
        }
    }
    sides.sort_unstable_by(|a, b| b.cmp(a));
    let rest = 8 - t.count_ones() as usize;
    base + rank_tuple(&sides, rest)
}

/// Une main représentative de la classe à atout désigné `id`, l'atout étant
/// rendu en couleur 0 (pique).
pub fn hand_from_class_id_trump(mut id: u32) -> CardSet {
    debug_assert!(id < NUM_HAND_CLASSES_TRUMP);
    // `TRUMP_BASE` est croissant : le masque d'atout est le dernier dont le
    // décalage ne dépasse pas `id`.
    let t = TRUMP_BASE.partition_point(|&b| b <= id) - 1;
    id -= TRUMP_BASE[t];
    let mut sides = [0u8; 3];
    unrank_tuple(id, 3, 8 - (t as u32).count_ones() as usize, &mut sides);
    hand_from_masks([t as u8, sides[0], sides[1], sides[2]])
}

// ===========================================================================
// Couche 2 : code lisible
// ===========================================================================

/// Rangs par force décroissante à l'atout : J 9 A 10 K Q 8 7.
///
/// Dérivé de [`TRUMP_STRENGTH`], qui reste la source unique de l'ordre.
pub const TRUMP_ORDER: [u8; 8] = build_trump_order();

const fn build_trump_order() -> [u8; 8] {
    let mut order = [0u8; 8];
    let mut r = 0;
    while r < 8 {
        order[7 - TRUMP_STRENGTH[r] as usize] = r as u8;
        r += 1;
    }
    order
}

/// Les quatre cartes d'atout mesurées au-dessus de 5 points DD : J, 9, A, 10.
const TOP4_RANKS: [u8; 4] = [
    TRUMP_ORDER[0],
    TRUMP_ORDER[1],
    TRUMP_ORDER[2],
    TRUMP_ORDER[3],
];

const RANK_K: u8 = 5;
const RANK_Q: u8 = 4;
const RANK_A: u8 = 7;
const RANK_T: u8 = 6;

/// Matadors, à la manière du « mit N / ohne N Spitzen » du Skat : longueur de la
/// série ininterrompue des plus gros atouts, comptée depuis le Valet dans
/// l'ordre [`TRUMP_ORDER`].
///
/// Positif si on la détient, négatif si l'adversaire la détient, `0` étant
/// impossible (soit on a le Valet, soit on ne l'a pas). `+8` = tous les atouts.
pub fn matadors(hand: CardSet, trump: u8) -> i8 {
    debug_assert!(trump < 4);
    let mask = suit_masks(hand)[trump as usize];
    let held = |r: u8| mask >> r & 1 == 1;
    if held(TRUMP_ORDER[0]) {
        TRUMP_ORDER.iter().take_while(|&&r| held(r)).count() as i8
    } else {
        -(TRUMP_ORDER.iter().take_while(|&&r| !held(r)).count() as i8)
    }
}

/// Résumé d'une couleur de côté : sa longueur et ses deux cartes à points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SideSuit {
    pub len: u8,
    pub ace: bool,
    pub ten: bool,
}

impl fmt::Display for SideSuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len == 0 {
            return f.write_str("-");
        }
        let top = match (self.ace, self.ten) {
            (true, true) => "AT",
            (true, false) => "A",
            (false, true) => "T",
            (false, false) => "x",
        };
        write!(f, "{}{}", top, self.len)
    }
}

/// Finesse du code. Chaque niveau contient les précédents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CodeLevel {
    /// Longueur d'atout seule — 9 codes.
    Length,
    /// + les quatre gros atouts détenus — 80 codes.
    Trump,
    /// + la forme des couleurs de côté — 339 codes.
    Shape,
    /// + l'As et le 10 de chaque couleur de côté — 5 277 codes.
    Tops,
    /// + la belote — 6 654 codes.
    Full,
}

/// Code de main insensible aux couleurs, atout désigné.
///
/// `Eq`/`Hash`/`Ord` en font directement une clé de regroupement : `coarsen`
/// puis `HashMap<HandCode, _>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct HandCode {
    /// Nombre d'atouts, 0 à 8.
    pub trump_len: u8,
    /// Bits 0..3 = J, 9, A, 10 d'atout, dans cet ordre de force.
    pub top4: u8,
    /// K et Q d'atout — 20 points de marque, invisibles au DD.
    pub belote: bool,
    /// Les trois couleurs de côté, triées (décroissant) pour être canoniques.
    pub sides: [SideSuit; 3],
}

impl HandCode {
    pub fn from_hand(hand: CardSet, trump: u8) -> Self {
        debug_assert_eq!(hand.count_ones(), 8, "main de 8 cartes attendue");
        debug_assert!(trump < 4);
        let masks = suit_masks(hand);
        let t = masks[trump as usize];

        let mut top4 = 0u8;
        for (i, &r) in TOP4_RANKS.iter().enumerate() {
            if t >> r & 1 == 1 {
                top4 |= 1 << i;
            }
        }

        let mut sides = [SideSuit::default(); 3];
        let mut n = 0;
        for (s, &m) in masks.iter().enumerate() {
            if s != trump as usize {
                sides[n] = SideSuit {
                    len: m.count_ones() as u8,
                    ace: m >> RANK_A & 1 == 1,
                    ten: m >> RANK_T & 1 == 1,
                };
                n += 1;
            }
        }
        sides.sort_unstable_by(|a, b| b.cmp(a));

        Self {
            trump_len: t.count_ones() as u8,
            top4,
            belote: (t >> RANK_K & 1 == 1) && (t >> RANK_Q & 1 == 1),
            sides,
        }
    }

    /// Efface tout ce qui est plus fin que `level`.
    pub fn coarsen(&self, level: CodeLevel) -> Self {
        let mut c = *self;
        if level < CodeLevel::Full {
            c.belote = false;
        }
        if level < CodeLevel::Tops {
            for s in &mut c.sides {
                s.ace = false;
                s.ten = false;
            }
        }
        if level < CodeLevel::Shape {
            c.sides = [SideSuit::default(); 3];
        }
        if level < CodeLevel::Trump {
            c.top4 = 0;
        }
        c
    }

    /// Série des plus gros atouts détenue, **plafonnée à 4** — le code ne garde
    /// que J, 9, A, 10. Pour la valeur exacte, y compris `+8`, voir [`matadors`].
    pub fn top_run(&self) -> u8 {
        (0..4).take_while(|i| self.top4 >> i & 1 == 1).count() as u8
    }

    /// La forme des couleurs de côté, longueurs triées décroissant.
    pub fn side_shape(&self) -> [u8; 3] {
        [self.sides[0].len, self.sides[1].len, self.sides[2].len]
    }
}

impl fmt::Display for HandCode {
    /// `T5.J9AT.A1/A1/x1`, plus `.B` si belote.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}.", self.trump_len)?;
        if self.top4 == 0 {
            f.write_str("-")?;
        } else {
            for (i, name) in ["J", "9", "A", "T"].iter().enumerate() {
                if self.top4 >> i & 1 == 1 {
                    f.write_str(name)?;
                }
            }
        }
        write!(
            f,
            ".{}/{}/{}",
            self.sides[0], self.sides[1], self.sides[2]
        )?;
        if self.belote {
            f.write_str(".B")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Les 24 permutations de couleurs appliquées à une main.
    fn permute(hand: CardSet, sigma: [usize; 4]) -> CardSet {
        let m = suit_masks(hand);
        let mut out = [0u8; 4];
        for s in 0..4 {
            out[sigma[s]] = m[s];
        }
        hand_from_masks(out)
    }

    fn all_perms() -> Vec<[usize; 4]> {
        let mut v = Vec::new();
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let p = [a, b, c, d];
                        let mut seen = [false; 4];
                        for &x in &p {
                            seen[x] = true;
                        }
                        if seen.iter().all(|&s| s) {
                            v.push(p);
                        }
                    }
                }
            }
        }
        v
    }

    fn random_hands(n: usize) -> Vec<CardSet> {
        // LCG : pas de dépendance à `rand`, ce module est compilé sans features.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..n)
            .map(|_| {
                let mut deck: Vec<u8> = (0..32).collect();
                for i in (1..32).rev() {
                    let j = (next() % (i as u64 + 1)) as usize;
                    deck.swap(i, j);
                }
                deck[..8].iter().fold(0u32, |acc, &c| acc | 1 << c)
            })
            .collect()
    }

    #[test]
    fn class_totals_match_burnside() {
        assert_eq!(NUM_HAND_CLASSES, 472_579);
        assert_eq!(NUM_HAND_CLASSES_TRUMP, 1_820_803);
    }

    #[test]
    fn class_id_is_invariant_under_suit_permutation() {
        let perms = all_perms();
        assert_eq!(perms.len(), 24);
        for hand in random_hands(500) {
            let id = hand_class_id(hand);
            for &p in &perms {
                assert_eq!(hand_class_id(permute(hand, p)), id, "main {hand:#010x}");
            }
        }
    }

    #[test]
    fn trump_class_id_is_invariant_under_side_permutation_only() {
        for hand in random_hands(300) {
            let id = hand_class_id_trump(hand, 0);
            // Les 6 permutations qui fixent le pique laissent l'index inchangé.
            for p in [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1]] {
                assert_eq!(hand_class_id_trump(permute(hand, p), 0), id);
            }
            // Désigner une autre couleur comme atout doit être vu comme la même
            // classe que permuter la main pour amener cette couleur en pique.
            for t in 1..4u8 {
                let mut sigma = [0usize; 4];
                sigma[t as usize] = 0;
                let mut k = 1;
                for s in 0..4 {
                    if s != t as usize {
                        sigma[s] = k;
                        k += 1;
                    }
                }
                assert_eq!(
                    hand_class_id_trump(hand, t),
                    hand_class_id_trump(permute(hand, sigma), 0)
                );
            }
        }
    }

    /// Échantillon régulier + les deux bords, qui sont les cas où un décalage
    /// d'index se voit. La version exhaustive est plus bas, ignorée par défaut.
    fn strided(total: u32, n: u32) -> impl Iterator<Item = u32> {
        let step = total / n;
        (0..n).map(move |i| i * step).chain([total - 1, 0])
    }

    #[test]
    fn class_id_round_trips() {
        for id in strided(NUM_HAND_CLASSES, 4_000) {
            let hand = hand_from_class_id(id);
            assert_eq!(hand.count_ones(), 8, "classe {id}");
            assert_eq!(hand_class_id(hand), id);
        }
        // Et dans l'autre sens, depuis de vraies mains.
        for hand in random_hands(2_000) {
            let id = hand_class_id(hand);
            assert!(id < NUM_HAND_CLASSES);
            assert_eq!(hand_class_id(hand_from_class_id(id)), id);
        }
    }

    #[test]
    fn trump_class_id_round_trips() {
        for id in strided(NUM_HAND_CLASSES_TRUMP, 4_000) {
            let hand = hand_from_class_id_trump(id);
            assert_eq!(hand.count_ones(), 8, "classe {id}");
            assert_eq!(hand_class_id_trump(hand, 0), id);
        }
        for hand in random_hands(2_000) {
            for t in 0..4u8 {
                let id = hand_class_id_trump(hand, t);
                assert!(id < NUM_HAND_CLASSES_TRUMP);
                assert_eq!(hand_class_id_trump(hand_from_class_id_trump(id), 0), id);
            }
        }
    }

    /// Bijectivité stricte sur tout l'espace : à lancer avec
    /// `cargo test -p colver-core --release --lib hand_class -- --ignored`
    /// (600× plus lent en debug, d'où l'exclusion du tour de piste habituel).
    #[test]
    #[ignore]
    fn class_id_round_trips_over_the_whole_space() {
        for id in 0..NUM_HAND_CLASSES {
            let hand = hand_from_class_id(id);
            assert_eq!(hand.count_ones(), 8, "classe {id}");
            assert_eq!(hand_class_id(hand), id);
        }
    }

    /// Idem à atout désigné. Voir la note ci-dessus.
    #[test]
    #[ignore]
    fn trump_class_id_round_trips_over_the_whole_space() {
        for id in 0..NUM_HAND_CLASSES_TRUMP {
            let hand = hand_from_class_id_trump(id);
            assert_eq!(hand.count_ones(), 8, "classe {id}");
            assert_eq!(hand_class_id_trump(hand, 0), id);
        }
    }

    /// Une carte de rang `r` dans la couleur `s`.
    fn card(s: u8, r: u8) -> CardSet {
        1 << (s * 8 + r)
    }

    #[test]
    fn trump_order_matches_the_engine() {
        // J 9 A 10 K Q 8 7 — l'ordre du jeu, pas l'ordre naturel des rangs.
        assert_eq!(TRUMP_ORDER, [3, 2, 7, 6, 5, 4, 1, 0]);
        for (pos, &r) in TRUMP_ORDER.iter().enumerate() {
            assert_eq!(TRUMP_STRENGTH[r as usize] as usize, 7 - pos);
        }
    }

    #[test]
    fn matadors_follows_the_trump_order() {
        // J 9 A 10 K d'atout + 3 cartes ailleurs : série de 5.
        let hand = card(0, 3) | card(0, 2) | card(0, 7) | card(0, 6) | card(0, 5)
            | card(1, 0) | card(1, 1) | card(2, 0);
        assert_eq!(matadors(hand, 0), 5);
        // Sans le Valet mais avec le 9 : l'adversaire tient la première, donc -1.
        let hand = card(0, 2) | card(0, 7) | card(0, 6) | card(0, 5) | card(0, 4)
            | card(1, 0) | card(1, 1) | card(2, 0);
        assert_eq!(matadors(hand, 0), -1);
        // Les huit atouts.
        let all: CardSet = 0xFF;
        assert_eq!(matadors(all, 0), 8);
        // Aucun atout : l'adversaire tient les huit premières.
        assert_eq!(matadors(0xFF00, 0), -8);
    }

    #[test]
    fn code_renders_the_reference_hand() {
        // A T K J 9 de pique, A de coeur, A de carreau, 7 de trefle — atout pique.
        let hand = card(0, 7) | card(0, 6) | card(0, 5) | card(0, 3) | card(0, 2)
            | card(1, 7) | card(2, 7) | card(3, 0);
        let c = HandCode::from_hand(hand, 0);
        assert_eq!(c.to_string(), "T5.J9AT.A1/A1/x1");
        assert_eq!(c.trump_len, 5);
        assert_eq!(c.top_run(), 4);
        assert_eq!(c.side_shape(), [1, 1, 1]);
        assert!(!c.belote, "pas de Dame d'atout");
    }

    #[test]
    fn code_is_invariant_under_suit_permutation() {
        for hand in random_hands(400) {
            for t in 0..4u8 {
                let c = HandCode::from_hand(hand, t);
                for p in all_perms() {
                    let moved = permute(hand, p);
                    assert_eq!(HandCode::from_hand(moved, p[t as usize] as u8), c);
                }
            }
        }
    }

    #[test]
    fn belote_needs_both_king_and_queen_of_trump() {
        let base = card(1, 0) | card(1, 1) | card(1, 2) | card(2, 0) | card(2, 1) | card(2, 2);
        let kq = base | card(0, 5) | card(0, 4);
        assert!(HandCode::from_hand(kq, 0).belote);
        // Roi et Dame, mais dans une couleur de côté : pas de belote.
        let side = base | card(3, 5) | card(3, 4);
        assert!(!HandCode::from_hand(side, 0).belote);
    }

    #[test]
    fn coarsen_is_monotone_and_idempotent() {
        use CodeLevel::*;
        let levels = [Length, Trump, Shape, Tops, Full];
        for hand in random_hands(200) {
            let c = HandCode::from_hand(hand, 0);
            for (i, &lvl) in levels.iter().enumerate() {
                let a = c.coarsen(lvl);
                assert_eq!(a.coarsen(lvl), a, "coarsen doit être idempotent");
                // Grossir davantage ne peut que confondre : deux mains égales à
                // un niveau fin le restent à un niveau grossier.
                for &coarser in &levels[..i] {
                    assert_eq!(a.coarsen(coarser), c.coarsen(coarser));
                }
            }
            assert_eq!(c.coarsen(Full), c);
        }
    }

    /// Le niveau `Trump` ne dépend que du masque d'atout : 80 codes, comptés sur
    /// les 256 masques possibles. Les niveaux plus fins exigent d'énumérer les
    /// 1 820 803 classes — c'est `code_level_cardinalities`, ignoré par défaut.
    #[test]
    fn trump_level_has_80_codes() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for m in 0..256u32 {
            let len = m.count_ones() as usize;
            // Complète la main avec des cartes de côté quelconques.
            let mut hand = m;
            let mut need = 8 - len;
            let mut c = 8;
            while need > 0 {
                hand |= 1 << c;
                c += 1;
                need -= 1;
            }
            seen.insert(HandCode::from_hand(hand, 0).coarsen(CodeLevel::Trump));
        }
        assert_eq!(seen.len(), 80);
    }

    /// Cardinalités annoncées dans la doc du module. Parcourt tout l'espace à
    /// atout désigné : à lancer avec `--release --ignored`.
    #[test]
    #[ignore]
    fn code_level_cardinalities() {
        use std::collections::HashSet;
        let levels = [
            (CodeLevel::Length, 9),
            (CodeLevel::Trump, 80),
            (CodeLevel::Shape, 339),
            (CodeLevel::Tops, 5_277),
            (CodeLevel::Full, 6_654),
        ];
        let mut sets: Vec<HashSet<HandCode>> = vec![HashSet::new(); levels.len()];
        for id in 0..NUM_HAND_CLASSES_TRUMP {
            let code = HandCode::from_hand(hand_from_class_id_trump(id), 0);
            for (i, &(lvl, _)) in levels.iter().enumerate() {
                sets[i].insert(code.coarsen(lvl));
            }
        }
        for (i, &(lvl, expected)) in levels.iter().enumerate() {
            assert_eq!(sets[i].len(), expected, "niveau {lvl:?}");
        }
    }
}
