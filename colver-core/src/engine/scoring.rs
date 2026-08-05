//! Deal scoring: "points faits + points demandés" mode (FFB section 9.1).
//!
//! A coinche multiplier applies to the **contract value only**, never to the base.
//!
//! **Contrat réussi** (taker points + belote >= contract value, or 8 tricks if capot):
//! - Standard: preneurs = their points + contrat + their belote; défense = their points + their belote.
//! - Contré: preneurs = `TOTAL_PTS` (`CAPOT_PTS` si capot réalisé) + contrat×2 + toute belote; défense 0.
//! - Surcontré: same with contrat×3.
//!
//! **Chute** (taker points < contract value, or fewer than 8 tricks for capot):
//! - Preneurs 0. Défense = `TOTAL_PTS` + contrat×mult + toute belote — the defense takes the
//!   contract *and* every card point of the deal, whatever the actual trick split.
//!
//! Contract value = bid × 10, or 250 for capot (a regular contract, not a flat bonus).
//!
//! Belote: on a chute, or under a coinche, all of it goes to the winning side.
//!
//! **Scores are exact — nothing is rounded** (changed 2026-07-31). The FFB rounds the marque to
//! the nearest 10 (§9.2), but this engine and the web score sheet both keep the raw sum, so a
//! donne marks the same number everywhere. This is what makes `TOTAL_PTS` visible: a chute now
//! marks 162 + contrat, not 160 + contrat.

use crate::state::*;

/// Every card point of a deal, dix de der (10) included.
pub const TOTAL_PTS: i16 = 162;
/// Same total when the taker wins all 8 tricks — dix de der is worth 100 on a capot.
pub const CAPOT_PTS: i16 = 252;

/// Result of scoring a deal.
#[derive(Debug, Clone, Copy)]
pub struct DealScore {
    pub scores: [i16; 2], // [NS, EW] — can be negative conceptually but we use signed for flexibility
}

/// Compute belote bonus per team (0 or 20).
fn belote_bonus(state: &GameState) -> [i16; 2] {
    let mut bonus = [0i16; 2];
    for team in 0..2 {
        if state.belote[team] == 2 {
            bonus[team] = 20;
        }
    }
    bonus
}

pub fn compute_deal_score(state: &GameState) -> DealScore {
    let taker = state.contract.team as usize;
    deal_score_from_card_points(
        &state.contract,
        [state.points[0] as i16, state.points[1] as i16],
        belote_bonus(state),
        state.tricks_won[taker] == 8,
    )
}

/// Le barème, à partir des seuls points cartes — sans état terminal.
///
/// Même arithmétique que [`compute_deal_score`], qui n'en est plus que le
/// lecteur d'état. Elle est publique parce qu'IS-DD en a besoin : le solveur DD
/// rend des **points cartes** par carte et par monde, or ce ne sont pas eux qui
/// décident une donne. L'écart entre les deux est une marche de `4V` au seuil du
/// contrat, et une pente nulle en dessous — voir [docs/classement_et_scoring.md].
///
/// `card_pts` : [N-S, E-O], de somme 162 (ou 252 sur capot réalisé).
/// `capot_realise` : le preneur a fait les 8 plis.
pub fn deal_score_from_card_points(
    contract: &Contract,
    card_pts: [i16; 2],
    belote: [i16; 2],
    capot_realise: bool,
) -> DealScore {
    let taker = contract.team as usize;
    let defense = 1 - taker;

    let total_belote = belote[0] + belote[1];

    let contract_value = contract.point_value() as i16;
    let coinche = contract.coinche;
    let is_capot_contract = contract.is_capot();

    let taker_pts = card_pts[taker];
    let defense_pts = card_pts[defense];

    // Total points including belote for determining réussi/chute
    let taker_total = taker_pts + belote[taker];

    let mut scores = [0i16; 2];

    if is_capot_contract {
        let reussi = capot_realise;

        if reussi {
            // Capot announced and made: taker_pts=252 (dix de der=100) + contract(250)
            match coinche {
                0 => {
                    // Preneurs: 252 + 250 + belote = 502. Défenseurs: their belote only.
                    scores[taker] = taker_pts + contract_value + belote[taker];
                    scores[defense] = belote[defense];
                }
                1 => {
                    // Contré réussi: 252 (capot réalisé base) + 250×2 + belote = 752
                    scores[taker] = CAPOT_PTS + contract_value * 2 + total_belote;
                    scores[defense] = 0;
                }
                2 => {
                    // Surcontré réussi: 252 (capot réalisé base) + 250×3 + belote = 1002
                    scores[taker] = CAPOT_PTS + contract_value * 3 + total_belote;
                    scores[defense] = 0;
                }
                _ => unreachable!(),
            }
        } else {
            // Capot announced but chute (< 8 tricks)
            scores[taker] = 0;
            match coinche {
                0 => scores[defense] = TOTAL_PTS + contract_value + total_belote,
                1 => scores[defense] = TOTAL_PTS + contract_value * 2 + total_belote,
                2 => scores[defense] = TOTAL_PTS + contract_value * 3 + total_belote,
                _ => unreachable!(),
            }
        }
    } else {
        // Non-capot contract
        let reussi = taker_total >= contract_value;

        if reussi {
            // Base for contré/surcontré réussi: every card point of the deal — 252 if the
            // taker took all 8 tricks (even when capot was not announced), 162 otherwise.
            let contre_base = if capot_realise { CAPOT_PTS } else { TOTAL_PTS };

            match coinche {
                0 => {
                    // Standard réussi: preneurs = their_points + contract + their_belote
                    // Défenseurs = their_points + their_belote
                    scores[taker] = taker_pts + contract_value + belote[taker];
                    scores[defense] = defense_pts + belote[defense];
                }
                1 => {
                    // Contré réussi: base + contract×2 + belote
                    scores[taker] = contre_base + contract_value * 2 + total_belote;
                    scores[defense] = 0;
                }
                2 => {
                    // Surcontré réussi: base + contract×3 + belote
                    scores[taker] = contre_base + contract_value * 3 + total_belote;
                    scores[defense] = 0;
                }
                _ => unreachable!(),
            }
        } else {
            // Chute: the defense takes the contract and all 162 card points, × multiplier on
            // the contract only, plus every belote.
            scores[taker] = 0;
            match coinche {
                0 => scores[defense] = TOTAL_PTS + contract_value + total_belote,
                1 => scores[defense] = TOTAL_PTS + contract_value * 2 + total_belote,
                2 => scores[defense] = TOTAL_PTS + contract_value * 3 + total_belote,
                _ => unreachable!(),
            }
        }
    }

    DealScore { scores }
}

// ── Du solveur DD au barème ──
//
// Le solveur rend des **points cartes**, et ce ne sont pas eux qui décident une
// donne : l'écart entre les deux est une marche de `4V` au seuil du contrat et
// une pente nulle en dessous. Toute lecture d'une valeur DD comme « ce que ce
// coup rapporte » doit donc passer par ici.
//
// Ces trois fonctions vivaient en privé dans `is_dd.rs`, seul appelant jusqu'au
// 2026-08-05 ; les pages d'analyse en ont besoin aussi, et le barème n'a le
// droit d'exister qu'une fois.

/// Total des points cartes de la donne, dix de der compris, connaissant le
/// total N-S final.
///
/// 162 normalement, 252 quand un camp fait capot (le dix de der y vaut 100).
///
/// **Une seule situation reste ambiguë** : `ns == 0` peut vouloir dire « E-O a
/// fait capot » (252) ou « N-S n'a ramassé que des plis à zéro point » (162) —
/// il y a 11 cartes sans valeur dans un jeu, donc le second cas existe. Les
/// plis déjà joués tranchent la plupart du temps : si N-S en a gagné un, le
/// capot est impossible. Sinon on retient le capot, de loin le plus fréquent
/// quand un camp finit à zéro. L'erreur résiduelle vaut 90 points de score sur
/// un écart qui en fait plusieurs centaines, et seulement dans ce cas-là.
#[inline]
pub fn total_card_points(state: &GameState, ns_card_pts: i16) -> i16 {
    if ns_card_pts == CAPOT_PTS {
        return CAPOT_PTS; // N-S a tout pris : sans ambiguïté
    }
    if ns_card_pts == 0 && state.tricks_won[0] == 0 {
        return CAPOT_PTS;
    }
    TOTAL_PTS
}

/// Belote/rebelote **finale**, par camp, depuis une donne en cours.
///
/// [`compute_deal_score`] lit `state.belote`, qui ne compte que ce qui a **déjà
/// été joué** (`check_belote` dans `apply_play`) et sous-estime donc en cours de
/// donne. Ici on veut la belote finale : elle est acquise dès qu'un joueur
/// détient Dame **et** Roi d'atout, puisqu'il finira forcément par jouer les
/// deux — d'où `hands | played_by`, les mains initiales reconstituées.
#[inline]
pub fn final_belote(
    hands: &[crate::card::CardSet; 4],
    played_by: &[crate::card::CardSet; 4],
    trump: u8,
) -> [i16; 2] {
    // Bits de rang : Dame = 4, Roi = 5 ; indice de carte = couleur × 8 + rang.
    let mask = (1u32 << (trump * 8 + 4)) | (1u32 << (trump * 8 + 5));
    let mut bonus = [0i16; 2];
    for p in 0..4usize {
        if (hands[p] | played_by[p]) & mask == mask {
            bonus[p % 2] = 20;
        }
    }
    bonus
}

/// Écart de score marqué **N-S − E-O** correspondant à un total N-S en points
/// cartes — typiquement une valeur rendue par le solveur DD.
///
/// C'est la conversion complète : réussite ou chute, valeur du contrat,
/// contré/surcontré, capot, dix de der, belote. Le résultat est signé et vit sur
/// une échelle de ±500 (davantage sous coinche), sans rapport avec les 0-252 des
/// points cartes — **les deux ne se soustraient pas**.
///
/// `played_by[seat]` = les cartes que ce siège a déjà posées, nécessaires pour
/// reconstituer la belote finale.
#[inline]
pub fn deal_score_delta(
    state: &GameState,
    played_by: &[crate::card::CardSet; 4],
    ns_card_pts: i16,
) -> i16 {
    let total = total_card_points(state, ns_card_pts);
    let card = [ns_card_pts, total - ns_card_pts];
    let taker = state.contract.team as usize;
    let belote = final_belote(&state.hands, played_by, state.contract.trump);
    let s = deal_score_from_card_points(
        &state.contract,
        card,
        belote,
        card[taker] == CAPOT_PTS,
    );
    s.scores[0] - s.scores[1]
}

/// Le contrat est-il tenu, pour un total N-S en points cartes donné ?
///
/// C'est le seul prédicat qui sépare les deux régimes du barème, donc **le seul
/// qui distingue une erreur qui coûte des points d'une erreur qui renverse la
/// donne**. Il ne se déduit pas de l'écart rendu par [`deal_score_delta`] : un
/// écart négatif peut aussi bien être une chute du preneur N-S qu'un contrat
/// tenu par E-O.
///
/// La belote compte : `scoring` l'ajoute au total du preneur pour décider de la
/// réussite, donc elle **déplace le seuil** au lieu d'ajouter 20 points au bout.
#[inline]
pub fn contract_made(
    state: &GameState,
    played_by: &[crate::card::CardSet; 4],
    ns_card_pts: i16,
) -> bool {
    let total = total_card_points(state, ns_card_pts);
    let taker = state.contract.team as usize;
    let card = [ns_card_pts, total - ns_card_pts];
    if state.contract.is_capot() {
        return card[taker] == CAPOT_PTS;
    }
    let belote = final_belote(&state.hands, played_by, state.contract.trump);
    card[taker] + belote[taker] >= state.contract.point_value() as i16
}

/// Convert deal score to rewards for RL: team 0 gets positive/negative based on comparison.
pub fn deal_rewards(state: &GameState) -> [f32; 2] {
    if state.contract.value == 0 {
        // No contract was made (4 passes) → no points
        return [0.0, 0.0];
    }
    let score = compute_deal_score(state);
    // Simple reward: normalized score difference
    [score.scores[0] as f32, score.scores[1] as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scored_state(
        taker_team: u8,
        bid_value: u8,
        trump: u8,
        coinche: u8,
        taker_pts: u8,
        defense_pts: u8,
        taker_tricks: u8,
        taker_belote: u8,
        defense_belote: u8,
    ) -> GameState {
        let hands = [0; 4]; // empty hands (done)
        let mut state = GameState::new(0, hands);
        state.phase = Phase::Done;
        state.contract = Contract {
            trump,
            value: bid_value,
            team: taker_team,
            coinche,
        };
        let defense = 1 - taker_team;
        state.points[taker_team as usize] = taker_pts;
        state.points[defense as usize] = defense_pts;
        state.tricks_won[taker_team as usize] = taker_tricks;
        state.tricks_won[defense as usize] = 8 - taker_tricks;
        state.belote[taker_team as usize] = taker_belote;
        state.belote[defense as usize] = defense_belote;
        state
    }

    /// Un état de jeu nu, sans plis joués : ce qu'il faut à `deal_score_delta`.
    fn make_playing_state(taker_team: u8, bid_value: u8, trump: u8, coinche: u8) -> GameState {
        let mut state = GameState::new(0, [0; 4]);
        state.phase = Phase::Playing;
        state.contract = Contract {
            trump,
            value: bid_value,
            team: taker_team,
            coinche,
        };
        state
    }

    #[test]
    fn deal_score_is_flat_below_the_contract() {
        // La propriété qui rend les points cartes trompeurs : sous le seuil, ils
        // ne valent RIEN. La défense qui fait chuter encaisse 162 + contrat quel
        // que soit le partage réel des plis, donc un coup qui « perd 30 points
        // cartes » sans remettre le contrat en cause n'a rien coûté.
        let state = make_playing_state(0, 10, 1, 0); // 100♥ par N-S
        let played = [0u32; 4];
        let floor = deal_score_delta(&state, &played, 0);
        for ns in 1..100i16 {
            assert_eq!(
                deal_score_delta(&state, &played, ns),
                floor,
                "le score bouge à {ns} points cartes, sous un contrat à 100"
            );
        }
    }

    #[test]
    fn deal_score_step_at_the_threshold_is_four_times_the_contract() {
        // Et la contrepartie : au seuil, UN point carte vaut `4V`. C'est ce qui
        // fait qu'un coup noté « −3 points cartes » peut coûter 400 points de
        // score. Vérifié sur tous les paliers, contrat normal non contré.
        for contract in [80i16, 90, 100, 110, 120, 130, 140, 150, 160] {
            let value = (contract / 10) as u8;
            let state = make_playing_state(0, value, 1, 0);
            let played = [0u32; 4];
            let below = deal_score_delta(&state, &played, contract - 1);
            let at = deal_score_delta(&state, &played, contract);
            assert_eq!(
                at - below,
                4 * contract,
                "la marche d'un contrat à {contract} devrait valoir 4V"
            );
        }
    }

    #[test]
    fn deal_score_delta_agrees_with_the_terminal_scoring() {
        // `deal_score_delta` sert à lire une valeur DD *avant* la fin de la
        // donne ; elle doit rendre exactement ce que `compute_deal_score` dira
        // une fois la donne finie. Si les deux divergent, une page d'analyse
        // annonce un écart que la feuille de marque contredira.
        // `(points cartes du preneur, plis du preneur)` **cohérents entre eux** :
        // un capot réalisé vaut 252 points cartes et non 162, et zéro pli vaut
        // zéro point. Une première version de ce test passait 162 points avec 8
        // plis — un état qu'aucune donne ne produit, et sur lequel
        // `total_card_points` n'a aucune raison de tomber juste.
        for contract in [80i16, 100, 130, 160] {
            for taker_team in 0..2u8 {
                for coinche in 0..3u8 {
                    for (taker_pts, taker_tricks) in
                        [(0i16, 0u8), (40, 2), (contract - 1, 4), (contract, 4), (152, 7), (252, 8)]
                    {
                        let value = (contract / 10) as u8;
                        let total = if taker_pts == 252 || taker_pts == 0 {
                            CAPOT_PTS
                        } else {
                            TOTAL_PTS
                        };
                        let defense_pts = total - taker_pts;
                        let terminal = make_scored_state(
                            taker_team,
                            value,
                            1,
                            coinche,
                            taker_pts as u8,
                            defense_pts as u8,
                            taker_tricks,
                            0,
                            0,
                        );
                        let expected = {
                            let s = compute_deal_score(&terminal);
                            s.scores[0] - s.scores[1]
                        };
                        let mut playing = make_playing_state(taker_team, value, 1, coinche);
                        playing.tricks_won = terminal.tricks_won;
                        let ns_pts = if taker_team == 0 { taker_pts } else { defense_pts };
                        assert_eq!(
                            deal_score_delta(&playing, &[0u32; 4], ns_pts),
                            expected,
                            "contrat {contract}, preneur {taker_team}, coinche {coinche}, \
                             preneur à {taker_pts} points cartes"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn final_belote_sees_a_holding_the_state_has_not_scored_yet() {
        // `state.belote` ne compte que ce qui a déjà été joué ; `final_belote`
        // regarde `hands | played_by`. Les deux moitiés doivent compter : la
        // carte encore en main, et celle déjà posée.
        let trump = 1u8; // ♥ → Dame = 12, Roi = 13
        let (dame, roi) = (trump * 8 + 4, trump * 8 + 5);

        let mut hands = [0u32; 4];
        hands[1] = (1 << dame) | (1 << roi); // Est tient les deux
        assert_eq!(final_belote(&hands, &[0; 4], trump), [0, 20]);

        // La Dame est déjà tombée : la belote reste acquise à Est-Ouest.
        let mut hands = [0u32; 4];
        hands[1] = 1 << roi;
        let mut played = [0u32; 4];
        played[1] = 1 << dame;
        assert_eq!(final_belote(&hands, &played, trump), [0, 20]);

        // Partagée entre les deux partenaires : ce n'est pas une belote.
        let mut hands = [0u32; 4];
        hands[1] = 1 << dame;
        hands[3] = 1 << roi;
        assert_eq!(final_belote(&hands, &[0; 4], trump), [0, 0]);
    }

    #[test]
    fn test_scores_are_exact_never_rounded() {
        // Le moteur ne marque plus à la dizaine (2026-07-31). Une donne dont la somme ne tombe
        // pas rond reste telle quelle des deux côtés — c'est ce test qui casse si un arrondi
        // revient quelque part.
        let state = make_scored_state(0, 8, 1, 0, 87, 75, 5, 0, 0);
        let score = compute_deal_score(&state);
        assert_eq!(score.scores[0], 167); // 87 + 80, et non 170
        assert_eq!(score.scores[1], 75); //  et non 80
    }

    #[test]
    fn test_chute_marks_the_whole_162() {
        // La défense prend tous les points cartes de la donne. Sans arrondi ces 162 se voient
        // enfin dans la marque : 162 + contrat×mult + belote, au point près.
        for contract in [80i16, 90, 100, 110, 120, 130, 140, 150, 160, 250] {
            let value = if contract == 250 { 25u8 } else { (contract / 10) as u8 };
            for mult in 1..=3 {
                for belote in [0i16, 20, 40] {
                    let taker_belote = if belote >= 20 { 2 } else { 0 };
                    let defense_belote = if belote == 40 { 2 } else { 0 };
                    let state = make_scored_state(
                        0,
                        value,
                        1,
                        (mult - 1) as u8,
                        20,
                        142,
                        1,
                        taker_belote,
                        defense_belote,
                    );
                    let score = compute_deal_score(&state);
                    assert_eq!(score.scores[0], 0);
                    assert_eq!(score.scores[1], TOTAL_PTS + contract * mult + belote);
                }
            }
        }
    }

    #[test]
    fn test_standard_reussi() {
        // Taker (NS=0) bid 80 Hearts, scored 92 points (no belote), defense got 70 pts
        // (92 + 70 = 162 total)
        let state = make_scored_state(0, 8, 1, 0, 92, 70, 5, 0, 0);
        let score = compute_deal_score(&state);
        // Preneurs: 92 + 80 + 0 = 172. Défense: 70 + 0 = 70
        assert_eq!(score.scores[0], 172);
        assert_eq!(score.scores[1], 70);
    }

    #[test]
    fn test_standard_chute() {
        // Taker (NS=0) bid 100 Spades, scored 82 points, defense 80
        let state = make_scored_state(0, 10, 0, 0, 82, 80, 4, 0, 0);
        let score = compute_deal_score(&state);
        // Chute: preneurs get 0, defense gets 162 + 100 + 0 = 262
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 262);
    }

    #[test]
    fn test_contre_reussi() {
        // Taker (EW=1) bid 80 Hearts contré, scored 100 (not capot)
        let state = make_scored_state(1, 8, 1, 1, 100, 62, 5, 0, 0);
        let score = compute_deal_score(&state);
        // Contré réussi: 162 + 80×2 + 0 = 322
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 322);
    }

    #[test]
    fn test_contre_chute() {
        // Taker (NS=0) bid 100 Spades contré, scored 90
        let state = make_scored_state(0, 10, 0, 1, 90, 72, 4, 0, 0);
        let score = compute_deal_score(&state);
        // Contré chute: 162 + 100×2 + 0 = 362
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 362);
    }

    #[test]
    fn test_surcontre_reussi() {
        // Taker (NS=0) bid 80 surcontré, scored 100 (not capot)
        let state = make_scored_state(0, 8, 1, 2, 100, 62, 6, 0, 0);
        let score = compute_deal_score(&state);
        // Surcontré réussi: 162 + 80×3 + 0 = 402
        assert_eq!(score.scores[0], 402);
        assert_eq!(score.scores[1], 0);
    }

    #[test]
    fn test_capot_reussi() {
        // Taker (NS=0) bid capot Hearts, won all 8 tricks, scored 252
        let state = make_scored_state(0, 25, 1, 0, 252, 0, 8, 0, 0);
        let score = compute_deal_score(&state);
        // Capot réussi: 252 + 250 = 502
        assert_eq!(score.scores[0], 502);
        assert_eq!(score.scores[1], 0);
    }

    #[test]
    fn test_capot_chute() {
        // Taker bid capot, only won 7 tricks
        let state = make_scored_state(0, 25, 1, 0, 140, 22, 7, 0, 0);
        let score = compute_deal_score(&state);
        // Capot chute: 162 + 250 = 412
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 412);
    }

    #[test]
    fn test_capot_chute_contre() {
        let state = make_scored_state(0, 25, 1, 1, 140, 22, 7, 0, 0);
        let score = compute_deal_score(&state);
        // Capot contré chute: 162 + 250×2 = 662
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 662);
    }

    #[test]
    fn test_capot_chute_surcontre() {
        let state = make_scored_state(0, 25, 1, 2, 140, 22, 7, 0, 0);
        let score = compute_deal_score(&state);
        // Capot surcontré chute: 162 + 250×3 = 912
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 912);
    }

    #[test]
    fn test_capot_chute_with_belote() {
        // Taker bid capot, chute, taker has belote → belote prenable
        let state = make_scored_state(0, 25, 1, 0, 140, 22, 7, 2, 0);
        let score = compute_deal_score(&state);
        // Capot chute: 162 + 250 + 20 (belote) = 432
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 432);
    }

    #[test]
    fn test_capot_reussi_with_defense_belote() {
        // Taker bid capot, won all 8 tricks, defense has belote
        let state = make_scored_state(0, 25, 1, 0, 252, 0, 8, 0, 2);
        let score = compute_deal_score(&state);
        // Capot réussi: 252 + 250 + 0 (taker belote) = 502
        // Defense keeps their belote: 20
        assert_eq!(score.scores[0], 502);
        assert_eq!(score.scores[1], 20);
    }

    #[test]
    fn test_capot_contre_reussi() {
        // EW bid capot contré, won all 8 tricks
        let state = make_scored_state(1, 25, 0, 1, 252, 0, 8, 0, 0);
        let score = compute_deal_score(&state);
        // Capot contré réussi: 252 + 250×2 = 752
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 752);
    }

    #[test]
    fn test_capot_surcontre_reussi() {
        let state = make_scored_state(0, 25, 0, 2, 252, 0, 8, 0, 0);
        let score = compute_deal_score(&state);
        // Capot surcontré réussi: 252 + 250×3 = 1002
        assert_eq!(score.scores[0], 1002);
        assert_eq!(score.scores[1], 0);
    }

    #[test]
    fn test_belote_with_reussi() {
        // Taker bid 80, scored 92, has belote (20)
        // Total = 92 + 20 = 112 >= 80 → réussi
        let state = make_scored_state(0, 8, 1, 0, 92, 70, 5, 2, 0);
        let score = compute_deal_score(&state);
        // Preneurs: 92 + 80 + 20 = 192. Defense: 70 + 0 = 70
        assert_eq!(score.scores[0], 192);
        assert_eq!(score.scores[1], 70);
    }

    #[test]
    fn test_belote_saves_contract() {
        // Taker bid 100, scored 88, has belote (20). Total = 88+20 = 108 >= 100 → réussi!
        let state = make_scored_state(0, 10, 1, 0, 88, 74, 4, 2, 0);
        let score = compute_deal_score(&state);
        // réussi: 88 + 100 + 20 = 208
        assert_eq!(score.scores[0], 208);
        assert_eq!(score.scores[1], 74);
    }

    #[test]
    fn test_chute_belote_goes_to_defense() {
        // Taker bid 130, scored 82, no belote. Defense has belote (20).
        // Taker total = 82 < 130 → chute
        let state = make_scored_state(0, 13, 1, 0, 82, 80, 4, 0, 2);
        let score = compute_deal_score(&state);
        // Chute: preneurs 0, defense 162 + 130 + 20 = 312
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 312);
    }
}
