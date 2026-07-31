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
const TOTAL_PTS: i16 = 162;
/// Same total when the taker wins all 8 tricks — dix de der is worth 100 on a capot.
const CAPOT_PTS: i16 = 252;

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
    let defense = 1 - taker;

    let belote = belote_bonus(state);
    let total_belote = belote[0] + belote[1];

    let contract_value = state.contract.point_value() as i16;
    let coinche = state.contract.coinche;
    let is_capot_contract = state.contract.is_capot();

    // Trick points + dix de der (already included in state.points via resolve_trick)
    let taker_pts = state.points[taker] as i16;
    let defense_pts = state.points[defense] as i16;

    // Total points including belote for determining réussi/chute
    let taker_total = taker_pts + belote[taker];

    let mut scores = [0i16; 2];

    // Capot réalisé = 8 plis (whether or not capot was announced)
    let capot_realise = state.tricks_won[taker] == 8;

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
