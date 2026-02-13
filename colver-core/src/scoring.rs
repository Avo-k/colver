use crate::state::*;

/// Compute deal scores for both teams: [NS_score, EW_score].
///
/// Implements "points faits + points demandés" scoring mode.
///
/// Rules (from FFB official rules section 10.2):
///
/// **Contract réussi** (taker points >= contract value, or 8 tricks if capot):
/// - Standard: Preneurs get their_points + contract_value + belote. Défense gets their_points + belote.
/// - Contré: Preneurs get 320 + contract×2 + belote(both). Défense gets 0.
/// - Surcontré: Preneurs get 640 + contract×4 + belote(both). Défense gets 0.
/// - Capot standard: Preneurs get 500 + belote. Défense gets 0 + their_belote.
///   (Well, capot means defense has 0 points, so defense gets their belote only if they have it—
///    but capot means 8 tricks so defense can't have belote unless they held Q+K of trump
///    and played both cards... in 0 tricks? Not possible. So defense belote = 0 if capot réussi.)
/// - Capot contré: 1000 + belote(both). Défense 0.
/// - Capot surcontré: 2000 + belote(both). Défense 0.
///
/// **Chute** (taker points < contract value, or fewer than 8 tricks for capot):
/// - Standard: Preneurs get 0 + their_belote. Défense gets 160 + contract + belote(opponent's if applies).
///   Actually per rules: "belote annoncée par une équipe peut changer de camp"
///   - Preneurs get 0. Défense gets 160 + contract + all belote (both teams').
/// - Contré: Preneurs get 0. Défense gets 320 + contract×2 + all belote.
/// - Surcontré: Preneurs get 0. Défense gets 640 + contract×4 + all belote.
/// - Capot chute standard: Preneurs get their_points + belote. Défense gets their_points + 250 + belote.
///   Wait, this is tricky. Let me re-examine.
///
/// Let me simplify based on the rules:
/// - Contract value = bid × 10 (or 250 for capot)
/// - Points include: trick points + dix de der + belote
///
/// For non-capot contracts:
///   réussi: preneurs_score >= contract_value (including belote)
///   chute: preneurs_score < contract_value
///
/// For capot contracts:
///   réussi: preneurs have 8 tricks
///   chute: preneurs have < 8 tricks

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

/// Round to nearest 10 (85→90, 84→80).
fn round10(x: i16) -> i16 {
    (x + 5) / 10 * 10
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

    if is_capot_contract {
        let capot_reussi = state.tricks_won[taker] == 8;

        if capot_reussi {
            match coinche {
                0 => {
                    // Capot réussi standard: preneurs get 500 + belote(both)
                    scores[taker] = round10(500 + total_belote);
                    scores[defense] = 0;
                }
                1 => {
                    // Capot contré réussi: 1000 + belote(both)
                    scores[taker] = round10(1000 + total_belote);
                    scores[defense] = 0;
                }
                2 => {
                    // Capot surcontré réussi: 2000 + belote(both)
                    scores[taker] = round10(2000 + total_belote);
                    scores[defense] = 0;
                }
                _ => unreachable!(),
            }
        } else {
            // Capot chute
            match coinche {
                0 => {
                    // Capot chute standard: preneurs get 0, defense gets 160 + 250 + all belote
                    scores[taker] = 0;
                    scores[defense] = round10(160 + 250 + total_belote);
                }
                1 => {
                    // Capot contré chute: preneurs get 0, defense gets 320 + 500 + all belote
                    scores[taker] = 0;
                    scores[defense] = round10(320 + 500 + total_belote);
                }
                2 => {
                    // Capot surcontré chute: preneurs get 0, defense gets 640 + 1000 + all belote
                    scores[taker] = 0;
                    scores[defense] = round10(640 + 1000 + total_belote);
                }
                _ => unreachable!(),
            }
        }
    } else {
        // Non-capot contract
        let reussi = taker_total >= contract_value;

        if reussi {
            match coinche {
                0 => {
                    // Standard réussi: preneurs get their_points + contract + their_belote
                    // Defense gets their_points + their_belote
                    scores[taker] = round10(taker_pts + contract_value + belote[taker]);
                    scores[defense] = round10(defense_pts + belote[defense]);
                }
                1 => {
                    // Contré réussi: preneurs get 320 + contract×2 + all belote, defense 0
                    scores[taker] = round10(320 + contract_value * 2 + total_belote);
                    scores[defense] = 0;
                }
                2 => {
                    // Surcontré réussi: preneurs get 640 + contract×4 + all belote, defense 0
                    scores[taker] = round10(640 + contract_value * 4 + total_belote);
                    scores[defense] = 0;
                }
                _ => unreachable!(),
            }
        } else {
            // Chute
            match coinche {
                0 => {
                    // Standard chute: preneurs 0, defense gets 160 + contract + all belote
                    scores[taker] = 0;
                    scores[defense] = round10(160 + contract_value + total_belote);
                }
                1 => {
                    // Contré chute: preneurs 0, defense gets 320 + contract×2 + all belote
                    scores[taker] = 0;
                    scores[defense] = round10(320 + contract_value * 2 + total_belote);
                }
                2 => {
                    // Surcontré chute: preneurs 0, defense gets 640 + contract×4 + all belote
                    scores[taker] = 0;
                    scores[defense] = round10(640 + contract_value * 4 + total_belote);
                }
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
    fn test_round10() {
        assert_eq!(round10(85), 90);
        assert_eq!(round10(84), 80);
        assert_eq!(round10(80), 80);
        assert_eq!(round10(90), 90);
        assert_eq!(round10(162), 160);
        assert_eq!(round10(155), 160);
        assert_eq!(round10(0), 0);
    }

    #[test]
    fn test_standard_reussi() {
        // Taker (NS=0) bid 80 Hearts, scored 92 points (no belote), defense got 70 pts
        // (92 + 70 = 162 total)
        let state = make_scored_state(0, 8, 1, 0, 92, 70, 5, 0, 0);
        let score = compute_deal_score(&state);
        // Preneurs: round10(92 + 80 + 0) = round10(172) = 170
        // Défense: round10(70 + 0) = 70
        assert_eq!(score.scores[0], 170);
        assert_eq!(score.scores[1], 70);
    }

    #[test]
    fn test_standard_chute() {
        // Taker (NS=0) bid 100 Spades, scored 82 points, defense 80
        let state = make_scored_state(0, 10, 0, 0, 82, 80, 4, 0, 0);
        let score = compute_deal_score(&state);
        // Chute: preneurs get 0, defense gets round10(160 + 100 + 0) = round10(260) = 260
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 260);
    }

    #[test]
    fn test_contre_reussi() {
        // Taker (EW=1) bid 80 Hearts contré, scored 100
        let state = make_scored_state(1, 8, 1, 1, 100, 62, 5, 0, 0);
        let score = compute_deal_score(&state);
        // Contré réussi: preneurs get round10(320 + 80*2 + 0) = round10(480) = 480
        // Defense: 0
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 480);
    }

    #[test]
    fn test_contre_chute() {
        // Taker (NS=0) bid 100 Spades contré, scored 90
        let state = make_scored_state(0, 10, 0, 1, 90, 72, 4, 0, 0);
        let score = compute_deal_score(&state);
        // Contré chute: preneurs 0, defense round10(320 + 100*2 + 0) = round10(520) = 520
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 520);
    }

    #[test]
    fn test_surcontre_reussi() {
        // Taker (NS=0) bid 80 surcontré, scored 100
        let state = make_scored_state(0, 8, 1, 2, 100, 62, 6, 0, 0);
        let score = compute_deal_score(&state);
        // Surcontré réussi: round10(640 + 80*4 + 0) = round10(960) = 960
        assert_eq!(score.scores[0], 960);
        assert_eq!(score.scores[1], 0);
    }

    #[test]
    fn test_capot_reussi() {
        // Taker (NS=0) bid capot Hearts, won all 8 tricks, scored 252
        let state = make_scored_state(0, 25, 1, 0, 252, 0, 8, 0, 0);
        let score = compute_deal_score(&state);
        // Capot réussi standard: round10(500 + 0) = 500
        assert_eq!(score.scores[0], 500);
        assert_eq!(score.scores[1], 0);
    }

    #[test]
    fn test_capot_chute() {
        // Taker bid capot, only won 7 tricks
        let state = make_scored_state(0, 25, 1, 0, 140, 22, 7, 0, 0);
        let score = compute_deal_score(&state);
        // Capot chute standard: preneurs 0, defense round10(160 + 250 + 0) = round10(410) = 410
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 410);
    }

    #[test]
    fn test_capot_contre_reussi() {
        let state = make_scored_state(1, 25, 0, 1, 252, 0, 8, 0, 0);
        let score = compute_deal_score(&state);
        // 1000 + 0 belote = 1000
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 1000);
    }

    #[test]
    fn test_capot_surcontre_reussi() {
        let state = make_scored_state(0, 25, 0, 2, 252, 0, 8, 0, 0);
        let score = compute_deal_score(&state);
        assert_eq!(score.scores[0], 2000);
        assert_eq!(score.scores[1], 0);
    }

    #[test]
    fn test_belote_with_reussi() {
        // Taker bid 80, scored 92, has belote (20)
        // Total = 92 + 20 = 112 >= 80 → réussi
        let state = make_scored_state(0, 8, 1, 0, 92, 70, 5, 2, 0);
        let score = compute_deal_score(&state);
        // Preneurs: round10(92 + 80 + 20) = round10(192) = 190
        // Defense: round10(70 + 0) = 70
        assert_eq!(score.scores[0], 190);
        assert_eq!(score.scores[1], 70);
    }

    #[test]
    fn test_belote_saves_contract() {
        // Taker bid 100, scored 88, has belote (20). Total = 88+20 = 108 >= 100 → réussi!
        let state = make_scored_state(0, 10, 1, 0, 88, 74, 4, 2, 0);
        let score = compute_deal_score(&state);
        // réussi: round10(88 + 100 + 20) = round10(208) = 210
        assert_eq!(score.scores[0], 210);
        assert_eq!(score.scores[1], round10(74));
    }

    #[test]
    fn test_chute_belote_goes_to_defense() {
        // Taker bid 130, scored 82, no belote. Defense has belote (20).
        // Taker total = 82 < 130 → chute
        let state = make_scored_state(0, 13, 1, 0, 82, 80, 4, 0, 2);
        let score = compute_deal_score(&state);
        // Chute: preneurs 0, defense round10(160 + 130 + 20) = round10(310) = 310
        assert_eq!(score.scores[0], 0);
        assert_eq!(score.scores[1], 310);
    }
}
