/// Game replay storage and extraction.
///
/// A game is fully deterministic from `(dealer, hands, action_sequence)` — just ~66 bytes.
/// Storing raw replays lets us re-extract any task-specific data cheaply without re-playing
/// expensive DMC/NN inference.
///
/// Binary format `COLVGM02` — variable-length records:
/// ```text
/// Header (16 bytes):
///   magic: "COLVGM02" (8 bytes)
///   num_games: u64 LE (8 bytes)
///
/// Per game:
///   dealer: u8 (1 byte)
///   hands: [u32; 4] LE (16 bytes)
///   score_ns: u16 LE (2 bytes)   ← cumul de partie AVANT la donne
///   score_ew: u16 LE (2 bytes)
///   num_actions: u8 (1 byte)
///   actions: [u8; num_actions] (variable, typically 36-48)
/// ```
///
/// ## Pourquoi le score de partie est dans le format (2026-08-04)
///
/// `COLVGM01` ne portait que la donne, ce qui est exact tant qu'on tire des
/// donnes **indépendantes**. Dès qu'on les enchaîne en parties de 2000 points,
/// c'est faux : bid v6 lit une observation *score-aware* et annonce autrement à
/// 1800-600 qu'à 0-0. Un modèle entraîné sur un tel corpus verrait donc des
/// enchères qu'il ne peut pas expliquer — la variable qui les décide n'est pas
/// dans son entrée. **Ce n'est pas de l'information manquante, c'est de
/// l'entropie irréductible ajoutée**, et elle est chiffrée : +0,074 à +0,121 de
/// perplexité sur la tête d'enchère dès 1200 points d'écart, contre +0,0028 pour
/// diviser les paramètres du modèle par 3,3 (`bench_playgen_ppl`, 2026-08-04).
///
/// **Le score stocké est celui d'AVANT la donne**, parce que c'est celui que
/// l'annonceur a vu. Le score d'après est une conséquence de la donne, donc
/// le donner au modèle serait lui montrer la réponse.
///
/// **On stocke N-S et E-O bruts, pas « moi » et « l'adversaire »** : le fichier
/// porte le fait objectif, la mise en perspective appartient au tokeniseur, qui
/// seul connaît l'observateur. C'est la même erreur que celle épinglée côté
/// enchères — passer `(ns, ew)` là où la fonction attend `(mien, adverse)` fait
/// croire aux quatre sièges qu'ils sont dans le même camp.
///
/// Un `COLVGM01` se relit toujours, avec un score 0-0. Ce n'est pas une valeur
/// de repli arbitraire : **tous** les corpus existants ont été produits en
/// donnes indépendantes, donc 0-0 y est la vérité, pas une approximation.

use std::io::{self, Write};

use crate::belief_obs::{self, BELIEF_OBS_DIM, BELIEF_OBS_DIM_V3};
use crate::card::{self, card_rank, card_suit, card_suit_u8, EMPTY, HIGHER_TRUMP_MASK, TRUMP_STRENGTH};
use crate::trick;
use crate::dmc_obs::EnvTracking;
use crate::state::{GameState, Phase};

const MAGIC_V1: &[u8; 8] = b"COLVGM01";
const MAGIC: &[u8; 8] = b"COLVGM02";

/// A complete game replay: initial state + all actions taken.
pub struct GameReplay {
    pub dealer: u8,
    pub hands: [u32; 4],
    /// Cumul de partie **avant** cette donne, `[N-S, E-O]`. `0-0` pour une
    /// donne tirée indépendamment — ce qui est le cas de tout `COLVGM01`.
    pub score_ns: u16,
    pub score_ew: u16,
    pub actions: Vec<u8>,
}

impl GameReplay {
    /// Une donne tirée **indépendamment**, hors de toute partie : score 0-0.
    ///
    /// Existe pour que le cas « pas de contexte de partie » se dise, au lieu de
    /// s'obtenir par omission. Un générateur qui enchaîne des donnes et oublie
    /// de renseigner le score produirait un corpus faux *sans que rien ne le
    /// signale* : les enchères de fin de partie y seraient étiquetées 0-0.
    pub fn independent(dealer: u8, hands: [u32; 4], actions: Vec<u8>) -> Self {
        GameReplay { dealer, hands, score_ns: 0, score_ew: 0, actions }
    }
}

/// A single belief training sample extracted from a replay.
pub struct BeliefSample {
    pub obs: Vec<f32>,
    pub target: [u8; 32],
    pub mask: u32,
    /// V2: hard constraint mask (3 hidden players × 32 cards), 1.0 = impossible.
    pub hard_constraints: Option<[f32; 96]>,
    /// Number of completed tricks when this sample was taken (0-7).
    pub trick_idx: u8,
    /// Position within current trick (0-3).
    pub pos_in_trick: u8,
}

impl GameReplay {
    /// Write a collection of game replays to a COLVGM02 binary file.
    pub fn write_all(path: &str, replays: &[GameReplay]) -> io::Result<()> {
        // Estimate total size: header + per-game (22 + avg ~40 actions)
        let est_size = 16 + replays.len() * 64;
        let mut buf = Vec::with_capacity(est_size);

        // Header
        buf.write_all(MAGIC)?;
        buf.write_all(&(replays.len() as u64).to_le_bytes())?;

        // Per game
        for replay in replays {
            buf.push(replay.dealer);
            for &hand in &replay.hands {
                buf.write_all(&hand.to_le_bytes())?;
            }
            buf.write_all(&replay.score_ns.to_le_bytes())?;
            buf.write_all(&replay.score_ew.to_le_bytes())?;
            buf.push(replay.actions.len() as u8);
            buf.write_all(&replay.actions)?;
        }

        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(path, &buf)
    }

    /// Load all game replays from a COLVGM01 or COLVGM02 binary file.
    ///
    /// Les deux versions se relisent : un `COLVGM01` rend un score 0-0, ce qui
    /// est la vérité pour ces corpus (donnes indépendantes) et non un défaut.
    pub fn load_all(path: &str) -> io::Result<Vec<GameReplay>> {
        let data = std::fs::read(path)?;
        if data.len() < 16 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        let with_scores = match &data[..8] {
            m if m == MAGIC => true,
            m if m == MAGIC_V1 => false,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid magic: expected COLVGM01 or COLVGM02, got {other:?}"),
                ));
            }
        };

        let num_games = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        let mut replays = Vec::with_capacity(num_games);
        let mut pos = 16;

        for _ in 0..num_games {
            if pos >= data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated game data"));
            }

            let dealer = data[pos];
            pos += 1;

            if pos + 16 > data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated hands"));
            }
            let mut hands = [0u32; 4];
            for h in &mut hands {
                *h = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                pos += 4;
            }

            let (score_ns, score_ew) = if with_scores {
                if pos + 4 > data.len() {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated scores"));
                }
                let ns = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
                let ew = u16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap());
                pos += 4;
                (ns, ew)
            } else {
                (0, 0)
            };

            if pos >= data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated num_actions"));
            }
            let num_actions = data[pos] as usize;
            pos += 1;

            if pos + num_actions > data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated actions"));
            }
            let actions = data[pos..pos + num_actions].to_vec();
            pos += num_actions;

            replays.push(GameReplay { dealer, hands, score_ns, score_ew, actions });
        }

        Ok(replays)
    }

    /// Replay this game, calling `callback` at each step with
    /// `(&GameState, &EnvTracking, action)` before the action is applied.
    pub fn replay_with<F>(&self, mut callback: F)
    where
        F: FnMut(&GameState, &EnvTracking, u8),
    {
        let mut state = GameState::new(self.dealer, self.hands);
        let mut tracking = EnvTracking::new();
        tracking.dealer = self.dealer;

        for &action in &self.actions {
            callback(&state, &tracking, action);
            tracking.track_action(&state, action);
            state.step(action);
        }
    }

    /// Extract V2 belief samples with hard constraints.
    fn extract_belief_samples_v2_into(&self, samples: &mut Vec<BeliefSample>) {
        use crate::belief_obs::BELIEF_OBS_DIM_V2;

        let mut state = GameState::new(self.dealer, self.hands);
        let mut tracking = EnvTracking::new();
        tracking.dealer = self.dealer;

        let true_hands = self.hands;
        let mut obs_buf = vec![0.0f32; BELIEF_OBS_DIM_V2];
        let mut tracker = TrumpCeilingTracker::new();

        for &action in &self.actions {
            if state.phase == Phase::Playing {
                let observer = state.current_player();

                let hard_constraints = tracker.compute_hard_constraints(&state, observer);

                // Write V2 obs (304 floats) directly
                belief_obs::write_belief_observation_v2(
                    &mut obs_buf, 0, &state, &tracking, observer, &hard_constraints,
                );

                // Target: player-relative card locations (3-class: 0=left, 1=partner, 2=right)
                let mut target = [0u8; 32];
                for p in 0..4u8 {
                    for c in 0..32u8 {
                        if true_hands[p as usize] & (1u32 << c) != 0 {
                            let rel_p = (p + 4 - observer) % 4;
                            if rel_p > 0 {
                                target[c as usize] = rel_p - 1;
                            }
                            // rel_p == 0 means observer's own card, masked out — target irrelevant
                        }
                    }
                }

                // Unknown mask
                let observer_hand = state.hands[observer as usize];
                let mut played = state.played_cards;
                for j in 0..4 {
                    let c = state.current_trick[j];
                    if c != card::EMPTY {
                        played |= 1u32 << c;
                    }
                }
                let unknown_mask = !observer_hand & !played;

                if unknown_mask != 0 {
                    let completed_tricks = tracking.play_order.len() / 4;
                    samples.push(BeliefSample {
                        obs: obs_buf.clone(),
                        target,
                        mask: unknown_mask,
                        hard_constraints: None, // embedded in obs already
                        trick_idx: completed_tricks as u8,
                        pos_in_trick: state.trick_count,
                    });
                }

                // Update tracker after recording sample
                tracker.record_play(&state, observer, action);
            }

            tracking.track_action(&state, action);
            state.step(action);
        }
    }

    /// Extract V3 belief samples with temporal features.
    fn extract_belief_samples_v3_into(&self, samples: &mut Vec<BeliefSample>) {
        let mut state = GameState::new(self.dealer, self.hands);
        let mut tracking = EnvTracking::new();
        tracking.dealer = self.dealer;

        let true_hands = self.hands;
        let mut obs_buf = vec![0.0f32; BELIEF_OBS_DIM_V3];
        let mut tracker = TrumpCeilingTracker::new();

        // V3-specific tracking
        let mut trick_leads: Vec<u8> = Vec::with_capacity(8);
        let mut trick_winners: Vec<u8> = Vec::with_capacity(8);
        // suit_fail_counts[hidden_player_rel_idx][suit] = count of non-following
        // hidden_player_rel_idx: 0=left, 1=partner, 2=right (relative to current observer)
        // We track per absolute player and remap at observation time.
        let mut suit_fail_counts_abs = [[0u8; 4]; 4]; // [abs_player][suit]

        // Track current trick info for completion detection
        let mut prev_trick_count = 0usize;

        for &action in &self.actions {
            if state.phase == Phase::Playing {
                let observer = state.current_player();

                // Detect trick completion: if play_order grew by 4, a trick just completed
                let completed_tricks = tracking.play_order.len() / 4;
                while prev_trick_count < completed_tricks {
                    // A trick just completed. Extract lead suit and winner.
                    let trick_start = prev_trick_count * 4;
                    let lead_card_idx = tracking.play_order[trick_start];
                    let lead_suit_val = card_suit(lead_card_idx) as u8;
                    trick_leads.push(lead_suit_val);

                    // Find the trick cards and winner using trick_history
                    // We have state.trick_history[prev_trick_count] available
                    // But state has already stepped past, so we must use play_order
                    let mut trick_cards = [EMPTY; 4];
                    let first_player_seat = {
                        // The lead player for this trick: seat of first card in play_order
                        // We need to identify the lead seat from play tracking
                        // Since play_order records cards in order played, and trick_lead
                        // was already consumed, we use the tracking info.
                        // For past tricks, the trick_lead info is gone from state.
                        // But we tracked the played_by info.
                        let card_0 = tracking.play_order[trick_start];
                        let mut lead_seat = 0u8;
                        for p in 0..4u8 {
                            if tracking.played_by[p as usize] & (1u32 << card_0) != 0 {
                                lead_seat = p;
                                break;
                            }
                        }
                        lead_seat
                    };

                    // Reconstruct trick cards in seat order
                    for j in 0..4usize {
                        let card_j = tracking.play_order[trick_start + j];
                        let seat_j = (first_player_seat as usize + j) % 4;
                        trick_cards[seat_j] = card_j;
                    }

                    let winner = trick::trick_winner(&trick_cards, first_player_seat, &state.contract);
                    trick_winners.push(winner);

                    // Update suit failure counts for this completed trick
                    for j in 1..4usize {
                        let card_j = tracking.play_order[trick_start + j];
                        let card_j_suit = card_suit(card_j) as u8;
                        let player_j = (first_player_seat as usize + j) % 4;
                        if card_j_suit != lead_suit_val {
                            suit_fail_counts_abs[player_j][lead_suit_val as usize] =
                                suit_fail_counts_abs[player_j][lead_suit_val as usize].saturating_add(1);
                        }
                    }

                    prev_trick_count += 1;
                }

                let hard_constraints = tracker.compute_hard_constraints(&state, observer);

                // Build suit_fail_counts relative to observer
                let rel_seats = [
                    ((observer as usize + 1) % 4), // left
                    ((observer as usize + 2) % 4), // partner
                    ((observer as usize + 3) % 4), // right
                ];
                let mut suit_fail_rel = [[0u8; 4]; 3];
                for (i, &seat) in rel_seats.iter().enumerate() {
                    suit_fail_rel[i] = suit_fail_counts_abs[seat];
                }

                belief_obs::write_belief_observation_v3(
                    &mut obs_buf, 0, &state, &tracking, observer,
                    &hard_constraints,
                    &trick_leads,
                    &trick_winners,
                    &suit_fail_rel,
                );

                // Target: player-relative card locations (3-class: 0=left, 1=partner, 2=right)
                let mut target = [0u8; 32];
                for p in 0..4u8 {
                    for c in 0..32u8 {
                        if true_hands[p as usize] & (1u32 << c) != 0 {
                            let rel_p = (p + 4 - observer) % 4;
                            if rel_p > 0 {
                                target[c as usize] = rel_p - 1;
                            }
                            // rel_p == 0 means observer's own card, masked out — target irrelevant
                        }
                    }
                }

                // Unknown mask
                let observer_hand = state.hands[observer as usize];
                let mut played = state.played_cards;
                for j in 0..4 {
                    let c = state.current_trick[j];
                    if c != card::EMPTY {
                        played |= 1u32 << c;
                    }
                }
                let unknown_mask = !observer_hand & !played;

                if unknown_mask != 0 {
                    let completed_tricks_v3 = tracking.play_order.len() / 4;
                    samples.push(BeliefSample {
                        obs: obs_buf.clone(),
                        target,
                        mask: unknown_mask,
                        hard_constraints: None,
                        trick_idx: completed_tricks_v3 as u8,
                        pos_in_trick: state.trick_count,
                    });
                }

                tracker.record_play(&state, observer, action);
            }

            tracking.track_action(&state, action);
            state.step(action);
        }
    }

    /// Extract all belief training samples from this replay.
    ///
    /// At each playing-phase step, records:
    /// - Belief observation from the current player's perspective
    /// - Ground truth target: which player holds each card (player-relative)
    /// - Unknown mask: cards the observer doesn't know about
    fn extract_belief_samples_into(&self, samples: &mut Vec<BeliefSample>) {
        let mut state = GameState::new(self.dealer, self.hands);
        let mut tracking = EnvTracking::new();
        tracking.dealer = self.dealer;

        let true_hands = self.hands;
        let mut obs_buf = vec![0.0f32; BELIEF_OBS_DIM];

        for &action in &self.actions {
            if state.phase == Phase::Playing {
                let observer = state.current_player();

                belief_obs::write_belief_observation(&mut obs_buf, 0, &state, &tracking, observer);

                // Target: player-relative card locations (3-class: 0=left, 1=partner, 2=right)
                let mut target = [0u8; 32];
                for p in 0..4u8 {
                    for c in 0..32u8 {
                        if true_hands[p as usize] & (1u32 << c) != 0 {
                            let rel_p = (p + 4 - observer) % 4;
                            if rel_p > 0 {
                                target[c as usize] = rel_p - 1;
                            }
                            // rel_p == 0 means observer's own card, masked out — target irrelevant
                        }
                    }
                }

                // Unknown mask
                let observer_hand = state.hands[observer as usize];
                let mut played = state.played_cards;
                for j in 0..4 {
                    let c = state.current_trick[j];
                    if c != card::EMPTY {
                        played |= 1u32 << c;
                    }
                }
                let unknown_mask = !observer_hand & !played;

                if unknown_mask != 0 {
                    let completed_tricks_v1 = tracking.play_order.len() / 4;
                    samples.push(BeliefSample {
                        obs: obs_buf.clone(),
                        target,
                        mask: unknown_mask,
                        hard_constraints: None,
                        trick_idx: completed_tricks_v1 as u8,
                        pos_in_trick: state.trick_count,
                    });
                }
            }

            tracking.track_action(&state, action);
            state.step(action);
        }
    }
}

/// Tracks trump ceiling deductions during game replay.
///
/// Accumulates per-player bitmasks of trump ranks proven impossible
/// (from inability to overtrump), plus deduced trump voids (from not
/// trumping when required).
pub struct TrumpCeilingTracker {
    /// Per-player (absolute index) bitmask of impossible trump ranks.
    ceiling_mask: [u8; 4],
    /// Per-player deduced void mask (suit bitmask, same format as state.voids).
    /// Captures "void in trump" deduction that state.voids doesn't track.
    deduced_voids: [u8; 4],
}

impl TrumpCeilingTracker {
    pub fn new() -> Self {
        TrumpCeilingTracker {
            ceiling_mask: [0; 4],
            deduced_voids: [0; 4],
        }
    }

    /// Record a play action and update trump ceiling / deduced voids.
    /// `state` is the state BEFORE the action is applied.
    pub fn record_play(&mut self, state: &GameState, player: u8, card_played: u8) {
        if state.phase != Phase::Playing {
            return;
        }

        let card_s = card_suit_u8(card_played);
        let trump_suit = state.contract.trump;

        if state.trick_count > 0 {
            // Not the leader
            let lead_card = state.current_trick[state.trick_lead as usize];
            let lead_suit_idx = card_suit(lead_card) as u8;

            if card_s != lead_suit_idx {
                // Didn't follow lead suit

                if card_s != trump_suit {
                    // Didn't play trump either.
                    if !partner_is_master_before_play(state, player) {
                        if let Some(best_rank) = best_trump_rank_on_trick(state, trump_suit) {
                            // "Ne pisse pas": an opponent cut and the player discarded.
                            // Discarding is legal while holding lower trumps, so this
                            // only proves no trump above the best trump on the trick.
                            self.ceiling_mask[player as usize] |=
                                HIGHER_TRUMP_MASK[best_rank as usize];
                        } else {
                            // No trump on the trick: trumping was mandatory if
                            // possible -> void in trump.
                            self.deduced_voids[player as usize] |= 1 << trump_suit;
                        }
                    }
                }

                // Trump ceiling: played trump but couldn't overtrump.
                // Only a fact when partner isn't master — with partner master,
                // any card is legal, so undertrumping can be voluntary (e.g.
                // dumping the lowest trump while holding higher ones).
                if card_s == trump_suit && !partner_is_master_before_play(state, player) {
                    if let Some(best_rank) = best_trump_rank_on_trick(state, trump_suit) {
                        let played_strength = TRUMP_STRENGTH[card_rank(card_played) as usize];
                        let best_strength = TRUMP_STRENGTH[best_rank as usize];
                        if played_strength < best_strength {
                            self.ceiling_mask[player as usize] |=
                                HIGHER_TRUMP_MASK[best_rank as usize];
                        }
                    }
                }
            } else if lead_suit_idx == trump_suit {
                // Following trump suit - check overtrump constraint
                if let Some(best_rank) = best_trump_rank_on_trick(state, trump_suit) {
                    let played_strength = TRUMP_STRENGTH[card_rank(card_played) as usize];
                    let best_strength = TRUMP_STRENGTH[best_rank as usize];
                    if played_strength < best_strength {
                        self.ceiling_mask[player as usize] |=
                            HIGHER_TRUMP_MASK[best_rank as usize];
                    }
                }
            }
        }
    }

    /// Compute hard constraints: 96 floats (3 hidden players x 32 cards).
    /// Layout: [left_opp(32), partner(32), right_opp(32)] in player-relative order.
    /// 1.0 = impossible (player cannot hold this card).
    pub fn compute_hard_constraints(&self, state: &GameState, observer: u8) -> [f32; 96] {
        let mut hc = [0.0f32; 96];
        let seats = [
            ((observer as usize + 1) % 4),
            ((observer as usize + 2) % 4),
            ((observer as usize + 3) % 4),
        ];

        let observer_hand = state.hands[observer as usize];
        let mut played = state.played_cards;
        for j in 0..4 {
            let c = state.current_trick[j];
            if c != EMPTY {
                played |= 1u32 << c;
            }
        }
        let known = observer_hand | played;
        let trump_suit = state.contract.trump;

        for (i, &seat) in seats.iter().enumerate() {
            let combined_voids = state.voids[seat] | self.deduced_voids[seat];

            for card_idx in 0..32u32 {
                let offset = i * 32 + card_idx as usize;
                let suit = (card_idx / 8) as u8;
                let rank = (card_idx % 8) as u8;

                // Known cards (observer hand, played, current trick)
                if known & (1 << card_idx) != 0 {
                    hc[offset] = 1.0;
                    continue;
                }

                // Void constraint (game-tracked + deduced)
                if combined_voids & (1 << suit) != 0 {
                    hc[offset] = 1.0;
                    continue;
                }

                // Trump ceiling
                if suit == trump_suit && self.ceiling_mask[seat] & (1 << rank) != 0 {
                    hc[offset] = 1.0;
                }
            }
        }

        hc
    }
}

/// Find the best trump rank currently on the trick (helper for TrumpCeilingTracker).
fn best_trump_rank_on_trick(state: &GameState, trump_suit: u8) -> Option<u8> {
    let trump = card::Suit::from_u8(trump_suit);
    let mut best: Option<u8> = None;
    let mut best_strength = 0u8;

    for i in 0..state.trick_count {
        let seat = (state.trick_lead + i) % 4;
        let c = state.current_trick[seat as usize];
        if c == EMPTY {
            continue;
        }
        if card_suit(c) == trump {
            let rank = card_rank(c);
            let strength = TRUMP_STRENGTH[rank as usize];
            if best.is_none() || strength > best_strength {
                best_strength = strength;
                best = Some(rank);
            }
        }
    }

    best
}

/// Check if player's partner is currently winning the trick (before player plays).
fn partner_is_master_before_play(state: &GameState, player: u8) -> bool {
    if state.trick_count < 2 {
        return false;
    }

    let partner = GameState::partner(player);
    let lead = state.trick_lead;
    let lead_card = state.current_trick[lead as usize];
    let lead_suit = card_suit(lead_card);
    let trump_suit = state.contract.trump_suit();

    let mut best_seat = lead;
    let mut has_trump = false;
    let mut best_trump_strength = 0u8;
    let mut best_lead_rank = card_rank(lead_card);
    let mut best_lead_seat = lead;

    if lead_suit == trump_suit {
        has_trump = true;
        best_trump_strength = TRUMP_STRENGTH[card_rank(lead_card) as usize];
        best_seat = lead;
    }

    for i in 1..state.trick_count {
        let seat = (lead + i) % 4;
        let c = state.current_trick[seat as usize];
        if c == EMPTY {
            continue;
        }
        let suit = card_suit(c);

        if suit == trump_suit {
            let s = TRUMP_STRENGTH[card_rank(c) as usize];
            if !has_trump || s > best_trump_strength {
                best_trump_strength = s;
                best_seat = seat;
                has_trump = true;
            }
        } else if suit == lead_suit && !has_trump {
            let r = card_rank(c);
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

/// Extract all belief samples from a set of game replays (single-threaded).
pub fn extract_belief_samples(replays: &[GameReplay]) -> Vec<BeliefSample> {
    let est = replays.len() * 31;
    let mut samples = Vec::with_capacity(est);
    for replay in replays {
        replay.extract_belief_samples_into(&mut samples);
    }
    samples
}

/// Extract all belief samples using rayon for parallelism.
#[cfg(feature = "parallel")]
pub fn extract_belief_samples_parallel(replays: &[GameReplay]) -> Vec<BeliefSample> {
    use rayon::prelude::*;

    replays
        .par_iter()
        .fold(
            || Vec::with_capacity(1024),
            |mut acc, replay| {
                replay.extract_belief_samples_into(&mut acc);
                acc
            },
        )
        .reduce(
            || Vec::new(),
            |mut a, b| {
                a.extend(b);
                a
            },
        )
}

/// Extract V2 belief samples (with hard constraints) from game replays (single-threaded).
pub fn extract_belief_samples_v2(replays: &[GameReplay]) -> Vec<BeliefSample> {
    let est = replays.len() * 31;
    let mut samples = Vec::with_capacity(est);
    for replay in replays {
        replay.extract_belief_samples_v2_into(&mut samples);
    }
    samples
}

/// Extract V3 belief samples (with temporal features) from game replays (single-threaded).
pub fn extract_belief_samples_v3(replays: &[GameReplay]) -> Vec<BeliefSample> {
    let est = replays.len() * 31;
    let mut samples = Vec::with_capacity(est);
    for replay in replays {
        replay.extract_belief_samples_v3_into(&mut samples);
    }
    samples
}

/// Extract V3 belief samples using rayon for parallelism.
#[cfg(feature = "parallel")]
pub fn extract_belief_samples_v3_parallel(replays: &[GameReplay]) -> Vec<BeliefSample> {
    use rayon::prelude::*;

    replays
        .par_iter()
        .fold(
            || Vec::with_capacity(1024),
            |mut acc, replay| {
                replay.extract_belief_samples_v3_into(&mut acc);
                acc
            },
        )
        .reduce(
            || Vec::new(),
            |mut a, b| {
                a.extend(b);
                a
            },
        )
}

/// Extract V2 belief samples using rayon for parallelism.
#[cfg(feature = "parallel")]
pub fn extract_belief_samples_v2_parallel(replays: &[GameReplay]) -> Vec<BeliefSample> {
    use rayon::prelude::*;

    replays
        .par_iter()
        .fold(
            || Vec::with_capacity(1024),
            |mut acc, replay| {
                replay.extract_belief_samples_v2_into(&mut acc);
                acc
            },
        )
        .reduce(
            || Vec::new(),
            |mut a, b| {
                a.extend(b);
                a
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_write_load() {
        let replays = vec![
            GameReplay {
                dealer: 0,
                hands: [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000],
                score_ns: 1780,
                score_ew: 420,
                actions: vec![0, 0, 0, 5, 0, 0, 0], // 3 passes + bid + 3 passes
            },
            GameReplay::independent(
                2,
                [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000],
                vec![0, 0, 0, 0], // 4 passes (void deal)
            ),
        ];

        let path = "/tmp/test_game_replay_roundtrip.bin";
        GameReplay::write_all(path, &replays).unwrap();
        let loaded = GameReplay::load_all(path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].dealer, 0);
        assert_eq!(loaded[0].hands, [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000]);
        assert_eq!(loaded[0].actions, vec![0, 0, 0, 5, 0, 0, 0]);
        assert_eq!((loaded[0].score_ns, loaded[0].score_ew), (1780, 420));
        assert_eq!(loaded[1].dealer, 2);
        assert_eq!(loaded[1].actions, vec![0, 0, 0, 0]);
        assert_eq!((loaded[1].score_ns, loaded[1].score_ew), (0, 0));

        std::fs::remove_file(path).ok();
    }

    /// Un corpus `COLVGM01` doit continuer de se relire — les 9 M de donnes de
    /// `playgen_games_9M.bin` ne seront pas régénérées pour un champ de score
    /// qui y vaut zéro de toute façon.
    #[test]
    fn colvgm01_still_loads_with_zero_scores() {
        // Écrit un COLVGM01 à la main : deux donnes, sans champ de score.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(MAGIC_V1);
        buf.extend_from_slice(&2u64.to_le_bytes());
        for (dealer, actions) in [(1u8, vec![0u8, 0, 0, 9, 0, 0, 0]), (3u8, vec![0u8, 0, 0, 0])] {
            buf.push(dealer);
            for h in [0xFFu32, 0xFF00, 0xFF_0000, 0xFF00_0000] {
                buf.extend_from_slice(&h.to_le_bytes());
            }
            buf.push(actions.len() as u8);
            buf.extend_from_slice(&actions);
        }
        let path = "/tmp/test_game_replay_v1_compat.bin";
        std::fs::write(path, &buf).unwrap();

        let loaded = GameReplay::load_all(path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].dealer, 1);
        assert_eq!(loaded[0].actions, vec![0, 0, 0, 9, 0, 0, 0]);
        assert_eq!(loaded[1].dealer, 3);
        for r in &loaded {
            assert_eq!((r.score_ns, r.score_ew), (0, 0), "COLVGM01 = donnes indépendantes");
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_empty_file() {
        let path = "/tmp/test_game_replay_empty.bin";
        GameReplay::write_all(path, &[]).unwrap();
        let loaded = GameReplay::load_all(path).unwrap();
        assert_eq!(loaded.len(), 0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_replay_with_callback() {
        let replay = GameReplay::independent(
            0,
            [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000],
            vec![0, 0, 0, 0], // 4 passes
        );

        let mut step_count = 0;
        replay.replay_with(|_state, _tracking, _action| {
            step_count += 1;
        });
        assert_eq!(step_count, 4);
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_replay_produces_valid_states() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(123);

        for _ in 0..20 {
            let dealer = rng.gen_range(0..4u8);
            let mut state = GameState::deal_random(dealer, &mut rng);
            let hands = state.hands;
            let mut tracking = EnvTracking::new();
            tracking.dealer = dealer;
            let mut actions = Vec::new();

            while !state.is_terminal() {
                let legal = state.legal_actions();
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                let action = crate::rollout::select_nth_bit(legal, idx);
                actions.push(action);
                tracking.track_action(&state, action);
                state.step(action);
            }

            let replay = GameReplay::independent(dealer, hands, actions.clone());

            // Verify replay produces same terminal state
            let mut replayed_state = GameState::new(dealer, hands);
            let mut replayed_tracking = EnvTracking::new();
            replayed_tracking.dealer = dealer;

            for &a in &actions {
                replayed_tracking.track_action(&replayed_state, a);
                replayed_state.step(a);
            }
            assert!(replayed_state.is_terminal());
            assert_eq!(replayed_state.points, state.points);

            // Verify replay_with visits correct number of steps
            let mut count = 0;
            replay.replay_with(|_, _, _| count += 1);
            assert_eq!(count, actions.len());
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_extract_belief_samples() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(456);
        let mut replays = Vec::new();

        for _ in 0..10 {
            let dealer = rng.gen_range(0..4u8);
            let mut state = GameState::deal_random(dealer, &mut rng);
            let hands = state.hands;
            let mut actions = Vec::new();

            while !state.is_terminal() {
                let legal = state.legal_actions();
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                let action = crate::rollout::select_nth_bit(legal, idx);
                actions.push(action);
                state.step(action);
            }

            replays.push(GameReplay::independent(dealer, hands, actions));
        }

        let samples = extract_belief_samples(&replays);
        assert!(!samples.is_empty());

        for sample in &samples {
            assert_eq!(sample.obs.len(), BELIEF_OBS_DIM);
            assert_ne!(sample.mask, 0);
            for &t in &sample.target {
                assert!(t < 3);
            }
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_extract_belief_samples_v2() {
        use crate::belief_obs::BELIEF_OBS_DIM_V2;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(789);
        let mut replays = Vec::new();

        for _ in 0..10 {
            let dealer = rng.gen_range(0..4u8);
            let mut state = GameState::deal_random(dealer, &mut rng);
            let hands = state.hands;
            let mut actions = Vec::new();

            while !state.is_terminal() {
                let legal = state.legal_actions();
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                let action = crate::rollout::select_nth_bit(legal, idx);
                actions.push(action);
                state.step(action);
            }

            replays.push(GameReplay::independent(dealer, hands, actions));
        }

        let samples = extract_belief_samples_v2(&replays);
        assert!(!samples.is_empty());

        for sample in &samples {
            assert_eq!(sample.obs.len(), BELIEF_OBS_DIM_V2);
            assert_ne!(sample.mask, 0);
            for &t in &sample.target {
                assert!(t < 3);
            }
            // V2: hard constraints are embedded in obs[208..304]
            assert!(sample.hard_constraints.is_none(), "V2 embeds hard constraints in obs");
            // Check that embedded hard constraints have some impossible cards
            let hc = &sample.obs[208..304];
            let mut has_some_impossible = false;
            for &v in hc.iter() {
                assert!(v == 0.0 || v == 1.0, "hard constraint should be 0 or 1, got {}", v);
                if v == 1.0 {
                    has_some_impossible = true;
                }
            }
            assert!(has_some_impossible, "V2 hard constraints should have some impossible cards");
        }
    }
}
