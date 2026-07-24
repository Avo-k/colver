//! Driving four [`Player`]s through a deal or a match.
//!
//! This is the loop that used to be copied into every tournament binary and
//! the web server, each copy re-deriving who observes what and when. There is
//! exactly one rule and it is easy to get wrong: **every player observes every
//! action**, including its own and including the auction, because that is how
//! belief states, world samplers and credibility judges stay in sync with the
//! game. A copy that forgets one `observe` produces an agent that plays
//! slightly worse for reasons nobody can find.

use rand::rngs::StdRng;
use rand::Rng;

use crate::agent::{AgentError, MatchContext, Player};
use crate::state::{GameState, Phase};

/// Points scored by each team on one deal, `[NS, EW]`.
pub type DealScore = [i32; 2];

/// First team to this many cumulative points wins a match.
pub const MATCH_TARGET: i32 = 2000;

/// Play one deal to completion, from a state that has just been dealt.
///
/// `ctx` carries the running match score in and the deal's action tracking
/// out. Returns the deal score; a passed-out deal scores `[0, 0]`.
pub fn play_deal(
    state: &mut GameState,
    players: &mut [Box<dyn Player>; 4],
    ctx: &mut MatchContext,
) -> Result<DealScore, AgentError> {
    ctx.reset_deal(state.dealer);
    for p in players.iter_mut() {
        p.init_deal(state);
    }

    while !state.is_terminal() {
        let seat = state.current_player();
        let before = *state;
        let action = players[seat as usize].action(&before, ctx)?;

        for p in players.iter_mut() {
            p.observe(&before, seat, action);
        }
        ctx.track(&before, action);
        state.step(action);
    }

    let score = state.deal_score();
    Ok([score.scores[0] as i32, score.scores[1] as i32])
}

/// Same as [`play_deal`], but also returns every decision's stats in play
/// order. For analysis and traces; the extra allocation is why it is separate.
pub fn play_deal_traced(
    state: &mut GameState,
    players: &mut [Box<dyn Player>; 4],
    ctx: &mut MatchContext,
) -> Result<(DealScore, Vec<(u8, crate::agent::Decision)>), AgentError> {
    ctx.reset_deal(state.dealer);
    for p in players.iter_mut() {
        p.init_deal(state);
    }

    let mut trace = Vec::new();
    while !state.is_terminal() {
        let seat = state.current_player();
        let before = *state;
        let decision = players[seat as usize].decide(&before, ctx)?;
        let action = decision.action;
        trace.push((seat, decision));

        for p in players.iter_mut() {
            p.observe(&before, seat, action);
        }
        ctx.track(&before, action);
        state.step(action);
    }

    let score = state.deal_score();
    Ok(([score.scores[0] as i32, score.scores[1] as i32], trace))
}

/// Outcome of a match to [`MATCH_TARGET`].
#[derive(Clone, Copy, Debug)]
pub struct MatchResult {
    /// Winning team, 0 = NS, 1 = EW.
    pub winner: u8,
    pub ns_final: i32,
    pub ew_final: i32,
    pub deals: u32,
}

/// Play deals until one team reaches [`MATCH_TARGET`].
///
/// The same four players are reused across deals — `init_deal` is what clears
/// their per-deal state — so the models are loaded once per match rather than
/// once per deal.
pub fn play_match(
    players: &mut [Box<dyn Player>; 4],
    dealer: u8,
    rng: &mut StdRng,
) -> Result<MatchResult, AgentError> {
    let mut ctx = MatchContext::new(dealer);
    let mut dealer = dealer;
    let mut deals = 0u32;

    while ctx.scores[0] < MATCH_TARGET && ctx.scores[1] < MATCH_TARGET {
        let mut state = GameState::deal_random(dealer, rng);
        let score = play_deal(&mut state, players, &mut ctx)?;
        // A passed-out deal scores nothing and is simply redealt.
        ctx.scores[0] += score[0];
        ctx.scores[1] += score[1];
        dealer = (dealer + 1) % 4;
        deals += 1;
    }

    let (ns, ew) = (ctx.scores[0], ctx.scores[1]);
    // Both teams can cross the line on the same deal; the higher total wins.
    let winner = if ns >= MATCH_TARGET && ew >= MATCH_TARGET {
        if ns >= ew {
            0
        } else {
            1
        }
    } else if ns >= MATCH_TARGET {
        0
    } else {
        1
    };
    Ok(MatchResult { winner, ns_final: ns, ew_final: ew, deals })
}

/// Bidding-phase check used by callers that need to know whether a deal was
/// passed out before inspecting the contract.
pub fn was_passed_out(state: &GameState) -> bool {
    state.phase == Phase::Done && state.contract.value == 0
}

/// Random dealer for a fresh match.
pub fn random_dealer(rng: &mut StdRng) -> u8 {
    rng.gen_range(0..4)
}
