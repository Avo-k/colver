/// deal_bias: measure the bias of traditional dealing vs competition shuffling.
///
/// Traditional ("maison"): tricks are gathered into one pile per team in the order
/// they were won, the two team piles are stacked (coin flip for which goes on top),
/// the deck is cut once (min 3 cards), then dealt 3-3-2. The deck is only fully
/// shuffled at the start of each 2000-point match.
///
/// Competition: full Fisher-Yates shuffle before every deal, then 3-3-2.
///
/// A "soirée" = 3 matches to 2000 points. Deals are played by NN bots
/// (bid v6 + DouDou50 by default) so trick composition is realistic.
///
/// Usage:
///   cargo run --bin deal_bias --release -- --soirees 1000 [--mode both] [--csv out.csv]
use colver_core::bid_net::BidNet;
use colver_core::bid_obs::{
    self, BID_OBS_DIM, BID_OBS_DIM_SCORE_AWARE, BID_OBS_DIM_SCORE_AWARE_V2,
    BID_OBS_DIM_SCORE_AWARE_V3,
};
use colver_core::card::{card_to_bit, Card, PLAIN_POINTS};
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM, OBS_DIM_TR};
use colver_core::state::{GameState, Phase};
use colver_core::trick::trick_winner;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const MATCH_TARGET: i32 = 2000;
const MAX_DEALS_PER_MATCH: u32 = 300;
const IDX_BUCKETS: usize = 25; // per-deal-index convergence tracking (last bucket = 24+)

// ══════════════════════════════════════════════════════════════════════
//  Shared weights (thread-safe blueprints, one net instance per thread)
// ══════════════════════════════════════════════════════════════════════

struct DmcWeights {
    floats: Vec<f32>,
    hidden: usize,
    obs_dim: usize,
    dueling: bool,
    residual: bool,
}

impl DmcWeights {
    fn load(path: &str, residual: bool) -> std::io::Result<Self> {
        let net = DmcNet::load(path)?;
        let obs_dim = net.obs_dim();
        let hidden = net.hidden();
        let dueling = net.is_dueling();
        drop(net);
        let data = std::fs::read(path)?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(DmcWeights { floats, hidden, obs_dim, dueling, residual })
    }

    fn make_net(&self) -> DmcNet {
        let mut net = DmcNet::from_floats(&self.floats, self.hidden, self.obs_dim, self.dueling).unwrap();
        if self.residual {
            net.set_residual(true);
        }
        net
    }
}

struct BidNetWeights {
    floats: Vec<f32>,
    hidden: usize,
    obs_dim: usize,
    dueling: bool,
    layers: usize,
}

impl BidNetWeights {
    fn load(path: &str, hidden: usize) -> std::io::Result<Self> {
        let net = BidNet::load_with_hidden(path, hidden)?;
        let obs_dim = net.obs_dim();
        let hidden = net.hidden();
        let dueling = net.is_dueling();
        let layers = net.layers();
        drop(net);
        let data = std::fs::read(path)?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(BidNetWeights { floats, hidden, obs_dim, dueling, layers })
    }

    fn make_net(&self) -> BidNet {
        BidNet::from_floats_with_layers(&self.floats, self.hidden, self.obs_dim, self.dueling, self.layers).unwrap()
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Physical deck model
// ══════════════════════════════════════════════════════════════════════

/// A deck is a Vec<Card> stack: index 0 = bottom, last = top. Dealing takes from the top.
#[derive(Clone, Copy, PartialEq)]
enum TrickOrder {
    Play,
    Reverse,
    Random,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Tradition,
    Shuffle,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::Tradition => "tradition",
            Mode::Shuffle => "shuffle",
        }
    }
}

fn fresh_shuffled(rng: &mut StdRng) -> Vec<Card> {
    let mut stack: Vec<Card> = (0..32u8).collect();
    stack.shuffle(rng);
    stack
}

/// Deal 3-3-2 from the top of the stack, starting left of the dealer.
fn deal_332(stack: &[Card], dealer: u8) -> [u32; 4] {
    debug_assert_eq!(stack.len(), 32);
    let mut hands = [0u32; 4];
    let mut next = stack.len();
    for &round in &[3usize, 3, 2] {
        for p in 0..4u8 {
            let seat = ((dealer + 1 + p) % 4) as usize;
            for _ in 0..round {
                next -= 1;
                hands[seat] |= card_to_bit(stack[next]);
            }
        }
    }
    hands
}

/// Single cut: the top k cards go under the rest. k uniform in [min_cut, 32-min_cut].
fn cut(stack: &mut Vec<Card>, rng: &mut StdRng, min_cut: usize) {
    let n = stack.len();
    let k = rng.gen_range(min_cut..=n - min_cut);
    stack.rotate_right(k);
}

/// Rebuild the physical deck from a finished deal.
///
/// Each team keeps one pile: tricks are stacked on it in the order they were won,
/// cards within a trick in play order (configurable). The two piles are then
/// stacked, coin flip for which team's pile ends on top. On a passed-out deal the
/// four hands are tossed back in seat order from dealer+1, each hand suit-grouped.
fn reconstruct_stack(state: &GameState, rng: &mut StdRng, order: TrickOrder) -> Vec<Card> {
    if state.contract.value == 0 {
        let mut stack = Vec::with_capacity(32);
        for p in 0..4u8 {
            let seat = ((state.dealer + 1 + p) % 4) as usize;
            let mut set = state.hands[seat];
            while set != 0 {
                stack.push(set.trailing_zeros() as u8);
                set &= set - 1;
            }
        }
        return stack;
    }

    let mut piles: [Vec<Card>; 2] = [Vec::with_capacity(32), Vec::with_capacity(32)];
    let mut lead = (state.dealer + 1) % 4;
    for i in 0..8 {
        let trick = &state.trick_history[i];
        let winner = trick_winner(trick, lead, &state.contract);
        let mut cards: [Card; 4] =
            core::array::from_fn(|j| trick[(lead as usize + j) % 4]);
        match order {
            TrickOrder::Play => {}
            TrickOrder::Reverse => cards.reverse(),
            TrickOrder::Random => cards.shuffle(rng),
        }
        piles[(winner % 2) as usize].extend_from_slice(&cards);
        lead = winner;
    }

    let (bottom, top) = if rng.gen_bool(0.5) { (0, 1) } else { (1, 0) };
    let mut stack = std::mem::take(&mut piles[bottom]);
    stack.extend_from_slice(&piles[top]);
    stack
}

// ══════════════════════════════════════════════════════════════════════
//  Bot play (bid NN + DMC, mirrors arena's fast path)
// ══════════════════════════════════════════════════════════════════════

struct Bots {
    bid: BidNet,
    bid_obs_dim: usize,
    dmc: DmcNet,
    dmc_obs_dim: usize,
    bid_obs_buf: Vec<f32>,
    obs_buf: Vec<f32>,
}

impl Bots {
    fn new(bid_w: &BidNetWeights, dmc_w: &DmcWeights) -> Self {
        Bots {
            bid: bid_w.make_net(),
            bid_obs_dim: bid_w.obs_dim,
            dmc: dmc_w.make_net(),
            dmc_obs_dim: dmc_w.obs_dim,
            bid_obs_buf: vec![0.0; bid_w.obs_dim.max(BID_OBS_DIM_SCORE_AWARE_V3)],
            obs_buf: vec![0.0; dmc_w.obs_dim.max(OBS_DIM)],
        }
    }
}

/// Play one deal to completion with the shared bots (all 4 seats).
fn play_deal(state: &mut GameState, bots: &mut Bots, cum: [i32; 2]) {
    let mut tracking = EnvTracking::new();
    tracking.reset(state.dealer);

    while !state.is_terminal() {
        let action = if state.phase == Phase::Bidding {
            let team = (state.current_player() % 2) as usize;
            let (my_score, opp_score) = (cum[team], cum[1 - team]);
            let legal = state.legal_actions();
            let dim = bots.bid_obs_dim;
            match dim {
                BID_OBS_DIM => {
                    bid_obs::write_bid_observation(&mut bots.bid_obs_buf, 0, state, &tracking.bid_history)
                }
                BID_OBS_DIM_SCORE_AWARE => bid_obs::write_bid_observation_score_aware(
                    &mut bots.bid_obs_buf, 0, state, &tracking.bid_history, my_score, opp_score,
                ),
                BID_OBS_DIM_SCORE_AWARE_V2 => bid_obs::write_bid_observation_score_aware_v2(
                    &mut bots.bid_obs_buf, 0, state, &tracking.bid_history, my_score, opp_score,
                ),
                BID_OBS_DIM_SCORE_AWARE_V3 => bid_obs::write_bid_observation_score_aware_v3(
                    &mut bots.bid_obs_buf, 0, state, &tracking.bid_history, my_score, opp_score,
                ),
                other => panic!("unsupported bid obs dim {}", other),
            }
            bots.bid.best_action_fast(&bots.bid_obs_buf[..dim], legal)
        } else if bots.dmc_obs_dim == OBS_DIM_TR {
            dmc_obs::write_observation_tr(&mut bots.obs_buf, 0, state, &tracking);
            let order = dmc_obs::current_player_order(state, &tracking);
            let mask = dmc_obs::cardset_to_canonical(state.legal_actions() as u32, &order);
            let (best, _) = bots.dmc.best_action(&bots.obs_buf, mask);
            dmc_obs::card_to_physical(best, &order)
        } else {
            dmc_obs::write_observation(&mut bots.obs_buf, 0, state, &tracking);
            let (best, _) = bots.dmc.best_action(&bots.obs_buf, state.legal_actions() as u32);
            best
        };
        tracking.track_action(state, action);
        state.step(action);
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Metrics
// ══════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct MatchRec {
    mode: Mode,
    soiree: u32,
    match_idx: u8,
    deals: u32,
    void_deals: u32,
    suit_voids: u32,     // suit-holdings of length 0 (out of 16 per deal)
    long5: u32,          // suit-holdings of length >= 5
    longest_sum: f64,    // sum over hands of longest-suit length (4 per deal)
    belote_hands: u32,   // hands holding K+Q of at least one suit (4 per deal)
    contracted: u32,     // deals with a contract
    contract_pts_sum: f64,
    coinche: u32,
    surcoinche: u32,
    chutes: u32,
    capots: u32,
}

#[derive(Clone)]
struct Stats {
    deals: u64,
    matches: u64,
    suit_len: [u64; 9],
    longest_suit: [u64; 9],
    hand_pts_sum: f64,
    hand_pts_sq: f64,
    contracts: [u64; 10], // 80..160 step 10, then capot
    abs_diff_sum: f64,
    idx_deals: [u64; IDX_BUCKETS],
    idx_suit_voids: [u64; IDX_BUCKETS],
    idx_long5: [u64; IDX_BUCKETS],
    idx_longest_sum: [f64; IDX_BUCKETS],
    per_match: Vec<MatchRec>,
}

impl Stats {
    fn new() -> Self {
        Stats {
            deals: 0,
            matches: 0,
            suit_len: [0; 9],
            longest_suit: [0; 9],
            hand_pts_sum: 0.0,
            hand_pts_sq: 0.0,
            contracts: [0; 10],
            abs_diff_sum: 0.0,
            idx_deals: [0; IDX_BUCKETS],
            idx_suit_voids: [0; IDX_BUCKETS],
            idx_long5: [0; IDX_BUCKETS],
            idx_longest_sum: [0.0; IDX_BUCKETS],
            per_match: Vec::new(),
        }
    }

    fn merge(&mut self, other: Stats) {
        self.deals += other.deals;
        self.matches += other.matches;
        for i in 0..9 {
            self.suit_len[i] += other.suit_len[i];
            self.longest_suit[i] += other.longest_suit[i];
        }
        self.hand_pts_sum += other.hand_pts_sum;
        self.hand_pts_sq += other.hand_pts_sq;
        for i in 0..10 {
            self.contracts[i] += other.contracts[i];
        }
        self.abs_diff_sum += other.abs_diff_sum;
        for i in 0..IDX_BUCKETS {
            self.idx_deals[i] += other.idx_deals[i];
            self.idx_suit_voids[i] += other.idx_suit_voids[i];
            self.idx_long5[i] += other.idx_long5[i];
            self.idx_longest_sum[i] += other.idx_longest_sum[i];
        }
        self.per_match.extend(other.per_match);
    }
}

/// Record hand-structure metrics for a freshly dealt (unplayed) deal.
fn record_hands(stats: &mut Stats, rec: &mut MatchRec, hands: &[u32; 4], deal_idx: u32) {
    let bucket = (deal_idx as usize).min(IDX_BUCKETS - 1);
    stats.idx_deals[bucket] += 1;
    for &hand in hands {
        let mut longest = 0usize;
        for s in 0..4 {
            let len = ((hand >> (8 * s)) & 0xFF).count_ones() as usize;
            stats.suit_len[len] += 1;
            longest = longest.max(len);
            if len == 0 {
                rec.suit_voids += 1;
                stats.idx_suit_voids[bucket] += 1;
            }
            if len >= 5 {
                rec.long5 += 1;
                stats.idx_long5[bucket] += 1;
            }
        }
        stats.longest_suit[longest] += 1;
        rec.longest_sum += longest as f64;
        stats.idx_longest_sum[bucket] += longest as f64;

        let mut pts = 0u32;
        let mut set = hand;
        while set != 0 {
            let c = set.trailing_zeros() as u8;
            pts += PLAIN_POINTS[(c & 7) as usize] as u32;
            set &= set - 1;
        }
        stats.hand_pts_sum += pts as f64;
        stats.hand_pts_sq += (pts * pts) as f64;

        let mut has_belote = false;
        for s in 0..4u32 {
            let qk = (1u32 << (s * 8 + 4)) | (1u32 << (s * 8 + 5));
            if hand & qk == qk {
                has_belote = true;
            }
        }
        if has_belote {
            rec.belote_hands += 1;
        }
    }
}

/// Record outcome metrics for a finished deal.
fn record_outcome(stats: &mut Stats, rec: &mut MatchRec, state: &GameState, scores: [i16; 2]) {
    stats.deals += 1;
    rec.deals += 1;
    if state.contract.value == 0 {
        rec.void_deals += 1;
        return;
    }
    rec.contracted += 1;
    let v = state.contract.value;
    let idx = if v == 25 { 9 } else { (v - 8) as usize };
    stats.contracts[idx] += 1;
    rec.contract_pts_sum += state.contract.point_value() as f64;
    match state.contract.coinche {
        1 => rec.coinche += 1,
        2 => {
            rec.coinche += 1;
            rec.surcoinche += 1;
        }
        _ => {}
    }
    if scores[state.contract.team as usize] == 0 {
        rec.chutes += 1;
    }
    let declarer_team = state.contract.team as usize;
    if state.tricks_won[declarer_team] == 8 {
        rec.capots += 1;
    }
    stats.abs_diff_sum += (scores[0] as f64 - scores[1] as f64).abs();
}

// ══════════════════════════════════════════════════════════════════════
//  Simulation
// ══════════════════════════════════════════════════════════════════════

struct Config {
    soirees: u32,
    matches_per_soiree: u8,
    seed: u64,
    threads: usize,
    modes: Vec<Mode>,
    trick_order: TrickOrder,
    min_cut: usize,
    csv: Option<String>,
    bid_model: String,
    bid_hidden: usize,
    play_model: String,
    residual: bool,
}

fn run_match(
    mode: Mode,
    cfg: &Config,
    bots: &mut Bots,
    rng: &mut StdRng,
    dealer: &mut u8,
    stats: &mut Stats,
    soiree: u32,
    match_idx: u8,
) {
    let mut rec = MatchRec {
        mode,
        soiree,
        match_idx,
        deals: 0,
        void_deals: 0,
        suit_voids: 0,
        long5: 0,
        longest_sum: 0.0,
        belote_hands: 0,
        contracted: 0,
        contract_pts_sum: 0.0,
        coinche: 0,
        surcoinche: 0,
        chutes: 0,
        capots: 0,
    };

    // Every match starts from a freshly shuffled deck (even in tradition mode).
    let mut stack = fresh_shuffled(rng);
    let mut cum = [0i32; 2];
    let mut deal_idx = 0u32;

    while cum[0] < MATCH_TARGET && cum[1] < MATCH_TARGET && deal_idx < MAX_DEALS_PER_MATCH {
        let hands = deal_332(&stack, *dealer);
        let mut state = GameState::new(*dealer, hands);
        record_hands(stats, &mut rec, &hands, deal_idx);

        play_deal(&mut state, bots, cum);
        let score = state.deal_score();
        cum[0] += score.scores[0] as i32;
        cum[1] += score.scores[1] as i32;
        record_outcome(stats, &mut rec, &state, score.scores);

        match mode {
            Mode::Tradition => {
                stack = reconstruct_stack(&state, rng, cfg.trick_order);
                cut(&mut stack, rng, cfg.min_cut);
            }
            Mode::Shuffle => {
                stack.shuffle(rng);
            }
        }
        *dealer = (*dealer + 1) % 4;
        deal_idx += 1;
    }

    stats.matches += 1;
    stats.per_match.push(rec);
}

fn run_soiree(mode: Mode, cfg: &Config, bots: &mut Bots, soiree: u32) -> Stats {
    let mode_salt = match mode {
        Mode::Tradition => 1u64,
        Mode::Shuffle => 2u64,
    };
    let mut rng = StdRng::seed_from_u64(
        cfg.seed ^ (soiree as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ mode_salt.wrapping_mul(0x517C_C1B7_2722_0A95),
    );
    let mut stats = Stats::new();
    let mut dealer: u8 = rng.gen_range(0..4);
    for m in 0..cfg.matches_per_soiree {
        run_match(mode, cfg, bots, &mut rng, &mut dealer, &mut stats, soiree, m);
    }
    stats
}

// ══════════════════════════════════════════════════════════════════════
//  Reporting
// ══════════════════════════════════════════════════════════════════════

/// Match-level mean ± standard error for a per-match metric.
fn cluster_se(recs: &[&MatchRec], f: impl Fn(&MatchRec) -> f64) -> (f64, f64) {
    let n = recs.len() as f64;
    if n < 2.0 {
        return (f64::NAN, f64::NAN);
    }
    let vals: Vec<f64> = recs.iter().map(|r| f(r)).filter(|v| v.is_finite()).collect();
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

fn print_report(all: &[(Mode, Stats)]) {
    for (mode, stats) in all {
        let d = stats.deals as f64;
        let hands = d * 4.0;
        let holdings = d * 16.0;
        println!();
        println!("═══ {} — {} soirée-matches, {} deals ({:.1} deals/match) ═══",
            mode.label(), stats.matches, stats.deals, d / stats.matches as f64);

        print!("suit-length distribution (% of the 16 suit-holdings per deal):\n  len ");
        for l in 0..9 {
            print!("{:>7}", l);
        }
        print!("\n  %   ");
        for l in 0..9 {
            print!("{:>7.3}", 100.0 * stats.suit_len[l] as f64 / holdings);
        }
        println!();

        print!("longest-suit-per-hand distribution (%):\n  len ");
        for l in 2..9 {
            print!("{:>7}", l);
        }
        print!("\n  %   ");
        for l in 2..9 {
            print!("{:>7.3}", 100.0 * stats.longest_suit[l] as f64 / hands);
        }
        println!();

        let pts_mean = stats.hand_pts_sum / hands;
        let pts_var = stats.hand_pts_sq / hands - pts_mean * pts_mean;
        println!("hand plain-points: mean {:.3}, std {:.3}", pts_mean, pts_var.sqrt());

        let recs: Vec<&MatchRec> = stats.per_match.iter().collect();
        let (voids, voids_se) = cluster_se(&recs, |r| 100.0 * r.suit_voids as f64 / (16 * r.deals) as f64);
        let (long5, long5_se) = cluster_se(&recs, |r| 100.0 * r.long5 as f64 / (16 * r.deals) as f64);
        let (longest, longest_se) = cluster_se(&recs, |r| r.longest_sum / (4 * r.deals) as f64);
        let (belote, belote_se) = cluster_se(&recs, |r| 100.0 * r.belote_hands as f64 / (4 * r.deals) as f64);
        let (voidd, voidd_se) = cluster_se(&recs, |r| 100.0 * r.void_deals as f64 / r.deals as f64);
        let (cval, cval_se) = cluster_se(&recs, |r| {
            if r.contracted > 0 { r.contract_pts_sum / r.contracted as f64 } else { f64::NAN }
        });
        let (coin, coin_se) = cluster_se(&recs, |r| {
            if r.contracted > 0 { 100.0 * r.coinche as f64 / r.contracted as f64 } else { f64::NAN }
        });
        let (chute, chute_se) = cluster_se(&recs, |r| {
            if r.contracted > 0 { 100.0 * r.chutes as f64 / r.contracted as f64 } else { f64::NAN }
        });
        let (capot, capot_se) = cluster_se(&recs, |r| {
            if r.contracted > 0 { 100.0 * r.capots as f64 / r.contracted as f64 } else { f64::NAN }
        });

        println!("key metrics (match-level mean ± SE):");
        println!("  void suits          {:.3} ± {:.3} %", voids, voids_se);
        println!("  suits of 5+ cards   {:.3} ± {:.3} %", long5, long5_se);
        println!("  longest suit / hand {:.4} ± {:.4}", longest, longest_se);
        println!("  hands with K+Q suit {:.3} ± {:.3} %", belote, belote_se);
        println!("  passed-out deals    {:.3} ± {:.3} %", voidd, voidd_se);
        println!("  mean contract       {:.2} ± {:.2} pts", cval, cval_se);
        println!("  coinche rate        {:.3} ± {:.3} % of contracts", coin, coin_se);
        println!("  chute rate          {:.3} ± {:.3} % of contracts", chute, chute_se);
        println!("  capot réalisé       {:.3} ± {:.3} % of contracts", capot, capot_se);
        println!("  mean |NS-EW| gap    {:.1} pts/deal", stats.abs_diff_sum / d);

        let total_contracts: u64 = stats.contracts.iter().sum();
        if total_contracts > 0 {
            print!("contract distribution (%):\n      ");
            for v in 0..9 {
                print!("{:>7}", 80 + v * 10);
            }
            print!("{:>7}", "capot");
            print!("\n  %   ");
            for v in 0..10 {
                print!("{:>7.3}", 100.0 * stats.contracts[v] as f64 / total_contracts as f64);
            }
            println!();
        }
    }

    // Convergence within a match: does the bias build up deal after deal?
    println!();
    println!("═══ convergence by deal index within a match ═══");
    println!("(void-suit % of 16 holdings / mean longest suit — deal 0 is freshly shuffled in both modes)");
    print!("  idx        ");
    for i in 0..IDX_BUCKETS.min(15) {
        print!("{:>12}", i);
    }
    println!();
    for (mode, stats) in all {
        print!("  {:<9}", mode.label());
        for i in 0..IDX_BUCKETS.min(15) {
            let n = stats.idx_deals[i] as f64;
            if n > 0.0 {
                print!("{:>6.2}/{:<5.3}",
                    100.0 * stats.idx_suit_voids[i] as f64 / (16.0 * n),
                    stats.idx_longest_sum[i] / (4.0 * n));
            } else {
                print!("{:>12}", "-");
            }
        }
        println!();
    }
}

fn write_csv(path: &str, all: &[(Mode, Stats)]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "mode,soiree,match,deals,void_deals,suit_voids,long5,longest_sum,belote_hands,contracted,contract_pts_sum,coinche,surcoinche,chutes,capots")?;
    for (_, stats) in all {
        for r in &stats.per_match {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{:.1},{},{},{:.0},{},{},{},{}",
                r.mode.label(), r.soiree, r.match_idx, r.deals, r.void_deals, r.suit_voids,
                r.long5, r.longest_sum, r.belote_hands, r.contracted, r.contract_pts_sum,
                r.coinche, r.surcoinche, r.chutes, r.capots
            )?;
        }
    }
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════
//  Main
// ══════════════════════════════════════════════════════════════════════

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = Config {
        soirees: parse_flag(&args, "--soirees").and_then(|v| v.parse().ok()).unwrap_or(500),
        matches_per_soiree: parse_flag(&args, "--matches-per-soiree").and_then(|v| v.parse().ok()).unwrap_or(3),
        seed: parse_flag(&args, "--seed").and_then(|v| v.parse().ok()).unwrap_or(42),
        threads: parse_flag(&args, "--threads").and_then(|v| v.parse().ok()).unwrap_or_else(|| {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
        }),
        modes: match parse_flag(&args, "--mode").as_deref() {
            Some("tradition") => vec![Mode::Tradition],
            Some("shuffle") => vec![Mode::Shuffle],
            _ => vec![Mode::Tradition, Mode::Shuffle],
        },
        trick_order: match parse_flag(&args, "--trick-order").as_deref() {
            Some("reverse") => TrickOrder::Reverse,
            Some("random") => TrickOrder::Random,
            _ => TrickOrder::Play,
        },
        min_cut: parse_flag(&args, "--min-cut").and_then(|v| v.parse().ok()).unwrap_or(3),
        csv: parse_flag(&args, "--csv"),
        bid_model: parse_flag(&args, "--bid-model")
            .unwrap_or_else(|| "models/bid_v6_isdd_resume/bid_nn_final.bin".to_string()),
        bid_hidden: parse_flag(&args, "--bid-hidden").and_then(|v| v.parse().ok()).unwrap_or(512),
        play_model: parse_flag(&args, "--play-model")
            .unwrap_or_else(|| "models/play_v2/play_final.bin".to_string()),
        residual: parse_flag(&args, "--residual").map(|v| v == "true").unwrap_or(true),
    };

    let bid_w = Arc::new(BidNetWeights::load(&cfg.bid_model, cfg.bid_hidden).unwrap_or_else(|e| {
        eprintln!("cannot load bid model {}: {}", cfg.bid_model, e);
        std::process::exit(1);
    }));
    let dmc_w = Arc::new(DmcWeights::load(&cfg.play_model, cfg.residual).unwrap_or_else(|e| {
        eprintln!("cannot load play model {}: {}", cfg.play_model, e);
        std::process::exit(1);
    }));
    println!(
        "deal_bias: {} soirées × {} matches, modes [{}], bid {} (obs {}), play {} (obs {}), {} threads, seed {}",
        cfg.soirees,
        cfg.matches_per_soiree,
        cfg.modes.iter().map(|m| m.label()).collect::<Vec<_>>().join(", "),
        cfg.bid_model, bid_w.obs_dim, cfg.play_model, dmc_w.obs_dim, cfg.threads, cfg.seed
    );

    let start = Instant::now();
    let done = Arc::new(AtomicU64::new(0));
    let total_units = cfg.soirees as u64 * cfg.modes.len() as u64;
    let cfg = Arc::new(cfg);

    let mut handles = Vec::new();
    for t in 0..cfg.threads {
        let cfg = Arc::clone(&cfg);
        let bid_w = Arc::clone(&bid_w);
        let dmc_w = Arc::clone(&dmc_w);
        let done = Arc::clone(&done);
        handles.push(std::thread::spawn(move || {
            let mut bots = Bots::new(&bid_w, &dmc_w);
            let mut local: Vec<(Mode, Stats)> =
                cfg.modes.iter().map(|&m| (m, Stats::new())).collect();
            let mut s = t as u32;
            while s < cfg.soirees {
                for (mi, &mode) in cfg.modes.iter().enumerate() {
                    let stats = run_soiree(mode, &cfg, &mut bots, s);
                    local[mi].1.merge(stats);
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if n % 100 == 0 {
                        eprintln!("  {}/{} soirée-mode units done", n, total_units);
                    }
                }
                s += cfg.threads as u32;
            }
            local
        }));
    }

    let mut all: Vec<(Mode, Stats)> = cfg.modes.iter().map(|&m| (m, Stats::new())).collect();
    for h in handles {
        let local = h.join().unwrap();
        for (i, (_, stats)) in local.into_iter().enumerate() {
            all[i].1.merge(stats);
        }
    }

    println!("simulated in {:.1}s", start.elapsed().as_secs_f64());
    print_report(&all);

    if let Some(path) = &cfg.csv {
        write_csv(path, &all).unwrap();
        println!("\nper-match CSV written to {}", path);
    }
}
