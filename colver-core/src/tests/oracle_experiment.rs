/// Test whether smart_bid contracts are achievable with perfect play.
///
/// Uses perfect-information MCTS (sees all hands) as an "oracle" to play
/// the cards after bidding. If the oracle makes the contracts, the bids
/// are calibrated correctly and the IS-MCTS play is the bottleneck.
use colver_core::bid_eval::{heuristic_bid, smart_bid};
use colver_core::bidding::{decode_bid, BID_COINCHE, BID_PASS, BID_SURCOINCHE};
use colver_core::mcts::{MctsConfig, MctsSearch, RolloutPolicy};
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::rollout::select_nth_bit;
use colver_core::scoring::compute_deal_score;
use colver_core::state::{GameState, Phase};
use rand::Rng;
use std::time::Instant;

const SUIT_NAMES: [&str; 4] = ["S", "H", "D", "C"];
const PLAYER_NAMES: [&str; 4] = ["N", "E", "S", "W"];

fn team_name(team: u8) -> &'static str {
    if team == 0 { "NS" } else { "EW" }
}

fn action_str(action: u8) -> String {
    if action == BID_PASS {
        "Pass".into()
    } else if action == BID_COINCHE {
        "X".into()
    } else if action == BID_SURCOINCHE {
        "XX".into()
    } else if action <= 40 {
        let (val, suit) = decode_bid(action);
        if val == 25 {
            format!("Capot{}", SUIT_NAMES[suit as usize])
        } else {
            format!("{}{}", val as u16 * 10, SUIT_NAMES[suit as usize])
        }
    } else {
        format!("?{}", action)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum PlayMode {
    Oracle,    // Perfect-info MCTS (sees all hands)
    Naive,     // Naive IS-MCTS
    Random,    // Random legal moves
}

fn mode_name(m: PlayMode) -> &'static str {
    match m {
        PlayMode::Oracle => "Oracle",
        PlayMode::Naive => "Naive IS-MCTS",
        PlayMode::Random => "Random",
    }
}

struct GameResult {
    bid_seq: String,
    contract_team: u8,
    contract_value: u16,
    contract_trump: u8,
    contract_coinche: u8,
    void_deal: bool,
    taker_points: u8,
    taker_tricks: u8,
    reussi: bool,
    ns_score: i16,
    ew_score: i16,
}

/// Do the bidding phase with specified bid functions, return the state after bidding.
fn do_bidding(
    state: &mut GameState,
    ns_smart: bool,
    ew_smart: bool,
) -> String {
    let mut seq = String::new();
    while state.phase == Phase::Bidding && !state.is_terminal() {
        let player = state.current_player();
        let team = GameState::player_team(player);
        let action = if team == 0 {
            if ns_smart { smart_bid(state) } else { heuristic_bid(state) }
        } else {
            if ew_smart { smart_bid(state) } else { heuristic_bid(state) }
        };
        if !seq.is_empty() { seq.push(' '); }
        seq.push_str(&format!("{}={}", PLAYER_NAMES[player as usize], action_str(action)));
        state.step(action);
    }
    seq
}

/// Play the card phase with specified modes for taker and defense.
fn do_play(
    state: &mut GameState,
    taker_team: u8,
    taker_mode: PlayMode,
    defense_mode: PlayMode,
    oracle_iters: u32,
    time_ms: u32,
    rng: &mut impl Rng,
) {
    let mut oracle = MctsSearch::new();
    let mut naive = NaiveIsMctsSearch::new();
    let naive_config = NaiveIsMctsConfig {
        iterations_per_det: 50,
        time_limit_ms: Some(time_ms),
        ..Default::default()
    };
    let oracle_config = MctsConfig {
        iterations: oracle_iters,
        exploration: std::f32::consts::SQRT_2,
        rollout_policy: RolloutPolicy::HeuristicPlay,
        ..Default::default()
    };

    while !state.is_terminal() {
        let player = state.current_player();
        let team = GameState::player_team(player);
        let mode = if team == taker_team { taker_mode } else { defense_mode };

        let action = match mode {
            PlayMode::Oracle => oracle.search(state, &oracle_config, rng),
            PlayMode::Naive => naive.search(state, &naive_config, rng),
            PlayMode::Random => {
                let legal = state.legal_actions();
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                select_nth_bit(legal, idx)
            }
        };
        state.step(action);
    }
}

fn play_game(
    ns_smart_bid: bool,
    ew_smart_bid: bool,
    taker_mode: PlayMode,
    defense_mode: PlayMode,
    oracle_iters: u32,
    time_ms: u32,
    dealer: u8,
    rng: &mut impl Rng,
) -> GameResult {
    let mut state = GameState::deal_random(dealer, rng);
    let bid_seq = do_bidding(&mut state, ns_smart_bid, ew_smart_bid);

    if state.is_terminal() && state.contract.value == 0 {
        return GameResult {
            bid_seq,
            contract_team: 0, contract_value: 0, contract_trump: 0,
            contract_coinche: 0, void_deal: true,
            taker_points: 0, taker_tricks: 0, reussi: false,
            ns_score: 0, ew_score: 0,
        };
    }

    let taker_team = state.contract.team;
    let contract_value = state.contract.point_value();
    let contract_trump = state.contract.trump;
    let contract_coinche = state.contract.coinche;

    do_play(&mut state, taker_team, taker_mode, defense_mode, oracle_iters, time_ms, rng);

    let taker = taker_team as usize;
    let taker_points = state.points[taker];
    let taker_tricks = state.tricks_won[taker];
    let reussi = if state.contract.is_capot() {
        taker_tricks == 8
    } else {
        let belote = if state.belote[taker] == 2 { 20 } else { 0 };
        (taker_points as u16 + belote) >= contract_value
    };
    let score = compute_deal_score(&state);

    GameResult {
        bid_seq,
        contract_team: taker_team,
        contract_value,
        contract_trump,
        contract_coinche,
        void_deal: false,
        taker_points,
        taker_tricks,
        reussi,
        ns_score: score.scores[0],
        ew_score: score.scores[1],
    }
}

struct AggStats {
    n: u32,
    void_deals: u32,
    ns_contracts: u32,
    ew_contracts: u32,
    ns_reussi: u32,
    ew_reussi: u32,
    ns_total_bid: u32,
    ew_total_bid: u32,
    ns_total_score: i32,
    ew_total_score: i32,
    coinches: u32,
}

impl AggStats {
    fn new() -> Self {
        AggStats {
            n: 0, void_deals: 0, ns_contracts: 0, ew_contracts: 0,
            ns_reussi: 0, ew_reussi: 0, ns_total_bid: 0, ew_total_bid: 0,
            ns_total_score: 0, ew_total_score: 0, coinches: 0,
        }
    }

    fn add(&mut self, r: &GameResult) {
        self.n += 1;
        if r.void_deal { self.void_deals += 1; return; }
        self.ns_total_score += r.ns_score as i32;
        self.ew_total_score += r.ew_score as i32;
        if r.contract_coinche > 0 { self.coinches += 1; }
        if r.contract_team == 0 {
            self.ns_contracts += 1;
            self.ns_total_bid += r.contract_value as u32;
            if r.reussi { self.ns_reussi += 1; }
        } else {
            self.ew_contracts += 1;
            self.ew_total_bid += r.contract_value as u32;
            if r.reussi { self.ew_reussi += 1; }
        }
    }

    fn print(&self, label: &str) {
        let played = self.n - self.void_deals;
        println!("  {}", label);

        if self.ns_contracts > 0 {
            println!(
                "    NS contracts: {} | avg bid: {:.0} | reussi: {} ({:.0}%) | chute: {} ({:.0}%)",
                self.ns_contracts,
                self.ns_total_bid as f64 / self.ns_contracts as f64,
                self.ns_reussi,
                self.ns_reussi as f64 / self.ns_contracts as f64 * 100.0,
                self.ns_contracts - self.ns_reussi,
                (self.ns_contracts - self.ns_reussi) as f64 / self.ns_contracts as f64 * 100.0,
            );
        }
        if self.ew_contracts > 0 {
            println!(
                "    EW contracts: {} | avg bid: {:.0} | reussi: {} ({:.0}%) | chute: {} ({:.0}%)",
                self.ew_contracts,
                self.ew_total_bid as f64 / self.ew_contracts as f64,
                self.ew_reussi,
                self.ew_reussi as f64 / self.ew_contracts as f64 * 100.0,
                self.ew_contracts - self.ew_reussi,
                (self.ew_contracts - self.ew_reussi) as f64 / self.ew_contracts as f64 * 100.0,
            );
        }
        if played > 0 {
            println!(
                "    Avg score: NS {:.0}  EW {:.0}  (delta {:+.0})  | void: {}",
                self.ns_total_score as f64 / played as f64,
                self.ew_total_score as f64 / played as f64,
                (self.ns_total_score - self.ew_total_score) as f64 / played as f64,
                self.void_deals,
            );
        }
    }
}

fn run_experiment(
    label: &str,
    n_games: u32,
    ns_smart_bid: bool,
    ew_smart_bid: bool,
    taker_mode: PlayMode,
    defense_mode: PlayMode,
    oracle_iters: u32,
    time_ms: u32,
    show_games: usize,
    rng: &mut impl Rng,
) -> AggStats {
    println!();
    println!("=== {} ===", label);
    println!(
        "  Bid: NS={} EW={} | Play: taker={} defense={}",
        if ns_smart_bid { "smart" } else { "heuristic" },
        if ew_smart_bid { "smart" } else { "heuristic" },
        mode_name(taker_mode),
        mode_name(defense_mode),
    );

    let start = Instant::now();
    let mut stats = AggStats::new();
    let mut shown = 0usize;

    for i in 0..n_games {
        let r = play_game(
            ns_smart_bid, ew_smart_bid,
            taker_mode, defense_mode,
            oracle_iters, time_ms,
            (i % 4) as u8, rng,
        );

        if shown < show_games && !r.void_deal {
            println!(
                "    Game {:3}: {} => {}{}  taker={} made={}/{} tricks={} {}  NS={} EW={}",
                i + 1,
                r.bid_seq,
                r.contract_value,
                SUIT_NAMES[r.contract_trump as usize],
                team_name(r.contract_team),
                r.taker_points,
                r.contract_value,
                r.taker_tricks,
                if r.reussi { "REUSSI" } else { "CHUTE" },
                r.ns_score,
                r.ew_score,
            );
            shown += 1;
        }

        stats.add(&r);

        if (i + 1) % 50 == 0 {
            let elapsed = start.elapsed();
            eprint!("\r  [{:.0}ms/game] {}/{}   ", elapsed.as_millis() as f64 / (i + 1) as f64, i + 1, n_games);
        }
    }

    let elapsed = start.elapsed();
    eprintln!();
    println!(
        "  {} games in {:.1?} ({:.0}ms/game)",
        n_games, elapsed, elapsed.as_millis() as f64 / n_games as f64
    );
    stats.print(label);
    stats
}

fn main() {
    let n_games: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let oracle_iters: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let time_ms: u32 = 30; // for naive IS-MCTS play

    let mut rng = rand::thread_rng();

    println!("Oracle Experiment: {} games, oracle={} iters, naive={}ms/move", n_games, oracle_iters, time_ms);

    // 1. smart_bid + oracle taker vs oracle defense
    //    "Can a perfect player make these contracts against a perfect defender?"
    let s1 = run_experiment(
        "smart_bid + Oracle vs Oracle",
        n_games, true, true,
        PlayMode::Oracle, PlayMode::Oracle,
        oracle_iters, time_ms, 10, &mut rng,
    );

    // 2. smart_bid + oracle taker vs random defense
    //    "Can a perfect player make these contracts against weak defense?"
    let s2 = run_experiment(
        "smart_bid + Oracle vs Random",
        n_games, true, true,
        PlayMode::Oracle, PlayMode::Random,
        oracle_iters, time_ms, 10, &mut rng,
    );

    // 3. smart_bid + naive taker vs random defense
    //    "Can IS-MCTS make these contracts against weak defense?"
    let s3 = run_experiment(
        "smart_bid + Naive vs Random",
        n_games, true, true,
        PlayMode::Naive, PlayMode::Random,
        oracle_iters, time_ms, 5, &mut rng,
    );

    // 4. heuristic_bid + oracle taker vs oracle defense (calibration)
    //    "Are heuristic bids easier to fulfill?"
    let s4 = run_experiment(
        "heuristic_bid + Oracle vs Oracle",
        n_games, false, false,
        PlayMode::Oracle, PlayMode::Oracle,
        oracle_iters, time_ms, 5, &mut rng,
    );

    // 5. heuristic_bid + naive taker vs random defense (reference)
    let s5 = run_experiment(
        "heuristic_bid + Naive vs Random",
        n_games, false, false,
        PlayMode::Naive, PlayMode::Random,
        oracle_iters, time_ms, 5, &mut rng,
    );

    // Summary comparison
    println!();
    println!("============================================================");
    println!("  COMPARISON: Contract success rates");
    println!("============================================================");
    println!();

    let success_rate = |s: &AggStats| -> (f64, f64) {
        let total_c = s.ns_contracts + s.ew_contracts;
        let total_r = s.ns_reussi + s.ew_reussi;
        let overall = if total_c > 0 { total_r as f64 / total_c as f64 * 100.0 } else { 0.0 };
        (overall, if total_c > 0 { (s.ns_total_bid + s.ew_total_bid) as f64 / total_c as f64 } else { 0.0 })
    };

    let (r1, b1) = success_rate(&s1);
    let (r2, b2) = success_rate(&s2);
    let (r3, b3) = success_rate(&s3);
    let (r4, b4) = success_rate(&s4);
    let (r5, b5) = success_rate(&s5);

    println!("  {:42} | avg bid | success", "Experiment");
    println!("  {:42} | ------- | -------", "----------");
    println!("  {:42} | {:>5.0}   | {:>5.1}%", "smart_bid  + Oracle vs Oracle",  b1, r1);
    println!("  {:42} | {:>5.0}   | {:>5.1}%", "smart_bid  + Oracle vs Random",  b2, r2);
    println!("  {:42} | {:>5.0}   | {:>5.1}%", "smart_bid  + Naive  vs Random",  b3, r3);
    println!("  {:42} | {:>5.0}   | {:>5.1}%", "heuristic  + Oracle vs Oracle",  b4, r4);
    println!("  {:42} | {:>5.0}   | {:>5.1}%", "heuristic  + Naive  vs Random",  b5, r5);
    println!();

    if r1 > 70.0 {
        println!("  => Oracle makes smart_bid contracts {:.0}% of the time — bids are ACHIEVABLE.", r1);
        println!("     The problem is IS-MCTS play quality, not bid calibration.");
    } else if r1 > 50.0 {
        println!("  => Oracle makes smart_bid contracts {:.0}% — bids are BORDERLINE.", r1);
        println!("     Some overbidding, but play quality also matters.");
    } else {
        println!("  => Oracle only makes smart_bid contracts {:.0}% — bids are TOO HIGH.", r1);
        println!("     Even perfect play can't save these contracts.");
    }
}
