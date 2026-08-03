//! What do the auctions of a given bidder actually look like?
//!
//! Written to answer a fair question about the tie-break arena sweep: were those real auctions,
//! or a degenerate "80 then three passes" that would make the whole comparison a fixed-contract
//! exercise? Asserting "it is a heuristic bidder, so it is realistic" is not an answer — the
//! auction is a measurable object.
//!
//! Reports, over N deals of self-play: the share of void deals, the contract value and suit
//! distribution, how many bids an auction contains, how often the two teams contest it, and how
//! often the contract is made. A degenerate auction shows up immediately as "1 bid, 100 % at 80,
//! never contested".
//!
//! Usage:
//!   cargo run --release --features parallel --bin bench_auction_profile -- \
//!       --strategy improved_v2 --deals 4000

use std::collections::BTreeMap;

use colver_core::agent::spec::AgentSpec;
use colver_core::agent::MatchContext;
use colver_core::game_loop;
use colver_core::state::GameState;

use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut strategy = "improved_v2".to_string();
    let mut deals = 4000usize;
    let mut seed = 11u64;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--strategy" => { i += 1; strategy = argv[i].clone(); }
            "--deals" => { i += 1; deals = argv[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = argv[i].parse().unwrap(); }
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }

    // "nn:<path>" builds the neural bidder, anything else is a named heuristic strategy.
    let toml = match strategy.strip_prefix("nn:") {
        Some(model) => format!(
            "[bid]\nstrategy = \"nn\"\nmodel = \"{model}\"\nhidden = 512\nscore_aware = true\n\n\
             [play]\nmethod = \"heuristic\"\n"
        ),
        None => format!("[bid]\nstrategy = \"{strategy}\"\n\n[play]\nmethod = \"heuristic\"\n"),
    };
    let spec = AgentSpec::from_toml_str(&toml).expect("spec");

    let mut rng = StdRng::seed_from_u64(seed);
    let mut values: BTreeMap<u16, usize> = BTreeMap::new();
    let mut suits = [0usize; 4];
    let mut bids_per_auction: BTreeMap<usize, usize> = BTreeMap::new();
    let (mut void, mut contested, mut coinched, mut made) = (0usize, 0usize, 0usize, 0usize);
    let mut played = 0usize;

    for d in 0..deals {
        let mut players = [
            spec.build(0).expect("p0"),
            spec.build(1).expect("p1"),
            spec.build(2).expect("p2"),
            spec.build(3).expect("p3"),
        ];
        let dealer = (d % 4) as u8;
        let mut state = GameState::deal_random(dealer, &mut rng);
        let mut ctx = MatchContext::new(dealer);
        let (_score, trace) = game_loop::play_deal_traced(&mut state, &mut players, &mut ctx)
            .expect("deal");


        if state.contract.value == 0 {
            void += 1;
            continue;
        }
        played += 1;
        let v = state.contract.value as u16 * 10;
        *values.entry(v).or_default() += 1;
        suits[state.contract.trump as usize] += 1;
        if state.contract.coinche > 0 {
            coinched += 1;
        }
        // The trace holds the auction then the 32 cards; the auction is therefore everything
        // before the last 32 entries. Counting by action value alone would not work — a card
        // index and a bid index live in the same 0..43 range.
        let n_auction = trace.len().saturating_sub(32);
        let auction = &trace[..n_auction];
        let n_bids = auction.iter().filter(|(_, dec)| dec.action != 0).count();

        // Contested = both teams put a bid in. Seats are 0=N,1=E,2=S,3=W, teams seat%2.
        let mut teams_bidding = [false; 2];
        for (seat, dec) in auction.iter() {
            if dec.action >= 1 && dec.action <= 40 {
                teams_bidding[(*seat % 2) as usize] = true;
            }
        }
        *bids_per_auction.entry(n_bids.min(9)).or_default() += 1;
        if teams_bidding[0] && teams_bidding[1] {
            contested += 1;
        }
        // Contract made: the declaring team's card points reach the contract.
        let taker = state.contract.team as usize;
        if state.points[taker] as u16 >= v.min(162) {
            made += 1;
        }
    }

    println!("bidder = {strategy} | {deals} deals\n");
    println!("donnes passées (4 passes) : {void} ({:.1} %)", 100.0 * void as f64 / deals as f64);
    println!("donnes jouées             : {played}");
    println!("  dont enchère contestée (les deux camps annoncent) : {:.1} %",
             100.0 * contested as f64 / played.max(1) as f64);
    println!("  dont contrées                                     : {:.1} %",
             100.0 * coinched as f64 / played.max(1) as f64);
    println!("  dont contrat réussi                               : {:.1} %",
             100.0 * made as f64 / played.max(1) as f64);

    println!("\nvaleur du contrat :");
    for (v, n) in &values {
        println!("  {v:>3} : {:>5}  {:>5.1} %  {}", n, 100.0 * *n as f64 / played as f64,
                 "#".repeat((60.0 * *n as f64 / played as f64) as usize));
    }
    println!("\ncouleur d'atout : ♠ {:.1} %  ♥ {:.1} %  ♦ {:.1} %  ♣ {:.1} %",
             100.0 * suits[0] as f64 / played as f64, 100.0 * suits[1] as f64 / played as f64,
             100.0 * suits[2] as f64 / played as f64, 100.0 * suits[3] as f64 / played as f64);

    println!("\nnombre d'annonces (hors passes) dans l'enchère :");
    for (k, n) in &bids_per_auction {
        println!("  {k:>2} : {:>5}  {:>5.1} %", n, 100.0 * *n as f64 / played as f64);
    }
}
