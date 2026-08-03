/// Distill IS-DD play decisions to a binary log for interpretability analysis.
///
/// For each (deal, forced trump suit), plays the deal with IS-DD and records
/// every non-forced decision: state bitmasks, legal mask, IS-DD Q-values per
/// legal card, the chosen card, and the final NS points.
///
/// Output format: COLVPD01 (see end of file for full schema).
///
/// Usage:
///   cargo run --bin distill_play --release --features parallel -- [options]
///
/// Options:
///   --pool PATH       Input pool (default: data/deals/base_5M.bin)
///   --output PATH     Output binary log (default: data/distill/play_distill.bin)
///   --deals N         Number of deals (default: 1000)
///   --offset N        Skip first N deals of pool (default: 0)
///   --time-ms N       IS-DD time per move (default: 20)
///   --dets N          Determinizations (default: 20)
///   --seed N          RNG seed (default: 42)
///   --skip-forced     If set (default), don't write rows where n_legal == 1.
///                     Forced rows are uninteresting for interpretability.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use colver_core::bid_train_env::DealPool;
use colver_core::is_dd::{IsDdConfig, IsDdSearch, PlayObjective};
use colver_core::state::{GameState, Phase};

/// One per-decision record (variable-length).
struct Record {
    deal_id: u32,
    forced_suit: u8,
    dealer: u8,
    trick_idx: u8,
    play_idx: u8,
    seat: u8,
    trick_lead: u8,
    chosen: u8,
    n_legal: u8,
    final_ns_pts: u8,
    hand: u32,
    legal: u32,
    played_cards: u32,
    /// 4 bytes packed: card_played_by_seat[0..3], 0xFF = empty
    trick_packed: u32,
    /// 4 bytes packed: voids[0..3] (engine-tracked)
    voids_packed: u32,
    /// (card, q_value) for each legal card. Q-value is from NS perspective (0..252).
    q_values: Vec<(u8, f32)>,
}

impl Record {
    fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.deal_id.to_le_bytes());
        out.push(self.forced_suit);
        out.push(self.dealer);
        out.push(self.trick_idx);
        out.push(self.play_idx);
        out.push(self.seat);
        out.push(self.trick_lead);
        out.push(self.chosen);
        out.push(self.n_legal);
        out.push(self.final_ns_pts);
        out.push(0u8); // _pad
        out.extend_from_slice(&self.hand.to_le_bytes());
        out.extend_from_slice(&self.legal.to_le_bytes());
        out.extend_from_slice(&self.played_cards.to_le_bytes());
        out.extend_from_slice(&self.trick_packed.to_le_bytes());
        out.extend_from_slice(&self.voids_packed.to_le_bytes());
        for &(card, q) in &self.q_values {
            out.push(card);
            out.extend_from_slice(&q.to_le_bytes());
        }
    }
}

fn pack_trick(trick: &[u8; 4]) -> u32 {
    (trick[0] as u32)
        | ((trick[1] as u32) << 8)
        | ((trick[2] as u32) << 16)
        | ((trick[3] as u32) << 24)
}

fn pack_voids(voids: &[u8; 4]) -> u32 {
    (voids[0] as u32)
        | ((voids[1] as u32) << 8)
        | ((voids[2] as u32) << 16)
        | ((voids[3] as u32) << 24)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut pool_path = String::from("data/deals/base_5M.bin");
    let mut output_path = String::from("data/distill/play_distill.bin");
    let mut num_deals: usize = 1000;
    let mut offset: usize = 0;
    let mut time_ms: u32 = 20;
    let mut dets: u32 = 20;
    let mut seed: u64 = 42;
    let mut skip_forced: bool = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pool" => { i += 1; pool_path = args[i].clone(); }
            "--output" => { i += 1; output_path = args[i].clone(); }
            "--deals" => { i += 1; num_deals = args[i].parse().unwrap(); }
            "--offset" => { i += 1; offset = args[i].parse().unwrap(); }
            "--time-ms" => { i += 1; time_ms = args[i].parse().unwrap(); }
            "--dets" => { i += 1; dets = args[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            "--include-forced" => { skip_forced = false; }
            _ => { eprintln!("Unknown arg: {}", args[i]); std::process::exit(1); }
        }
        i += 1;
    }

    eprintln!("Loading pool from {}...", pool_path);
    let pool = DealPool::load(&pool_path).expect("Failed to load pool");
    eprintln!("  Pool has {} deals", pool.len());

    let end = (offset + num_deals).min(pool.len());
    let actual = end - offset;
    if actual < num_deals {
        eprintln!("  WARNING: only {} deals available from offset {}", actual, offset);
    }
    let num_deals = actual;
    let sampled: Vec<(u8, [u32; 4])> = (offset..end)
        .map(|idx| {
            let deal = pool.get(idx);
            (deal.dealer, deal.hands)
        })
        .collect();
    eprintln!("  Taking deals [{}, {}) ({} deals)", offset, end, num_deals);

    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let config = IsDdConfig {
        determinizations: dets,
        time_limit_ms: Some(time_ms),
        // Explicite, et non hérité du défaut : c'est l'échelle des `q_values`
        // écrits dans le fichier. Sous `DealScore` ce sont des écarts de score
        // de donne (±500), pas des points cartes — d'où la version 2 du format.
        objective: PlayObjective::DealScore,
        ..Default::default()
    };

    let total_games = num_deals * 4;
    eprintln!(
        "\nDistilling {} deals × 4 suits = {} games (IS-DD {}ms × {} dets, skip_forced={})",
        num_deals, total_games, time_ms, dets, skip_forced
    );
    eprintln!("  q_values scale: deal-score margin NS-EW (COLVPD01 v2)");

    let start = Instant::now();
    let progress = AtomicUsize::new(0);

    // Each game produces a Vec<Record>. Collect all in parallel.
    let games: Vec<(usize, u8)> = (0..num_deals)
        .flat_map(|d| (0..4u8).map(move |s| (d, s)))
        .collect();

    use rayon::prelude::*;

    let all_records: Vec<Vec<Record>> = games
        .par_iter()
        .map(|&(deal_idx, suit)| {
            let (dealer, hands) = sampled[deal_idx];
            let mut rng = StdRng::seed_from_u64(seed + deal_idx as u64 * 100 + suit as u64);
            let mut state = GameState::setup_dd(dealer, hands, suit);
            let mut search = IsDdSearch::new();

            // Buffer records as we play; we'll backfill final_ns_pts at the end.
            let mut buf: Vec<Record> = Vec::with_capacity(32);

            while state.phase == Phase::Playing {
                let result = search.search_with_stats(&state, &config, &mut rng);
                let chosen = result.best_action;
                let n_legal = result.card_scores.len() as u8;

                let write_row = !(skip_forced && n_legal <= 1);
                if write_row {
                    let deal_id = (offset + deal_idx) as u32;
                    let seat = state.current_player;
                    let mut trick_bytes = [0xFFu8; 4];
                    for s in 0..4 {
                        trick_bytes[s] = state.current_trick[s];
                    }
                    let q_values: Vec<(u8, f32)> = result.card_scores.iter().copied().collect();

                    buf.push(Record {
                        deal_id,
                        forced_suit: suit,
                        dealer,
                        trick_idx: state.tricks_won[0] + state.tricks_won[1],
                        play_idx: state.trick_count,
                        seat,
                        trick_lead: state.trick_lead,
                        chosen,
                        n_legal,
                        final_ns_pts: 0, // backfilled below
                        hand: state.hands[seat as usize],
                        legal: state.legal_actions() as u32,
                        played_cards: state.played_cards,
                        trick_packed: pack_trick(&trick_bytes),
                        voids_packed: pack_voids(&state.voids),
                        q_values,
                    });
                }

                state.step(chosen);
            }

            let final_ns = state.points[0];
            for r in buf.iter_mut() {
                r.final_ns_pts = final_ns;
            }

            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 400 == 0 || done == total_games {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = done as f64 / elapsed;
                let eta = (total_games - done) as f64 / rate;
                eprintln!(
                    "  {}/{} games ({:.1}/s) {:.1}s elapsed, ETA {:.0}s",
                    done, total_games, rate, elapsed, eta
                );
            }

            buf
        })
        .collect();

    let elapsed = start.elapsed().as_secs_f64();
    let total_records: usize = all_records.iter().map(|v| v.len()).sum();
    eprintln!(
        "\nDone: {} games in {:.1}s ({:.1} games/s), {} records",
        total_games,
        elapsed,
        total_games as f64 / elapsed,
        total_records,
    );

    // Serialize all records into one buffer, then write.
    let mut payload: Vec<u8> = Vec::with_capacity(total_records * 64);
    for game_records in &all_records {
        for r in game_records {
            r.write_to(&mut payload);
        }
    }

    let mut f = BufWriter::new(File::create(&output_path).expect("Cannot create output"));
    f.write_all(b"COLVPD01").unwrap();
    // v2 (2026-08-03) : `q_values` passe des points cartes N-S (0-252) à
    // l'écart de score de donne N-S − E-O, en même temps que l'objectif
    // par défaut d'IS-DD. Même schéma binaire, autre échelle.
    f.write_all(&2u8.to_le_bytes()).unwrap(); // version
    f.write_all(&[0u8; 7]).unwrap(); // pad to 16 bytes
    f.write_all(&(total_records as u64).to_le_bytes()).unwrap();
    f.write_all(&payload).unwrap();
    f.flush().unwrap();

    let size_mb = (16 + 8 + payload.len()) as f64 / 1_048_576.0;
    eprintln!(
        "Wrote {} records to {} ({:.1} MB, avg {:.1} bytes/record)",
        total_records,
        output_path,
        size_mb,
        payload.len() as f64 / total_records.max(1) as f64,
    );
}

// ============================================================================
// COLVPD01 binary format (Play Distill v2)
// ============================================================================
//
// Header (24 bytes):
//   magic        : [u8; 8]  = "COLVPD01"
//   version      : u8       = 2   (1 = q_values en points cartes N-S 0-252)
//   _pad         : [u8; 7]  = 0
//   n_records    : u64      (little-endian)
//
// Each record (variable length, 34 + n_legal*5 bytes):
//   deal_id      : u32  (le)  index into source pool
//   forced_suit  : u8         trump suit forced for this game (0..3)
//   dealer       : u8         dealer seat (0..3)
//   trick_idx    : u8         which trick (0..7)
//   play_idx     : u8         position in current trick (0..3)
//   seat         : u8         current player (0..3) — NS = {0,2}, EW = {1,3}
//   trick_lead   : u8         seat that led the current trick
//   chosen       : u8         card chosen by IS-DD (0..31)
//   n_legal      : u8         number of legal cards (>= 1; usually >= 2 since
//                              forced rows are skipped by default)
//   final_ns_pts : u8         NS team's final points for this game (0..252)
//   _pad         : u8         = 0
//   hand         : u32 (le)   bitmask of current player's remaining cards
//   legal        : u32 (le)   bitmask of legal cards at this decision
//   played_cards : u32 (le)   bitmask of all cards played this deal so far
//   trick_packed : u32 (le)   4 packed u8: cards already played in current
//                              trick, indexed by seat (0xFF = empty)
//   voids_packed : u32 (le)   4 packed u8: engine's known voids per seat
//                              (bit i = void in suit i)
//   q_values     : [(u8, f32); n_legal]
//                              For each legal card: (card_idx, value).
//                              v2: value = deal-score margin NS-EW (contract,
//                                  chute, belote, capot included; roughly ±500).
//                              v1: value = NS card points (0-252).
//                              ns_points is from NS team's perspective (0..252).
//                              Defenders pick min, declarer picks max.
//
// Card representation: see colver_core::card. 32 cards laid out by suit.
//   Spades [0..7], Hearts [8..15], Diamonds [16..23], Clubs [24..31].
//   Trump rank order: J(3) > 9(2) > A(7) > 10(6) > K(5) > Q(4) > 8(1) > 7(0).
// ============================================================================
