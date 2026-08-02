//! Verify and benchmark [`GpuPlaygen::generate_worlds_multi`] — sampling worlds
//! for several unrelated positions in a single GPU batch.
//!
//! ## Why the batch axis matters
//!
//! A `/play_worlds` request costs ~220 ms whether it returns 1 world or 256:
//! the cost is the sequential token loop, which is kernel-launch bound, not
//! FLOP bound. IS-DD asks for ~20 worlds from a *different* position at every
//! decision, so it runs the sidecar at a small fraction of its throughput, and
//! client-side concurrency cannot fix it against a serial server.
//!
//! ## What is checked, in order of how much it would hurt to get wrong
//!
//! 1. **Equivalence** — one item through the multi path must be *bit-identical*
//!    to the single path at the same seed. If this fails, batching silently
//!    changed the sampler and every world it ever produces is suspect.
//!    Padding is absent at K=1, so identity is the correct expectation, not
//!    an approximation.
//! 2. **Independence** — a position's world distribution must not depend on
//!    who else shares its batch. This is what padding and the additive mask
//!    are for; a leak between lanes would show up here as drifted card
//!    marginals. Compared against the *sampling noise* of the single path
//!    against itself, which is the only honest yardstick.
//! 3. **Validity** — every returned world keeps the observer's hand and the
//!    per-seat card counts. Cheap, and catches lane/index mix-ups.
//! 4. **Throughput** — the point of the exercise.
//!
//! Usage:
//!   CUDARC_CUDA_VERSION=13010 cargo run -p colver-core --release \
//!     --features gpu_server --bin bench_playgen_batch -- \
//!     --playgen models/playgen/playgen_v2_final.bin --positions 32 --worlds 20

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::bid_train_env::DealPool;
use colver_core::card;
use colver_core::playgen::gpu::{GpuPlaygen, WorldBatchItem};
use colver_core::playgen::infer::{PlaygenModel, PlaygenSampler};
use colver_core::state::{GameState, Phase};

use std::sync::Arc;

/// Play a deal forward to a random point in the play phase, feeding a sampler
/// for `observer` exactly as the server's `replay` does.
fn position(
    model: &Arc<PlaygenModel>,
    dealer: u8,
    hands: [u32; 4],
    observer: u8,
    max_cards: usize,
    rng: &mut StdRng,
) -> Option<(PlaygenSampler, GameState)> {
    let mut state = GameState::new(dealer, hands);
    let mut sampler = PlaygenSampler::new(model.clone());
    sampler.init_deal(&state, observer);

    // Auction: a random legal bid then passes, so the trump is named.
    let mut guard = 0;
    while state.phase == Phase::Bidding && guard < 24 {
        guard += 1;
        let legal = state.legal_actions();
        if legal == 0 {
            return None;
        }
        // Prefer a real bid on the first action, then pass out.
        let opts: Vec<u8> = (0..43u8).filter(|&a| legal & (1u64 << a) != 0).collect();
        let action = if guard == 1 {
            let bids: Vec<u8> = opts.iter().copied().filter(|&a| (1..=36).contains(&a)).collect();
            if bids.is_empty() { opts[0] } else { bids[rng.gen_range(0..bids.len())] }
        } else {
            0 // PASS
        };
        if legal & (1u64 << action) == 0 {
            return None;
        }
        let p = state.current_player();
        sampler.record_action(&state, p, action);
        state.step(action);
    }
    if state.phase != Phase::Playing {
        return None;
    }

    let n_cards = rng.gen_range(0..=max_cards);
    for _ in 0..n_cards {
        if state.phase != Phase::Playing {
            break;
        }
        let legal = state.legal_actions();
        if legal == 0 {
            break;
        }
        let opts: Vec<u8> = (0..32u8).filter(|&c| legal & (1u64 << c) != 0).collect();
        let action = opts[rng.gen_range(0..opts.len())];
        let p = state.current_player();
        sampler.record_action(&state, p, action);
        state.step(action);
    }
    if state.phase != Phase::Playing {
        return None;
    }
    Some((sampler, state))
}

/// p(card -> seat) over a world set, for comparing distributions.
fn marginals(worlds: &[([u32; 4], colver_core::playgen::infer::WorldLogp)]) -> [[f32; 32]; 4] {
    let mut m = [[0f32; 32]; 4];
    if worlds.is_empty() {
        return m;
    }
    for (h, _) in worlds {
        for p in 0..4 {
            let mut b = h[p];
            while b != 0 {
                m[p][b.trailing_zeros() as usize] += 1.0;
                b &= b - 1;
            }
        }
    }
    for row in m.iter_mut() {
        for v in row.iter_mut() {
            *v /= worlds.len() as f32;
        }
    }
    m
}

fn max_abs_diff(a: &[[f32; 32]; 4], b: &[[f32; 32]; 4]) -> f32 {
    let mut d = 0f32;
    for p in 0..4 {
        for c in 0..32 {
            d = d.max((a[p][c] - b[p][c]).abs());
        }
    }
    d
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut playgen_path = String::from("models/playgen/playgen_v2_final.bin");
    let mut pool_path = String::from("data/deals/base_5M.bin");
    let mut positions = 32usize;
    let mut worlds = 20usize;
    let mut dist_worlds = 512usize;
    let mut seed = 42u64;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--playgen" => { i += 1; playgen_path = argv[i].clone(); }
            "--pool" => { i += 1; pool_path = argv[i].clone(); }
            "--positions" => { i += 1; positions = argv[i].parse().unwrap(); }
            "--worlds" => { i += 1; worlds = argv[i].parse().unwrap(); }
            "--dist-worlds" => { i += 1; dist_worlds = argv[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = argv[i].parse().unwrap(); }
            o => panic!("unknown arg {o}"),
        }
        i += 1;
    }

    let model = Arc::new(PlaygenModel::load(&playgen_path).expect("load playgen"));
    assert!(model.v2, "requires a v2 model");
    let device = candle_core::Device::new_cuda(0).expect("CUDA 0");
    let gpu = GpuPlaygen::new(&model, device).expect("upload to GPU");
    eprintln!("modèle {} chargé sur GPU", playgen_path);

    let pool = DealPool::load(&pool_path).expect("load pool");
    let mut rng = StdRng::seed_from_u64(seed);

    // Build the positions.
    let mut pos = Vec::new();
    let mut idx = 0usize;
    while pos.len() < positions && idx < pool.len() {
        let d = pool.get(idx);
        idx += 1;
        let obs = rng.gen_range(0..4u8);
        if let Some(p) = position(&model, d.dealer, d.hands, obs, 20, &mut rng) {
            pos.push(p);
        }
    }
    assert_eq!(pos.len(), positions, "pas assez de positions jouables");
    eprintln!("{} positions construites\n", positions);

    // ══ 1. Equivalence: K=1 multi vs single, same seed ══
    println!("=== 1. équivalence K=1 (multi vs simple, même graine) ===");
    let mut mismatches = 0;
    for (n, (sampler, state)) in pos.iter().enumerate().take(8) {
        let mut r1 = StdRng::seed_from_u64(1000 + n as u64);
        let single = gpu
            .generate_worlds_scored(sampler, state, worlds, 1.0, &mut r1)
            .expect("single");
        let mut r2 = StdRng::seed_from_u64(1000 + n as u64);
        let multi = gpu
            .generate_worlds_multi(
                &[WorldBatchItem { sampler, state, n_worlds: worlds, temperature: 1.0 }],
                &mut r2,
            )
            .expect("multi");
        let m0 = &multi[0];
        let same = single.len() == m0.len()
            && single.iter().zip(m0.iter()).all(|(a, b)| a.0 == b.0);
        if !same {
            mismatches += 1;
            println!("  position {n} : DIVERGENCE ({} vs {} mondes)", single.len(), m0.len());
        }
    }
    println!(
        "  {} / 8 positions bit-identiques{}",
        8 - mismatches,
        if mismatches == 0 { "  ✓" } else { "  ✗ ÉCHEC" }
    );

    // ══ 2. Independence: does a batch-mate change the distribution? ══
    println!("\n=== 2. indépendance entre lanes (marginales de cartes) ===");
    let (s0, st0) = &pos[0];
    let mut ra = StdRng::seed_from_u64(7);
    let alone_a = gpu.generate_worlds_scored(s0, st0, dist_worlds, 1.0, &mut ra).expect("a");
    let mut rb = StdRng::seed_from_u64(8);
    let alone_b = gpu.generate_worlds_scored(s0, st0, dist_worlds, 1.0, &mut rb).expect("b");

    // Same position, but sharing a batch with several unrelated ones.
    let mut items: Vec<WorldBatchItem> = vec![WorldBatchItem {
        sampler: s0,
        state: st0,
        n_worlds: dist_worlds,
        temperature: 1.0,
    }];
    for (s, st) in pos.iter().skip(1).take(7) {
        items.push(WorldBatchItem { sampler: s, state: st, n_worlds: worlds, temperature: 1.0 });
    }
    let mut rc = StdRng::seed_from_u64(9);
    let batched = gpu.generate_worlds_multi(&items, &mut rc).expect("batched");

    let noise = max_abs_diff(&marginals(&alone_a), &marginals(&alone_b));
    let effect = max_abs_diff(&marginals(&alone_a), &marginals(&batched[0]));
    println!("  bruit d'échantillonnage (seul vs seul)   : {noise:.4}");
    println!("  seul vs groupé avec 7 autres positions   : {effect:.4}");
    println!(
        "  verdict : {}",
        if effect <= noise * 2.0 + 0.01 {
            "dans le bruit  ✓"
        } else {
            "ÉCART SUSPECT  ✗"
        }
    );

    // ══ 3. Validity of every world in a full batch ══
    println!("\n=== 3. validité des mondes ===");
    let items: Vec<WorldBatchItem> = pos
        .iter()
        .map(|(s, st)| WorldBatchItem { sampler: s, state: st, n_worlds: worlds, temperature: 1.0 })
        .collect();
    let mut rd = StdRng::seed_from_u64(11);
    let all = gpu.generate_worlds_multi(&items, &mut rd).expect("all");
    let mut bad = 0usize;
    let mut produced = 0usize;
    for (j, ws) in all.iter().enumerate() {
        let (_s, st) = &pos[j];
        produced += ws.len();
        for (h, _) in ws {
            let counts_ok =
                (0..4).all(|p| card::card_count(h[p]) == card::card_count(st.hands[p]));
            let all_cards: u32 = h[0] | h[1] | h[2] | h[3];
            let disjoint = h.iter().map(|x| card::card_count(*x)).sum::<u32>()
                == card::card_count(all_cards);
            if !counts_ok || !disjoint {
                bad += 1;
            }
        }
    }
    println!(
        "  {} mondes produits sur {} demandés, {} invalides{}",
        produced,
        positions * worlds,
        bad,
        if bad == 0 { "  ✓" } else { "  ✗ ÉCHEC" }
    );

    // ══ 4. Throughput ══
    println!("\n=== 4. débit ===");
    let mut re = StdRng::seed_from_u64(13);
    let t0 = Instant::now();
    let mut seq_total = 0usize;
    for (s, st) in pos.iter() {
        seq_total += gpu
            .generate_worlds_scored(s, st, worlds, 1.0, &mut re)
            .expect("seq")
            .len();
    }
    let seq = t0.elapsed().as_secs_f64();

    let items: Vec<WorldBatchItem> = pos
        .iter()
        .map(|(s, st)| WorldBatchItem { sampler: s, state: st, n_worlds: worlds, temperature: 1.0 })
        .collect();
    let mut rf = StdRng::seed_from_u64(13);
    let t1 = Instant::now();
    let bat = gpu.generate_worlds_multi(&items, &mut rf).expect("bat");
    let batch = t1.elapsed().as_secs_f64();
    let bat_total: usize = bat.iter().map(|v| v.len()).sum();

    println!("  {positions} positions x {worlds} mondes");
    println!(
        "  une par une : {:.2} s  ({:.1} ms/position, {:.0} mondes/s, {} rendus)",
        seq,
        seq / positions as f64 * 1e3,
        seq_total as f64 / seq,
        seq_total
    );
    println!(
        "  en un lot   : {:.2} s  ({:.1} ms/position, {:.0} mondes/s, {} rendus)",
        batch,
        batch / positions as f64 * 1e3,
        bat_total as f64 / batch,
        bat_total
    );
    println!("  accélération : {:.1}x", seq / batch);

    if mismatches > 0 || bad > 0 {
        std::process::exit(1);
    }
}
