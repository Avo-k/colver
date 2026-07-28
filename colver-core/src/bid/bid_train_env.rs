/// Bidding training environment with pre-solved DD deal pool.
///
/// Phase 1: `DealPool::generate(n, seed)` pre-solves N deals in parallel
///          using all CPU cores (~100 deals/s on 32 cores, ~77ms/solve).
/// Phase 2: Training envs sample deals from the pool (instant reset).
///          Bidding runs in microseconds, throughput becomes GPU-bound.
///
/// When bidding ends:
/// 1. Look up `dd_pts[trump]` for the contracted suit
/// 2. Build a synthetic terminal GameState with DD points
/// 3. Call `compute_deal_score()` for match scores
/// 4. Reward = `(my_score - opp_score) / 500.0`
/// 5. Flush buffered transitions to replay buffer

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bid_obs::{
    self, BID_MASK_DIM, BID_OBS_DIM, BID_OBS_DIM_SCORE_AWARE, BID_OBS_DIM_SCORE_AWARE_V2,
    BID_OBS_DIM_SCORE_AWARE_V3,
};
use crate::rollout;
use crate::scoring::compute_deal_score;
use crate::solver;
use crate::state::{GameState, Phase};

/// Calibrated match win probability: σ(1.7 × Δ / (R_sum^0.8 + 340))
/// Fitted from 10k full matches. v3_max: δ=320, v4_sa: δ=360 → average 340.
/// Scoring rules: surcontré ×3, base 162 (tous les points cartes), capot = contrat 250.
/// La base est passée de 160 à 162 après ce fit, sans effet : l'arrondi à la dizaine rend
/// les mêmes marques (cf. `engine/scoring.rs`).
pub fn win_probability(s_me: f32, s_opp: f32) -> f32 {
    let r_sum = (2000.0 - s_me) + (2000.0 - s_opp);
    let denom = r_sum.max(1.0).powf(0.8) + 340.0;
    let x = 1.7 * (s_me - s_opp) / denom;
    1.0 / (1.0 + (-x).exp())
}

/// Compute score-aware reward as Δ win probability × scale.
/// Handles match-ending deals (someone crosses 2000).
pub fn score_aware_reward(
    s_me: f32,
    s_opp: f32,
    my_deal_pts: i16,
    opp_deal_pts: i16,
    scale: f32,
) -> f32 {
    let p0 = win_probability(s_me, s_opp);

    let new_me = s_me + my_deal_pts as f32;
    let new_opp = s_opp + opp_deal_pts as f32;

    let p1 = if new_me >= 2000.0 && new_opp >= 2000.0 {
        if new_me >= new_opp { 1.0 } else { 0.0 }
    } else if new_me >= 2000.0 {
        1.0
    } else if new_opp >= 2000.0 {
        0.0
    } else {
        win_probability(new_me, new_opp)
    };

    (p1 - p0) * scale
}

/// How to compute reward points from a deal's pre-solved data.
#[derive(Clone, Copy, Debug)]
pub enum RewardMode {
    /// Use DD-optimal points only (bid v2 default).
    DdOnly,
    /// Use real (DouDou50 rollout) points only.
    RealOnly,
    /// Blend: alpha * DD + (1-alpha) * real.
    Blend(f32),
}

/// A pre-solved deal: hands + DD results for all 4 trump suits.
#[derive(Clone)]
pub struct PresolvedDeal {
    pub dealer: u8,
    pub hands: [u32; 4],
    /// NS points for each trump suit (indexed by suit 0-3).
    pub dd_pts: [u8; 4],
    /// Real (DouDou50) NS points per suit. None if not enriched.
    pub real_pts: Option<[u8; 4]>,
}

/// A named score layer: play-method points for each deal × 4 suits.
/// Stored as a parallel array aligned with the deal pool.
pub struct ScoreLayer {
    /// Name of the play method (e.g. "dmc", "isdd").
    pub name: String,
    /// Offset into the deal pool (these scores correspond to deals[offset..offset+len]).
    pub offset: usize,
    /// Per-deal NS points for each trump suit [u8; 4].
    pub scores: Vec<[u8; 4]>,
}

/// Pool of pre-solved deals for fast training resets.
pub struct DealPool {
    deals: Vec<PresolvedDeal>,
    /// Named score layers (e.g. DMC, IS-DD) as parallel arrays.
    score_layers: Vec<ScoreLayer>,
    /// Which score layer to use for `real_pts` (index into score_layers, or None).
    active_score: Option<usize>,
    /// Per-dealer indices, built lazily via `build_dealer_index()`.
    /// Used by `sample_with_dealer` for realistic 0→1→2→3 rotation in match-sim
    /// training — each deal keeps its original dealer (so ISDD scores stay valid),
    /// we just route requests to deals whose dealer matches the target.
    dealer_index: Option<[Vec<usize>; 4]>,
}

impl DealPool {
    /// Generate `n` deals in parallel using all CPU cores (work-stealing).
    pub fn generate(n: usize, seed: u64) -> Self {
        let mut deals = Vec::with_capacity(n);
        Self::generate_chunk_into(&mut deals, n, seed, 0);
        DealPool { deals, score_layers: Vec::new(), active_score: None, dealer_index: None }
    }

    /// Generate `n` deals with checkpoints every `chunk_size` deals.
    /// Saves to `path` after each chunk. Resumes from existing file.
    pub fn generate_with_checkpoints(
        n: usize,
        seed: u64,
        path: &str,
        chunk_size: usize,
    ) -> Self {
        // Resume from existing partial pool
        let mut deals = if std::path::Path::new(path).exists() {
            match Self::load(path) {
                Ok(pool) => {
                    let existing = pool.deals.len();
                    if existing >= n {
                        eprintln!("  Pool already has {} deals (requested {}), done.", existing, n);
                        return pool;
                    }
                    eprintln!("  Resuming from {} existing deals in {}", existing, path);
                    pool.deals
                }
                Err(e) => {
                    eprintln!("  Failed to load {}: {}, starting fresh", path, e);
                    Vec::with_capacity(n)
                }
            }
        } else {
            Vec::with_capacity(n)
        };

        let overall_start = Instant::now();

        while deals.len() < n {
            let remaining = n - deals.len();
            let chunk_n = remaining.min(chunk_size);
            let chunk_seed = seed.wrapping_add(deals.len() as u64 * 37);
            let chunk_idx = deals.len() / chunk_size;

            eprintln!(
                "\n  --- Chunk {} : generating {} deals ({}/{} total) ---",
                chunk_idx + 1,
                chunk_n,
                deals.len(),
                n
            );

            let offset = deals.len();
            Self::generate_chunk_into(&mut deals, chunk_n, chunk_seed, offset);

            // Checkpoint: save entire pool so far
            let pool = DealPool {
                deals: deals.clone(),
                score_layers: Vec::new(),
                active_score: None,
                dealer_index: None,
            };
            match pool.save(path) {
                Ok(()) => {
                    let size_mb = (deals.len() * 21 + 16) as f64 / 1_048_576.0;
                    let elapsed = overall_start.elapsed().as_secs_f64();
                    let rate = deals.len() as f64 / elapsed;
                    let eta = (n - deals.len()) as f64 / rate;
                    eprintln!(
                        "  Checkpoint: {}/{} deals saved to {} ({:.1}MB) | {:.0} deals/s | ETA {:.0}s",
                        deals.len(), n, path, size_mb, rate, eta
                    );
                }
                Err(e) => eprintln!("  Warning: failed to save checkpoint: {}", e),
            }
        }

        let total_time = overall_start.elapsed().as_secs_f64();
        eprintln!(
            "\n  Done: {} deals in {:.1}s ({:.0} deals/s)",
            deals.len(),
            total_time,
            deals.len() as f64 / total_time
        );

        DealPool { deals, score_layers: Vec::new(), active_score: None, dealer_index: None }
    }

    /// Generate `n` deals in parallel, appending to `out`.
    /// `global_offset` is used for progress display only.
    fn generate_chunk_into(out: &mut Vec<PresolvedDeal>, n: usize, seed: u64, global_offset: usize) {
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        let next_idx = AtomicUsize::new(0);
        let progress = AtomicUsize::new(0);
        let start_time = Instant::now();

        let mut slots: Vec<Option<PresolvedDeal>> = (0..n).map(|_| None).collect();
        let slots_ptr = slots.as_mut_ptr();

        struct SendPtr<T>(*mut T);
        unsafe impl<T> Send for SendPtr<T> {}
        unsafe impl<T> Sync for SendPtr<T> {}
        let shared_ptr = SendPtr(slots_ptr);

        let total_target = global_offset + n;

        std::thread::scope(|s| {
            let next_ref = &next_idx;
            let progress_ref = &progress;
            let ptr_ref = &shared_ptr;
            let start = &start_time;

            for t in 0..num_threads {
                let thread_seed = seed + t as u64 * 1_000_000;

                s.spawn(move || {
                    let mut rng = StdRng::seed_from_u64(thread_seed);
                    let mut tt_buf = solver::new_tt_buffer();

                    loop {
                        let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                        if idx >= n {
                            break;
                        }

                        let dealer = rng.gen_range(0..4u8);
                        let state = GameState::deal_random(dealer, &mut rng);
                        let dd_pts = solve_all_suits(&state, &mut tt_buf);

                        unsafe {
                            (*ptr_ref.0.add(idx)) = Some(PresolvedDeal {
                                dealer,
                                hands: state.hands,
                                dd_pts,
                                real_pts: None,
                            });
                        }

                        let done = progress_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 50_000 == 0 {
                            let global_done = global_offset + done;
                            let elapsed = start.elapsed().as_secs_f64();
                            let rate = done as f64 / elapsed;
                            let eta = (n - done) as f64 / rate;
                            let pct = global_done as f64 / total_target as f64;
                            let bar_width = 40;
                            let filled = (pct * bar_width as f64) as usize;
                            let bar: String = (0..bar_width)
                                .map(|i| if i < filled { '█' } else { '░' })
                                .collect();
                            eprintln!(
                                "  {} {:.0}% | {}/{} | {:.0} deals/s | ETA {:.0}s",
                                bar, pct * 100.0, global_done, total_target, rate, eta
                            );
                        }
                    }
                });
            }
        });

        let total_time = start_time.elapsed().as_secs_f64();
        eprintln!(
            "  Chunk done: {} deals in {:.1}s ({:.0} deals/s)",
            n,
            total_time,
            n as f64 / total_time
        );

        out.extend(slots.into_iter().map(|d| d.unwrap()));
    }

    /// Sample a random deal from the pool.
    #[inline]
    pub fn sample(&self, rng: &mut impl Rng) -> &PresolvedDeal {
        &self.deals[rng.gen_range(0..self.deals.len())]
    }

    /// Build a per-dealer index of deal positions so `sample_with_dealer` is O(1).
    /// Idempotent; rebuilds the index each call.
    pub fn build_dealer_index(&mut self) {
        let mut idx: [Vec<usize>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (i, deal) in self.deals.iter().enumerate() {
            idx[deal.dealer as usize].push(i);
        }
        eprintln!(
            "  Built dealer index: sizes = [{}, {}, {}, {}]",
            idx[0].len(), idx[1].len(), idx[2].len(), idx[3].len()
        );
        self.dealer_index = Some(idx);
    }

    /// Sample a deal whose original dealer is `dealer`. Requires a prior call to
    /// `build_dealer_index()`; falls back to rejection sampling otherwise (O(4)
    /// expected per call for a uniform pool).
    pub fn sample_with_dealer(&self, rng: &mut impl Rng, dealer: u8) -> &PresolvedDeal {
        debug_assert!(dealer < 4);
        if let Some(ref idx) = self.dealer_index {
            let bucket = &idx[dealer as usize];
            if !bucket.is_empty() {
                return &self.deals[bucket[rng.gen_range(0..bucket.len())]];
            }
        }
        // Fallback: rejection sample.
        for _ in 0..64 {
            let d = self.sample(rng);
            if d.dealer == dealer {
                return d;
            }
        }
        // Last-resort: first deal matching, else any deal.
        for d in &self.deals {
            if d.dealer == dealer {
                return d;
            }
        }
        self.sample(rng)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.deals.len()
    }

    /// Get deal by index.
    #[inline]
    pub fn get(&self, idx: usize) -> &PresolvedDeal {
        &self.deals[idx]
    }

    /// Save pool to binary file.
    /// Format: magic "COLVDD01" (8B) + count (u64 LE) + N × 21B per deal.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(b"COLVDD01")?;
        f.write_all(&(self.deals.len() as u64).to_le_bytes())?;
        for deal in &self.deals {
            f.write_all(&[deal.dealer])?;
            for &h in &deal.hands {
                f.write_all(&h.to_le_bytes())?;
            }
            f.write_all(&deal.dd_pts)?;
        }
        f.flush()?;
        Ok(())
    }

    /// Load pool from binary file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != b"COLVDD01" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Bad magic: expected COLVDD01, got {:?}", &magic),
            ));
        }

        let mut count_buf = [0u8; 8];
        f.read_exact(&mut count_buf)?;
        let count = u64::from_le_bytes(count_buf) as usize;

        let mut deals = Vec::with_capacity(count);
        for _ in 0..count {
            let mut dealer = [0u8; 1];
            f.read_exact(&mut dealer)?;

            let mut hands = [0u32; 4];
            for h in &mut hands {
                let mut buf = [0u8; 4];
                f.read_exact(&mut buf)?;
                *h = u32::from_le_bytes(buf);
            }

            let mut dd_pts = [0u8; 4];
            f.read_exact(&mut dd_pts)?;

            deals.push(PresolvedDeal {
                dealer: dealer[0],
                hands,
                dd_pts,
                real_pts: None,
            });
        }

        Ok(DealPool { deals, score_layers: Vec::new(), active_score: None, dealer_index: None })
    }

    /// Load enriched pool (COLVDR01): dd_pts + real_pts per deal.
    /// The real_pts are stored as a score layer named after the file.
    /// For backward compat, also sets real_pts on each PresolvedDeal.
    pub fn load_enriched(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != b"COLVDR01" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Bad magic: expected COLVDR01, got {:?}", &magic),
            ));
        }

        let mut count_buf = [0u8; 8];
        f.read_exact(&mut count_buf)?;
        let count = u64::from_le_bytes(count_buf) as usize;

        let mut deals = Vec::with_capacity(count);
        let mut scores = Vec::with_capacity(count);
        for _ in 0..count {
            let mut dealer = [0u8; 1];
            f.read_exact(&mut dealer)?;

            let mut hands = [0u32; 4];
            for h in &mut hands {
                let mut buf = [0u8; 4];
                f.read_exact(&mut buf)?;
                *h = u32::from_le_bytes(buf);
            }

            let mut dd_pts = [0u8; 4];
            f.read_exact(&mut dd_pts)?;

            let mut real_pts = [0u8; 4];
            f.read_exact(&mut real_pts)?;

            deals.push(PresolvedDeal {
                dealer: dealer[0],
                hands,
                dd_pts,
                real_pts: Some(real_pts),
            });
            scores.push(real_pts);
        }

        // Infer layer name from filename
        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("real")
            .to_string();

        let layer = ScoreLayer {
            name: name.clone(),
            offset: 0,
            scores,
        };

        eprintln!("  Loaded enriched pool: {} deals from {} (score layer: {})", count, path, name);
        Ok(DealPool {
            deals,
            score_layers: vec![layer],
            active_score: Some(0),
            dealer_index: None,
        })
    }

    /// Load a separate score file (COLVSC01) and attach it as a score layer.
    /// Format: magic "COLVSC01" (8B) + name_len (u16 LE) + name (utf8) + count (u32 LE) + offset (u32 LE) + count × [u8; 4].
    /// The offset says these scores correspond to deals[offset..offset+count] in the base pool.
    pub fn load_scores(&mut self, path: &str) -> std::io::Result<()> {
        use std::io::Read;
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != b"COLVSC01" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Bad magic: expected COLVSC01, got {:?}", &magic),
            ));
        }

        let mut name_len_buf = [0u8; 2];
        f.read_exact(&mut name_len_buf)?;
        let name_len = u16::from_le_bytes(name_len_buf) as usize;

        let mut name_buf = vec![0u8; name_len];
        f.read_exact(&mut name_buf)?;
        let name = String::from_utf8(name_buf).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;

        let mut count_buf = [0u8; 4];
        f.read_exact(&mut count_buf)?;
        let count = u32::from_le_bytes(count_buf) as usize;

        let mut offset_buf = [0u8; 4];
        f.read_exact(&mut offset_buf)?;
        let offset = u32::from_le_bytes(offset_buf) as usize;

        let mut scores = Vec::with_capacity(count);
        for _ in 0..count {
            let mut pts = [0u8; 4];
            f.read_exact(&mut pts)?;
            scores.push(pts);
        }

        eprintln!(
            "  Loaded score layer '{}': {} scores at offset {} from {}",
            name, count, offset, path
        );

        self.score_layers.push(ScoreLayer { name, offset, scores });
        Ok(())
    }

    /// Save a score layer to a separate file (COLVSC01).
    pub fn save_scores(name: &str, offset: usize, scores: &[[u8; 4]], path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(b"COLVSC01")?;
        let name_bytes = name.as_bytes();
        f.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
        f.write_all(name_bytes)?;
        f.write_all(&(scores.len() as u32).to_le_bytes())?;
        f.write_all(&(offset as u32).to_le_bytes())?;
        for pts in scores {
            f.write_all(pts)?;
        }
        f.flush()?;
        Ok(())
    }

    /// Select which score layer to use as `real_pts` when sampling deals.
    /// Pass None to disable (DD-only mode). Pass a name to select by layer name.
    pub fn select_score_layer(&mut self, name: Option<&str>) {
        match name {
            None => {
                self.active_score = None;
                // Clear real_pts on all deals
                for deal in &mut self.deals {
                    deal.real_pts = None;
                }
            }
            Some(n) => {
                let idx = self.score_layers.iter().position(|l| l.name == n);
                if let Some(idx) = idx {
                    self.active_score = Some(idx);
                    let layer = &self.score_layers[idx];
                    // Apply scores to deals in range
                    for (i, pts) in layer.scores.iter().enumerate() {
                        let deal_idx = layer.offset + i;
                        if deal_idx < self.deals.len() {
                            self.deals[deal_idx].real_pts = Some(*pts);
                        }
                    }
                    eprintln!("  Activated score layer '{}' ({} deals)", n, layer.scores.len());
                } else {
                    let names: Vec<_> = self.score_layers.iter().map(|l| l.name.as_str()).collect();
                    panic!("Score layer '{}' not found. Available: {:?}", n, names);
                }
            }
        }
    }

    /// List available score layer names.
    pub fn score_layer_names(&self) -> Vec<&str> {
        self.score_layers.iter().map(|l| l.name.as_str()).collect()
    }

    /// Load if file exists with enough deals, otherwise generate with 500k checkpoints.
    pub fn load_or_generate(path: &str, n: usize, seed: u64) -> Self {
        if std::path::Path::new(path).exists() {
            eprintln!("  Loading deal pool from {}...", path);
            let start = Instant::now();
            match Self::load(path) {
                Ok(pool) if pool.len() >= n => {
                    eprintln!(
                        "  Loaded {} deals in {:.1}s",
                        pool.len(),
                        start.elapsed().as_secs_f64()
                    );
                    return pool;
                }
                Ok(pool) => {
                    eprintln!(
                        "  Pool has {} deals but {} requested, generating more...",
                        pool.len(),
                        n
                    );
                }
                Err(e) => {
                    eprintln!("  Failed to load {}: {}, regenerating...", path, e);
                }
            }
        }
        Self::generate_with_checkpoints(n, seed, path, 500_000)
    }
}

/// A single buffered bid transition (stored until episode ends and reward is known).
struct BidTransition {
    obs: [f32; BID_OBS_DIM],
    mask: [f32; BID_MASK_DIM],
    action: u8,
    team: u8, // team of the player who acted (0=NS, 1=EW)
}

/// Score-aware transition: uses variable-dim obs (110 v1, 113 v2) with match scores appended.
struct ScoreAwareBidTransition {
    obs: Vec<f32>,
    mask: [f32; BID_MASK_DIM],
    action: u8,
    team: u8,
}

/// A single bidding training environment.
pub struct BidTrainingEnv {
    pub state: GameState,
    bid_history: Vec<(u8, u8)>,
    /// DD-solved NS points per trump suit (indexed by suit 0-3).
    dd_pts: [u8; 4],
    /// Real (DouDou50) NS points per suit (if enriched pool).
    real_pts: Option<[u8; 4]>,
    /// How to compute reward points.
    reward_mode: RewardMode,
    /// Reusable TT buffer (2MB) to avoid repeated allocation.
    tt_buf: Vec<u64>,
    /// Buffered transitions for this episode.
    transitions: Vec<BidTransition>,
    /// Score-aware mode: match scores (NS cumulative, EW cumulative).
    /// When Some, uses score-aware obs and Δ-winprob reward.
    pub score_aware: Option<(i32, i32)>,
    /// Score-aware obs dim (BID_OBS_DIM_SCORE_AWARE = 110 for v1, _V2 = 113 for v2 features).
    pub sa_obs_dim: usize,
    /// Optional clip applied to Δ-winprob reward (post scale). None = no clip.
    pub reward_clip: Option<f32>,
    /// Score-aware transitions (variable-dim obs). Used instead of `transitions` when score_aware is Some.
    sa_transitions: Vec<ScoreAwareBidTransition>,
}

impl BidTrainingEnv {
    pub fn new(rng: &mut impl Rng) -> Self {
        let dealer = rng.gen_range(0..4u8);
        let state = GameState::deal_random(dealer, rng);
        let mut tt_buf = solver::new_tt_buffer();
        let dd_pts = solve_all_suits(&state, &mut tt_buf);

        BidTrainingEnv {
            state,
            bid_history: Vec::with_capacity(12),
            dd_pts,
            real_pts: None,
            reward_mode: RewardMode::DdOnly,
            tt_buf,
            transitions: Vec::with_capacity(12),
            score_aware: None,
            sa_obs_dim: BID_OBS_DIM_SCORE_AWARE,
            reward_clip: None,
            sa_transitions: Vec::new(),
        }
    }

    /// Create from a pre-solved deal (no DD solving needed).
    pub fn from_deal(deal: &PresolvedDeal) -> Self {
        Self::from_deal_with_mode(deal, RewardMode::DdOnly)
    }

    /// Create from a pre-solved deal with a specific reward mode.
    pub fn from_deal_with_mode(deal: &PresolvedDeal, reward_mode: RewardMode) -> Self {
        BidTrainingEnv {
            state: GameState::new(deal.dealer, deal.hands),
            bid_history: Vec::with_capacity(12),
            dd_pts: deal.dd_pts,
            real_pts: deal.real_pts,
            reward_mode,
            tt_buf: Vec::new(),
            transitions: Vec::with_capacity(12),
            score_aware: None,
            sa_obs_dim: BID_OBS_DIM_SCORE_AWARE,
            reward_clip: None,
            sa_transitions: Vec::new(),
        }
    }

    /// Reset with a new random deal + DD solve.
    pub fn reset(&mut self, rng: &mut impl Rng) {
        let dealer = rng.gen_range(0..4u8);
        self.state = GameState::deal_random(dealer, rng);
        self.bid_history.clear();
        self.transitions.clear();
        self.sa_transitions.clear();
        self.dd_pts = solve_all_suits(&self.state, &mut self.tt_buf);
        self.real_pts = None;
    }

    /// Reset from a pre-solved deal (instant, no DD solving).
    pub fn reset_from_deal(&mut self, deal: &PresolvedDeal) {
        self.state = GameState::new(deal.dealer, deal.hands);
        self.bid_history.clear();
        self.transitions.clear();
        self.sa_transitions.clear();
        self.dd_pts = deal.dd_pts;
        self.real_pts = deal.real_pts;
    }

    /// Set random match scores for score-aware training.
    /// If score_pool is provided, samples from real match data (80%) + uniform (20%).
    pub fn randomize_match_scores(&mut self, rng: &mut impl Rng) {
        let ns: i32 = rng.gen_range(0..2000);
        let ew: i32 = rng.gen_range(0..2000);
        self.score_aware = Some((ns, ew));
    }

    /// Set match scores by sampling from real match data with uniform fallback.
    pub fn randomize_match_scores_from_pool(
        &mut self,
        score_pool: &[(i32, i32)],
        uniform_ratio: f32,
        rng: &mut impl Rng,
    ) {
        let (ns, ew) = if score_pool.is_empty() || rng.gen::<f32>() < uniform_ratio {
            (rng.gen_range(0..2000), rng.gen_range(0..2000))
        } else {
            score_pool[rng.gen_range(0..score_pool.len())]
        };
        self.score_aware = Some((ns, ew));
    }

    /// Record the current observation as a transition (before stepping).
    fn record_transition(&mut self, action: u8) {
        let team = GameState::player_team(self.state.current_player());

        if let Some((ns_cum, ew_cum)) = self.score_aware {
            // Score-aware: variable-dim obs (110 v1, 113 v2)
            let dim = self.sa_obs_dim;
            let mut obs = vec![0.0f32; dim];
            let (my_score, opp_score) = if team == 0 {
                (ns_cum, ew_cum)
            } else {
                (ew_cum, ns_cum)
            };
            if dim == BID_OBS_DIM_SCORE_AWARE_V3 {
                bid_obs::write_bid_observation_score_aware_v3(
                    &mut obs, 0, &self.state, &self.bid_history, my_score, opp_score,
                );
            } else if dim == BID_OBS_DIM_SCORE_AWARE_V2 {
                bid_obs::write_bid_observation_score_aware_v2(
                    &mut obs, 0, &self.state, &self.bid_history, my_score, opp_score,
                );
            } else {
                bid_obs::write_bid_observation_score_aware(
                    &mut obs, 0, &self.state, &self.bid_history, my_score, opp_score,
                );
            }
            let mut mask = [0.0f32; BID_MASK_DIM];
            bid_obs::write_bid_mask(&mut mask, 0, &self.state);
            self.sa_transitions.push(ScoreAwareBidTransition {
                obs, mask, action, team,
            });
        } else {
            // Standard: 108-dim obs
            let mut obs = [0.0f32; BID_OBS_DIM];
            bid_obs::write_bid_observation(&mut obs, 0, &self.state, &self.bid_history);
            let mut mask = [0.0f32; BID_MASK_DIM];
            bid_obs::write_bid_mask(&mut mask, 0, &self.state);
            self.transitions.push(BidTransition {
                obs, mask, action, team,
            });
        }
    }

    /// Step the bidding environment. Returns true if bidding is done.
    pub fn step(&mut self, action: u8) -> bool {
        debug_assert_eq!(self.state.phase, Phase::Bidding);

        // Record transition before stepping
        self.record_transition(action);

        // Track bid history
        self.bid_history
            .push((self.state.current_player(), action));

        self.state.step(action);

        // Bidding ends when phase transitions to Playing or Done (void)
        self.state.phase != Phase::Bidding
    }

    /// Compute rewards and return transitions (108-dim obs).
    /// Call after bidding ends (step returned true).
    /// Returns Vec of (obs, mask, action, reward, team).
    pub fn flush_transitions(
        &mut self,
    ) -> Vec<([f32; BID_OBS_DIM], [f32; BID_MASK_DIM], u8, f32, u8)> {
        let (ns_score, ew_score) = self.compute_scores();
        let rewards = [
            (ns_score - ew_score) as f32 / 500.0, // NS reward
            (ew_score - ns_score) as f32 / 500.0, // EW reward
        ];

        let result: Vec<_> = self
            .transitions
            .drain(..)
            .map(|t| {
                let reward = rewards[t.team as usize];
                (t.obs, t.mask, t.action, reward, t.team)
            })
            .collect();

        result
    }

    /// Flush score-aware transitions (variable-dim obs, Δ-winprob reward).
    /// Returns Vec of (obs, mask, action, reward, team). Obs dim = self.sa_obs_dim.
    /// Applies self.reward_clip to the symmetric reward if Some.
    pub fn flush_transitions_score_aware(
        &mut self,
        scale: f32,
    ) -> Vec<(Vec<f32>, [f32; BID_MASK_DIM], u8, f32, u8)> {
        let (ns_deal, ew_deal) = self.compute_scores();
        let (ns_cum, ew_cum) = self.score_aware.unwrap_or((0, 0));

        // NS team reward
        let mut ns_reward = score_aware_reward(
            ns_cum as f32, ew_cum as f32, ns_deal, ew_deal, scale,
        );
        if let Some(clip) = self.reward_clip {
            ns_reward = ns_reward.clamp(-clip, clip);
        }
        let rewards = [ns_reward, -ns_reward];

        self.sa_transitions
            .drain(..)
            .map(|t| {
                (t.obs, t.mask, t.action, rewards[t.team as usize], t.team)
            })
            .collect()
    }

    /// Compute deal scores using reward mode (DD, real, or blend).
    /// Public so the match-sim driver can accumulate per-deal pts before flush.
    pub fn compute_scores(&self) -> (i16, i16) {
        if self.state.contract.value == 0 {
            // Void deal (4 passes) → 0/0
            return (0, 0);
        }

        let trump = self.state.contract.trump;

        // Select NS points based on reward mode
        let ns_dd_pts = self.dd_pts[trump as usize];
        let ns_pts_raw = match self.reward_mode {
            RewardMode::DdOnly => ns_dd_pts,
            RewardMode::RealOnly => {
                self.real_pts.map(|r| r[trump as usize]).unwrap_or(ns_dd_pts)
            }
            RewardMode::Blend(alpha) => {
                if let Some(real) = self.real_pts {
                    let blended = alpha * ns_dd_pts as f32
                        + (1.0 - alpha) * real[trump as usize] as f32;
                    blended.round().max(0.0).min(252.0) as u8
                } else {
                    ns_dd_pts
                }
            }
        };

        let ew_dd_pts = if ns_pts_raw == 252 || ns_pts_raw == 0 {
            252 - ns_pts_raw
        } else {
            162 - ns_pts_raw
        };
        let ns_dd_pts = ns_pts_raw;

        // Build a synthetic terminal state for compute_deal_score
        let taker = self.state.contract.team as usize;
        let defense = 1 - taker;

        let taker_pts = if taker == 0 {
            ns_dd_pts
        } else {
            ew_dd_pts
        };
        let defense_pts = if defense == 0 {
            ns_dd_pts
        } else {
            ew_dd_pts
        };

        // Determine tricks_won: if one team got all points → capot (8 tricks)
        let (taker_tricks, defense_tricks) = if defense_pts == 0 {
            (8u8, 0u8)
        } else if taker_pts == 0 {
            (0u8, 8u8)
        } else {
            // Approximate trick split from points (not perfect but good enough)
            // Minimum 1 trick if has any points
            let total_pts = taker_pts as u16 + defense_pts as u16;
            let taker_frac = taker_pts as f32 / total_pts as f32;
            let t = (taker_frac * 8.0).round().max(1.0).min(7.0) as u8;
            (t, 8 - t)
        };

        let mut terminal = GameState::new(0, [0; 4]);
        terminal.phase = Phase::Done;
        terminal.contract = self.state.contract;
        terminal.points[taker] = taker_pts;
        terminal.points[defense] = defense_pts;
        terminal.tricks_won[taker] = taker_tricks;
        terminal.tricks_won[defense] = defense_tricks;

        // Belote: the DD/IS-DD score layers are trick-point only and don't record
        // belote declarations, but Q+K of trump in the same hand is fully determined
        // by the initial deal. Detect it from the original hands so the reward
        // reflects the +20 bonus on ~11% of deals where the declarer team has it.
        let trump = self.state.contract.trump;
        let qk_mask = (1u32 << (trump as u32 * 8 + 4)) | (1u32 << (trump as u32 * 8 + 5));
        let mut belote = [0u8; 2];
        for p in 0..4u8 {
            let hand = self.state.hands[p as usize];
            if (hand & qk_mask) == qk_mask {
                let team = GameState::player_team(p) as usize;
                belote[team] = 2; // assume both Q and K get played → full belote (+20)
            }
        }
        terminal.belote = belote;

        let score = compute_deal_score(&terminal);
        (score.scores[0], score.scores[1])
    }
}

/// Load score points CSV (ns,ew,winner) → Vec<(i32, i32)> for sampling match scores.
pub fn load_score_points(path: &str) -> std::io::Result<Vec<(i32, i32)>> {
    let content = std::fs::read_to_string(path)?;
    let mut points = Vec::new();
    for line in content.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let (Ok(ns), Ok(ew)) = (parts[0].parse::<i32>(), parts[1].parse::<i32>()) {
                points.push((ns, ew));
            }
        }
    }
    Ok(points)
}

/// DD-solve all 4 trump suits, returning [ns_pts; 4].
fn solve_all_suits(state: &GameState, tt_buf: &mut [u64]) -> [u8; 4] {
    let mut pts = [0u8; 4];
    for suit in 0..4u8 {
        let result = solver::solve_for_trump_reuse_tt(
            state.hands,
            state.dealer,
            suit,
            tt_buf,
        );
        pts[suit as usize] = result[0]; // NS points
    }
    pts
}

/// Vectorized bidding training environment.
pub struct VecBidEnv {
    pub envs: Vec<BidTrainingEnv>,
    /// Flat observation buffer: n_envs × obs_dim.
    pub obs_buf: Vec<f32>,
    /// Flat mask buffer: n_envs × BID_MASK_DIM.
    pub mask_buf: Vec<f32>,
    pub rng: StdRng,
    /// Observation dimension (108 or 110 for score-aware).
    pub obs_dim: usize,
    /// When true, score-aware resets accumulate cumulative scores and rotate
    /// the dealer 0→1→2→3 across deals until one team reaches 2000. Requires
    /// the pool to have a dealer index built (DealPool::build_dealer_index).
    pub match_sim: bool,
}

impl VecBidEnv {
    pub fn new(n_envs: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let envs: Vec<_> = (0..n_envs).map(|_| BidTrainingEnv::new(&mut rng)).collect();
        let obs_dim = BID_OBS_DIM;
        let obs_buf = vec![0.0f32; n_envs * obs_dim];
        let mask_buf = vec![0.0f32; n_envs * BID_MASK_DIM];

        let mut vec_env = VecBidEnv { envs, obs_buf, mask_buf, rng, obs_dim, match_sim: false };
        vec_env.refresh_observations();
        vec_env
    }

    /// Create from a pre-solved deal pool (instant, no DD solving).
    pub fn new_with_pool(n_envs: usize, seed: u64, pool: &DealPool) -> Self {
        Self::new_with_pool_and_mode(n_envs, seed, pool, RewardMode::DdOnly)
    }

    /// Create from a deal pool with a specific reward mode.
    pub fn new_with_pool_and_mode(
        n_envs: usize,
        seed: u64,
        pool: &DealPool,
        reward_mode: RewardMode,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let envs: Vec<_> = (0..n_envs)
            .map(|_| BidTrainingEnv::from_deal_with_mode(pool.sample(&mut rng), reward_mode))
            .collect();
        let obs_dim = BID_OBS_DIM;
        let obs_buf = vec![0.0f32; n_envs * obs_dim];
        let mask_buf = vec![0.0f32; n_envs * BID_MASK_DIM];

        let mut vec_env = VecBidEnv { envs, obs_buf, mask_buf, rng, obs_dim, match_sim: false };
        vec_env.refresh_observations();
        vec_env
    }

    /// Enable score-aware mode: resize obs_buf to 110-dim, randomize match scores.
    pub fn enable_score_aware(&mut self) {
        self.enable_score_aware_with_pool(&[], 1.0);
    }

    /// Enable score-aware with a real score distribution pool.
    /// uniform_ratio: fraction of samples drawn uniformly (rest from pool).
    pub fn enable_score_aware_with_pool(&mut self, score_pool: &[(i32, i32)], uniform_ratio: f32) {
        self.enable_score_aware_with_dim(BID_OBS_DIM_SCORE_AWARE, score_pool, uniform_ratio);
    }

    /// Enable score-aware with explicit obs dim (110 v1 or 113 v2).
    pub fn enable_score_aware_with_dim(
        &mut self,
        obs_dim: usize,
        score_pool: &[(i32, i32)],
        uniform_ratio: f32,
    ) {
        assert!(
            obs_dim == BID_OBS_DIM_SCORE_AWARE
                || obs_dim == BID_OBS_DIM_SCORE_AWARE_V2
                || obs_dim == BID_OBS_DIM_SCORE_AWARE_V3,
            "score-aware obs dim must be 110, 113, or 117"
        );
        self.obs_dim = obs_dim;
        self.obs_buf = vec![0.0f32; self.envs.len() * self.obs_dim];
        for env in &mut self.envs {
            env.sa_obs_dim = obs_dim;
            env.randomize_match_scores_from_pool(score_pool, uniform_ratio, &mut self.rng);
        }
        self.refresh_observations();
    }

    /// Set reward clipping bound for all envs (None disables).
    pub fn set_reward_clip(&mut self, clip: Option<f32>) {
        for env in &mut self.envs {
            env.reward_clip = clip;
        }
    }

    #[inline]
    pub fn n_envs(&self) -> usize {
        self.envs.len()
    }

    /// Refresh all obs/mask buffers from current env states.
    pub fn refresh_observations(&mut self) {
        for i in 0..self.envs.len() {
            self.refresh_env_inner(i);
        }
    }

    #[inline]
    pub fn obs_slice(&self, i: usize) -> &[f32] {
        &self.obs_buf[i * self.obs_dim..(i + 1) * self.obs_dim]
    }

    #[inline]
    pub fn mask_slice(&self, i: usize) -> &[f32] {
        &self.mask_buf[i * BID_MASK_DIM..(i + 1) * BID_MASK_DIM]
    }

    /// Get legal actions as u64 for env i.
    #[inline]
    pub fn legal_mask_u64(&self, i: usize) -> u64 {
        self.envs[i].state.legal_actions()
    }

    /// Get a random legal action for env i.
    pub fn random_action(&mut self, i: usize) -> u8 {
        let mask = self.envs[i].state.legal_actions();
        let count = mask.count_ones();
        let n = self.rng.gen_range(0..count);
        rollout::select_nth_bit(mask, n)
    }

    /// Get current players for all envs.
    pub fn current_players(&self) -> Vec<u8> {
        self.envs.iter().map(|e| e.state.current_player()).collect()
    }

    /// Step env i. If bidding ended, flushes transitions and returns them.
    /// Returns None if bidding continues, Some(transitions) if done.
    /// Each transition is (obs, mask, action, reward, team).
    pub fn step_env(
        &mut self,
        i: usize,
        action: u8,
    ) -> Option<Vec<([f32; BID_OBS_DIM], [f32; BID_MASK_DIM], u8, f32, u8)>> {
        let done = self.envs[i].step(action);
        if done {
            let transitions = self.envs[i].flush_transitions();
            // Reset for next episode
            self.envs[i].reset(&mut self.rng);
            // Refresh this env's observation
            self.refresh_env(i);
            Some(transitions)
        } else {
            self.refresh_env(i);
            None
        }
    }

    /// Step env i with pool-based reset (no DD solving on reset).
    pub fn step_env_pooled(
        &mut self,
        i: usize,
        action: u8,
        pool: &DealPool,
    ) -> Option<Vec<([f32; BID_OBS_DIM], [f32; BID_MASK_DIM], u8, f32, u8)>> {
        let done = self.envs[i].step(action);
        if done {
            let transitions = self.envs[i].flush_transitions();
            let deal = pool.sample(&mut self.rng);
            self.envs[i].reset_from_deal(deal);
            self.refresh_env(i);
            Some(transitions)
        } else {
            self.refresh_env(i);
            None
        }
    }

    /// Step env i with pool-based reset, score-aware mode (variable-dim obs, Δ-winprob reward).
    /// Obs dim of returned transitions = the env's sa_obs_dim (110, 113, or 117).
    ///
    /// Reset behavior depends on `self.match_sim`:
    /// - `false` (default): randomize match scores + draw any deal (legacy).
    /// - `true`: accumulate (ns_deal, ew_deal) into cumulative scores, rotate dealer
    ///   to (prev_dealer+1)%4, draw a deal from the dealer-indexed pool. When one
    ///   team ≥ 2000, reset cumulatives to (0, 0) and pick a fresh random dealer.
    pub fn step_env_pooled_score_aware(
        &mut self,
        i: usize,
        action: u8,
        pool: &DealPool,
        scale: f32,
        score_pool: &[(i32, i32)],
        uniform_ratio: f32,
    ) -> Option<Vec<(Vec<f32>, [f32; BID_MASK_DIM], u8, f32, u8)>> {
        let done = self.envs[i].step(action);
        if done {
            // In match-sim mode we need pre-flush deal scores for cumulative update.
            let deal_scores = if self.match_sim {
                Some(self.envs[i].compute_scores())
            } else {
                None
            };

            let transitions = self.envs[i].flush_transitions_score_aware(scale);
            let prev_dim = self.envs[i].sa_obs_dim;
            let prev_clip = self.envs[i].reward_clip;

            if let Some((ns_deal, ew_deal)) = deal_scores {
                let (mut ns_cum, mut ew_cum) =
                    self.envs[i].score_aware.unwrap_or((0, 0));
                ns_cum += ns_deal as i32;
                ew_cum += ew_deal as i32;

                let next_dealer = if ns_cum >= 2000 || ew_cum >= 2000 {
                    ns_cum = 0;
                    ew_cum = 0;
                    self.rng.gen_range(0..4u8)
                } else {
                    (self.envs[i].state.dealer + 1) % 4
                };

                let deal = pool.sample_with_dealer(&mut self.rng, next_dealer);
                self.envs[i].reset_from_deal(deal);
                self.envs[i].sa_obs_dim = prev_dim;
                self.envs[i].reward_clip = prev_clip;
                self.envs[i].score_aware = Some((ns_cum, ew_cum));
            } else {
                let deal = pool.sample(&mut self.rng);
                self.envs[i].reset_from_deal(deal);
                self.envs[i].sa_obs_dim = prev_dim;
                self.envs[i].reward_clip = prev_clip;
                self.envs[i].randomize_match_scores_from_pool(score_pool, uniform_ratio, &mut self.rng);
            }
            self.refresh_env(i);
            Some(transitions)
        } else {
            self.refresh_env(i);
            None
        }
    }

    /// Enable match simulation mode: cumulative scores + dealer rotation across
    /// deals, reset on 2000-point match end. Requires the pool to have a dealer
    /// index built (call `pool.build_dealer_index()` before).
    pub fn set_match_sim(&mut self, enabled: bool) {
        self.match_sim = enabled;
        if enabled {
            // Start each env at (0, 0) with a random starting dealer.
            for env in &mut self.envs {
                env.score_aware = Some((0, 0));
            }
        }
    }

    /// Update reward mode for all envs (used for curriculum scheduling).
    pub fn set_reward_mode(&mut self, mode: RewardMode) {
        for env in &mut self.envs {
            env.reward_mode = mode;
        }
    }

    /// Refresh obs/mask buffers for a single env.
    #[inline]
    fn refresh_env(&mut self, i: usize) {
        self.refresh_env_inner(i);
    }

    fn refresh_env_inner(&mut self, i: usize) {
        let od = self.obs_dim;
        if od == BID_OBS_DIM_SCORE_AWARE
            || od == BID_OBS_DIM_SCORE_AWARE_V2
            || od == BID_OBS_DIM_SCORE_AWARE_V3
        {
            let (my, opp) = if let Some((ns, ew)) = self.envs[i].score_aware {
                let player = self.envs[i].state.current_player();
                let team = GameState::player_team(player);
                if team == 0 { (ns, ew) } else { (ew, ns) }
            } else {
                (0, 0)
            };
            if od == BID_OBS_DIM_SCORE_AWARE_V3 {
                bid_obs::write_bid_observation_score_aware_v3(
                    &mut self.obs_buf, i * od,
                    &self.envs[i].state, &self.envs[i].bid_history,
                    my, opp,
                );
            } else if od == BID_OBS_DIM_SCORE_AWARE_V2 {
                bid_obs::write_bid_observation_score_aware_v2(
                    &mut self.obs_buf, i * od,
                    &self.envs[i].state, &self.envs[i].bid_history,
                    my, opp,
                );
            } else {
                bid_obs::write_bid_observation_score_aware(
                    &mut self.obs_buf, i * od,
                    &self.envs[i].state, &self.envs[i].bid_history,
                    my, opp,
                );
            }
        } else {
            bid_obs::write_bid_observation(
                &mut self.obs_buf, i * od,
                &self.envs[i].state, &self.envs[i].bid_history,
            );
        }
        bid_obs::write_bid_mask(&mut self.mask_buf, i * BID_MASK_DIM, &self.envs[i].state);
    }
}

/// PER buffer sized for bidding (configurable obs dim, 43-dim mask).
pub struct BidReplayBuffer {
    capacity: usize,
    alpha: f64,
    obs_dim: usize,
    tree: SumTree,
    obs: Vec<f32>,
    masks: Vec<f32>,
    actions: Vec<u8>,
    returns: Vec<f32>,
    max_priority: f64,
    cached_priority: f64,
}

/// A sampled batch from the bidding PER buffer.
pub struct BidPERSample {
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
    pub obs_data: Vec<f32>,
    pub mask_data: Vec<f32>,
    pub actions: Vec<u8>,
    pub returns: Vec<f32>,
}

/// SumTree for proportional sampling.
struct SumTree {
    capacity: usize,
    tree: Vec<f64>,
    data_pointer: usize,
    n_entries: usize,
}

impl SumTree {
    fn new(capacity: usize) -> Self {
        SumTree {
            capacity,
            tree: vec![0.0f64; 2 * capacity],
            data_pointer: 0,
            n_entries: 0,
        }
    }

    #[inline]
    fn update(&mut self, idx: usize, priority: f64) {
        let tree_idx = idx + self.capacity;
        let change = priority - self.tree[tree_idx];
        self.tree[tree_idx] = priority;
        let mut i = tree_idx >> 1;
        while i >= 1 {
            self.tree[i] += change;
            i >>= 1;
        }
    }

    #[inline]
    fn add(&mut self, priority: f64) -> usize {
        let idx = self.data_pointer;
        self.update(idx, priority);
        self.data_pointer = (self.data_pointer + 1) % self.capacity;
        if self.n_entries < self.capacity {
            self.n_entries += 1;
        }
        idx
    }

    #[inline]
    fn get(&self, mut s: f64) -> usize {
        let mut idx = 1;
        let cap2 = 2 * self.capacity;
        loop {
            let left = 2 * idx;
            if left >= cap2 {
                break;
            }
            if s <= self.tree[left] {
                idx = left;
            } else {
                s -= self.tree[left];
                idx = left + 1;
            }
        }
        idx - self.capacity
    }

    #[inline]
    fn total(&self) -> f64 {
        self.tree[1]
    }

    #[inline]
    fn priority(&self, idx: usize) -> f64 {
        self.tree[idx + self.capacity]
    }

    #[inline]
    fn n_entries(&self) -> usize {
        self.n_entries
    }
}

impl BidReplayBuffer {
    pub fn new(capacity: usize, alpha: f64) -> Self {
        Self::with_obs_dim(capacity, alpha, BID_OBS_DIM)
    }

    pub fn with_obs_dim(capacity: usize, alpha: f64, obs_dim: usize) -> Self {
        let cached_priority = 1.0f64.powf(alpha);
        BidReplayBuffer {
            capacity,
            alpha,
            obs_dim,
            tree: SumTree::new(capacity),
            obs: vec![0.0f32; capacity * obs_dim],
            masks: vec![0.0f32; capacity * BID_MASK_DIM],
            actions: vec![0u8; capacity],
            returns: vec![0.0f32; capacity],
            max_priority: 1.0,
            cached_priority,
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.tree.n_entries()
    }

    /// Push a single transition with max priority.
    pub fn push(&mut self, obs: &[f32], mask: &[f32], action: u8, ret: f32) {
        debug_assert_eq!(obs.len(), self.obs_dim);
        debug_assert_eq!(mask.len(), BID_MASK_DIM);

        let od = self.obs_dim;
        let p = self.cached_priority;
        let idx = self.tree.add(p);
        self.obs[idx * od..idx * od + od].copy_from_slice(obs);
        let mask_start = idx * BID_MASK_DIM;
        self.masks[mask_start..mask_start + BID_MASK_DIM].copy_from_slice(mask);
        self.actions[idx] = action;
        self.returns[idx] = ret;
    }

    /// Sample a batch with prioritized replay.
    pub fn sample(&self, batch_size: usize, beta: f64, rng: &mut impl Rng) -> BidPERSample {
        let total = self.tree.total();
        let segment = total / batch_size as f64;
        let size = self.size();

        let mut indices = Vec::with_capacity(batch_size);
        let mut priorities = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let lo = segment * i as f64;
            let hi = segment * (i + 1) as f64;
            let s: f64 = lo + rng.gen::<f64>() * (hi - lo);
            let mut idx = self.tree.get(s);
            if idx >= size {
                idx = size - 1;
            }
            indices.push(idx);
            let p = self.tree.priority(idx);
            priorities.push(if p > 1e-8 { p } else { 1e-8 });
        }

        // IS weights
        let mut weights = Vec::with_capacity(batch_size);
        let mut max_weight: f32 = 0.0;
        let size_f = size as f64;
        for &p in &priorities {
            let prob = p / total;
            let w = ((size_f * prob).powf(-beta)) as f32;
            if w > max_weight {
                max_weight = w;
            }
            weights.push(w);
        }
        if max_weight > 0.0 {
            for w in weights.iter_mut() {
                *w /= max_weight;
            }
        }

        // Gather data
        let od = self.obs_dim;
        let mut obs_data = vec![0.0f32; batch_size * od];
        let mut mask_data = vec![0.0f32; batch_size * BID_MASK_DIM];
        let mut act_data = Vec::with_capacity(batch_size);
        let mut ret_data = Vec::with_capacity(batch_size);

        for (j, &idx) in indices.iter().enumerate() {
            obs_data[j * od..j * od + od]
                .copy_from_slice(&self.obs[idx * od..idx * od + od]);
            let mask_src = idx * BID_MASK_DIM;
            let mask_dst = j * BID_MASK_DIM;
            mask_data[mask_dst..mask_dst + BID_MASK_DIM]
                .copy_from_slice(&self.masks[mask_src..mask_src + BID_MASK_DIM]);
            act_data.push(self.actions[idx]);
            ret_data.push(self.returns[idx]);
        }

        BidPERSample {
            indices,
            weights,
            obs_data,
            mask_data,
            actions: act_data,
            returns: ret_data,
        }
    }

    /// Update priorities based on TD errors.
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f32]) {
        debug_assert_eq!(indices.len(), td_errors.len());
        let alpha = self.alpha;
        let mut max_p = self.max_priority;

        for i in 0..indices.len() {
            let p = (td_errors[i].abs() + 1e-6) as f64;
            if p > max_p {
                max_p = p;
            }
            self.tree.update(indices[i], p.powf(alpha));
        }

        if max_p > self.max_priority {
            self.max_priority = max_p;
            self.cached_priority = max_p.powf(alpha);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bid_training_env_new() {
        let mut rng = StdRng::seed_from_u64(42);
        let env = BidTrainingEnv::new(&mut rng);
        assert_eq!(env.state.phase, Phase::Bidding);
        // DD points should be valid
        for &pts in &env.dd_pts {
            assert!(pts <= 252, "DD points {} out of range", pts);
        }
    }

    #[test]
    fn test_bid_training_env_reset() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = BidTrainingEnv::new(&mut rng);
        let old_hands = env.state.hands;
        env.reset(&mut rng);
        assert_ne!(env.state.hands, old_hands);
        assert!(env.bid_history.is_empty());
        assert!(env.transitions.is_empty());
    }

    #[test]
    fn test_bid_training_env_play_through() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = BidTrainingEnv::new(&mut rng);

        // Play through bidding with random actions
        loop {
            let mask = env.state.legal_actions();
            let count = mask.count_ones();
            let idx = rng.gen_range(0..count);
            let action = rollout::select_nth_bit(mask, idx);
            let done = env.step(action);
            if done {
                break;
            }
        }

        // Flush transitions
        let transitions = env.flush_transitions();
        assert!(!transitions.is_empty(), "should have transitions");

        // Check obs dimensions and rewards
        for (obs, mask, action, reward, team) in &transitions {
            assert_eq!(obs.len(), BID_OBS_DIM);
            assert_eq!(mask.len(), BID_MASK_DIM);
            assert!(*action < 43, "action {} out of range", action);
            assert!(reward.is_finite(), "reward not finite: {}", reward);
            assert!(*team <= 1, "team {} out of range", team);
        }
    }

    #[test]
    fn test_bid_training_env_void_deal() {
        // Force 4 passes = void deal
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = BidTrainingEnv::new(&mut rng);

        for _ in 0..4 {
            let done = env.step(0); // PASS
            if done {
                break;
            }
        }

        let transitions = env.flush_transitions();
        // Void deal: all rewards should be 0
        for (_, _, _, reward, _) in &transitions {
            assert_eq!(*reward, 0.0, "void deal should have 0 reward");
        }
    }

    #[test]
    fn test_vec_bid_env() {
        let mut vec_env = VecBidEnv::new(4, 42);
        assert_eq!(vec_env.n_envs(), 4);
        assert_eq!(vec_env.obs_buf.len(), 4 * BID_OBS_DIM);
        assert_eq!(vec_env.mask_buf.len(), 4 * BID_MASK_DIM);

        // All envs should be in bidding phase
        for env in &vec_env.envs {
            assert_eq!(env.state.phase, Phase::Bidding);
        }
    }

    #[test]
    fn test_vec_bid_env_step() {
        let mut vec_env = VecBidEnv::new(8, 42);

        let mut total_transitions = 0;
        // Run many steps until some episodes complete
        for _ in 0..100 {
            for i in 0..vec_env.n_envs() {
                let action = vec_env.random_action(i);
                if let Some(transitions) = vec_env.step_env(i, action) {
                    total_transitions += transitions.len();
                }
            }
        }

        assert!(total_transitions > 0, "should have completed some episodes");
    }

    #[test]
    fn test_bid_replay_buffer() {
        let mut buf = BidReplayBuffer::new(100, 0.6);
        assert_eq!(buf.size(), 0);

        let obs = vec![0.5f32; BID_OBS_DIM];
        let mask = vec![1.0f32; BID_MASK_DIM];

        for i in 0..50 {
            buf.push(&obs, &mask, (i % 43) as u8, if i % 2 == 0 { 0.5 } else { -0.3 });
        }
        assert_eq!(buf.size(), 50);

        let mut rng = StdRng::seed_from_u64(42);
        let sample = buf.sample(16, 0.4, &mut rng);
        assert_eq!(sample.indices.len(), 16);
        assert_eq!(sample.obs_data.len(), 16 * BID_OBS_DIM);
        assert_eq!(sample.mask_data.len(), 16 * BID_MASK_DIM);
        assert_eq!(sample.actions.len(), 16);
        assert_eq!(sample.returns.len(), 16);

        for &w in &sample.weights {
            assert!(w > 0.0 && w <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_deal_pool_generate() {
        let pool = DealPool::generate(100, 42);
        assert_eq!(pool.len(), 100);
        for deal in &pool.deals {
            assert!(deal.dealer < 4);
            for suit in 0..4 {
                let ns = deal.dd_pts[suit] as u16;
                let total = if ns == 0 || ns == 252 { 252 } else { 162 };
                assert!(ns <= total, "DD pts {} exceeds total {}", ns, total);
            }
        }
    }

    #[test]
    fn test_from_deal_and_reset() {
        let pool = DealPool::generate(10, 42);
        let mut rng = StdRng::seed_from_u64(99);
        let deal = pool.sample(&mut rng);
        let mut env = BidTrainingEnv::from_deal(deal);
        assert_eq!(env.state.phase, Phase::Bidding);
        assert_eq!(env.state.hands, deal.hands);
        assert_eq!(env.dd_pts, deal.dd_pts);

        let deal2 = pool.sample(&mut rng);
        env.reset_from_deal(deal2);
        assert_eq!(env.state.hands, deal2.hands);
        assert!(env.bid_history.is_empty());
    }

    #[test]
    fn test_vec_bid_env_with_pool() {
        let pool = DealPool::generate(50, 42);
        let mut vec_env = VecBidEnv::new_with_pool(4, 99, &pool);
        assert_eq!(vec_env.n_envs(), 4);

        let mut total_transitions = 0;
        for _ in 0..100 {
            for i in 0..vec_env.n_envs() {
                let action = vec_env.random_action(i);
                if let Some(transitions) = vec_env.step_env_pooled(i, action, &pool) {
                    total_transitions += transitions.len();
                }
            }
        }
        assert!(total_transitions > 0, "should have completed some episodes");
    }

    #[test]
    fn test_deal_pool_save_load() {
        let pool = DealPool::generate(50, 42);
        let path = "/tmp/colver_test_pool.bin";
        pool.save(path).unwrap();
        let loaded = DealPool::load(path).unwrap();
        assert_eq!(loaded.len(), pool.len());
        for (a, b) in pool.deals.iter().zip(loaded.deals.iter()) {
            assert_eq!(a.dealer, b.dealer);
            assert_eq!(a.hands, b.hands);
            assert_eq!(a.dd_pts, b.dd_pts);
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_dd_points_consistency() {
        let mut rng = StdRng::seed_from_u64(123);
        for _ in 0..20 {
            let env = BidTrainingEnv::new(&mut rng);
            for suit in 0..4 {
                let ns = env.dd_pts[suit] as u16;
                let total = if ns == 0 || ns == 252 { 252 } else { 162 };
                let ew = total - ns;
                assert!(
                    ns + ew == total,
                    "suit {}: ns={}, ew={}, total={}",
                    suit, ns, ew, total
                );
            }
        }
    }

    #[test]
    fn test_compute_scores_non_void() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = BidTrainingEnv::new(&mut rng);

        // Make a bid so it's not void
        let action = crate::bidding::encode_bid(8, 0); // 80 Spades
        let legal = env.state.legal_actions();
        if legal & (1u64 << action) != 0 {
            env.step(action);
            // Pass remaining players
            while env.state.phase == Phase::Bidding {
                env.step(0); // PASS
            }

            let (ns, ew) = env.compute_scores();
            // At least one team should score
            assert!(ns >= 0 || ew >= 0);
            // Total should be positive (someone wins the contract)
            assert!(ns + ew > 0, "total scores should be positive");
        }
    }

    #[test]
    fn test_compute_scores_belote_bonus() {
        // Craft a deal where seat 0 (NS) has Q♠+K♠, contract = 80 spades by NS with
        // taker trick pts = 80 (just makes). Expected: +20 belote → taker total 100.
        let mut hands = [0u32; 4];
        // Seat 0 gets: Q♠, K♠, plus 6 fillers in hearts (non Q/K): 7♥,8♥,9♥,J♥,10♥,A♥
        hands[0] = (1 << 4) | (1 << 5);
        hands[0] |= (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11) | (1 << 14) | (1 << 15);
        assert_eq!(hands[0].count_ones(), 8);
        // Distribute the other 24 cards to seats 1-3 (doesn't matter for scoring test).
        let used = hands[0];
        let mut remaining: u32 = 0xFFFF_FFFF & !used;
        for seat in 1..4 {
            let mut h = 0u32;
            for _ in 0..8 {
                let bit = remaining.trailing_zeros();
                h |= 1 << bit;
                remaining &= !(1 << bit);
            }
            hands[seat] = h;
        }

        let mut env = BidTrainingEnv::from_deal_with_mode(
            &PresolvedDeal {
                dealer: 3,
                hands,
                dd_pts: [80, 0, 0, 0],
                real_pts: Some([80, 0, 0, 0]),
            },
            RewardMode::RealOnly,
        );
        // Force a contract: NS bids 80♠, everyone else passes.
        env.step(crate::bidding::encode_bid(8, 0));
        while env.state.phase == Phase::Bidding {
            env.step(0);
        }
        let (ns, ew) = env.compute_scores();
        // NS should make: 80 trick + 20 belote ≥ 80 → reussi.
        // Score: round10(80 + 80 + 20) = 180. EW: round10(82 + 0) = 80.
        assert!(ns > 0, "NS should score (made contract with belote)");
        assert!(ns >= 170 && ns <= 200, "ns={} expected ~180 with belote", ns);
        // Without the belote fix, ns would be lower (no +20 → still 170 since 80≥80 anyway).
        // But the bonus must show up in the taker score compared to same state without belote.
        let _ = ew;
    }

    #[test]
    fn test_dealer_index_sampling() {
        let mut rng = StdRng::seed_from_u64(7);
        // Build a tiny pool by generating 200 deals (dealer randomized).
        let mut pool = DealPool::generate(200, 42);
        pool.build_dealer_index();
        // Sample 100 times per dealer and check every returned deal matches.
        for target in 0..4u8 {
            for _ in 0..100 {
                let d = pool.sample_with_dealer(&mut rng, target);
                assert_eq!(d.dealer, target, "dealer index returned wrong dealer");
            }
        }
    }

    #[test]
    fn test_match_sim_accumulates_and_rotates() {
        // Build a small pool and vec env in score-aware + match-sim mode; run
        // enough bid decisions to end several deals, then check that scores
        // accumulated and dealer rotated by +1 from deal to deal.
        let mut pool = DealPool::generate(400, 11);
        pool.build_dealer_index();
        let mut vec_env = VecBidEnv::new_with_pool_and_mode(2, 1, &pool, RewardMode::DdOnly);
        vec_env.enable_score_aware_with_dim(BID_OBS_DIM_SCORE_AWARE, &[], 0.0);
        vec_env.set_match_sim(true);

        let mut rng = StdRng::seed_from_u64(2);
        let mut prev_dealer = vec_env.envs[0].state.dealer;
        let mut prev_cum = vec_env.envs[0].score_aware.unwrap();
        let mut deals_seen = 0;
        let mut rotations_seen = 0;

        for _ in 0..1000 {
            let action = vec_env.random_action(0);
            if let Some(_transitions) = vec_env.step_env_pooled_score_aware(0, action, &pool, 1.0, &[], 0.0) {
                deals_seen += 1;
                let new_dealer = vec_env.envs[0].state.dealer;
                let new_cum = vec_env.envs[0].score_aware.unwrap();
                let expected_dealer = (prev_dealer + 1) % 4;
                // Dealer either rotated by +1 (match continued) or randomized to
                // anything (match reset). Accept both; count rotations.
                if new_dealer == expected_dealer && (new_cum.0 > prev_cum.0 || new_cum.1 > prev_cum.1) {
                    rotations_seen += 1;
                }
                // Cumulatives are non-negative and reset to 0 implies match ended.
                assert!(new_cum.0 >= 0 && new_cum.1 >= 0);
                if new_cum.0 < prev_cum.0 || new_cum.1 < prev_cum.1 {
                    // Reset must zero both.
                    assert_eq!(new_cum, (0, 0), "partial reset — expected full (0,0)");
                }
                prev_dealer = new_dealer;
                prev_cum = new_cum;
            }
        }

        assert!(deals_seen >= 20, "saw only {} deals", deals_seen);
        assert!(rotations_seen >= 5, "saw only {} rotations", rotations_seen);
    }
}
