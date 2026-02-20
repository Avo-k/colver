/// Bidding training environment with pre-solved DD deal pool.
///
/// Phase 1: `DealPool::generate(n, seed)` pre-solves N deals in parallel
///          using all CPU cores (~5 min for 100K deals on 16 cores).
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

use crate::bid_obs::{self, BID_MASK_DIM, BID_OBS_DIM};
use crate::rollout;
use crate::scoring::compute_deal_score;
use crate::solver;
use crate::state::{GameState, Phase};

/// A pre-solved deal: hands + DD results for all 4 trump suits.
pub struct PresolvedDeal {
    pub dealer: u8,
    pub hands: [u32; 4],
    /// NS points for each trump suit (indexed by suit 0-3).
    pub dd_pts: [u8; 4],
}

/// Pool of pre-solved deals for fast training resets.
pub struct DealPool {
    deals: Vec<PresolvedDeal>,
}

impl DealPool {
    /// Generate `n` random deals with DD solutions using all CPU cores.
    /// Uses work-stealing (atomic counter) so fast threads do more work
    /// and no thread sits idle waiting for others with hard deals.
    pub fn generate(n: usize, seed: u64) -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        let next_idx = AtomicUsize::new(0);
        let progress = AtomicUsize::new(0);
        let start_time = Instant::now();

        // Pre-allocate output slots
        let mut deals: Vec<Option<PresolvedDeal>> = (0..n).map(|_| None).collect();
        let deals_ptr = deals.as_mut_ptr();

        // SAFETY: Each thread writes to disjoint indices (grabbed via atomic counter).
        // No two threads ever write to the same slot. The pointer is valid for the
        // entire scope lifetime since `deals` lives on the stack above.
        struct SendPtr<T>(*mut T);
        unsafe impl<T> Send for SendPtr<T> {}
        unsafe impl<T> Sync for SendPtr<T> {}
        let shared_ptr = SendPtr(deals_ptr);

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

                        // Write directly to our slot (no contention, unique index)
                        unsafe {
                            (*ptr_ref.0.add(idx)) = Some(PresolvedDeal {
                                dealer,
                                hands: state.hands,
                                dd_pts,
                            });
                        }

                        let done = progress_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 5_000 == 0 {
                            let elapsed = start.elapsed().as_secs_f64();
                            let rate = done as f64 / elapsed;
                            let eta = (n - done) as f64 / rate;
                            let pct = done as f64 / n as f64;
                            let bar_width = 40;
                            let filled = (pct * bar_width as f64) as usize;
                            let bar: String = (0..bar_width)
                                .map(|i| if i < filled { '█' } else { '░' })
                                .collect();
                            eprint!(
                                "\r  {} {:.0}% | {}/{} | {:.0} deals/s | ETA {:.0}s  ",
                                bar, pct * 100.0, done, n, rate, eta
                            );
                        }
                    }
                });
            }
        });

        let total_time = start_time.elapsed().as_secs_f64();
        eprintln!(
            "\r  {} 100% | {}/{} | {:.0} deals/s | {:.1}s total       ",
            "█".repeat(40),
            n,
            n,
            n as f64 / total_time,
            total_time
        );

        // Unwrap all Options — all slots are guaranteed filled after scope join
        let deals: Vec<PresolvedDeal> = deals.into_iter().map(|d| d.unwrap()).collect();
        DealPool { deals }
    }

    /// Sample a random deal from the pool.
    #[inline]
    pub fn sample(&self, rng: &mut impl Rng) -> &PresolvedDeal {
        &self.deals[rng.gen_range(0..self.deals.len())]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.deals.len()
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
            });
        }

        Ok(DealPool { deals })
    }

    /// Load if file exists, otherwise generate and save.
    pub fn load_or_generate(path: &str, n: usize, seed: u64) -> Self {
        if std::path::Path::new(path).exists() {
            eprintln!("  Loading deal pool from {}...", path);
            let start = Instant::now();
            let pool = Self::load(path).unwrap_or_else(|e| {
                eprintln!("  Failed to load {}: {}, regenerating...", path, e);
                let pool = Self::generate(n, seed);
                pool.save(path).ok();
                pool
            });
            eprintln!(
                "  Loaded {} deals in {:.1}s",
                pool.len(),
                start.elapsed().as_secs_f64()
            );
            pool
        } else {
            let pool = Self::generate(n, seed);
            if let Err(e) = pool.save(path) {
                eprintln!("  Warning: failed to save pool to {}: {}", path, e);
            } else {
                let size_mb = (pool.len() * 21 + 16) as f64 / 1_048_576.0;
                eprintln!("  Saved pool to {} ({:.1}MB)", path, size_mb);
            }
            pool
        }
    }
}

/// A single buffered bid transition (stored until episode ends and reward is known).
struct BidTransition {
    obs: [f32; BID_OBS_DIM],
    mask: [f32; BID_MASK_DIM],
    action: u8,
    team: u8, // team of the player who acted (0=NS, 1=EW)
}

/// A single bidding training environment.
pub struct BidTrainingEnv {
    pub state: GameState,
    bid_history: Vec<(u8, u8)>,
    /// DD-solved NS points per trump suit (indexed by suit 0-3).
    dd_pts: [u8; 4],
    /// Reusable TT buffer (2MB) to avoid repeated allocation.
    tt_buf: Vec<u64>,
    /// Buffered transitions for this episode.
    transitions: Vec<BidTransition>,
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
            tt_buf,
            transitions: Vec::with_capacity(12),
        }
    }

    /// Create from a pre-solved deal (no DD solving needed).
    pub fn from_deal(deal: &PresolvedDeal) -> Self {
        BidTrainingEnv {
            state: GameState::new(deal.dealer, deal.hands),
            bid_history: Vec::with_capacity(12),
            dd_pts: deal.dd_pts,
            tt_buf: Vec::new(), // not used with pool
            transitions: Vec::with_capacity(12),
        }
    }

    /// Reset with a new random deal + DD solve.
    pub fn reset(&mut self, rng: &mut impl Rng) {
        let dealer = rng.gen_range(0..4u8);
        self.state = GameState::deal_random(dealer, rng);
        self.bid_history.clear();
        self.transitions.clear();
        self.dd_pts = solve_all_suits(&self.state, &mut self.tt_buf);
    }

    /// Reset from a pre-solved deal (instant, no DD solving).
    pub fn reset_from_deal(&mut self, deal: &PresolvedDeal) {
        self.state = GameState::new(deal.dealer, deal.hands);
        self.bid_history.clear();
        self.transitions.clear();
        self.dd_pts = deal.dd_pts;
    }

    /// Record the current observation as a transition (before stepping).
    fn record_transition(&mut self, action: u8) {
        let mut obs = [0.0f32; BID_OBS_DIM];
        bid_obs::write_bid_observation(&mut obs, 0, &self.state, &self.bid_history);

        let mut mask = [0.0f32; BID_MASK_DIM];
        bid_obs::write_bid_mask(&mut mask, 0, &self.state);

        let team = GameState::player_team(self.state.current_player());

        self.transitions.push(BidTransition {
            obs,
            mask,
            action,
            team,
        });
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

    /// Compute rewards and return transitions.
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

    /// Compute deal scores from DD results.
    fn compute_scores(&self) -> (i16, i16) {
        if self.state.contract.value == 0 {
            // Void deal (4 passes) → 0/0
            return (0, 0);
        }

        let trump = self.state.contract.trump;
        let ns_dd_pts = self.dd_pts[trump as usize];
        let ew_dd_pts = if ns_dd_pts == 252 || ns_dd_pts == 0 {
            252 - ns_dd_pts
        } else {
            162 - ns_dd_pts
        };

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
        // No belote in DD simulation
        terminal.belote = [0; 2];

        let score = compute_deal_score(&terminal);
        (score.scores[0], score.scores[1])
    }
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
    /// Flat observation buffer: n_envs × BID_OBS_DIM.
    pub obs_buf: Vec<f32>,
    /// Flat mask buffer: n_envs × BID_MASK_DIM.
    pub mask_buf: Vec<f32>,
    pub rng: StdRng,
}

impl VecBidEnv {
    pub fn new(n_envs: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let envs: Vec<_> = (0..n_envs).map(|_| BidTrainingEnv::new(&mut rng)).collect();
        let obs_buf = vec![0.0f32; n_envs * BID_OBS_DIM];
        let mask_buf = vec![0.0f32; n_envs * BID_MASK_DIM];

        let mut vec_env = VecBidEnv {
            envs,
            obs_buf,
            mask_buf,
            rng,
        };
        vec_env.refresh_observations();
        vec_env
    }

    /// Create from a pre-solved deal pool (instant, no DD solving).
    pub fn new_with_pool(n_envs: usize, seed: u64, pool: &DealPool) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let envs: Vec<_> = (0..n_envs)
            .map(|_| BidTrainingEnv::from_deal(pool.sample(&mut rng)))
            .collect();
        let obs_buf = vec![0.0f32; n_envs * BID_OBS_DIM];
        let mask_buf = vec![0.0f32; n_envs * BID_MASK_DIM];

        let mut vec_env = VecBidEnv {
            envs,
            obs_buf,
            mask_buf,
            rng,
        };
        vec_env.refresh_observations();
        vec_env
    }

    #[inline]
    pub fn n_envs(&self) -> usize {
        self.envs.len()
    }

    /// Refresh all obs/mask buffers from current env states.
    pub fn refresh_observations(&mut self) {
        for i in 0..self.envs.len() {
            let env = &self.envs[i];
            bid_obs::write_bid_observation(
                &mut self.obs_buf,
                i * BID_OBS_DIM,
                &env.state,
                &env.bid_history,
            );
            bid_obs::write_bid_mask(&mut self.mask_buf, i * BID_MASK_DIM, &env.state);
        }
    }

    #[inline]
    pub fn obs_slice(&self, i: usize) -> &[f32] {
        &self.obs_buf[i * BID_OBS_DIM..(i + 1) * BID_OBS_DIM]
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

    /// Refresh obs/mask buffers for a single env.
    #[inline]
    fn refresh_env(&mut self, i: usize) {
        bid_obs::write_bid_observation(
            &mut self.obs_buf,
            i * BID_OBS_DIM,
            &self.envs[i].state,
            &self.envs[i].bid_history,
        );
        bid_obs::write_bid_mask(
            &mut self.mask_buf,
            i * BID_MASK_DIM,
            &self.envs[i].state,
        );
    }
}

/// PER buffer sized for bidding (114-dim obs, 43-dim mask).
pub struct BidReplayBuffer {
    capacity: usize,
    alpha: f64,
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
        let cached_priority = 1.0f64.powf(alpha);
        BidReplayBuffer {
            capacity,
            alpha,
            tree: SumTree::new(capacity),
            obs: vec![0.0f32; capacity * BID_OBS_DIM],
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
        debug_assert_eq!(obs.len(), BID_OBS_DIM);
        debug_assert_eq!(mask.len(), BID_MASK_DIM);

        let p = self.cached_priority;
        let idx = self.tree.add(p);
        let obs_start = idx * BID_OBS_DIM;
        let mask_start = idx * BID_MASK_DIM;
        self.obs[obs_start..obs_start + BID_OBS_DIM].copy_from_slice(obs);
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
        let mut obs_data = vec![0.0f32; batch_size * BID_OBS_DIM];
        let mut mask_data = vec![0.0f32; batch_size * BID_MASK_DIM];
        let mut act_data = Vec::with_capacity(batch_size);
        let mut ret_data = Vec::with_capacity(batch_size);

        for (j, &idx) in indices.iter().enumerate() {
            let obs_src = idx * BID_OBS_DIM;
            let obs_dst = j * BID_OBS_DIM;
            obs_data[obs_dst..obs_dst + BID_OBS_DIM]
                .copy_from_slice(&self.obs[obs_src..obs_src + BID_OBS_DIM]);
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
}
