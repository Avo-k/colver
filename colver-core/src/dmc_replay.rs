/// Prioritized Experience Replay buffer for DMC training.
///
/// Port of the SumTree + PrioritizedReplayBuffer from `colver-py/src/lib.rs`,
/// stripped of PyO3/numpy wrappers. Uses pre-allocated flat `Vec<f32>` storage.

use rand::Rng;

use crate::dmc_obs::OBS_DIM;

const NUM_CARDS: usize = 32;

/// Binary sum tree for O(log n) proportional sampling.
pub struct SumTree {
    capacity: usize,
    tree: Vec<f64>,
    data_pointer: usize,
    n_entries: usize,
}

impl SumTree {
    pub fn new(capacity: usize) -> Self {
        SumTree {
            capacity,
            tree: vec![0.0f64; 2 * capacity],
            data_pointer: 0,
            n_entries: 0,
        }
    }

    #[inline]
    pub fn update(&mut self, idx: usize, priority: f64) {
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
    pub fn add(&mut self, priority: f64) -> usize {
        let idx = self.data_pointer;
        self.update(idx, priority);
        self.data_pointer = (self.data_pointer + 1) % self.capacity;
        if self.n_entries < self.capacity {
            self.n_entries += 1;
        }
        idx
    }

    #[inline]
    pub fn get(&self, mut s: f64) -> usize {
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
    pub fn total(&self) -> f64 {
        self.tree[1]
    }

    #[inline]
    pub fn priority(&self, idx: usize) -> f64 {
        self.tree[idx + self.capacity]
    }

    #[inline]
    pub fn n_entries(&self) -> usize {
        self.n_entries
    }
}

/// A sampled batch from the PER buffer.
pub struct PERSample {
    /// Indices into the replay buffer.
    pub indices: Vec<usize>,
    /// Importance sampling weights, normalized to max=1.
    pub weights: Vec<f32>,
    /// Flat observation data: batch_size * OBS_DIM.
    pub obs_data: Vec<f32>,
    /// Flat mask data: batch_size * NUM_CARDS.
    pub mask_data: Vec<f32>,
    /// Actions taken (card indices 0-31).
    pub actions: Vec<u8>,
    /// Episode returns (binary: 0.0 or 1.0 or 0.5).
    pub returns: Vec<f32>,
}

/// Prioritized Experience Replay buffer with pre-allocated flat storage.
pub struct PrioritizedReplayBuffer {
    _capacity: usize,
    alpha: f64,
    tree: SumTree,
    obs: Vec<f32>,     // capacity * OBS_DIM, row-major
    masks: Vec<f32>,   // capacity * NUM_CARDS
    actions: Vec<u8>,
    returns: Vec<f32>,
    max_priority: f64,
    cached_priority: f64,
}

impl PrioritizedReplayBuffer {
    pub fn new(capacity: usize, alpha: f64) -> Self {
        let cached_priority = 1.0f64.powf(alpha);
        PrioritizedReplayBuffer {
            _capacity: capacity,
            alpha,
            tree: SumTree::new(capacity),
            obs: vec![0.0f32; capacity * OBS_DIM],
            masks: vec![0.0f32; capacity * NUM_CARDS],
            actions: vec![0u8; capacity],
            returns: vec![0.0f32; capacity],
            max_priority: 1.0,
            cached_priority,
        }
    }

    /// Current buffer size.
    #[inline]
    pub fn size(&self) -> usize {
        self.tree.n_entries()
    }

    /// Push a single transition with max priority.
    pub fn push(
        &mut self,
        obs: &[f32],
        mask: &[f32],
        action: u8,
        ret: f32,
    ) {
        debug_assert_eq!(obs.len(), OBS_DIM);
        debug_assert_eq!(mask.len(), NUM_CARDS);

        let p = self.cached_priority;
        let idx = self.tree.add(p);
        let obs_start = idx * OBS_DIM;
        let mask_start = idx * NUM_CARDS;
        self.obs[obs_start..obs_start + OBS_DIM].copy_from_slice(obs);
        self.masks[mask_start..mask_start + NUM_CARDS].copy_from_slice(mask);
        self.actions[idx] = action;
        self.returns[idx] = ret;
    }

    /// Push a batch of transitions with max priority.
    /// All slices must have `n` entries; obs is flat n*OBS_DIM, masks is flat n*NUM_CARDS.
    pub fn push_batch(
        &mut self,
        obs: &[f32],
        masks: &[f32],
        actions: &[u8],
        returns: &[f32],
    ) {
        let n = actions.len();
        debug_assert_eq!(obs.len(), n * OBS_DIM);
        debug_assert_eq!(masks.len(), n * NUM_CARDS);
        debug_assert_eq!(returns.len(), n);

        let p = self.cached_priority;
        for i in 0..n {
            let idx = self.tree.add(p);
            let obs_start = idx * OBS_DIM;
            let mask_start = idx * NUM_CARDS;
            self.obs[obs_start..obs_start + OBS_DIM]
                .copy_from_slice(&obs[i * OBS_DIM..(i + 1) * OBS_DIM]);
            self.masks[mask_start..mask_start + NUM_CARDS]
                .copy_from_slice(&masks[i * NUM_CARDS..(i + 1) * NUM_CARDS]);
            self.actions[idx] = actions[i];
            self.returns[idx] = returns[i];
        }
    }

    /// Sample a batch with prioritized replay.
    pub fn sample(&self, batch_size: usize, beta: f64, rng: &mut impl Rng) -> PERSample {
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
        let mut obs_data = vec![0.0f32; batch_size * OBS_DIM];
        let mut mask_data = vec![0.0f32; batch_size * NUM_CARDS];
        let mut act_data = Vec::with_capacity(batch_size);
        let mut ret_data = Vec::with_capacity(batch_size);

        for (j, &idx) in indices.iter().enumerate() {
            let obs_src = idx * OBS_DIM;
            let obs_dst = j * OBS_DIM;
            obs_data[obs_dst..obs_dst + OBS_DIM]
                .copy_from_slice(&self.obs[obs_src..obs_src + OBS_DIM]);
            let mask_src = idx * NUM_CARDS;
            let mask_dst = j * NUM_CARDS;
            mask_data[mask_dst..mask_dst + NUM_CARDS]
                .copy_from_slice(&self.masks[mask_src..mask_src + NUM_CARDS]);
            act_data.push(self.actions[idx]);
            ret_data.push(self.returns[idx]);
        }

        PERSample {
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
    fn test_sum_tree_basic() {
        let mut tree = SumTree::new(4);
        assert_eq!(tree.total(), 0.0);
        assert_eq!(tree.n_entries(), 0);

        tree.add(1.0);
        assert_eq!(tree.n_entries(), 1);
        assert!((tree.total() - 1.0).abs() < 1e-10);

        tree.add(2.0);
        assert_eq!(tree.n_entries(), 2);
        assert!((tree.total() - 3.0).abs() < 1e-10);

        tree.add(3.0);
        assert_eq!(tree.n_entries(), 3);
        assert!((tree.total() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_sum_tree_get() {
        let mut tree = SumTree::new(4);
        tree.add(1.0); // idx 0
        tree.add(2.0); // idx 1
        tree.add(3.0); // idx 2

        // s=0.5 should land in idx 0 (priority 1.0)
        assert_eq!(tree.get(0.5), 0);
        // s=1.5 should land in idx 1 (cumulative 1.0..3.0)
        assert_eq!(tree.get(1.5), 1);
        // s=4.0 should land in idx 2 (cumulative 3.0..6.0)
        assert_eq!(tree.get(4.0), 2);
    }

    #[test]
    fn test_sum_tree_update() {
        let mut tree = SumTree::new(4);
        tree.add(1.0);
        tree.add(2.0);
        assert!((tree.total() - 3.0).abs() < 1e-10);

        tree.update(0, 5.0);
        assert!((tree.total() - 7.0).abs() < 1e-10);
        assert!((tree.priority(0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sum_tree_wraparound() {
        let mut tree = SumTree::new(2);
        tree.add(1.0); // idx 0
        tree.add(2.0); // idx 1
        assert_eq!(tree.n_entries(), 2);

        // Should wrap around and overwrite idx 0
        tree.add(3.0);
        assert_eq!(tree.n_entries(), 2); // still 2 (capacity)
        assert!((tree.priority(0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_per_buffer_push_and_size() {
        let mut buf = PrioritizedReplayBuffer::new(100, 0.6);
        assert_eq!(buf.size(), 0);

        let obs = vec![0.0f32; OBS_DIM];
        let mask = vec![1.0f32; 32];
        buf.push(&obs, &mask, 5, 1.0);
        assert_eq!(buf.size(), 1);
    }

    #[test]
    fn test_per_buffer_sample() {
        let mut buf = PrioritizedReplayBuffer::new(100, 0.6);
        let obs = vec![0.5f32; OBS_DIM];
        let mask = vec![1.0f32; 32];

        for i in 0..50 {
            buf.push(&obs, &mask, (i % 32) as u8, if i % 2 == 0 { 1.0 } else { 0.0 });
        }
        assert_eq!(buf.size(), 50);

        let mut rng = rand::thread_rng();
        let sample = buf.sample(16, 0.4, &mut rng);
        assert_eq!(sample.indices.len(), 16);
        assert_eq!(sample.weights.len(), 16);
        assert_eq!(sample.obs_data.len(), 16 * OBS_DIM);
        assert_eq!(sample.mask_data.len(), 16 * 32);
        assert_eq!(sample.actions.len(), 16);
        assert_eq!(sample.returns.len(), 16);

        // All weights should be in (0, 1]
        for &w in &sample.weights {
            assert!(w > 0.0 && w <= 1.0 + 1e-6, "weight {} out of range", w);
        }
    }

    #[test]
    fn test_per_buffer_update_priorities() {
        let mut buf = PrioritizedReplayBuffer::new(100, 0.6);
        let obs = vec![0.0f32; OBS_DIM];
        let mask = vec![1.0f32; 32];

        for _ in 0..10 {
            buf.push(&obs, &mask, 0, 1.0);
        }

        let indices = vec![0, 1, 2];
        let td_errors = vec![0.5, 1.0, 2.0];
        buf.update_priorities(&indices, &td_errors);

        // Priority at idx 2 should be highest
        assert!(buf.tree.priority(2) > buf.tree.priority(0));
    }

    #[test]
    fn test_per_buffer_push_batch() {
        let mut buf = PrioritizedReplayBuffer::new(100, 0.6);
        let n = 5;
        let obs = vec![0.1f32; n * OBS_DIM];
        let masks = vec![1.0f32; n * 32];
        let actions = vec![0u8, 1, 2, 3, 4];
        let returns = vec![1.0f32, 0.0, 1.0, 0.0, 0.5];

        buf.push_batch(&obs, &masks, &actions, &returns);
        assert_eq!(buf.size(), 5);
    }
}
