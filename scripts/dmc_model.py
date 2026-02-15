"""Q-Network, Replay Buffer, and Prioritized Replay for DMC training."""

import numpy as np
import torch
import torch.nn as nn


OBS_DIM = 415
NUM_CARDS = 32


class QNetwork(nn.Module):
    """MLP Q-network with LayerNorm: obs -> Q-values (32), one per card index."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 1024, num_actions: int = NUM_CARDS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns Q-values (batch, 32)."""
        return self.net(obs)

    def act(self, obs: torch.Tensor, mask: torch.Tensor, eps: float = 0.0) -> np.ndarray:
        """Select actions with epsilon-greedy.

        Args:
            obs: (batch, obs_dim) float tensor
            mask: (batch, 32) float tensor, 1.0 for legal actions
            eps: exploration rate

        Returns:
            actions: (batch,) int numpy array
        """
        with torch.no_grad():
            q = self.forward(obs)
            q[mask == 0] = -1e9
            greedy = q.argmax(dim=1).cpu().numpy()

        batch_size = obs.shape[0]
        if eps > 0:
            # Sample random legal actions for exploration
            random_actions = _sample_legal(mask.cpu().numpy())
            use_random = np.random.rand(batch_size) < eps
            return np.where(use_random, random_actions, greedy)
        return greedy


def _sample_legal(mask: np.ndarray) -> np.ndarray:
    """Sample a random legal action per row from a binary mask (batch, 32)."""
    batch_size = mask.shape[0]
    actions = np.zeros(batch_size, dtype=np.int64)
    for i in range(batch_size):
        legal = np.where(mask[i] > 0)[0]
        actions[i] = np.random.choice(legal)
    return actions


class ReplayBuffer:
    """Circular replay buffer for DMC transitions (uniform sampling)."""

    def __init__(self, capacity: int = 500_000, obs_dim: int = OBS_DIM):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.masks = np.zeros((capacity, NUM_CARDS), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0

    def push(self, obs: np.ndarray, mask: np.ndarray, action: int, ret: float):
        """Add a single transition."""
        self.obs[self.pos] = obs
        self.masks[self.pos] = mask
        self.actions[self.pos] = action
        self.returns[self.pos] = ret
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, obs: np.ndarray, masks: np.ndarray,
                   actions: np.ndarray, returns: np.ndarray):
        """Add a batch of transitions efficiently."""
        n = len(obs)
        if n == 0:
            return
        # Handle wrap-around
        end = self.pos + n
        if end <= self.capacity:
            self.obs[self.pos:end] = obs
            self.masks[self.pos:end] = masks
            self.actions[self.pos:end] = actions
            self.returns[self.pos:end] = returns
        else:
            first = self.capacity - self.pos
            self.obs[self.pos:] = obs[:first]
            self.masks[self.pos:] = masks[:first]
            self.actions[self.pos:] = actions[:first]
            self.returns[self.pos:] = returns[:first]
            rest = n - first
            self.obs[:rest] = obs[first:]
            self.masks[:rest] = masks[first:]
            self.actions[:rest] = actions[first:]
            self.returns[:rest] = returns[first:]
        self.pos = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Sample a random batch. Returns (obs, masks, actions, returns) as tensors."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.masks[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.returns[idx]),
        )


class SumTree:
    """Binary sum tree for O(log n) proportional sampling. Iterative implementation."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data_pointer = 0
        self.n_entries = 0

    def update(self, idx: int, priority: float):
        """Update priority at leaf index (iterative propagation)."""
        tree_idx = idx + self.capacity
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        tree_idx >>= 1
        while tree_idx >= 1:
            self.tree[tree_idx] += change
            tree_idx >>= 1

    def add(self, priority: float) -> int:
        """Add new entry, returns data index."""
        idx = self.data_pointer
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return idx

    def get(self, s: float) -> int:
        """Sample a leaf index proportional to priority (iterative)."""
        idx = 1
        cap2 = 2 * self.capacity
        tree = self.tree
        while True:
            left = 2 * idx
            if left >= cap2:
                break
            if s <= tree[left]:
                idx = left
            else:
                s -= tree[left]
                idx = left + 1
        return idx - self.capacity

    def get_batch(self, s_values: np.ndarray) -> np.ndarray:
        """Batch sample: retrieve leaf indices for an array of s values."""
        n = len(s_values)
        results = np.empty(n, dtype=np.int64)
        cap2 = 2 * self.capacity
        tree = self.tree
        cap = self.capacity
        for i in range(n):
            s = s_values[i]
            idx = 1
            while True:
                left = 2 * idx
                if left >= cap2:
                    break
                if s <= tree[left]:
                    idx = left
                else:
                    s -= tree[left]
                    idx = left + 1
            results[i] = idx - cap
        return results

    @property
    def total(self) -> float:
        return self.tree[1]

    def priority(self, idx: int) -> float:
        return self.tree[idx + self.capacity]

    def priorities_batch(self, indices: np.ndarray) -> np.ndarray:
        """Get priorities for a batch of data indices."""
        return self.tree[indices + self.capacity]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using SumTree.

    Priority = |TD error|^alpha. Importance sampling weights for bias correction.
    """

    def __init__(self, capacity: int = 2_000_000, obs_dim: int = OBS_DIM,
                 alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.masks = np.zeros((capacity, NUM_CARDS), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        self.size = 0
        self._cached_priority = 1.0  # cached max_priority ** alpha

    def push_batch(self, obs: np.ndarray, masks: np.ndarray,
                   actions: np.ndarray, returns: np.ndarray):
        """Add a batch of transitions with max priority."""
        n = len(obs)
        p = self._cached_priority
        tree = self.tree
        for i in range(n):
            idx = tree.add(p)
            self.obs[idx] = obs[i]
            self.masks[idx] = masks[i]
            self.actions[idx] = actions[i]
            self.returns[idx] = returns[i]
        self.size = tree.n_entries

    def sample(self, batch_size: int, beta: float = 0.4
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                          torch.Tensor, torch.Tensor, np.ndarray]:
        """Sample with priorities. Returns (obs, masks, actions, returns, weights, indices)."""
        total = self.tree.total
        segment = total / batch_size

        # Vectorized random sampling
        lo = np.arange(batch_size, dtype=np.float64) * segment
        hi = lo + segment
        s_values = np.random.uniform(lo, hi)

        # Batch tree traversal
        indices = self.tree.get_batch(s_values)
        np.clip(indices, 0, self.size - 1, out=indices)

        # Batch priority lookup
        priorities = self.tree.priorities_batch(indices)
        np.maximum(priorities, 1e-8, out=priorities)

        # Importance sampling weights
        probs = priorities / total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()

        return (
            torch.from_numpy(self.obs[indices].copy()),
            torch.from_numpy(self.masks[indices].copy()),
            torch.from_numpy(self.actions[indices].copy()),
            torch.from_numpy(self.returns[indices].copy()),
            torch.from_numpy(weights.astype(np.float32)),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = np.abs(td_errors) + 1e-6
        max_p = priorities.max()
        if max_p > self.max_priority:
            self.max_priority = float(max_p)
            self._cached_priority = self.max_priority ** self.alpha
        alpha = self.alpha
        tree = self.tree
        for i in range(len(indices)):
            tree.update(int(indices[i]), float(priorities[i]) ** alpha)
