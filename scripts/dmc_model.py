"""Q-Network, Replay Buffer, and Prioritized Replay for DMC training."""

import numpy as np
import torch
import torch.nn as nn


OBS_DIM = 372
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
    """Binary sum tree for O(log n) proportional sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = idx >> 1
        self.tree[parent] += change
        if parent > 1:
            self._propagate(parent, change)

    def update(self, idx: int, priority: float):
        """Update priority at leaf index."""
        tree_idx = idx + self.capacity
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority: float) -> int:
        """Add new entry, returns data index."""
        idx = self.data_pointer
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return idx

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx
        right = left + 1
        if left >= 2 * self.capacity:
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def get(self, s: float) -> int:
        """Sample a leaf index proportional to priority."""
        tree_idx = self._retrieve(1, s)
        return tree_idx - self.capacity

    @property
    def total(self) -> float:
        return self.tree[1]

    def priority(self, idx: int) -> float:
        return self.tree[idx + self.capacity]


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

    def push_batch(self, obs: np.ndarray, masks: np.ndarray,
                   actions: np.ndarray, returns: np.ndarray):
        """Add a batch of transitions with max priority."""
        n = len(obs)
        for i in range(n):
            idx = self.tree.add(self.max_priority ** self.alpha)
            self.obs[idx] = obs[i]
            self.masks[idx] = masks[i]
            self.actions[idx] = actions[i]
            self.returns[idx] = returns[i]
        self.size = self.tree.n_entries

    def sample(self, batch_size: int, beta: float = 0.4
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                          torch.Tensor, torch.Tensor, np.ndarray]:
        """Sample with priorities. Returns (obs, masks, actions, returns, weights, indices)."""
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        total = self.tree.total
        segment = total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            idx = self.tree.get(s)
            # Clamp to valid range
            idx = max(0, min(idx, self.size - 1))
            indices[i] = idx
            priorities[i] = max(self.tree.priority(idx), 1e-8)

        # Importance sampling weights
        probs = priorities / total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()

        return (
            torch.from_numpy(self.obs[indices]),
            torch.from_numpy(self.masks[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.returns[indices]),
            torch.from_numpy(weights.astype(np.float32)),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = np.abs(td_errors) + 1e-6
        for i, idx in enumerate(indices):
            p = priorities[i] ** self.alpha
            self.max_priority = max(self.max_priority, priorities[i])
            self.tree.update(int(idx), p)
