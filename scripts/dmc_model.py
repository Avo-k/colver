"""Q-Network and Replay Buffer for DouZero-style Deep Monte-Carlo training."""

import numpy as np
import torch
import torch.nn as nn


OBS_DIM = 213
NUM_CARDS = 32


class QNetwork(nn.Module):
    """MLP Q-network: obs (222) -> Q-values (32), one per card index."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 512, num_actions: int = NUM_CARDS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns Q-values (batch, 32)."""
        return self.net(obs)

    def act(self, obs: torch.Tensor, mask: torch.Tensor, eps: float = 0.0) -> np.ndarray:
        """Select actions with epsilon-greedy.

        Args:
            obs: (batch, 222) float tensor
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
    """Circular replay buffer for DMC transitions.

    Stores: obs (222), mask32 (32), action (1), return (1).
    """

    def __init__(self, capacity: int = 500_000):
        self.capacity = capacity
        self.obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
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
