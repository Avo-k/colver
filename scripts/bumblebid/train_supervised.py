"""Bumblebid DQN training with DD episode returns + nn_v2 opponents.

BB plays one team (ε-greedy from its own policy), nn_v2 plays the other.
When an auction ends, BB's transitions get the DD return from the FINAL
contract. Loss = MSE(Q(s, a_taken), return).

nn_v2 opponents provide realistic, informative auctions so BB can learn
to decode bid history signals from competent partners/opponents.

Usage:
    PYTHONUNBUFFERED=1 PYTHONPATH=scripts python -m bumblebid.train_supervised \
        --pool-file data/deals/archive/bumblebid_5M_enriched.bin \
        --dd-pool data/deals/archive/dd_5M_enriched.bin
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .model import Bumblebid, NUM_BID_ACTIONS, MAX_SEQ_LEN, P_CLS, P_POS0, P_RANK0, P_NONE, S_NULL
    from .data import (
        load_bumblebid_pool, load_dd_pool, compute_deal_score, decode_bid,
        ACTION_PRIMARY_LUT, ACTION_SUIT_LUT, ACTION_SUIT2_LUT,
        SUIT_PERM_T, ACTION_PERM_TABLE,
    )
    from .nn_v2 import DuelingBidNet, encode_bid_obs_batch, select_actions_nn_v2
except ImportError:
    from model import Bumblebid, NUM_BID_ACTIONS, MAX_SEQ_LEN, P_CLS, P_POS0, P_RANK0, P_NONE, S_NULL
    from data import (
        load_bumblebid_pool, load_dd_pool, compute_deal_score, decode_bid,
        ACTION_PRIMARY_LUT, ACTION_SUIT_LUT, ACTION_SUIT2_LUT,
        SUIT_PERM_T, ACTION_PERM_TABLE,
    )
    from nn_v2 import DuelingBidNet, encode_bid_obs_batch, select_actions_nn_v2

BID_PASS = 0
BID_COINCHE = 41
BID_SURCOINCHE = 42

# Bid value/suit lookup
_BID_VAL_LUT = np.zeros(41, dtype=np.int32)
_BID_SUIT_LUT = np.zeros(41, dtype=np.int32)
for _a in range(1, 41):
    if _a <= 36:
        _BID_VAL_LUT[_a] = (_a - 1) // 4 + 8
        _BID_SUIT_LUT[_a] = (_a - 1) % 4
    else:
        _BID_VAL_LUT[_a] = 25
        _BID_SUIT_LUT[_a] = _a - 37


# ---------------------------------------------------------------------------
# Scoring: compute DD return from a final contract
# ---------------------------------------------------------------------------
def _round10(x):
    return ((x + 5) // 10) * 10


def _compute_returns(dd_pts, contract_val, contract_suit, taker_team, coinche):
    """Compute (ns_reward, ew_reward) for a completed auction.

    Args:
        dd_pts: [4] uint8 — NS DD points per suit
        contract_val: int — bid value encoding (8-16 or 25 for capot), -1 for void
        contract_suit: int — trump suit index
        taker_team: int — 0=NS, 1=EW
        coinche: int — 0/1/2

    Returns: (ns_reward, ew_reward) as float32, normalized by /500
    """
    if contract_val < 0:
        return 0.0, 0.0

    cv = contract_val * 10  # actual contract value (80-160 or 250)
    ns_pts = int(dd_pts[contract_suit])
    ew_pts = (252 - ns_pts) if ns_pts in (0, 252) else (162 - ns_pts)

    taker_pts = ns_pts if taker_team == 0 else ew_pts
    defense_pts = ew_pts if taker_team == 0 else ns_pts
    is_capot = cv == 250

    if is_capot:
        if defense_pts == 0:
            t_sc, d_sc = 500, 0
        else:
            t_sc, d_sc = 0, 500
    else:
        if taker_pts >= cv:
            t_sc = _round10(taker_pts + cv)
            d_sc = _round10(defense_pts)
        else:
            t_sc = 0
            d_sc = _round10(160 + cv)

    # Apply coinche multiplier
    mult = [1, 2, 4][coinche]
    t_sc *= mult
    d_sc *= mult

    ns_sc = t_sc if taker_team == 0 else d_sc
    ew_sc = d_sc if taker_team == 0 else t_sc
    ns_r = (ns_sc - ew_sc) / 500.0
    ew_r = (ew_sc - ns_sc) / 500.0
    return np.float32(ns_r), np.float32(ew_r)


# ---------------------------------------------------------------------------
# Vectorized bidding env with per-env transition buffering
# ---------------------------------------------------------------------------
MAX_BID_STEPS = 12  # max bid rounds in a Contrée auction


class VecBidEnv:
    """Vectorized bidding env for DQN training.

    BB plays one team, nn_v2 plays the other. BB team alternates per deal
    for balanced training. Only BB transitions are buffered.

    When an auction ends, computes DD return from the final contract
    and flushes BB's transitions to the replay buffer.
    """

    def __init__(self, n_envs, pool_dealers, pool_primary, pool_suits, pool_dd,
                 pool_hands, device, seed=42):
        self.n = n_envs
        self.pool_dealers = pool_dealers
        self.pool_dd = pool_dd
        self.pool_hands = pool_hands  # [N_pool, 4] uint32 for nn_v2 obs
        self.n_pool = len(pool_dealers)
        self.rng = np.random.default_rng(seed)
        self.device = device

        self.gpu_primary = torch.from_numpy(pool_primary.astype(np.int64)).to(device)
        self.gpu_suits = torch.from_numpy(pool_suits.astype(np.int64)).to(device)

        # Env state
        self.deal_idx = np.zeros(n_envs, dtype=np.int64)
        self.env_dealer = np.zeros(n_envs, dtype=np.int32)
        self.env_cur_player = np.zeros(n_envs, dtype=np.int32)
        self.env_consec_passes = np.zeros(n_envs, dtype=np.int32)
        self.env_bid_val = np.full(n_envs, -1, dtype=np.int32)
        self.env_bid_suit = np.zeros(n_envs, dtype=np.int32)
        self.env_bid_team = np.zeros(n_envs, dtype=np.int32)
        self.env_coinche = np.zeros(n_envs, dtype=np.int32)
        self.env_done = np.zeros(n_envs, dtype=bool)

        # Which team BB plays (0=NS, 1=EW), alternates per reset
        self.bb_team = np.zeros(n_envs, dtype=np.int32)
        self._reset_counter = 0

        # Token sequence state (for building BB batches)
        self.bid_primary = np.zeros((n_envs, 24), dtype=np.int64)
        self.bid_suits = np.zeros((n_envs, 24), dtype=np.int64)
        self.n_bid_tokens = np.zeros(n_envs, dtype=np.int32)

        # Raw bid history for nn_v2 obs encoding
        self.bid_actions_hist = np.zeros((n_envs, MAX_BID_STEPS), dtype=np.int32)
        self.bid_seats_hist = np.zeros((n_envs, MAX_BID_STEPS), dtype=np.int32)
        self.n_bids = np.zeros(n_envs, dtype=np.int32)

        # Per-env transition buffer (BB transitions only)
        S = MAX_SEQ_LEN
        A = NUM_BID_ACTIONS
        self.pend_count = np.zeros(n_envs, dtype=np.int32)
        self.pend_primary = np.zeros((n_envs, MAX_BID_STEPS, S), dtype=np.int64)
        self.pend_suits = np.zeros((n_envs, MAX_BID_STEPS, S), dtype=np.int64)
        self.pend_slens = np.zeros((n_envs, MAX_BID_STEPS), dtype=np.int64)
        self.pend_masks = np.zeros((n_envs, MAX_BID_STEPS, A), dtype=np.float32)
        self.pend_actions = np.zeros((n_envs, MAX_BID_STEPS), dtype=np.int64)
        self.pend_teams = np.zeros((n_envs, MAX_BID_STEPS), dtype=np.int32)

        self.reset_all()

    def reset_envs(self, indices):
        idx = np.asarray(indices)
        if len(idx) == 0:
            return
        di = self.rng.integers(self.n_pool, size=len(idx))
        self.deal_idx[idx] = di
        self.env_dealer[idx] = self.pool_dealers[di]
        self.env_cur_player[idx] = (self.pool_dealers[di] + 1) % 4
        self.env_consec_passes[idx] = 0
        self.env_bid_val[idx] = -1
        self.env_bid_suit[idx] = 0
        self.env_bid_team[idx] = 0
        self.env_coinche[idx] = 0
        self.env_done[idx] = False
        self.bid_primary[idx] = 0
        self.bid_suits[idx] = 0
        self.n_bid_tokens[idx] = 0
        self.bid_actions_hist[idx] = 0
        self.bid_seats_hist[idx] = 0
        self.n_bids[idx] = 0
        self.pend_count[idx] = 0
        # Alternate BB team per reset
        for i in idx:
            self.bb_team[i] = self._reset_counter % 2
            self._reset_counter += 1

    def is_bb_turn(self):
        """Returns [N] bool — True if current player is on BB's team."""
        return (self.env_cur_player % 2) == self.bb_team

    def reset_all(self):
        self.reset_envs(np.arange(self.n))

    _SEQ_BUCKETS = [10, 16, 24, 34]

    def get_batch(self):
        """Build GPU batch for model inference."""
        max_nb = int(self.n_bid_tokens.max())
        raw_len = 10 + max_nb
        seq_len = raw_len
        for b in self._SEQ_BUCKETS:
            if b >= raw_len:
                seq_len = b
                break

        deal_t = torch.from_numpy(self.deal_idx).to(self.device)
        seat_t = torch.from_numpy(self.env_cur_player.astype(np.int64)).to(self.device)
        hand_p = self.gpu_primary[deal_t, seat_t]
        hand_s = self.gpu_suits[deal_t, seat_t]

        if seq_len == 10:
            pad = torch.zeros(self.n, 10, dtype=torch.bool, device=self.device)
            return hand_p, hand_s, pad

        bid_p = torch.from_numpy(self.bid_primary[:, :max_nb].copy()).to(self.device)
        bid_s = torch.from_numpy(self.bid_suits[:, :max_nb].copy()).to(self.device)
        pad_cols = seq_len - 10 - max_nb
        if pad_cols > 0:
            z = torch.zeros(self.n, pad_cols, dtype=torch.long, device=self.device)
            batch_p = torch.cat([hand_p, bid_p, z], dim=1)
            batch_s = torch.cat([hand_s, bid_s, z], dim=1)
        else:
            batch_p = torch.cat([hand_p, bid_p], dim=1)
            batch_s = torch.cat([hand_s, bid_s], dim=1)

        lens = torch.from_numpy((10 + self.n_bid_tokens).astype(np.int64)).to(self.device)
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
        pad_mask = positions >= lens.unsqueeze(1)
        return batch_p, batch_s, pad_mask

    def compute_legal_masks(self):
        """Vectorized legal masks. Returns [N, 43] float32 numpy."""
        N = self.n
        masks = np.zeros((N, NUM_BID_ACTIONS), dtype=np.float32)
        masks[:, BID_PASS] = 1.0

        no_bid = (self.env_bid_val < 0) & ~self.env_done
        masks[no_bid, 1:41] = 1.0

        has_bid = (self.env_bid_val >= 0) & ~self.env_done
        not_coinched = (self.env_coinche == 0) & has_bid
        if np.any(not_coinched):
            nc = np.where(not_coinched)[0]
            cv = self.env_bid_val[nc]
            ct = self.env_bid_team[nc]
            bv = _BID_VAL_LUT[1:41]
            higher = bv[None, :] > cv[:, None]
            masks[nc[:, None], np.arange(1, 41)[None, :]] = higher.astype(np.float32)
            opp = (self.env_cur_player[nc] % 2) != ct
            masks[nc[opp], BID_COINCHE] = 1.0

        coinched = (self.env_coinche == 1) & has_bid
        if np.any(coinched):
            ci = np.where(coinched)[0]
            same = (self.env_cur_player[ci] % 2) == self.env_bid_team[ci]
            masks[ci[same], BID_SURCOINCHE] = 1.0

        return masks

    def buffer_transitions(self, prim_np, suit_np, seq_len, masks_np, actions,
                           active_idx):
        """Buffer the current step's transitions for active envs."""
        for ii, i in enumerate(active_idx):
            k = self.pend_count[i]
            if k >= MAX_BID_STEPS:
                continue
            S = prim_np.shape[1]
            self.pend_primary[i, k, :S] = prim_np[ii]
            self.pend_primary[i, k, S:] = 0
            self.pend_suits[i, k, :S] = suit_np[ii]
            self.pend_suits[i, k, S:] = 0
            self.pend_slens[i, k] = seq_len[ii]
            self.pend_masks[i, k] = masks_np[active_idx[ii]]
            self.pend_actions[i, k] = actions[i]
            self.pend_teams[i, k] = self.env_cur_player[i] % 2
            self.pend_count[i] = k + 1

    def step_and_flush(self, actions, replay):
        """Step all active envs, flush done ones to replay buffer.

        Returns (completed, flushed_rewards).
        """
        active = ~self.env_done
        act = actions

        # Record bid history (before advancing cur_player)
        for i in np.where(active)[0]:
            nb = self.n_bids[i]
            if nb < MAX_BID_STEPS:
                self.bid_actions_hist[i, nb] = act[i]
                self.bid_seats_hist[i, nb] = self.env_cur_player[i]
                self.n_bids[i] = nb + 1

        # Append bid tokens
        nb = self.n_bid_tokens.copy()
        can_append = active & (nb + 2 <= 24)
        if np.any(can_append):
            ca = np.where(can_append)[0]
            nbs = nb[ca]
            acts = act[ca]
            self.bid_primary[ca, nbs] = ACTION_PRIMARY_LUT[acts]
            self.bid_suits[ca, nbs] = ACTION_SUIT_LUT[acts]
            self.bid_primary[ca, nbs + 1] = P_NONE
            self.bid_suits[ca, nbs + 1] = ACTION_SUIT2_LUT[acts]
            self.n_bid_tokens[ca] = nbs + 2

        # Update state
        is_pass = (act == BID_PASS) & active
        self.env_consec_passes[is_pass] += 1

        is_coinche = (act == BID_COINCHE) & active
        self.env_coinche[is_coinche] = 1
        self.env_consec_passes[is_coinche] = 0

        is_surcoinche = (act == BID_SURCOINCHE) & active
        self.env_coinche[is_surcoinche] = 2
        self.env_consec_passes[is_surcoinche] = 0
        self.env_done[is_surcoinche] = True

        is_bid = active & (act >= 1) & (act <= 40)
        if np.any(is_bid):
            bi = np.where(is_bid)[0]
            ba = act[bi]
            self.env_bid_val[bi] = _BID_VAL_LUT[ba]
            self.env_bid_suit[bi] = _BID_SUIT_LUT[ba]
            self.env_bid_team[bi] = self.env_cur_player[bi] % 2
            self.env_coinche[bi] = 0
            self.env_consec_passes[bi] = 0

        still_active = active & ~is_surcoinche
        void_end = still_active & (self.env_bid_val < 0) & (self.env_consec_passes >= 4)
        bid_end = still_active & (self.env_bid_val >= 0) & (self.env_consec_passes >= 3)
        self.env_done[void_end | bid_end] = True

        self.env_cur_player[active] = (self.env_cur_player[active] + 1) % 4

        # Flush done envs to replay buffer
        just_done = np.where(self.env_done & active)[0]
        completed = len(just_done)
        flushed_rewards = []
        for i in just_done:
            n_trans = self.pend_count[i]
            if n_trans == 0:
                continue
            # Compute DD return from the final contract
            dd = self.pool_dd[self.deal_idx[i]]
            ns_r, ew_r = _compute_returns(
                dd, self.env_bid_val[i], self.env_bid_suit[i],
                self.env_bid_team[i], self.env_coinche[i],
            )
            rewards = np.where(
                self.pend_teams[i, :n_trans] == 0, ns_r, ew_r,
            ).astype(np.float32)
            replay.add(
                self.pend_primary[i, :n_trans],
                self.pend_suits[i, :n_trans],
                self.pend_slens[i, :n_trans],
                self.pend_masks[i, :n_trans],
                self.pend_actions[i, :n_trans],
                rewards,
            )
            flushed_rewards.extend(rewards.tolist())

        if completed > 0:
            self.reset_envs(just_done)
        return completed, flushed_rewards


# ---------------------------------------------------------------------------
# Replay buffer for DQN transitions: (state, mask, action, reward)
# ---------------------------------------------------------------------------
class BidReplayBuffer:
    """Ring buffer storing (state_tokens, mask, action_taken, reward).

    Stores non-augmented data; augmentation applied fresh at sample time.
    """

    def __init__(self, capacity):
        S = MAX_SEQ_LEN
        A = NUM_BID_ACTIONS
        self.capacity = capacity
        self.primary = np.zeros((capacity, S), dtype=np.int64)
        self.suits = np.zeros((capacity, S), dtype=np.int64)
        self.seq_lens = np.zeros(capacity, dtype=np.int64)
        self.masks = np.zeros((capacity, A), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add(self, primary_np, suits_np, seq_lens_np, masks_np, actions_np,
            rewards_np):
        """Add a batch of transitions. All inputs are numpy arrays."""
        n = len(actions_np)
        S = primary_np.shape[1]
        if self.pos + n <= self.capacity:
            sl = slice(self.pos, self.pos + n)
            self.primary[sl, :S] = primary_np
            self.primary[sl, S:] = 0
            self.suits[sl, :S] = suits_np
            self.suits[sl, S:] = 0
            self.seq_lens[sl] = seq_lens_np
            self.masks[sl] = masks_np
            self.actions[sl] = actions_np
            self.rewards[sl] = rewards_np
        else:
            first = self.capacity - self.pos
            self.primary[self.pos:, :S] = primary_np[:first]
            self.primary[self.pos:, S:] = 0
            self.suits[self.pos:, :S] = suits_np[:first]
            self.suits[self.pos:, S:] = 0
            self.seq_lens[self.pos:] = seq_lens_np[:first]
            self.masks[self.pos:] = masks_np[:first]
            self.actions[self.pos:] = actions_np[:first]
            self.rewards[self.pos:] = rewards_np[:first]
            rest = n - first
            if rest > 0:
                self.primary[:rest, :S] = primary_np[first:]
                self.primary[:rest, S:] = 0
                self.suits[:rest, :S] = suits_np[first:]
                self.suits[:rest, S:] = 0
                self.seq_lens[:rest] = seq_lens_np[first:]
                self.masks[:rest] = masks_np[first:]
                self.actions[:rest] = actions_np[first:]
                self.rewards[:rest] = rewards_np[first:]
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(self.size, size=batch_size)
        max_len = int(self.seq_lens[idx].max())
        for b in [10, 16, 24, 34]:
            if b >= max_len:
                max_len = b
                break
        return {
            "primary": torch.from_numpy(self.primary[idx, :max_len].copy()).to(device),
            "suits": torch.from_numpy(self.suits[idx, :max_len].copy()).to(device),
            "seq_lens": torch.from_numpy(self.seq_lens[idx].copy()).to(device),
            "masks": torch.from_numpy(self.masks[idx].copy()).to(device),
            "actions": torch.from_numpy(self.actions[idx].copy()).to(device),
            "rewards": torch.from_numpy(self.rewards[idx].copy()).to(device),
        }


# ---------------------------------------------------------------------------
# Suit augmentation for DQN (suits + masks + actions, no target vector)
# ---------------------------------------------------------------------------
_SUIT_PERM_GPU = None
_ACTION_PERM_GPU = None


def augment_dqn_batch(batch_suits, batch_masks, batch_actions, device):
    """Apply random 24x suit permutation to a DQN batch.

    Returns (suits_aug, masks_aug, actions_aug, inv_perm).
    inv_perm[b, new_action] = old_action — for un-permuting greedy actions.
    """
    global _SUIT_PERM_GPU, _ACTION_PERM_GPU
    if _SUIT_PERM_GPU is None:
        _SUIT_PERM_GPU = SUIT_PERM_T.to(device)
        _ACTION_PERM_GPU = ACTION_PERM_TABLE.to(device)

    B = batch_suits.shape[0]
    perm_idx = torch.randint(24, (B,), device=device)
    suit_perm = _SUIT_PERM_GPU[perm_idx]

    # Permute suit IDs in token sequence
    is_real = batch_suits < 4
    permuted = torch.gather(suit_perm, 1, batch_suits.clamp(0, 3))
    suits_aug = torch.where(is_real, permuted, batch_suits)

    # Permute actions and masks
    action_perm = _ACTION_PERM_GPU[perm_idx]  # [B, 43]: old → new
    inv_perm = torch.zeros_like(action_perm)
    inv_perm.scatter_(1, action_perm,
                      torch.arange(43, device=device).unsqueeze(0).expand(B, -1))

    # Permute masks: new_mask[new_action] = old_mask[old_action]
    masks_aug = torch.gather(batch_masks, 1, inv_perm)

    # Permute taken actions: old → new
    actions_aug = torch.gather(action_perm, 1,
                               batch_actions.unsqueeze(1)).squeeze(1)

    return suits_aug, masks_aug, actions_aug, inv_perm


# ---------------------------------------------------------------------------
# Arena eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def arena_evaluate(model, device, bid_model_path, play_model_path, n_matches=100):
    try:
        from colver import Env
    except ImportError:
        return None
    bb_total = v2_total = 0.0
    bb_wins = v2_wins = 0
    for i in range(n_matches):
        env_a = Env(); env_a.reset()
        env_a.load_bid_model(bid_model_path, 512); env_a.load_dmc_model(play_model_path)
        hands, dealer = env_a.get_hands(), env_a.get_dealer()
        ns_a, ew_a = _play_game(env_a, model, 0, device)
        env_b = Env.deal_with_hands(dealer, hands)
        env_b.load_bid_model(bid_model_path, 512); env_b.load_dmc_model(play_model_path)
        ns_b, ew_b = _play_game(env_b, model, 1, device)
        bb_score, v2_score = ns_a + ew_b, ew_a + ns_b
        bb_total += bb_score; v2_total += v2_score
        if bb_score > v2_score: bb_wins += 1
        elif v2_score > bb_score: v2_wins += 1
    return bb_wins, v2_wins, n_matches, (bb_total - v2_total) / n_matches


@torch.no_grad()
def arena_evaluate_heuristic(model, device, play_model_path, n_matches=100):
    try:
        from colver import Env
    except ImportError:
        return None
    bb_total = h_total = 0.0
    bb_wins = h_wins = 0
    for i in range(n_matches):
        env_a = Env(); env_a.reset()
        env_a.load_dmc_model(play_model_path)
        hands, dealer = env_a.get_hands(), env_a.get_dealer()
        ns_a, ew_a = _play_game(env_a, model, 0, device, opp="heuristic")
        env_b = Env.deal_with_hands(dealer, hands)
        env_b.load_dmc_model(play_model_path)
        ns_b, ew_b = _play_game(env_b, model, 1, device, opp="heuristic")
        bb_score, h_score = ns_a + ew_b, ew_a + ns_b
        bb_total += bb_score; h_total += h_score
        if bb_score > h_score: bb_wins += 1
        elif h_score > bb_score: h_wins += 1
    return bb_wins, h_wins, n_matches, (bb_total - h_total) / n_matches


def _play_game(env, model, bb_team, device, opp="nn"):
    while env.phase() == 0 and not env.is_terminal():
        seat = env.current_player()
        if seat % 2 == bb_team:
            action = _bb_action(env, model, device)
        elif opp == "nn":
            action = env.action_bid_nn()["best_action"]
        else:
            action = env.bid_improved_v2()
        env.step(action)
    while not env.is_terminal():
        action = env.action_dmc_with_stats()["best_action"]
        env.step(action)
    return env.rewards()


def _bb_action(env, model, device):
    seat = env.current_player()
    dealer = env.get_dealer()
    hand_cards = env.get_hands()[seat]
    bid_history = env.get_bid_history()
    cards = [(c % 8, c // 8) for c in hand_cards]
    cards.sort(key=lambda c: c[1] * 8 + c[0])
    pos = (seat + 4 - dealer) % 4
    primary = [P_CLS, P_POS0 + pos]
    suits = [S_NULL, S_NULL]
    for rank, suit in cards:
        primary.append(P_RANK0 + rank)
        suits.append(suit)
    for _s, act in bid_history:
        if len(primary) + 2 > MAX_SEQ_LEN:
            break
        primary.extend([int(ACTION_PRIMARY_LUT[act]), P_NONE])
        suits.extend([int(ACTION_SUIT_LUT[act]), int(ACTION_SUIT2_LUT[act])])
    while len(primary) < MAX_SEQ_LEN:
        primary.append(0)
        suits.append(0)
    p_t = torch.tensor(primary[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
    s_t = torch.tensor(suits[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
    mask = env.legal_action_mask()
    m_t = torch.from_numpy(mask[:43]).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(p_t, s_t)
    logits = logits.float().masked_fill(m_t == 0, -1e9)
    return logits.argmax(dim=-1).item()


def _save_model(model, path):
    sd = model.state_dict()
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    torch.save(sd, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Bumblebid DQN training")
    p.add_argument("--pool-file", default="data/deals/archive/bumblebid_5M_enriched.bin")
    p.add_argument("--dd-pool", default="data/deals/archive/dd_5M_enriched.bin")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--n-envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--buffer-capacity", type=int, default=200_000)
    p.add_argument("--warmup-steps", type=int, default=2000,
                   help="Fill buffer before training starts")
    p.add_argument("--updates-per-step", type=int, default=1,
                   help="Gradient updates per env step")
    p.add_argument("--eps-start", type=float, default=0.3)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--eps-anneal-steps", type=int, default=0,
                   help="Steps to anneal epsilon (default: 80%% of --steps)")
    p.add_argument("--arena-freq", type=int, default=50_000)
    p.add_argument("--arena-matches", type=int, default=100)
    p.add_argument("--arena-bid-model", default="models/bid_v2/bid_nn_final.bin")
    p.add_argument("--arena-play-model", default="models/dmc_50.bin")
    p.add_argument("--save-freq", type=int, default=100_000)
    p.add_argument("--save-dir", default="models/bumblebid/dqn")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.eps_anneal_steps == 0:
        args.eps_anneal_steps = int(args.steps * 0.8)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")

    # Load pools
    dealers_bb, hp, hs, dd_pts_bb, _ = load_bumblebid_pool(args.pool_file)
    dealers_dd, hands_dd, dd_pts_dd = load_dd_pool(args.dd_pool)
    print(f"Pool: {len(dealers_dd):,} deals", flush=True)

    # Load nn_v2 opponent model
    opp_model = DuelingBidNet.load_from_bin(
        args.arena_bid_model, obs_dim=108, hidden=512, n_layers=3,
    ).to(device)
    print(f"nn_v2 opponent: {sum(p.numel() for p in opp_model.parameters()):,} params")

    # BB model
    model = Bumblebid(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
    ).to(device)
    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"Bumblebid: {n_params:,} params  d={args.d_model} L={args.n_layers} H={args.n_heads}")
    print(f"Steps: {args.steps:,}  LR: {args.lr}  Envs: {args.n_envs}  "
          f"Buffer: {args.buffer_capacity:,}  Batch: {args.batch_size}")
    print(f"eps: {args.eps_start:.2f}->{args.eps_end:.2f} over {args.eps_anneal_steps:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.98), fused=True,
    )

    vec_env = VecBidEnv(
        args.n_envs, dealers_dd, hp, hs, dd_pts_dd, hands_dd, device,
        seed=args.seed,
    )
    replay = BidReplayBuffer(args.buffer_capacity)

    print(f"\n{'Step':>10} | {'Loss':>8} | {'AvgR':>7} | "
          f"{'eps':>5} | {'Buf':>7} | {'Steps/s':>7}", flush=True)
    print("-" * 65, flush=True)

    total_loss = 0.0
    loss_count = 0
    total_reward = 0.0
    reward_count = 0
    total_episodes = 0
    last_log_time = time.time()
    last_log_step = 0
    env_rng = np.random.default_rng(args.seed + 1)

    for step in range(args.steps):
        # --- Epsilon schedule ---
        frac = min(step / max(args.eps_anneal_steps, 1), 1.0)
        epsilon = args.eps_start + (args.eps_end - args.eps_start) * frac

        # --- Collect one step from all envs ---
        masks_np = vec_env.compute_legal_masks()
        primary, suits, pad_mask = vec_env.get_batch()

        active = ~vec_env.env_done
        is_bb = vec_env.is_bb_turn()
        bb_active = active & is_bb
        opp_active = active & ~is_bb

        actions = np.zeros(vec_env.n, dtype=np.int32)

        # --- BB seats: ε-greedy from transformer model ---
        bb_idx = np.where(bb_active)[0]
        if len(bb_idx) > 0:
            prim_np = primary[bb_idx].cpu().numpy()
            suit_np = suits[bb_idx].cpu().numpy()
            slens = (10 + vec_env.n_bid_tokens[bb_idx]).astype(np.int64)

            with torch.no_grad():
                dummy_act = torch.zeros(vec_env.n, dtype=torch.long, device=device)
                suits_sel, masks_sel, _, inv_perm_sel = augment_dqn_batch(
                    suits.clone(),
                    torch.from_numpy(masks_np).to(device),
                    dummy_act,
                    device,
                )
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits_sel = model(primary, suits_sel, pad_mask)
                aug_logits = logits_sel.float().masked_fill(masks_sel == 0, -1e9)
                aug_actions = aug_logits.argmax(dim=-1)
                greedy_all = torch.gather(
                    inv_perm_sel, 1, aug_actions.unsqueeze(1)
                ).squeeze(1).cpu().numpy().astype(np.int32)

            actions[bb_idx] = greedy_all[bb_idx]
            explore_mask = env_rng.random(vec_env.n) < epsilon
            for i in np.where(explore_mask & bb_active)[0]:
                legal = np.where(masks_np[i] > 0.5)[0]
                actions[i] = env_rng.choice(legal)

            # Buffer BB transitions only
            vec_env.buffer_transitions(prim_np, suit_np, slens, masks_np,
                                       actions, bb_idx)

        # --- Opponent seats: nn_v2 greedy ---
        opp_idx = np.where(opp_active)[0]
        if len(opp_idx) > 0:
            opp_hands = hands_dd[vec_env.deal_idx[opp_idx]]
            opp_obs = encode_bid_obs_batch(
                opp_hands,
                vec_env.env_dealer[opp_idx],
                vec_env.env_cur_player[opp_idx],
                vec_env.bid_actions_hist[opp_idx],
                vec_env.bid_seats_hist[opp_idx],
                vec_env.n_bids[opp_idx],
            )
            opp_actions = select_actions_nn_v2(
                opp_model, opp_obs, masks_np[opp_idx], device,
            )
            actions[opp_idx] = opp_actions

        # Step envs and flush done auctions to replay buffer
        completed, flushed_rewards = vec_env.step_and_flush(actions, replay)
        total_episodes += completed
        if flushed_rewards:
            total_reward += sum(flushed_rewards)
            reward_count += len(flushed_rewards)

        # --- Train on replay buffer batch ---
        for _update in range(args.updates_per_step):
            if replay.size < args.warmup_steps:
                break
            batch = replay.sample(args.batch_size, device)

            # Build pad mask from seq_lens
            bL = batch["primary"].shape[1]
            positions = torch.arange(bL, device=device).unsqueeze(0)
            b_pad = positions >= batch["seq_lens"].unsqueeze(1)

            # Fresh augmentation
            b_suits_aug, b_masks_aug, b_actions_aug, _ = augment_dqn_batch(
                batch["suits"].clone(), batch["masks"],
                batch["actions"], device,
            )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch["primary"], b_suits_aug, b_pad)
                logits_f = logits.float()

                # DQN loss: MSE(Q(s, a_taken), return)
                q_taken = logits_f.gather(
                    1, b_actions_aug.unsqueeze(1)
                ).squeeze(1)
                loss = F.mse_loss(q_taken, batch["rewards"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            loss_count += 1

        # Logging
        if (step + 1) % 1000 == 0:
            elapsed = time.time() - last_log_time
            sps = (step + 1 - last_log_step) / max(elapsed, 1e-6)
            if loss_count > 0:
                avg_loss = total_loss / loss_count
                avg_r = total_reward / max(reward_count, 1)
                print(f"{step+1:10,} | {avg_loss:8.5f} | {avg_r:+7.4f} | "
                      f"{epsilon:5.3f} | {replay.size:>7,} | {sps:7.0f}",
                      flush=True)
            else:
                print(f"{step+1:10,} | {'(warmup)':>8} | {'':>7} | "
                      f"{epsilon:5.3f} | {replay.size:>7,} | {sps:7.0f}",
                      flush=True)
            total_loss = 0.0
            total_reward = 0.0
            reward_count = 0
            loss_count = 0
            last_log_step = step + 1
            last_log_time = time.time()

        if args.arena_freq > 0 and (step + 1) % args.arena_freq == 0:
            result = arena_evaluate(
                model, device, args.arena_bid_model, args.arena_play_model,
                n_matches=args.arena_matches,
            )
            if result:
                bb_w, v2_w, n, diff = result
                print(f"  >>> Arena vs nn_v2 ({n} matches): BB={bb_w} V2={v2_w} "
                      f"diff={diff:+.1f}", flush=True)
            result_h = arena_evaluate_heuristic(
                model, device, args.arena_play_model, n_matches=100,
            )
            if result_h:
                bb_w, h_w, n, diff = result_h
                print(f"  >>> Arena vs heuristic ({n} matches): BB={bb_w} H={h_w} "
                      f"diff={diff:+.1f}", flush=True)
            _save_model(model, save_dir / "latest.pt")

        if (step + 1) % args.save_freq == 0:
            _save_model(model, save_dir / f"step_{step+1}.pt")

    _save_model(model, save_dir / "final.pt")
    print(f"\nDone. Episodes: {total_episodes:,}  Buffer: {replay.size:,}",
          flush=True)


if __name__ == "__main__":
    main()
