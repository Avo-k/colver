"""
Load a COLVPD01 play-distill binary and compute interpretable features.

Input:  data/distill/play_distill_*.bin  (COLVPD01 — see distill_play.rs)
Output: data/distill/play_features.npz   (compressed numpy archive)
        + data/distill/play_qvalues.npz  (sparse Q-value records, optional)

Per-decision features (atomic + tactical, all trump-relative where appropriate):

Identifiers / context
  deal_id, forced_suit, dealer, trick_idx, play_idx, seat, trick_lead, chosen,
  n_legal, final_ns_pts, hand, legal, played_cards

Role / position
  is_declarer_team   (NS = declarer in setup_dd)
  is_partner_of_lead (current player is partner of trick leader)
  partner_seat
  is_lead, is_last_to_play

Hand composition (trump-relative)
  trump_count
  has_trump_jack, has_trump_nine, has_trump_ace, has_trump_ten,
    has_trump_king, has_trump_queen, has_trump_eight, has_trump_seven
  trump_points_in_hand (sum of TRUMP_POINTS over remaining trump cards)
  trump_strength_max (highest trump remaining in hand, 0..7 in trump-strength order)
  side_aces, side_tens, side_voids, side_singletons, side_doubletons,
    best_side_length

Trick state
  lead_suit          (suit_led, or 4 if I'm leading)
  is_trump_led
  pts_in_trick_so_far
  trick_winner_so_far_seat   (seat that holds the highest card so far, or 255)
  partner_winning            (trick_winner_so_far is my partner)
  current_winner_card        (255 if empty trick)
  trick_has_trump            (any trump played in the trick so far)

Master / control (vs cards still outstanding)
  holds_master_trump   (highest trump remaining anywhere is in my hand)
  holds_master_lead    (highest card remaining in lead suit is in my hand)
  holds_master_per_side[3]  (highest card remaining in each side suit, in trump-relative slots)
    -> exposed as: holds_master_in_each_side_count (0..3)

Tactics
  can_follow_lead   (I have at least one card in lead suit; 1 if I'm leading)
  can_cut           (I'm void in lead suit AND I have trump AND lead suit != trump)
  has_higher_trump_than_trick (trick has trump and my best trump beats current best)
  partner_has_master_in_lead  (partner holds master of lead suit, inferred from voids+played)
  trumps_remaining_outside     (trumps not yet played and not in my hand)
  outstanding_trumps_in_opps   (estimate via voids: opps still have trump = both opps not void in trump)

Voids (from engine tracking)
  partner_void_in_lead, opp_left_void_in_lead, opp_right_void_in_lead
  partner_void_in_trump, any_opp_void_in_trump

Game progression
  pts_so_far_ns, pts_so_far_ew, tricks_remaining

Q-value summary (per row)
  q_chosen, q_max, q_min, q_2nd_best, q_margin, q_best_minus_chosen

The Q-vector summary is computed in NS-perspective. Defenders pick MIN, declarer
picks MAX; q_margin is always |chosen - 2nd_best| with sign convention:
positive = chose better than 2nd best (in role's direction).
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

# --- Card layout (matches colver_core::card) ---
# Bits 0-7 = spades, 8-15 = hearts, 16-23 = diamonds, 24-31 = clubs.
# Ranks: 0=7, 1=8, 2=9, 3=J, 4=Q, 5=K, 6=10, 7=A.

TRUMP_POINTS = np.array([0, 0, 14, 20, 3, 4, 10, 11], dtype=np.int8)  # by rank
PLAIN_POINTS = np.array([0, 0, 0, 2, 3, 4, 10, 11], dtype=np.int8)

# Trump strength: J(3)>9(2)>A(7)>10(6)>K(5)>Q(4)>8(1)>7(0)
# Map: rank -> strength (0..7, higher = stronger)
TRUMP_STRENGTH_BY_RANK = np.array([0, 1, 6, 7, 2, 3, 4, 5], dtype=np.int8)
# Inverse: which rank is the i-th strongest trump (rank index)
# strength 7 -> rank 3 (J), 6 -> rank 2 (9), 5 -> rank 7 (A), 4 -> rank 6 (10),
# 3 -> rank 5 (K), 2 -> rank 4 (Q), 1 -> rank 1 (8), 0 -> rank 0 (7)

# Plain strength = rank itself (A=7 highest, 7=0 lowest)


def load_records(path: Path):
    """Single-pass loader. Returns (fixed_arrays, qvals_flat)."""
    raw = path.read_bytes()
    if raw[:8] != b"COLVPD01":
        raise ValueError(f"Bad magic: {raw[:8]!r}")
    version = raw[8]
    n_records = struct.unpack_from("<Q", raw, 16)[0]
    # L'échelle des q_values dépend de la version : v1 = points cartes N-S
    # (0-252), v2 = écart de score de donne N-S − E-O (contrat compris, ±500).
    # Deux échelles, pas deux unités du même axe — ne pas mélanger deux fichiers.
    scale = "deal-score margin NS-EW" if version >= 2 else "NS card points"
    print(f"  magic OK, version={version}, n_records={n_records:,}, size={len(raw):,} bytes")
    print(f"  q_values scale: {scale}")

    # Pre-allocate fixed-size arrays
    deal_id = np.empty(n_records, dtype=np.uint32)
    forced_suit = np.empty(n_records, dtype=np.uint8)
    dealer = np.empty(n_records, dtype=np.uint8)
    trick_idx = np.empty(n_records, dtype=np.uint8)
    play_idx = np.empty(n_records, dtype=np.uint8)
    seat = np.empty(n_records, dtype=np.uint8)
    trick_lead = np.empty(n_records, dtype=np.uint8)
    chosen = np.empty(n_records, dtype=np.uint8)
    n_legal_arr = np.empty(n_records, dtype=np.uint8)
    final_ns_pts = np.empty(n_records, dtype=np.uint8)
    hand = np.empty(n_records, dtype=np.uint32)
    legal = np.empty(n_records, dtype=np.uint32)
    played_cards = np.empty(n_records, dtype=np.uint32)
    trick_packed = np.empty(n_records, dtype=np.uint32)
    voids_packed = np.empty(n_records, dtype=np.uint32)

    # Q-vals: variable length. Pre-scan size by summing n_legal would require
    # a pass; instead grow with array module.
    q_card_chunks = []
    q_value_chunks = []
    q_offsets = np.empty(n_records + 1, dtype=np.int64)
    q_offsets[0] = 0
    cur_q_offset = 0

    record_struct = struct.Struct("<I 10B 5I")
    rec_size_fixed = record_struct.size  # 4 + 10 + 20 = 34
    qpair_struct = struct.Struct("<Bf")
    qpair_size = qpair_struct.size  # 5

    pos = 24  # past header
    t0 = time.time()
    chunk_card_buf = []
    chunk_value_buf = []

    for i in range(n_records):
        (
            deal_id[i],
            forced_suit[i], dealer[i], trick_idx[i], play_idx[i], seat[i],
            trick_lead[i], chosen[i], n_legal_arr[i], final_ns_pts[i], _pad,
            hand[i], legal[i], played_cards[i], trick_packed[i], voids_packed[i],
        ) = record_struct.unpack_from(raw, pos)
        pos += rec_size_fixed

        nl = int(n_legal_arr[i])
        for _ in range(nl):
            card, q = qpair_struct.unpack_from(raw, pos)
            chunk_card_buf.append(card)
            chunk_value_buf.append(q)
            pos += qpair_size

        cur_q_offset += nl
        q_offsets[i + 1] = cur_q_offset

        if i % 1_000_000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (n_records - i) / rate
            print(f"    parsed {i:,}/{n_records:,} ({rate/1e6:.2f}M/s, ETA {eta:.0f}s)")
            # Flush chunks to keep Python list size manageable
            q_card_chunks.append(np.asarray(chunk_card_buf, dtype=np.uint8))
            q_value_chunks.append(np.asarray(chunk_value_buf, dtype=np.float32))
            chunk_card_buf = []
            chunk_value_buf = []

    if chunk_card_buf:
        q_card_chunks.append(np.asarray(chunk_card_buf, dtype=np.uint8))
        q_value_chunks.append(np.asarray(chunk_value_buf, dtype=np.float32))

    q_card = np.concatenate(q_card_chunks) if q_card_chunks else np.empty(0, dtype=np.uint8)
    q_value = np.concatenate(q_value_chunks) if q_value_chunks else np.empty(0, dtype=np.float32)

    print(f"  parsed in {time.time()-t0:.1f}s, q_total={len(q_card):,}")

    return {
        "deal_id": deal_id,
        "forced_suit": forced_suit,
        "dealer": dealer,
        "trick_idx": trick_idx,
        "play_idx": play_idx,
        "seat": seat,
        "trick_lead": trick_lead,
        "chosen": chosen,
        "n_legal": n_legal_arr,
        "final_ns_pts": final_ns_pts,
        "hand": hand,
        "legal": legal,
        "played_cards": played_cards,
        "trick_packed": trick_packed,
        "voids_packed": voids_packed,
        "q_offsets": q_offsets,
        "q_card": q_card,
        "q_value": q_value,
    }


# ---------- Feature engineering (vectorized) ----------

def suit_bits(cardset_u32: np.ndarray, suit: np.ndarray | int) -> np.ndarray:
    """Extract the 8 bits for `suit` (0..3) per row. Returns u8 per row."""
    if isinstance(suit, np.ndarray):
        shifts = (suit.astype(np.uint32) * 8)
        return ((cardset_u32 >> shifts) & 0xFF).astype(np.uint8)
    else:
        return ((cardset_u32 >> (int(suit) * 8)) & 0xFF).astype(np.uint8)


def popcount8(x: np.ndarray) -> np.ndarray:
    """Population count of 8-bit values."""
    x = x.astype(np.uint8)
    x = (x & 0x55) + ((x >> 1) & 0x55)
    x = (x & 0x33) + ((x >> 2) & 0x33)
    x = (x & 0x0F) + ((x >> 4) & 0x0F)
    return x


def popcount32(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint32)
    x = (x & 0x55555555) + ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x & 0x0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F)
    x = (x & 0x00FF00FF) + ((x >> 8) & 0x00FF00FF)
    x = (x & 0x0000FFFF) + ((x >> 16) & 0x0000FFFF)
    return x


def highest_set_bit_or_minus1(bits8: np.ndarray) -> np.ndarray:
    """For each u8 row, return the index of the highest set bit (0..7) or -1."""
    out = np.full(len(bits8), -1, dtype=np.int8)
    for r in range(7, -1, -1):
        mask = (bits8 >> r) & 1 == 1
        sel = (out == -1) & mask
        out[sel] = r
    return out


def highest_trump_strength_in_bits(bits8: np.ndarray) -> np.ndarray:
    """For each u8 (one suit's cards), return the max trump-strength (0..7) or -1."""
    out = np.full(len(bits8), -1, dtype=np.int8)
    # Iterate strength from high to low: J=str7,9=6,A=5,10=4,K=3,Q=2,8=1,7=0
    rank_by_strength = [3, 2, 7, 6, 5, 4, 1, 0]
    for str_idx, rank in enumerate(rank_by_strength):
        strength = 7 - str_idx
        mask = (bits8 >> rank) & 1 == 1
        sel = (out == -1) & mask
        out[sel] = strength
    return out


def lead_suit_from_trick(trick_packed: np.ndarray, trick_lead: np.ndarray) -> np.ndarray:
    """Suit of the lead card; returns 4 if trick is empty (current player leading)."""
    # Extract card at trick_lead position
    shifts = trick_lead.astype(np.uint32) * 8
    lead_card = ((trick_packed >> shifts) & 0xFF).astype(np.int16)
    # 0xFF = empty
    suit = np.where(lead_card == 0xFF, 4, lead_card // 8).astype(np.int8)
    return suit


def card_at_seat(trick_packed: np.ndarray, seat: np.ndarray) -> np.ndarray:
    """Extract card played at given seat from packed trick (0xFF if empty)."""
    shifts = seat.astype(np.uint32) * 8
    return ((trick_packed >> shifts) & 0xFF).astype(np.int16)


def card_points(card: np.ndarray, trump: np.ndarray) -> np.ndarray:
    """Belote card points for given (card, trump_suit). card in 0..31, -1 ignored."""
    out = np.zeros(len(card), dtype=np.int16)
    valid = card >= 0
    suit_of_card = (card // 8).astype(np.int8)
    rank_of_card = (card % 8).astype(np.int8)
    is_trump = valid & (suit_of_card == trump)
    out[is_trump] = TRUMP_POINTS[rank_of_card[is_trump]]
    is_plain = valid & (suit_of_card != trump)
    out[is_plain] = PLAIN_POINTS[rank_of_card[is_plain]]
    return out


def compute_trick_winner(trick_packed: np.ndarray, trick_lead: np.ndarray, play_idx: np.ndarray, trump: np.ndarray) -> np.ndarray:
    """Seat that currently leads the trick after `play_idx` cards played.
       Returns 255 if no card played yet."""
    n = len(trick_packed)
    out = np.full(n, 255, dtype=np.uint8)
    has_play = play_idx > 0
    if not has_play.any():
        return out

    # Order cards by play sequence: lead, lead+1, lead+2, lead+3
    lead_suit = lead_suit_from_trick(trick_packed, trick_lead)

    # For each row, walk through play order and find the current best.
    # Vectorized version: build (card, seat) pairs in play order.
    seats_in_order = np.zeros((n, 4), dtype=np.int16)
    cards_in_order = np.full((n, 4), -1, dtype=np.int16)
    for k in range(4):
        seat_k = (trick_lead.astype(np.uint8) + k) & 3
        seats_in_order[:, k] = seat_k
        shifts = seat_k.astype(np.uint32) * 8
        c = ((trick_packed >> shifts) & 0xFF).astype(np.int16)
        in_play = (k < play_idx)
        cards_in_order[in_play, k] = c[in_play]

    # Score each card: trump=1000+strength, follow=100+rank, off-suit=0
    suit_each = np.where(cards_in_order < 0, -1, cards_in_order // 8).astype(np.int16)
    rank_each = np.where(cards_in_order < 0, -1, cards_in_order % 8).astype(np.int16)
    score = np.full((n, 4), -1, dtype=np.int32)
    trump_b = trump[:, None]
    lead_b = lead_suit[:, None]
    is_trump_card = (suit_each == trump_b) & (cards_in_order >= 0)
    is_lead_card = (suit_each == lead_b) & (cards_in_order >= 0) & (~is_trump_card)
    # Trump-strength
    rank_safe = np.clip(rank_each, 0, 7).astype(np.int8)
    trump_str = TRUMP_STRENGTH_BY_RANK[rank_safe].astype(np.int32)
    score[is_trump_card] = (1000 + trump_str)[is_trump_card]
    score[is_lead_card] = (100 + rank_each.astype(np.int32))[is_lead_card]
    # Off-suit (cardless playable but doesn't beat lead)
    other = (~is_trump_card) & (~is_lead_card) & (cards_in_order >= 0)
    score[other] = 0  # can't win

    # Argmax over k
    winner_k = np.argmax(score, axis=1)
    winner_seat = seats_in_order[np.arange(n), winner_k]
    out[has_play] = winner_seat[has_play]
    return out


def points_in_trick_so_far(trick_packed: np.ndarray, play_idx: np.ndarray, trump: np.ndarray) -> np.ndarray:
    """Sum of belote points of cards already played in current trick."""
    n = len(trick_packed)
    out = np.zeros(n, dtype=np.int16)
    for s in range(4):
        shifts = s * 8
        c = ((trick_packed >> shifts) & 0xFF).astype(np.int16)
        # Only count if trick_lead+k corresponds to a play that happened.
        # Simpler: card played iff c != 0xFF
        played_mask = c != 0xFF
        c_pts = card_points(np.where(played_mask, c, 0), trump)
        out[played_mask] += c_pts[played_mask]
    return out


def void_bits(voids_packed: np.ndarray, seat: np.ndarray) -> np.ndarray:
    """Extract per-seat voids u8 (4 bits = 1 if void in suit i)."""
    shifts = seat.astype(np.uint32) * 8
    return ((voids_packed >> shifts) & 0xFF).astype(np.uint8)


def has_void_in(voids_packed: np.ndarray, seat: np.ndarray, suit: np.ndarray) -> np.ndarray:
    """Boolean: seat is void in suit. If suit==4 (no lead yet), returns False."""
    vb = void_bits(voids_packed, seat)
    has_v = (vb >> suit.astype(np.uint8)) & 1 == 1
    return has_v & (suit < 4)


def compute_features(rec: dict) -> dict:
    n = len(rec["deal_id"])
    print(f"  computing features for {n:,} rows...")
    t0 = time.time()

    trump = rec["forced_suit"].astype(np.int8)  # 0..3
    seat = rec["seat"].astype(np.int8)
    hand = rec["hand"]
    played = rec["played_cards"]
    legal = rec["legal"]
    trick_packed = rec["trick_packed"]
    voids_packed = rec["voids_packed"]
    trick_lead = rec["trick_lead"].astype(np.int8)
    play_idx = rec["play_idx"].astype(np.int8)
    chosen = rec["chosen"].astype(np.int16)

    out = {}

    # Role / position
    is_declarer_team = ((seat & 1) == 0).astype(np.uint8)  # NS=0/2 are declarer in setup_dd
    out["is_declarer_team"] = is_declarer_team
    partner_seat = (seat ^ 2).astype(np.uint8)
    out["partner_seat"] = partner_seat
    out["is_lead"] = (play_idx == 0).astype(np.uint8)
    out["is_last_to_play"] = (play_idx == 3).astype(np.uint8)
    out["is_partner_of_lead"] = (trick_lead == partner_seat).astype(np.uint8)

    # Hand composition (trump-relative)
    trump_bits = suit_bits(hand, trump)
    out["trump_count"] = popcount8(trump_bits).astype(np.int8)
    out["has_trump_seven"] = ((trump_bits >> 0) & 1).astype(np.uint8)
    out["has_trump_eight"] = ((trump_bits >> 1) & 1).astype(np.uint8)
    out["has_trump_nine"] = ((trump_bits >> 2) & 1).astype(np.uint8)
    out["has_trump_jack"] = ((trump_bits >> 3) & 1).astype(np.uint8)
    out["has_trump_queen"] = ((trump_bits >> 4) & 1).astype(np.uint8)
    out["has_trump_king"] = ((trump_bits >> 5) & 1).astype(np.uint8)
    out["has_trump_ten"] = ((trump_bits >> 6) & 1).astype(np.uint8)
    out["has_trump_ace"] = ((trump_bits >> 7) & 1).astype(np.uint8)
    out["trump_strength_max"] = highest_trump_strength_in_bits(trump_bits)

    # Trump points in hand (sum of TRUMP_POINTS for each trump card)
    tpts = np.zeros(n, dtype=np.int16)
    for r in range(8):
        has_r = (trump_bits >> r) & 1 == 1
        tpts[has_r] += int(TRUMP_POINTS[r])
    out["trump_points_in_hand"] = tpts

    # Side-suit aggregates
    side_aces = np.zeros(n, dtype=np.int8)
    side_tens = np.zeros(n, dtype=np.int8)
    side_voids = np.zeros(n, dtype=np.int8)
    side_singletons = np.zeros(n, dtype=np.int8)
    side_doubletons = np.zeros(n, dtype=np.int8)
    best_side_length = np.zeros(n, dtype=np.int8)
    for s in range(4):
        is_side = trump != s
        sb = suit_bits(hand, s)
        sc = popcount8(sb).astype(np.int8)
        has_ace = ((sb >> 7) & 1 == 1) & is_side
        has_ten = ((sb >> 6) & 1 == 1) & is_side
        side_aces[has_ace] += 1
        side_tens[has_ten] += 1
        sel = is_side & (sc == 0)
        side_voids[sel] += 1
        sel = is_side & (sc == 1)
        side_singletons[sel] += 1
        sel = is_side & (sc == 2)
        side_doubletons[sel] += 1
        sel = is_side & (sc > best_side_length)
        best_side_length[sel] = sc[sel]
    out["side_aces"] = side_aces
    out["side_tens"] = side_tens
    out["side_voids"] = side_voids
    out["side_singletons"] = side_singletons
    out["side_doubletons"] = side_doubletons
    out["best_side_length"] = best_side_length

    # Trick state
    lead_suit = lead_suit_from_trick(trick_packed, trick_lead)  # 0..4
    out["lead_suit"] = lead_suit
    out["is_trump_led"] = ((lead_suit == trump) & (lead_suit < 4)).astype(np.uint8)

    out["pts_in_trick_so_far"] = points_in_trick_so_far(trick_packed, play_idx, trump)
    winner = compute_trick_winner(trick_packed, trick_lead, play_idx, trump)
    out["trick_winner_so_far_seat"] = winner
    valid_winner = winner != 255
    partner_winning = np.zeros(n, dtype=np.uint8)
    partner_winning[valid_winner] = (winner[valid_winner] == partner_seat[valid_winner]).astype(np.uint8)
    out["partner_winning"] = partner_winning

    # Trick has trump? Any card in the trick is in trump suit.
    trick_has_trump = np.zeros(n, dtype=np.uint8)
    for s_idx in range(4):
        c = ((trick_packed >> (s_idx * 8)) & 0xFF).astype(np.int16)
        played_mask = c != 0xFF
        c_suit = np.where(played_mask, c // 8, -1).astype(np.int8)
        trick_has_trump |= ((c_suit == trump) & played_mask).astype(np.uint8)
    out["trick_has_trump"] = trick_has_trump

    # Master / control: highest remaining card outside of `played_cards`
    # outstanding = 0xFFFFFFFF & ~played_cards (but only 32 bits)
    outstanding = (~played) & 0xFFFFFFFF
    # In each suit, find highest remaining card; check if it's in our hand.
    # For trump: max by trump strength. For side suits: max by rank.
    # holds_master_trump
    trump_outstanding = suit_bits(outstanding, trump)
    trump_in_hand = trump_bits
    holds_master_trump = np.zeros(n, dtype=np.uint8)
    # For each row: which is the highest trump remaining? Is it in trump_in_hand?
    # Iterate trump strength high→low: rank order [3,2,7,6,5,4,1,0]
    found = np.zeros(n, dtype=bool)
    for rank in [3, 2, 7, 6, 5, 4, 1, 0]:
        is_present = (trump_outstanding >> rank) & 1 == 1
        new_find = is_present & (~found)
        in_hand = (trump_in_hand >> rank) & 1 == 1
        holds_master_trump[new_find & in_hand] = 1
        found |= new_find
    out["holds_master_trump"] = holds_master_trump

    # holds_master_lead: for each row with a lead suit (lead_suit < 4 and != trump),
    # check if the highest remaining card in lead suit is in our hand.
    has_lead = (lead_suit >= 0) & (lead_suit < 4) & (lead_suit != trump)
    holds_master_lead = np.zeros(n, dtype=np.uint8)
    if has_lead.any():
        ls = lead_suit.astype(np.uint8)
        lead_outstanding = ((outstanding >> (ls * 8)) & 0xFF).astype(np.uint8)
        lead_in_hand = ((hand >> (ls * 8)) & 0xFF).astype(np.uint8)
        # plain rank order high→low: 7,6,5,4,3,2,1,0
        found2 = np.zeros(n, dtype=bool)
        for rank in [7, 6, 5, 4, 3, 2, 1, 0]:
            is_present = (lead_outstanding >> rank) & 1 == 1
            new_find = is_present & (~found2) & has_lead
            in_hand = (lead_in_hand >> rank) & 1 == 1
            holds_master_lead[new_find & in_hand] = 1
            found2 |= new_find
    out["holds_master_lead"] = holds_master_lead

    # holds_master_in_each_side_count: of the 3 side suits, count how many we hold the master of
    holds_master_side_count = np.zeros(n, dtype=np.int8)
    for s in range(4):
        is_side = trump != s
        side_outstanding = suit_bits(outstanding, s)
        side_in_hand = suit_bits(hand, s)
        # check highest remaining
        found_s = np.zeros(n, dtype=bool)
        master_in_hand = np.zeros(n, dtype=bool)
        for rank in [7, 6, 5, 4, 3, 2, 1, 0]:
            is_present = (side_outstanding >> rank) & 1 == 1
            new_find = is_present & (~found_s) & is_side
            in_hand = (side_in_hand >> rank) & 1 == 1
            master_in_hand[new_find & in_hand] = True
            found_s |= new_find
        holds_master_side_count[master_in_hand] += 1
    out["holds_master_side_count"] = holds_master_side_count

    # Tactics
    # can_follow_lead: I have a card in lead suit (or I'm leading)
    can_follow_lead = np.zeros(n, dtype=np.uint8)
    # if lead_suit == 4 (I'm leading), set 1
    can_follow_lead[lead_suit == 4] = 1
    sel = (lead_suit >= 0) & (lead_suit < 4)
    if sel.any():
        ls = np.where(sel, lead_suit, 0).astype(np.uint8)
        my_lead_bits = ((hand >> (ls * 8)) & 0xFF).astype(np.uint8)
        can_follow_lead[sel & (my_lead_bits != 0)] = 1
    out["can_follow_lead"] = can_follow_lead

    # can_cut: void in lead suit, lead suit != trump, I have trump
    has_trump = trump_bits != 0
    is_offsuit_lead = (lead_suit < 4) & (lead_suit != trump)
    am_void_in_lead = is_offsuit_lead & (can_follow_lead == 0)
    out["can_cut"] = (am_void_in_lead & has_trump).astype(np.uint8)

    # has_higher_trump_than_trick: trick has trump and my best trump beats current best trump on table
    has_higher_trump = np.zeros(n, dtype=np.uint8)
    if trick_has_trump.any():
        # find max trump strength on table
        best_table = np.full(n, -1, dtype=np.int8)
        for s_idx in range(4):
            c = ((trick_packed >> (s_idx * 8)) & 0xFF).astype(np.int16)
            played_mask = c != 0xFF
            c_suit = np.where(played_mask, c // 8, -1).astype(np.int8)
            c_rank = np.where(played_mask, c % 8, 0).astype(np.int8)
            is_trump_c = (c_suit == trump) & played_mask
            cs = TRUMP_STRENGTH_BY_RANK[c_rank]
            sel = is_trump_c & (cs > best_table)
            best_table[sel] = cs[sel]
        my_best = out["trump_strength_max"]  # int8, -1 if no trump
        has_higher_trump = ((trick_has_trump == 1) & (my_best > best_table)).astype(np.uint8)
    out["has_higher_trump_than_trick"] = has_higher_trump

    # trumps_remaining_outside: trumps not yet played AND not in my hand
    trump_remaining_anywhere = popcount8(trump_outstanding).astype(np.int8)
    trump_in_hand_count = popcount8(trump_bits).astype(np.int8)
    out["trumps_remaining_outside"] = (trump_remaining_anywhere - trump_in_hand_count).astype(np.int8)

    # outstanding_trumps_in_opps: estimate from voids
    # opponents = the two seats with parity != my parity
    vb_partner = void_bits(voids_packed, partner_seat)
    opp_left = (seat + 1) & 3
    opp_right = (seat + 3) & 3
    vb_oppL = void_bits(voids_packed, opp_left)
    vb_oppR = void_bits(voids_packed, opp_right)
    out["partner_void_in_lead"] = (((vb_partner >> np.where(lead_suit < 4, lead_suit, 0).astype(np.uint8)) & 1) & (lead_suit < 4)).astype(np.uint8)
    out["opp_left_void_in_lead"] = (((vb_oppL >> np.where(lead_suit < 4, lead_suit, 0).astype(np.uint8)) & 1) & (lead_suit < 4)).astype(np.uint8)
    out["opp_right_void_in_lead"] = (((vb_oppR >> np.where(lead_suit < 4, lead_suit, 0).astype(np.uint8)) & 1) & (lead_suit < 4)).astype(np.uint8)
    out["partner_void_in_trump"] = ((vb_partner >> trump.astype(np.uint8)) & 1).astype(np.uint8)
    any_opp_void_trump = (((vb_oppL >> trump.astype(np.uint8)) & 1) | ((vb_oppR >> trump.astype(np.uint8)) & 1)).astype(np.uint8)
    out["any_opp_void_in_trump"] = any_opp_void_trump

    # Game progression
    out["tricks_remaining"] = (8 - rec["trick_idx"].astype(np.int8))

    # Q-value summary per row
    qoff = rec["q_offsets"]
    qcard = rec["q_card"]
    qval = rec["q_value"]
    n_legal = rec["n_legal"].astype(np.int16)
    q_chosen = np.zeros(n, dtype=np.float32)
    q_max = np.full(n, -1e9, dtype=np.float32)
    q_min = np.full(n, 1e9, dtype=np.float32)
    q_2nd = np.full(n, np.nan, dtype=np.float32)
    q_2nd_min = np.full(n, np.nan, dtype=np.float32)

    # Vectorize via a per-record loop is unavoidable for variable lengths;
    # but we can do it efficiently with numpy reductions per row.
    # Instead: compute via segment_reduce-like approach.
    print("    Q-value summary...")
    for i in range(n):
        s = qoff[i]
        e = qoff[i + 1]
        if e == s:
            continue
        qs = qval[s:e]
        cs = qcard[s:e]
        # find chosen
        ch = chosen[i]
        idx = np.where(cs == ch)[0]
        if idx.size:
            q_chosen[i] = qs[idx[0]]
        q_max[i] = qs.max()
        q_min[i] = qs.min()
        if qs.size > 1:
            sorted_q = np.sort(qs)
            q_2nd[i] = sorted_q[-2]      # 2nd highest
            q_2nd_min[i] = sorted_q[1]   # 2nd lowest

    out["q_chosen"] = q_chosen
    out["q_max"] = q_max
    out["q_min"] = q_min
    out["q_2nd_max"] = q_2nd
    out["q_2nd_min"] = q_2nd_min
    # Margin: in role's direction (declarer maximizes, defender minimizes NS pts).
    declarer_team = is_declarer_team == 1
    margin = np.zeros(n, dtype=np.float32)
    margin[declarer_team] = (q_chosen[declarer_team] - q_2nd[declarer_team])
    margin[~declarer_team] = (q_2nd_min[~declarer_team] - q_chosen[~declarer_team])
    out["q_margin"] = margin
    # how much chose-vs-best (in role direction): 0 = optimal
    role_best = np.where(declarer_team, q_max, q_min)
    out["q_chosen_vs_best"] = (np.where(declarer_team, q_chosen - role_best, role_best - q_chosen)).astype(np.float32)

    print(f"  features computed in {time.time()-t0:.1f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/distill/play_distill_200k.bin")
    ap.add_argument("--output", default="data/distill/play_features.npz")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    print(f"Loading {in_path}...")
    rec = load_records(in_path)

    feat = compute_features(rec)

    # Pass-through identifiers + raw bitmasks (handy for re-derivation in notebooks)
    passthrough = ["deal_id", "forced_suit", "dealer", "trick_idx", "play_idx",
                   "seat", "trick_lead", "chosen", "n_legal", "final_ns_pts",
                   "hand", "legal", "played_cards", "trick_packed", "voids_packed"]
    arrays = {k: rec[k] for k in passthrough}
    arrays.update(feat)

    print(f"Saving {len(arrays)} arrays to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    sz = out_path.stat().st_size
    print(f"  saved ({sz/1024/1024:.1f} MB)")
    print(f"\nColumns: {list(arrays.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
