"""Bumblebid: Encoder-only transformer for Belote Contree bidding.

ModernBERT-inspired architecture:
  - Pre-norm RMSNorm (no bias)
  - GeGLU FFN (2/3 * 4 * d_model intermediate)
  - Learned positional embeddings
  - Shared suit embeddings between cards and bids

Token sequence:
  [CLS] [POS_x] [card_1 ... card_8] [bid_val bid_suit] ...

Each token embedding = primary_emb[id] + suit_emb[suit] + pos_emb[pos].
Output: [CLS] -> RMSNorm -> Linear -> 43 bid action logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Primary token IDs (first embedding component)
# ---------------------------------------------------------------------------
P_NONE = 0  # placeholder for suit-only tokens (2nd slot of bids)
P_CLS = 1
P_POS0 = 2  # ..P_POS3 = 5  (dealer-relative seat 0-3)
P_RANK0 = 6  # ..P_RANK7 = 13 (card ranks: 7,8,9,J,Q,K,10,A)
P_VAL0 = 14  # ..P_VAL8 = 22  (bid values: 80,90,...,160)
P_CAPOT = 23
P_PASS = 24
P_COINCHE = 25
P_SURCOINCHE = 26
NUM_PRIMARY = 27

# ---------------------------------------------------------------------------
# Suit IDs (shared between card tokens and bid suit tokens)
# ---------------------------------------------------------------------------
S_SPADES = 0
S_HEARTS = 1
S_DIAMONDS = 2
S_CLUBS = 3
S_NULL = 4  # for CLS, POS, bid-value slot, PASS/COINCHE/SURCOINCHE
NUM_SUITS = 5

MAX_SEQ_LEN = 34  # CLS + POS + 8 cards + 12 bid rounds * 2 tokens
NUM_BID_ACTIONS = 43


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class GeGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.gelu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_norm = RMSNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = GeGLU(d_model, d_ff)

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        h = self.attn_norm(x)

        qkv = self.qkv_proj(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if pad_mask is not None:
            # pad_mask: [B, L] bool, True = padding → convert to float -inf mask
            attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_mask = attn_mask.to(dtype=q.dtype) * torch.finfo(q.dtype).min

        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        h = h.transpose(1, 2).contiguous().view(B, L, D)
        h = self.out_proj(h)

        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class Bumblebid(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = int(2 / 3 * 4 * d_model)

        self.primary_emb = nn.Embedding(NUM_PRIMARY, d_model)
        self.suit_emb = nn.Embedding(NUM_SUITS, d_model)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, d_model)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        self.out_norm = RMSNorm(d_model)
        self.out_head = nn.Linear(d_model, NUM_BID_ACTIONS, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        primary_ids: torch.Tensor,  # [B, L] long
        suit_ids: torch.Tensor,  # [B, L] long
        pad_mask: torch.Tensor | None = None,  # [B, L] bool, True = padding
    ) -> torch.Tensor:
        """Returns raw logits [B, 43] for bid actions."""
        B, L = primary_ids.shape
        positions = torch.arange(L, device=primary_ids.device)

        x = (
            self.primary_emb(primary_ids)
            + self.suit_emb(suit_ids)
            + self.pos_emb(positions)
        )

        for layer in self.layers:
            x = layer(x, pad_mask)

        cls = x[:, 0]
        return self.out_head(self.out_norm(cls))
