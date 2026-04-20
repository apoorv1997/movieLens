"""Biased Matrix Factorization: r_ui = mu + b_u + b_i + <p_u, q_i>."""
from __future__ import annotations

import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int, global_mean: float) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.register_buffer("global_mean", torch.tensor(global_mean, dtype=torch.float32))

        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Raw prediction, no clipping — used during training."""
        pu = self.user_emb(users)
        qi = self.item_emb(items)
        dot = (pu * qi).sum(dim=-1)
        return self.global_mean + self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1) + dot

    @torch.no_grad()
    def predict(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Inference: clamp to the observed rating range."""
        return torch.clamp(self.forward(users, items), 1.0, 5.0)

    @torch.no_grad()
    def score_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Return a [len(user_ids), n_items] score matrix (no clamping — we only care about ranking)."""
        pu = self.user_emb(user_ids)                          # [B, F]
        bu = self.user_bias(user_ids).squeeze(-1)             # [B]
        Q = self.item_emb.weight                              # [n_items, F]
        bi = self.item_bias.weight.squeeze(-1)                # [n_items]
        # [B, n_items] = pu @ Q.T + bu[:, None] + bi[None, :] + mu
        return pu @ Q.T + bu.unsqueeze(-1) + bi.unsqueeze(0) + self.global_mean
