"""Matrix factorization trainer: batched Adam + val-loss early stopping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import CFG
from .model import MatrixFactorization


@dataclass
class TrainHistory:
    train_losses: list
    val_losses: list
    best_epoch: int
    best_val_loss: float


def _df_to_tensors(df: pd.DataFrame, device: str) -> tuple:
    u = torch.from_numpy(df["user_id"].to_numpy(dtype=np.int64)).to(device)
    i = torch.from_numpy(df["item_id"].to_numpy(dtype=np.int64)).to(device)
    r = torch.from_numpy(df["rating"].to_numpy(dtype=np.float32)).to(device)
    return u, i, r


def train_mf(
    n_users: int,
    n_items: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_factors: int = CFG.N_FACTORS,
    epochs: int = CFG.EPOCHS,
    patience: int = CFG.PATIENCE,
    lr: float = CFG.LR,
    batch: int = CFG.BATCH,
    weight_decay: float = CFG.WEIGHT_DECAY,
    verbose: bool = True,
) -> tuple[MatrixFactorization, TrainHistory]:
    device = CFG.DEVICE
    global_mean = float(train_df["rating"].mean())
    model = MatrixFactorization(n_users, n_items, n_factors, global_mean).to(device)

    u_tr, i_tr, r_tr = _df_to_tensors(train_df, device)
    u_val, i_val, r_val = _df_to_tensors(val_df, device)

    ds = TensorDataset(u_tr, i_tr, r_tr)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_state: Optional[dict] = None
    best_val = float("inf")
    best_epoch = -1
    patience_left = patience
    train_hist, val_hist = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for ub, ib, rb in loader:
            pred = model(ub, ib)
            loss = loss_fn(pred, rb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * ub.numel()
            n += ub.numel()
        train_loss = total / n

        model.eval()
        with torch.no_grad():
            val_pred = model(u_val, i_val)
            val_loss = float(nn.functional.mse_loss(val_pred, r_val).item())

        train_hist.append(train_loss)
        val_hist.append(val_loss)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1

        if verbose:
            print(
                f"  epoch {epoch:2d}/{epochs} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f}"
                + ("  *" if improved else "")
            )
        if patience_left <= 0:
            if verbose:
                print(f"  early stop at epoch {epoch} (best={best_epoch})")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, TrainHistory(train_hist, val_hist, best_epoch, best_val)
