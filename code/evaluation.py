"""Task (b) rating metrics and Task (c) ranking metrics.

The vectorized `generate_topk_mf` is the fix for the original project's disqualifying bug:
items the user already rated in training MUST be excluded from top-K, per the project spec.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .config import CFG
from .model import MatrixFactorization


def eval_rating(
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    test_df: pd.DataFrame,
) -> Dict[str, float]:
    """MAE and RMSE on the held-out test ratings."""
    u = test_df["user_id"].to_numpy()
    i = test_df["item_id"].to_numpy()
    r = test_df["rating"].to_numpy(dtype=np.float32)
    pred = np.asarray(predict_fn(u, i), dtype=np.float32)
    err = pred - r
    return {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean()))}


def mf_predict_fn(model: MatrixFactorization) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    device = CFG.DEVICE
    model.eval()

    def _fn(users: np.ndarray, items: np.ndarray) -> np.ndarray:
        u = torch.from_numpy(np.asarray(users, dtype=np.int64)).to(device)
        i = torch.from_numpy(np.asarray(items, dtype=np.int64)).to(device)
        with torch.no_grad():
            pred = model.predict(u, i).cpu().numpy()
        return pred

    return _fn


def generate_topk_mf(
    model: MatrixFactorization,
    train_matrix: sp.csr_matrix,
    k: int = CFG.K,
    batch: int = 512,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Return per-user top-K item indices (excluding train items) and the full score matrix.

    The score matrix is returned on CPU as float32 so fairness rerank can reuse it
    without recomputation.
    """
    device = CFG.DEVICE
    model.eval()
    n_users, n_items = train_matrix.shape
    all_scores = np.empty((n_users, n_items), dtype=np.float32)
    recs: Dict[int, np.ndarray] = {}

    train_csr = train_matrix.tocsr()

    for start in range(0, n_users, batch):
        end = min(start + batch, n_users)
        ids = torch.arange(start, end, device=device, dtype=torch.long)
        with torch.no_grad():
            scores = model.score_users(ids)  # [B, n_items]
        scores_cpu = scores.cpu().numpy()
        all_scores[start:end] = scores_cpu

        # Mask already-rated items with -inf for ranking only.
        masked = scores.clone()
        for row, u in enumerate(range(start, end)):
            seen = train_csr[u].indices
            if seen.size:
                masked[row, torch.from_numpy(seen.astype(np.int64)).to(device)] = float("-inf")

        _, topk_idx = torch.topk(masked, k=k, dim=1)
        topk_idx_cpu = topk_idx.cpu().numpy()
        for row, u in enumerate(range(start, end)):
            recs[u] = topk_idx_cpu[row].astype(np.int32)

    return recs, all_scores


def _dcg(rel: np.ndarray) -> float:
    if rel.size == 0:
        return 0.0
    gains = (2.0 ** rel - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float((gains * discounts).sum())


def eval_ranking(
    recs: Dict[int, np.ndarray],
    test_df: pd.DataFrame,
    k: int = CFG.K,
    relevance_threshold: float = 4.0,
) -> Dict[str, float]:
    """Macro-averaged Precision/Recall/F/NDCG@k over users with ≥1 relevant test item.

    An item is "relevant" if the test rating is >= `relevance_threshold` (4-5 stars).
    """
    test_by_user: Dict[int, pd.DataFrame] = {u: g for u, g in test_df.groupby("user_id", sort=False)}

    precisions, recalls, ndcgs = [], [], []
    for u, items in recs.items():
        if u not in test_by_user:
            continue
        g = test_by_user[u]
        relevant = set(g.loc[g["rating"] >= relevance_threshold, "item_id"].astype(int).tolist())
        if not relevant:
            continue
        topk = items[:k]
        hits = [1 if int(it) in relevant else 0 for it in topk]
        n_hits = sum(hits)

        precisions.append(n_hits / k)
        recalls.append(n_hits / len(relevant))

        ideal_hits = min(len(relevant), k)
        rel_vec = np.asarray(hits, dtype=np.float32)
        ideal_vec = np.zeros(k, dtype=np.float32)
        ideal_vec[:ideal_hits] = 1.0
        idcg = _dcg(ideal_vec)
        ndcgs.append(_dcg(rel_vec) / idcg if idcg > 0 else 0.0)

    p = float(np.mean(precisions)) if precisions else 0.0
    r = float(np.mean(recalls)) if recalls else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    n = float(np.mean(ndcgs)) if ndcgs else 0.0
    return {f"P@{k}": p, f"R@{k}": r, f"F@{k}": f, f"NDCG@{k}": n, "n_users_eval": len(precisions)}
