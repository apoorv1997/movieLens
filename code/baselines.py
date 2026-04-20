"""Non-MF baselines: GlobalMean, MostPopular, UserCF.

All top-K generators enforce the "exclude already-rated items" rule from the
project description.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


class GlobalMean:
    """Predict the training mean for every (u, i). Useful as a rating-prediction floor."""

    def fit(self, train_df: pd.DataFrame, train_matrix: sp.csr_matrix) -> "GlobalMean":
        self.mu_ = float(train_df["rating"].mean())
        self.n_items_ = train_matrix.shape[1]
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return np.full(len(users), self.mu_, dtype=np.float32)

    def recommend_all(self, train_matrix: sp.csr_matrix, k: int) -> Dict[int, np.ndarray]:
        # All items tie at mu_; pick the first k unseen items per user (arbitrary but deterministic).
        n_users, n_items = train_matrix.shape
        recs: Dict[int, np.ndarray] = {}
        for u in range(n_users):
            seen = set(train_matrix[u].indices.tolist())
            picked = [i for i in range(n_items) if i not in seen][:k]
            recs[u] = np.asarray(picked, dtype=np.int32)
        return recs


class MostPopular:
    """Recommend items with the highest training-interaction count, excluding rated items."""

    def fit(self, train_df: pd.DataFrame, train_matrix: sp.csr_matrix) -> "MostPopular":
        counts = np.asarray(train_matrix.getnnz(axis=0)).ravel()
        self.order_ = np.argsort(-counts, kind="stable")  # descending
        self.n_items_ = train_matrix.shape[1]
        return self

    def recommend_all(self, train_matrix: sp.csr_matrix, k: int) -> Dict[int, np.ndarray]:
        n_users = train_matrix.shape[0]
        recs: Dict[int, np.ndarray] = {}
        for u in range(n_users):
            seen = set(train_matrix[u].indices.tolist())
            out = []
            for i in self.order_:
                if i in seen:
                    continue
                out.append(int(i))
                if len(out) == k:
                    break
            recs[u] = np.asarray(out, dtype=np.int32)
        return recs


class UserCF:
    """User-based collaborative filtering with cosine similarity on mean-centered ratings.

    Predicts r(u, i) = mu_u + sum_v sim(u,v) * (r(v,i) - mu_v) / sum_v |sim(u,v)|
    over v in top-N neighbors of u who rated i.
    """

    def __init__(self, n_neighbors: int = 50) -> None:
        self.n_neighbors = n_neighbors

    def fit(self, train_df: pd.DataFrame, train_matrix: sp.csr_matrix) -> "UserCF":
        n_users, n_items = train_matrix.shape

        # Per-user mean (over observed entries only). Users with zero ratings
        # would divide by zero; default to global mean for them.
        sums = np.asarray(train_matrix.sum(axis=1)).ravel()
        counts = np.asarray(train_matrix.getnnz(axis=1)).ravel()
        global_mean = float(train_df["rating"].mean())
        means = np.where(counts > 0, sums / np.maximum(counts, 1), global_mean)

        # Mean-center only observed entries (keep sparsity).
        centered = train_matrix.copy().astype(np.float32)
        centered.data = centered.data - np.repeat(means.astype(np.float32), counts)

        # Dense user-user cosine similarity. 6040 x 6040 float32 ~ 140MB, fine.
        sim = cosine_similarity(centered, dense_output=True).astype(np.float32)
        np.fill_diagonal(sim, 0.0)

        # Keep only top-N neighbors per user; zero the rest.
        if self.n_neighbors < n_users:
            # argpartition puts the top n_neighbors indices in the last positions (unordered).
            part = np.argpartition(-sim, self.n_neighbors, axis=1)[:, : self.n_neighbors]
            mask = np.zeros_like(sim, dtype=bool)
            rows = np.arange(n_users)[:, None]
            mask[rows, part] = True
            sim = np.where(mask, sim, 0.0)

        # For score_all_items: precompute a {0,1} mask so denom is a simple matmul.
        rated_mask = train_matrix.copy().astype(np.float32)
        rated_mask.data = np.ones_like(rated_mask.data)

        self.sim_ = sim
        self.means_ = means.astype(np.float32)
        self.centered_ = centered.tocsr()
        self.rated_mask_ = rated_mask.tocsr()
        self.train_matrix_ = train_matrix
        self.global_mean_ = global_mean
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        preds = np.empty(len(users), dtype=np.float32)
        # Vectorize per unique user for speed.
        for idx, (u, i) in enumerate(zip(users, items)):
            preds[idx] = self._predict_one(int(u), int(i))
        return preds

    def _predict_one(self, u: int, i: int) -> float:
        col = self.centered_[:, i]  # sparse column of deviations
        raters = col.nonzero()[0]
        if raters.size == 0:
            return self.means_[u]
        sims = self.sim_[u, raters]
        denom = np.abs(sims).sum()
        if denom == 0.0:
            return self.means_[u]
        devs = np.asarray(col[raters].todense()).ravel()
        pred = self.means_[u] + float((sims * devs).sum() / denom)
        return float(np.clip(pred, 1.0, 5.0))

    def score_all_items(self, u: int) -> np.ndarray:
        """Return predicted ratings for every item for user u (for top-K ranking)."""
        sim_row = self.sim_[u].astype(np.float32)  # (n_users,)
        num = self.centered_.T.dot(sim_row)  # (n_items,)
        denom = self.rated_mask_.T.dot(np.abs(sim_row))  # (n_items,)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = self.means_[u] + np.where(denom > 0, num / np.maximum(denom, 1e-8), 0.0)
        return scores.astype(np.float32)

    def recommend_all(self, train_matrix: sp.csr_matrix, k: int) -> Dict[int, np.ndarray]:
        n_users, n_items = train_matrix.shape
        recs: Dict[int, np.ndarray] = {}
        for u in range(n_users):
            scores = self.score_all_items(u)
            # Mask already-rated with -inf.
            seen = train_matrix[u].indices
            scores[seen] = -np.inf
            top = np.argpartition(-scores, k)[:k]
            top = top[np.argsort(-scores[top])]
            recs[u] = top.astype(np.int32)
        return recs
