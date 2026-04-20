"""Long-tail fairness: tail detection, post-hoc rerank for coverage, Pareto sweep, user-group analysis."""
from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import CFG
from .evaluation import eval_ranking


def identify_tail_items(train_df: pd.DataFrame, percentile: int = CFG.TAIL_PERCENTILE) -> Set[int]:
    """Items in the bottom `percentile`% by training rating count are "tail"."""
    counts = train_df.groupby("item_id").size()
    threshold = float(np.percentile(counts.values, percentile))
    return set(counts[counts <= threshold].index.astype(int).tolist())


def compute_tail_coverage(recs: Dict[int, np.ndarray], tail_items: Set[int], k: int = CFG.K) -> float:
    """Fraction of top-K slots occupied by tail items, averaged across users."""
    if not recs:
        return 0.0
    per_user = []
    for items in recs.values():
        topk = items[:k]
        per_user.append(sum(1 for it in topk if int(it) in tail_items) / k)
    return float(np.mean(per_user))


def rerank_for_coverage(
    recs: Dict[int, np.ndarray],
    all_scores: np.ndarray,
    train_matrix: sp.csr_matrix,
    tail_items: Set[int],
    target: float,
    k: int = CFG.K,
) -> Dict[int, np.ndarray]:
    """For each user, swap lowest-scored non-tail items for highest-scored unseen tail items
    until tail share in top-K >= target.

    `all_scores` is [n_users, n_items] from `generate_topk_mf`. Training items are NOT masked here
    so we mask them out in this routine.
    """
    tail_arr = np.fromiter(tail_items, dtype=np.int32)
    if tail_arr.size == 0 or target <= 0.0:
        return {u: v.copy() for u, v in recs.items()}

    needed = int(np.ceil(target * k))
    out: Dict[int, np.ndarray] = {}

    for u, items in recs.items():
        topk = list(int(x) for x in items[:k])
        tail_in_top = [idx for idx, it in enumerate(topk) if it in tail_items]
        if len(tail_in_top) >= needed:
            out[u] = np.asarray(topk, dtype=np.int32)
            continue

        # Candidate tail items: not already in topk, not rated in train. Rank by score desc.
        seen = set(train_matrix[u].indices.tolist())
        seen |= set(topk)
        scores_u = all_scores[u]
        cand_mask = np.array([it not in seen for it in tail_arr])
        cand_tail = tail_arr[cand_mask]
        if cand_tail.size == 0:
            out[u] = np.asarray(topk, dtype=np.int32)
            continue
        cand_scores = scores_u[cand_tail]
        order = np.argsort(-cand_scores)
        cand_tail = cand_tail[order]

        # Positions of non-tail items in topk, sorted by score ascending (swap worst first).
        non_tail_positions = [idx for idx, it in enumerate(topk) if it not in tail_items]
        non_tail_positions.sort(key=lambda idx: scores_u[topk[idx]])

        swaps = needed - len(tail_in_top)
        for cand, pos in zip(cand_tail[:swaps], non_tail_positions[:swaps]):
            topk[pos] = int(cand)

        out[u] = np.asarray(topk, dtype=np.int32)

    return out


def run_pareto_sweep(
    recs: Dict[int, np.ndarray],
    all_scores: np.ndarray,
    train_matrix: sp.csr_matrix,
    tail_items: Set[int],
    test_df: pd.DataFrame,
    targets: tuple = CFG.PARETO_TARGETS,
    k: int = CFG.K,
) -> List[dict]:
    """Return list of {target, coverage, ndcg, precision, recall, f} for each target."""
    results = []
    for t in targets:
        reranked = rerank_for_coverage(recs, all_scores, train_matrix, tail_items, t, k)
        cov = compute_tail_coverage(reranked, tail_items, k)
        metrics = eval_ranking(reranked, test_df, k)
        results.append({
            "target": float(t),
            "coverage": cov,
            f"NDCG@{k}": metrics[f"NDCG@{k}"],
            f"P@{k}": metrics[f"P@{k}"],
            f"R@{k}": metrics[f"R@{k}"],
            f"F@{k}": metrics[f"F@{k}"],
        })
    return results


def user_group_analysis(
    recs: Dict[int, np.ndarray],
    all_scores: np.ndarray,
    train_matrix: sp.csr_matrix,
    tail_items: Set[int],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: tuple = CFG.PARETO_TARGETS,
    k: int = CFG.K,
) -> Dict[str, List[dict]]:
    """Per-group (cold/warm/heavy) NDCG and coverage across fairness targets."""
    counts = train_df.groupby("user_id").size()
    all_users = set(recs.keys())
    cold = {u for u in all_users if counts.get(u, 0) < 20}
    warm = {u for u in all_users if 20 <= counts.get(u, 0) <= 100}
    heavy = {u for u in all_users if counts.get(u, 0) > 100}

    groups = {"cold": cold, "warm": warm, "heavy": heavy}
    out: Dict[str, List[dict]] = {g: [] for g in groups}

    for t in targets:
        reranked = rerank_for_coverage(recs, all_scores, train_matrix, tail_items, t, k)
        for gname, gusers in groups.items():
            g_recs = {u: reranked[u] for u in gusers if u in reranked}
            g_test = test_df[test_df["user_id"].isin(gusers)]
            cov = compute_tail_coverage(g_recs, tail_items, k)
            metrics = eval_ranking(g_recs, g_test, k)
            out[gname].append({
                "target": float(t),
                "coverage": cov,
                f"NDCG@{k}": metrics[f"NDCG@{k}"],
                "n_users": len(gusers),
            })

    return out


def plot_pareto(pareto: List[dict], path: str, k: int = CFG.K) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cov = [p["coverage"] for p in pareto]
    ndcg = [p[f"NDCG@{k}"] for p in pareto]
    targets = [p["target"] for p in pareto]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cov, ndcg, marker="o", linestyle="-", color="#1f77b4")
    for c, n, t in zip(cov, ndcg, targets):
        ax.annotate(f"t={t:.2f}", (c, n), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Long-tail coverage in top-K")
    ax.set_ylabel(f"NDCG@{k}")
    ax.set_title("Fairness - accuracy trade-off (Pareto curve)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_user_groups(group_results: Dict[str, List[dict]], path: str, k: int = CFG.K) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"cold": "#d62728", "warm": "#2ca02c", "heavy": "#1f77b4"}
    for gname, rows in group_results.items():
        targets = [r["target"] for r in rows]
        ndcg = [r[f"NDCG@{k}"] for r in rows]
        n = rows[0]["n_users"] if rows else 0
        ax.plot(targets, ndcg, marker="o", label=f"{gname} (n={n})", color=colors.get(gname))
    ax.set_xlabel("Tail-coverage target")
    ax.set_ylabel(f"NDCG@{k}")
    ax.set_title("Accuracy loss by user activity group")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
