"""Task (d) demo: Streamlit app that serves top-K recommendations for a MovieLens user.

Run:  ~/.virtualenvs/movieLens/bin/streamlit run movieLens/code/demo/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO.parent))

from code.config import CFG
from code.data import load_and_split, load_movies
from code.fairness import identify_tail_items, rerank_for_coverage
from code.model import MatrixFactorization


@st.cache_resource
def load_assets():
    ds = load_and_split()
    movies = load_movies()
    # Map internal item_id -> raw id -> title
    raw_by_item = {int(v): k for k, v in ds.item_id_map.items()}
    movies_by_raw = movies.set_index("item_raw")

    ckpt_path = CFG.REPORTS_DIR / "mf_model.pt"
    if not ckpt_path.exists():
        st.error(
            f"Model checkpoint missing at {ckpt_path}. Run `python -m code.main` first."
        )
        st.stop()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MatrixFactorization(
        ckpt["n_users"], ckpt["n_items"], ckpt["n_factors"], ckpt["global_mean"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tail = identify_tail_items(ds.train_df, CFG.TAIL_PERCENTILE)
    return ds, movies_by_raw, raw_by_item, model, tail


def titles_for(item_ids, raw_by_item, movies_by_raw):
    rows = []
    for it in item_ids:
        raw = raw_by_item.get(int(it))
        if raw is None or raw not in movies_by_raw.index:
            rows.append({"title": f"<unknown id={it}>", "genres": []})
        else:
            r = movies_by_raw.loc[raw]
            rows.append({"title": r["title"], "genres": r["genres"]})
    return rows


def score_user(model: MatrixFactorization, user_idx: int, n_items: int) -> np.ndarray:
    with torch.no_grad():
        ids = torch.tensor([user_idx], dtype=torch.long)
        return model.score_users(ids).squeeze(0).cpu().numpy()


def main():
    st.set_page_config(page_title="MovieLens MF recommender", layout="wide")
    st.title("MovieLens Matrix Factorization Recommender")
    st.caption(
        "CS550 project, Task (d) demo. Already-rated items are excluded from the top-K, "
        "per the project spec."
    )

    ds, movies_by_raw, raw_by_item, model, tail = load_assets()

    # Sidebar controls
    st.sidebar.header("Controls")
    user_options = sorted(int(u) for u in ds.user_id_map.values)
    user_idx = st.sidebar.selectbox("User (internal id)", user_options, index=0)
    k = st.sidebar.slider("k (top-K size)", 5, 30, 10)
    use_fair = st.sidebar.checkbox(
        f"Apply long-tail rerank (target {CFG.TARGET_COVERAGE:.0%})",
        value=False,
    )

    # Training history panel
    train_row = ds.train_df[ds.train_df["user_id"] == user_idx]
    st.sidebar.metric("Training ratings", len(train_row))
    if len(train_row):
        st.sidebar.metric("Mean rating", f"{train_row['rating'].mean():.2f}")

    # Score and rank
    scores = score_user(model, user_idx, ds.n_items)
    seen = set(ds.train_matrix[user_idx].indices.tolist())
    masked = scores.copy()
    if seen:
        masked[list(seen)] = -np.inf
    top_idx = np.argpartition(-masked, k)[:k]
    top_idx = top_idx[np.argsort(-masked[top_idx])]
    recs_dict = {user_idx: top_idx.astype(np.int32)}

    if use_fair:
        all_scores = np.zeros((ds.n_users, ds.n_items), dtype=np.float32)
        all_scores[user_idx] = scores
        recs_dict = rerank_for_coverage(
            recs_dict, all_scores, ds.train_matrix, tail,
            target=CFG.TARGET_COVERAGE, k=k,
        )

    final_top = recs_dict[user_idx]

    # Build display table
    title_info = titles_for(final_top, raw_by_item, movies_by_raw)
    rows = []
    for rank, (it, info) in enumerate(zip(final_top, title_info), start=1):
        rows.append({
            "rank": rank,
            "title": info["title"],
            "genres": ", ".join(info["genres"]),
            "pred_rating": round(float(np.clip(scores[int(it)], 1.0, 5.0)), 2),
            "tail": "yes" if int(it) in tail else "",
        })
    df = pd.DataFrame(rows)

    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"Top-{k} for user {user_idx}")
        st.dataframe(df, hide_index=True, use_container_width=True)

    with right:
        st.subheader("This user's highest-rated training items")
        top_seen = (
            train_row.sort_values("rating", ascending=False).head(10)
            .assign(title=lambda x: titles_for(x["item_id"], raw_by_item, movies_by_raw))
        )
        if len(top_seen):
            hist_rows = [
                {"rating": float(r.rating), "title": t["title"]}
                for r, t in zip(top_seen.itertuples(), top_seen["title"])
            ]
            st.dataframe(pd.DataFrame(hist_rows), hide_index=True, use_container_width=True)
        else:
            st.write("No training ratings (cold user).")


if __name__ == "__main__":
    main()
