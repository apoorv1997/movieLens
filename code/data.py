"""Task (a): dataset download, load, and per-user train/val/test split."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import CFG

_ML1M_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"


@dataclass
class Dataset:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_matrix: sp.csr_matrix  # [n_users, n_items] rating values
    n_users: int
    n_items: int
    user_id_map: pd.Series  # raw -> idx
    item_id_map: pd.Series  # raw -> idx


def download_ml1m(data_dir: Path = CFG.DATA_DIR) -> Path:
    ml_dir = data_dir / "ml-1m"
    if (ml_dir / "ratings.dat").exists():
        return ml_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-1m.zip"
    urlretrieve(_ML1M_URL, zip_path)
    with ZipFile(zip_path) as z:
        z.extractall(data_dir)
    zip_path.unlink()
    return ml_dir


def load_ratings(ml_dir: Path = CFG.ML1M_DIR) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(
        ml_dir / "ratings.dat",
        sep="::",
        header=None,
        names=["user_raw", "item_raw", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    df["user_id"] = df["user_raw"].astype("category").cat.codes.astype(np.int32)
    df["item_id"] = df["item_raw"].astype("category").cat.codes.astype(np.int32)
    df["rating"] = df["rating"].astype(np.float32)

    user_id_map = pd.Series(
        df["user_id"].values, index=df["user_raw"].values
    ).drop_duplicates()
    item_id_map = pd.Series(
        df["item_id"].values, index=df["item_raw"].values
    ).drop_duplicates()

    return df[["user_id", "item_id", "rating"]], user_id_map, item_id_map


def load_movies(ml_dir: Path = CFG.ML1M_DIR) -> pd.DataFrame:
    """For the demo: raw movie IDs -> (title, genres list)."""
    df = pd.read_csv(
        ml_dir / "movies.dat",
        sep="::",
        header=None,
        names=["item_raw", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    df["genres"] = df["genres"].str.split("|")
    return df


def split(ratings: pd.DataFrame) -> Dataset:
    rng = np.random.default_rng(CFG.SEED)
    n_users = int(ratings["user_id"].max()) + 1
    n_items = int(ratings["item_id"].max()) + 1

    train_parts, val_parts, test_parts = [], [], []
    for _, grp in ratings.groupby("user_id", sort=False):
        if len(grp) < CFG.MIN_RATINGS_PER_USER:
            train_parts.append(grp)
            continue
        idx = rng.permutation(len(grp))
        shuffled = grp.iloc[idx]
        n = len(shuffled)
        n_test = int(round((1 - CFG.TRAIN_FRAC) * n))
        n_val = int(round(CFG.TRAIN_FRAC * n * CFG.VAL_FRAC_OF_TRAIN))
        test_parts.append(shuffled.iloc[:n_test])
        val_parts.append(shuffled.iloc[n_test : n_test + n_val])
        train_parts.append(shuffled.iloc[n_test + n_val :])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    train_matrix = sp.csr_matrix(
        (
            train_df["rating"].values,
            (train_df["user_id"].values, train_df["item_id"].values),
        ),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    # Placeholder id maps built from the raw ratings - we'll overwrite from load_ratings caller
    user_id_map = pd.Series(range(n_users), index=range(n_users))
    item_id_map = pd.Series(range(n_items), index=range(n_items))

    return Dataset(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_matrix=train_matrix,
        n_users=n_users,
        n_items=n_items,
        user_id_map=user_id_map,
        item_id_map=item_id_map,
    )


def load_and_split() -> Dataset:
    download_ml1m()
    ratings, user_map, item_map = load_ratings()
    ds = split(ratings)
    ds.user_id_map = user_map
    ds.item_id_map = item_map
    return ds
