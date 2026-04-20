"""Hyperparameters, paths, and seed helpers."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _REPO_ROOT.parent


@dataclass(frozen=True)
class Config:
    SEED: int = 42

    N_FACTORS: int = 50
    LR: float = 0.01
    BATCH: int = 4096
    EPOCHS: int = 50
    PATIENCE: int = 5
    WEIGHT_DECAY: float = 2e-5

    TRAIN_FRAC: float = 0.8
    VAL_FRAC_OF_TRAIN: float = 0.1
    MIN_RATINGS_PER_USER: int = 5

    K: int = 10
    TAIL_PERCENTILE: int = 20
    TARGET_COVERAGE: float = 0.25

    PARETO_TARGETS: tuple = (0.0, 0.10, 0.20, 0.25, 0.30, 0.40)
    ABLATION_FACTORS: tuple = (20, 50, 100)

    DATA_DIR: Path = _PROJECT_ROOT / "data"
    ML1M_DIR: Path = _PROJECT_ROOT / "data" / "ml-1m"
    REPORTS_DIR: Path = _PROJECT_ROOT / "reports"

    @property
    def DEVICE(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


CFG = Config()
