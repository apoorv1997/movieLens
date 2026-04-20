# MovieLens Matrix-Factorization Recommender (CS550 project)

A PyTorch matrix-factorization recommender for MovieLens 1M, with three baselines, a long-tail fairness intervention, a Pareto trade-off sweep, a user-group activity-level analysis, and a Streamlit demo.

The module layout maps one-to-one to the project description's Required Tasks (a) - (d).

## Quick start

```bash
# One-time setup (CUDA 12.x build of torch + rest of deps)
python -m venv ~/.virtualenvs/movieLens
~/.virtualenvs/movieLens/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
~/.virtualenvs/movieLens/bin/pip install -r requirements.txt

# Reproduce every number and figure
cd movieLens
~/.virtualenvs/movieLens/bin/python -m code.main

# Launch the demo
~/.virtualenvs/movieLens/bin/streamlit run code/demo/app.py
```

`python -m code.main` writes `reports/results.json`, `reports/pareto.png`, `reports/user_groups.png`, and `reports/mf_model.pt`. All outputs go to `../reports/` (outside the repo).

## File map

| File | Task | Purpose |
|---|---|---|
| `code/config.py` | - | Hyperparameters, paths, seeding helper |
| `code/data.py` | (a) | Download + load ML-1M, per-user 80/10/10 train/val/test split |
| `code/baselines.py` | (b/c) | GlobalMean, MostPopular, UserCF |
| `code/model.py` | (b) | Biased MF as an `nn.Module` |
| `code/train.py` | (b) | Batched Adam with val-loss early stopping |
| `code/evaluation.py` | (b/c) | MAE/RMSE, vectorized top-K with exclude-rated masking, P/R/F/NDCG@10 |
| `code/fairness.py` | + | Long-tail detection, post-hoc rerank, Pareto sweep, user-group analysis |
| `code/main.py` | - | End-to-end pipeline orchestrator |
| `code/demo/app.py` | (d) | Streamlit demo |

## Bugs fixed vs. the original implementation

1. `generate_recommendations` now excludes items the user already rated in training, via a `-inf` mask before `torch.topk`. The spec explicitly requires this ("you should avoid recommending an item that the user has already rated in the training dataset"); the original allowed the user's training items back into the top-K, which destroyed ranking metrics.
2. The MostPopular baseline now applies the same exclude-rated filter.
3. The MF forward pass no longer clamps during training, which previously killed gradients at the 1/5 boundary. Clamping moved to inference (`predict`).
4. Manual SGD has been replaced by `torch.optim.Adam` + autograd, which eliminates the item-update-uses-already-updated-user-embedding bug in the original.
5. Training runs up to 50 epochs with val-loss early stopping (patience 5), vs. 10 fixed epochs previously.

## Reproducibility

- `SEED=42` is set across numpy, torch, and Python's `random`, including `torch.cuda.manual_seed_all`.
- Per-user random split uses a seeded `np.random.default_rng`.
- CUDA non-determinism is not fully suppressed (no `deterministic=True`), so run-to-run NDCG may differ at the 4th decimal.

## Expected results

Running on a single RTX 4060 (batched Adam, N=50 factors, 50 max epochs, early stop):

- MF MAE ~ 0.68, RMSE ~ 0.87, beating UserCF (0.74 / 0.95) and GlobalMean (0.93 / 1.12).
- MF ranking NDCG@10 ~ 0.12; MostPopular ~ 0.15; UserCF ~ 0.00 (UserCF's cosine-weighted predictions over items rated by few neighbors produce degenerate scores at the top).
- Fair-MF reaches the requested long-tail coverage (e.g., 0.30 at target 0.25) with a ~15% NDCG drop.
- The Pareto curve has six points sweeping coverage from 0.00 to 0.40.
- User-group NDCG: heavy users (>100 training ratings) ~ 0.20, warm users ~ 0.07, cold users ~ 0.05. MF beats MostPopular for heavy users; MostPopular wins the overall average because it dominates on cold users.
