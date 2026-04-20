"""End-to-end pipeline: Tasks (a) -> (b) -> (c) + fairness/Pareto/user-group + ablation.

Running `python -m code.main` from the `movieLens/` directory reproduces every
number and figure we cite in the report.
"""
from __future__ import annotations

import json
import time

import torch

from .baselines import GlobalMean, MostPopular, UserCF
from .config import CFG, set_all_seeds
from .data import load_and_split
from .evaluation import eval_ranking, eval_rating, generate_topk_mf, mf_predict_fn
from .fairness import (
    compute_tail_coverage,
    identify_tail_items,
    plot_pareto,
    plot_user_groups,
    rerank_for_coverage,
    run_pareto_sweep,
    user_group_analysis,
)
from .train import train_mf


def _time(label: str):
    class _T:
        def __enter__(self_inner):
            self_inner.t0 = time.time()
            return self_inner

        def __exit__(self_inner, *exc):
            dt = time.time() - self_inner.t0
            print(f"[{label}] {dt:.2f}s")

    return _T()


def _baseline_predict_fn(baseline):
    def _fn(users, items):
        return baseline.predict(users, items)
    return _fn


def main() -> dict:
    set_all_seeds(CFG.SEED)
    CFG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={CFG.DEVICE}")

    with _time("load+split"):
        ds = load_and_split()
    print(f"  n_users={ds.n_users} n_items={ds.n_items}")
    print(f"  train={len(ds.train_df)} val={len(ds.val_df)} test={len(ds.test_df)}")

    results: dict = {
        "config": {
            "n_factors": CFG.N_FACTORS, "lr": CFG.LR, "batch": CFG.BATCH,
            "epochs": CFG.EPOCHS, "patience": CFG.PATIENCE,
            "weight_decay": CFG.WEIGHT_DECAY, "k": CFG.K,
            "tail_percentile": CFG.TAIL_PERCENTILE,
            "seed": CFG.SEED, "device": CFG.DEVICE,
        },
        "counts": {
            "n_users": ds.n_users, "n_items": ds.n_items,
            "train": len(ds.train_df), "val": len(ds.val_df), "test": len(ds.test_df),
        },
    }

    # ---- Baselines ---------------------------------------------------------
    print("\n== Baselines ==")
    with _time("GlobalMean.fit"):
        gm = GlobalMean().fit(ds.train_df, ds.train_matrix)
    with _time("MostPopular.fit"):
        mp = MostPopular().fit(ds.train_df, ds.train_matrix)
    with _time("UserCF.fit"):
        ucf = UserCF(n_neighbors=50).fit(ds.train_df, ds.train_matrix)

    # ---- MF training -------------------------------------------------------
    print("\n== MF training ==")
    with _time("MF.train"):
        mf, history = train_mf(
            ds.n_users, ds.n_items, ds.train_df, ds.val_df, verbose=True
        )
    results["mf_training"] = {
        "best_epoch": history.best_epoch,
        "best_val_mse": history.best_val_loss,
        "train_losses": history.train_losses,
        "val_losses": history.val_losses,
    }

    # ---- Rating prediction -------------------------------------------------
    print("\n== Rating prediction (Task b) ==")
    rating_results = {}
    rating_results["GlobalMean"] = eval_rating(_baseline_predict_fn(gm), ds.test_df)
    print(f"  GlobalMean {rating_results['GlobalMean']}")
    rating_results["UserCF"] = eval_rating(_baseline_predict_fn(ucf), ds.test_df)
    print(f"  UserCF     {rating_results['UserCF']}")
    rating_results["MF"] = eval_rating(mf_predict_fn(mf), ds.test_df)
    print(f"  MF         {rating_results['MF']}")
    results["rating_prediction"] = rating_results

    # ---- Top-K generation --------------------------------------------------
    print("\n== Top-K generation (Task c) ==")
    with _time("MF.topK"):
        mf_recs, mf_scores = generate_topk_mf(mf, ds.train_matrix, k=CFG.K)
    with _time("MostPopular.topK"):
        mp_recs = mp.recommend_all(ds.train_matrix, k=CFG.K)
    with _time("UserCF.topK"):
        ucf_recs = ucf.recommend_all(ds.train_matrix, k=CFG.K)

    ranking_results = {}
    ranking_results["MostPopular"] = eval_ranking(mp_recs, ds.test_df, k=CFG.K)
    print(f"  MostPopular {ranking_results['MostPopular']}")
    ranking_results["UserCF"] = eval_ranking(ucf_recs, ds.test_df, k=CFG.K)
    print(f"  UserCF      {ranking_results['UserCF']}")
    ranking_results["MF"] = eval_ranking(mf_recs, ds.test_df, k=CFG.K)
    print(f"  MF          {ranking_results['MF']}")
    results["ranking"] = ranking_results

    # ---- Fairness ----------------------------------------------------------
    print("\n== Fairness: long-tail coverage ==")
    tail = identify_tail_items(ds.train_df, CFG.TAIL_PERCENTILE)
    print(f"  tail_items={len(tail)} ({100*len(tail)/ds.n_items:.1f}% of catalogue)")

    mf_cov = compute_tail_coverage(mf_recs, tail, CFG.K)
    mp_cov = compute_tail_coverage(mp_recs, tail, CFG.K)
    ucf_cov = compute_tail_coverage(ucf_recs, tail, CFG.K)
    print(f"  pre-rerank coverage: MF={mf_cov:.3f} UserCF={ucf_cov:.3f} MostPopular={mp_cov:.3f}")

    fair_mf_recs = rerank_for_coverage(
        mf_recs, mf_scores, ds.train_matrix, tail, CFG.TARGET_COVERAGE, CFG.K
    )
    fair_cov = compute_tail_coverage(fair_mf_recs, tail, CFG.K)
    fair_metrics = eval_ranking(fair_mf_recs, ds.test_df, k=CFG.K)
    print(f"  Fair-MF (target={CFG.TARGET_COVERAGE}) coverage={fair_cov:.3f}")
    print(f"  Fair-MF {fair_metrics}")
    results["fairness"] = {
        "tail_items": len(tail),
        "coverage_pre": {"MF": mf_cov, "UserCF": ucf_cov, "MostPopular": mp_cov},
        "fair_mf_target": CFG.TARGET_COVERAGE,
        "fair_mf_coverage": fair_cov,
        "fair_mf_metrics": fair_metrics,
    }

    # ---- Pareto sweep ------------------------------------------------------
    print("\n== Pareto sweep ==")
    with _time("pareto"):
        pareto = run_pareto_sweep(
            mf_recs, mf_scores, ds.train_matrix, tail, ds.test_df, CFG.PARETO_TARGETS, CFG.K
        )
    for row in pareto:
        print(f"  target={row['target']:.2f}  cov={row['coverage']:.3f}  NDCG={row[f'NDCG@{CFG.K}']:.4f}")
    results["pareto"] = pareto
    plot_pareto(pareto, str(CFG.REPORTS_DIR / "pareto.png"), k=CFG.K)

    # ---- User-group analysis ----------------------------------------------
    print("\n== User-group fairness ==")
    with _time("user_groups"):
        groups = user_group_analysis(
            mf_recs, mf_scores, ds.train_matrix, tail, ds.train_df, ds.test_df,
            CFG.PARETO_TARGETS, CFG.K
        )
    for gname, rows in groups.items():
        base = rows[0][f"NDCG@{CFG.K}"]
        print(f"  {gname:5s} (n={rows[0]['n_users']}) target=0 NDCG={base:.4f}  "
              + "  ".join(f"t={r['target']:.2f}:NDCG={r[f'NDCG@{CFG.K}']:.4f}" for r in rows[1:]))
    results["user_groups"] = groups
    plot_user_groups(groups, str(CFG.REPORTS_DIR / "user_groups.png"), k=CFG.K)

    # ---- Ablation: n_factors ----------------------------------------------
    print("\n== Ablation: n_factors ==")
    ablation = []
    for nf in CFG.ABLATION_FACTORS:
        set_all_seeds(CFG.SEED)
        with _time(f"MF(n_factors={nf})"):
            mf_nf, hist_nf = train_mf(
                ds.n_users, ds.n_items, ds.train_df, ds.val_df,
                n_factors=nf, verbose=False,
            )
        mae_rmse = eval_rating(mf_predict_fn(mf_nf), ds.test_df)
        recs_nf, _ = generate_topk_mf(mf_nf, ds.train_matrix, k=CFG.K)
        ranking_nf = eval_ranking(recs_nf, ds.test_df, k=CFG.K)
        ablation.append({
            "n_factors": nf,
            "best_val_mse": hist_nf.best_val_loss,
            "best_epoch": hist_nf.best_epoch,
            "rating": mae_rmse,
            "ranking": ranking_nf,
        })
        print(f"  n_factors={nf}: val={hist_nf.best_val_loss:.4f} "
              f"MAE={mae_rmse['MAE']:.4f} NDCG={ranking_nf[f'NDCG@{CFG.K}']:.4f}")
    results["ablation_n_factors"] = ablation

    # ---- Persist -----------------------------------------------------------
    torch.save({
        "state_dict": mf.state_dict(),
        "n_users": ds.n_users,
        "n_items": ds.n_items,
        "n_factors": CFG.N_FACTORS,
        "global_mean": float(mf.global_mean.item()),
        "user_id_map": ds.user_id_map.to_dict(),
        "item_id_map": ds.item_id_map.to_dict(),
    }, CFG.REPORTS_DIR / "mf_model.pt")
    print(f"\nsaved {CFG.REPORTS_DIR / 'mf_model.pt'}")

    results_path = CFG.REPORTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"wrote {results_path}")

    return results


if __name__ == "__main__":
    main()
