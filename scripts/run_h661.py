"""H661: DeployableFoldRobust661 -- deployable mean-minus-lambda*std of per-fold Sharpe objective.

Same lag-aware, interior-only construction as H650, but the pool weights are fit to maximize
mean(per-fold OOS Sharpe) - lambda*std(per-fold Sharpe) on the 1-day-lagged signal. Optimizes the program's
own overfitting metric (low fold-Sharpe dispersion) directly, rather than the pooled full-sample Sharpe.

Run: uv run python scripts/run_h661.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from set8_lev import (
    TRADING_DAYS,
    apply_leverage,
    finalize_run,
    mdr_warm_start,
    portfolio_metrics,
    slsqp,
    two_state_cap,
    vix_momentum_mask,
)
from set8_pool import build_pool

LOOKBACK = 20
LAMBDA = 0.5  # penalty on per-fold Sharpe dispersion


def fold_masks_from(idx, folds):
    return [np.asarray((idx >= f.test_start) & (idx < f.test_end)) for f in folds]


def _fold_sharpes(r, masks):
    out = []
    for m in masks:
        fr = r[m]
        if len(fr) > 10:
            s = fr.std()
            out.append(fr.mean() / s * np.sqrt(TRADING_DAYS) if s > 0 else 0.0)
    return np.asarray(out)


def lev_foldrobust_neg(mat, idx, tv, cap, masks, lam=LAMBDA):
    def neg(w):
        fss = _fold_sharpes(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK), masks)
        return -(fss.mean() - lam * fss.std()) if len(fss) else 0.0

    return neg


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print(
        "H661: DeployableFoldRobust661 -- deployable fold-variance-penalized objective\n" + "=" * 64
    )
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)
    masks = fold_masks_from(idx, folds)

    print(
        f"\nInterior-only candidates, weights FIT (mean - {LAMBDA}*std of fold Sharpe) on lagged signal:"
    )
    cands = {}
    for thr in [15.0, 16.0, 17.0]:
        for asym in (False, True):
            tv, cap = lagged_caps(vix, idx, thr, asym)
            w = slsqp(lev_foldrobust_neg(mat, idx, tv, cap, masks), w0, n)
            fss = _fold_sharpes(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK), masks)
            score = fss.mean() - LAMBDA * fss.std()
            key = f"t{thr:.0f}w20{'-asym' if asym else ''}"
            cands[key] = (-score, w, tv, cap, thr, asym)
            print(f"  {key}: mean={fss.mean():.3f} std={fss.std():.3f} score={score:.3f}")

    best_key = min(cands, key=lambda k: cands[k][0])  # highest mean-lambda*std score
    _, w, tv, cap, thr, asym = cands[best_key]
    print(f"\nDeployable fold-robust champion config: {best_key}")
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)
    fss = _fold_sharpes(r, masks)

    scores = []
    for seed in range(5):
        ws = np.random.RandomState(seed).dirichlet(np.ones(n))
        wr = slsqp(lev_foldrobust_neg(mat, idx, tv, cap, masks), ws, n)
        f2 = _fold_sharpes(apply_leverage(mat @ wr, idx, tv, cap, LOOKBACK), masks)
        scores.append(round(float(f2.mean() - LAMBDA * f2.std()), 4))
    print(f"Restart scores: {scores} (range {max(scores) - min(scores):.4f})")

    sens = {}
    for d in [-1.0, 0.0, 1.0]:
        tvd, capd = lagged_caps(vix, idx, thr + d, asym)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, tvd, capd, LOOKBACK))[0]), 3
        )
    print(f"Sensitivity (Sharpe): {sens}")
    print(
        f"Fold mean {fss.mean():.3f} std {fss.std():.3f} range {fss.min():.2f}..{fss.max():.2f}; "
        f"Sharpe {sh:.3f}; vs H650 fold range (1.21..2.75)"
    )

    finalize_run(
        "661",
        "deployable-fold-robust",
        "DeployableFoldRobust661",
        "Set14 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable fold-robust: maximize mean - {LAMBDA}*std of per-fold OOS Sharpe on the "
            f"1-day-LAGGED signal, interior-only grid. Config {best_key}. Fold mean {fss.mean():.3f}, "
            f"std {fss.std():.3f}, range {fss.min():.2f}..{fss.max():.2f}. 5-start scores {scores} "
            f"(range {max(scores) - min(scores):.4f}). Sensitivity {sens}. Sharpe {sh:.3f}, "
            f"Calmar {cal:.3f}, Sortino {so:.3f}. vs H650 fold range (1.21..2.75)."
        ),
        extra={
            "objective": "deployable-fold-mean-minus-lambda-std-lagged-interior",
            "champion_config": best_key,
            "lambda": LAMBDA,
            "fold_sharpe_mean": round(float(fss.mean()), 4),
            "fold_sharpe_std": round(float(fss.std()), 4),
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "restart_scores": scores,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableFoldRobust661",
    )


if __name__ == "__main__":
    main()
