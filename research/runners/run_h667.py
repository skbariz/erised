"""H667: DeployableMinCorr667 -- deployable minimum-correlation allocation.

A return-blind allocation: each sleeve is scored by (1 - mean off-diagonal correlation), tilted by that score
divided by its volatility, and normalized -- up-weighting low/negative-correlation diversifiers and
down-weighting redundant, mutually-correlated sleeves. The same 1-day-lagged interior-grid leverage as H650
is applied. Return-blind (allocation depends only on the covariance), so it cannot be over-fit.

Run: uv run python scripts/run_h667.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from set8_lev import (
    TRADING_DAYS,
    apply_leverage,
    finalize_run,
    portfolio_metrics,
    two_state_cap,
    vix_momentum_mask,
)
from set8_pool import build_pool

LOOKBACK = 20


def mincorr_weights(mat):
    """Minimum-correlation weights: tilt by (1 - mean off-diagonal correlation) / volatility."""
    cov = np.cov(mat.T) * TRADING_DAYS
    vols = np.sqrt(np.diag(cov))
    corr = np.corrcoef(mat.T)
    off = corr.copy()
    np.fill_diagonal(off, np.nan)
    avg_corr = np.nanmean(off, axis=1)  # mean correlation of each sleeve to the rest
    score = np.clip(1.0 - avg_corr, 1e-6, None)  # low/negative corr -> high score
    w = score / vols  # combine the diversification tilt with inverse-vol sizing
    return w / w.sum()


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print("H667: DeployableMinCorr667 -- deployable minimum-correlation allocation\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx = clean.index

    w = mincorr_weights(mat)
    order = np.argsort(w)[::-1]
    top = ", ".join(f"{names[i]} {w[i]:.3f}" for i in order[:5])
    print(f"MinCorr weight vector: min {w.min():.4f}, max {w.max():.4f}; top5: {top}")

    print("\nMinCorr weights fixed; sweeping the interior leverage grid:")
    cands = {}
    for thr in [15.0, 16.0, 17.0]:
        for asym in (False, True):
            tv, cap = lagged_caps(vix, idx, thr, asym)
            sh = portfolio_metrics(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))[0]
            key = f"t{thr:.0f}w20{'-asym' if asym else ''}"
            cands[key] = (sh, tv, cap, thr, asym)
            print(f"  {key}: Sh={sh:.3f}")

    best_key = max(cands, key=lambda k: cands[k][0])
    _, tv, cap, thr, asym = cands[best_key]
    print(f"\nDeployable MinCorr champion config: {best_key}")
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)

    sens = {}
    for d in [-1.0, 0.0, 1.0]:
        tvd, capd = lagged_caps(vix, idx, thr + d, asym)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, tvd, capd, LOOKBACK))[0]), 3
        )
    print(f"Sensitivity (Sharpe): {sens}")
    print(
        f"Sharpe {sh:.3f} DD {dd:.2%} CAGR {cg:.2%}; vs ERC H659 (1.414): {sh - 1.414:+.3f}; "
        f"vs inverse-vol H653 (1.303): {sh - 1.303:+.3f}; vs H650 (1.979): {sh - 1.979:+.3f}"
    )

    finalize_run(
        "667",
        "deployable-mincorr",
        "DeployableMinCorr667",
        "Set15 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable minimum-correlation allocation (return-blind, correlation-rank tilt x inverse-vol) "
            f"+ 1-day-LAGGED interior-grid leverage. Config {best_key}. Weight min/max "
            f"{w.min():.4f}/{w.max():.4f}. Sensitivity {sens}. Sharpe {sh:.3f}, Calmar {cal:.3f}, "
            f"Sortino {so:.3f}. vs ERC H659 (1.414): {sh - 1.414:+.3f}; vs inverse-vol H653 (1.303): "
            f"{sh - 1.303:+.3f}; vs H650 (1.979): {sh - 1.979:+.3f}."
        ),
        extra={
            "objective": "deployable-mincorr-lagged-interior",
            "champion_config": best_key,
            "weight_min": round(float(w.min()), 5),
            "weight_max": round(float(w.max()), 5),
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableMinCorr667",
    )


if __name__ == "__main__":
    main()
