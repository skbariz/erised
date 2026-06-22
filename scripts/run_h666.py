"""H666: DeployableHRP666 -- deployable Hierarchical Risk Parity allocation.

A return-blind allocation: pool weights set by HRP (Lopez de Prado 2016) -- cluster the 55-sleeve
correlation-distance matrix, quasi-diagonalize the covariance, and split weight by recursive bisection on
cluster inverse-variance-portfolio variance. The same 1-day-lagged interior-grid leverage as H650 is applied.
Return-blind (allocation depends only on the covariance), so it cannot be over-fit to in-sample returns.

Run: uv run python scripts/run_h666.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
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


def _ivp(cov, items):
    """Inverse-variance-portfolio weights within a sub-cluster."""
    iv = 1.0 / np.diag(cov)[items]
    return iv / iv.sum()


def _cluster_var(cov, items):
    """Variance of the inverse-variance portfolio of a sub-cluster."""
    w = _ivp(cov, items)
    sub = cov[np.ix_(items, items)]
    return float(w @ sub @ w)


def _quasi_diag(link):
    """Seriate the linkage tree into the quasi-diagonal leaf order (Lopez de Prado)."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.to_numpy() - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def hrp_weights(mat):
    """Hierarchical Risk Parity weights over the sleeve return matrix."""
    cov = np.cov(mat.T) * TRADING_DAYS
    corr = np.corrcoef(mat.T)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    sort_ix = _quasi_diag(link)

    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        clusters = [
            c[s:e]
            for c in clusters
            for s, e in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for k in range(0, len(clusters), 2):
            c0, c1 = clusters[k], clusters[k + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    return w.sort_index().to_numpy()


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print("H666: DeployableHRP666 -- deployable Hierarchical Risk Parity allocation\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx = clean.index

    w = hrp_weights(mat)
    order = np.argsort(w)[::-1]
    top = ", ".join(f"{names[i]} {w[i]:.3f}" for i in order[:5])
    print(f"HRP weight vector: min {w.min():.4f}, max {w.max():.4f}; top5: {top}")

    print("\nHRP weights fixed; sweeping the interior leverage grid:")
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
    print(f"\nDeployable HRP champion config: {best_key}")
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
        "666",
        "deployable-hrp",
        "DeployableHRP666",
        "Set15 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable Hierarchical Risk Parity (return-blind, cluster-aware) + 1-day-LAGGED "
            f"interior-grid leverage. Config {best_key}. Weight min/max {w.min():.4f}/{w.max():.4f}. "
            f"Sensitivity {sens}. Sharpe {sh:.3f}, Calmar {cal:.3f}, Sortino {so:.3f}. "
            f"vs ERC H659 (1.414): {sh - 1.414:+.3f}; vs inverse-vol H653 (1.303): {sh - 1.303:+.3f}; "
            f"vs H650 (1.979): {sh - 1.979:+.3f}."
        ),
        extra={
            "objective": "deployable-hrp-lagged-interior",
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
        label="DeployableHRP666",
    )


if __name__ == "__main__":
    main()
