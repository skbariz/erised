"""H668: DeployableMaxSortino668 -- deployable max-Sortino allocation.

Deployable counterpart of H650: pool weights fit by SLSQP (long-only, sum-to-1, MDR warm-start) to maximize
the Sortino ratio (annualized mean / downside deviation) instead of the Sharpe ratio's total volatility, then
the same 1-day-lagged interior-grid leverage as H650. Downside-only fitting keeps the convex, right-skewed
trend/crisis sleeves that a symmetric Sharpe solve trims.

Run: uv run python scripts/run_h668.py
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


def sortino_weights(mat):
    """Long-only weights maximizing the Sortino ratio of the blended sleeve returns."""
    n = mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    def neg_sortino(w):
        r = mat @ w
        mu = r.mean() * TRADING_DAYS
        ds = np.minimum(r, 0.0)
        dd = np.sqrt((ds**2).mean()) * np.sqrt(TRADING_DAYS)
        return 0.0 if dd <= 0 else -mu / dd

    return slsqp(neg_sortino, w0, n)


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print("H668: DeployableMaxSortino668 -- deployable max-Sortino allocation\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx = clean.index

    # 5-start weight stability: confirm the SLSQP optimum is not start-dependent.
    starts = [mdr_warm_start(mat, mat.shape[1])] + [
        np.random.RandomState(s).dirichlet(np.ones(mat.shape[1])) for s in range(4)
    ]
    sols = []
    for w0 in starts:

        def neg_sortino(w):
            r = mat @ w
            mu = r.mean() * TRADING_DAYS
            ds = np.minimum(r, 0.0)
            dd = np.sqrt((ds**2).mean()) * np.sqrt(TRADING_DAYS)
            return 0.0 if dd <= 0 else -mu / dd

        sols.append(slsqp(neg_sortino, w0, mat.shape[1]))
    sols = np.array(sols)
    max_spread = float(np.abs(sols - sols[0]).max())
    print(f"5-start max weight spread vs MDR-start solution: {max_spread:.4f}")

    w = sols[0]
    order = np.argsort(w)[::-1]
    top = ", ".join(f"{names[i]} {w[i]:.3f}" for i in order[:5])
    print(f"Sortino weight vector: nonzero {int((w > 1e-4).sum())}/{len(w)}; top5: {top}")

    print("\nSortino weights fixed; sweeping the interior leverage grid:")
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
    print(f"\nDeployable max-Sortino champion config: {best_key}")
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
        f"Sharpe {sh:.3f} DD {dd:.2%} CAGR {cg:.2%}; vs H650 (1.979): {sh - 1.979:+.3f}; "
        f"vs MAD H658 (1.953): {sh - 1.953:+.3f}"
    )

    finalize_run(
        "668",
        "deployable-max-sortino",
        "DeployableMaxSortino668",
        "Set15 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable max-Sortino allocation (SLSQP, MDR warm-start, return/downside-dev objective) + "
            f"1-day-LAGGED interior-grid leverage. Config {best_key}. 5-start max weight spread "
            f"{max_spread:.4f}. Sensitivity {sens}. Sharpe {sh:.3f}, Calmar {cal:.3f}, Sortino {so:.3f}. "
            f"vs H650 (1.979): {sh - 1.979:+.3f}; vs MAD H658 (1.953): {sh - 1.953:+.3f}."
        ),
        extra={
            "objective": "deployable-max-sortino-lagged-interior",
            "champion_config": best_key,
            "weight_spread_5start": round(max_spread, 5),
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableMaxSortino668",
    )


if __name__ == "__main__":
    main()
