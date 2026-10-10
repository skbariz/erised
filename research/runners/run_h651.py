"""H651: DeployableSortino651 -- deployable champion optimized for leveraged Sortino.

Same lag-aware, interior-only construction as H650 (the terminal Sharpe champion), but the pool weights
are fit to maximize the *leveraged Sortino ratio* (downside-deviation denominator) on the 1-day-lagged
regime signal. Tests whether targeting downside risk rather than total vol yields a better-protected
deployable portfolio. Full battery: 5-start stability, sensitivity, gap vs H650.

Run: uv run python scripts/run_h651.py
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
    mdr_warm_start,
    portfolio_metrics,
    slsqp,
    two_state_cap,
    vix_momentum_mask,
)
from set8_pool import build_pool

LOOKBACK = 20


def lev_sortino_neg(mat, idx, tv, cap):
    """Negative leveraged Sortino: annualized mean / annualized downside deviation."""

    def neg(w):
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        ds = r[r < 0]
        dd = np.sqrt((ds**2).mean()) * np.sqrt(TRADING_DAYS) if len(ds) else 0.0
        return -(r.mean() * TRADING_DAYS / dd) if dd > 0 else 0.0

    return neg


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print("H651: DeployableSortino651 -- deployable Sortino-optimal strategy\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    print("\nInterior-only candidates, weights FIT (Sortino) on the lagged signal:")
    cands = {}
    for thr in [15.0, 16.0, 17.0]:
        for asym in (False, True):
            tv, cap = lagged_caps(vix, idx, thr, asym)
            w = slsqp(lev_sortino_neg(mat, idx, tv, cap), w0, n)
            so = portfolio_metrics(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))[4]
            key = f"t{thr:.0f}w20{'-asym' if asym else ''}"
            cands[key] = (so, w, tv, cap, thr, asym)
            print(f"  {key}: Sortino={so:.3f}")

    best_key = max(cands, key=lambda k: cands[k][0])
    _, w, tv, cap, thr, asym = cands[best_key]
    print(f"\nDeployable Sortino champion config: {best_key}")
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)

    shs = []
    for seed in range(5):
        ws = np.random.RandomState(seed).dirichlet(np.ones(n))
        wr = slsqp(lev_sortino_neg(mat, idx, tv, cap), ws, n)
        shs.append(
            round(float(portfolio_metrics(apply_leverage(mat @ wr, idx, tv, cap, LOOKBACK))[4]), 4)
        )
    print(f"Restart Sortinos: {shs} (range {max(shs) - min(shs):.4f})")

    sens = {}
    for d in [-1.0, 0.0, 1.0]:
        tvd, capd = lagged_caps(vix, idx, thr + d, asym)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, tvd, capd, LOOKBACK))[0]), 3
        )
    print(f"Sensitivity (Sharpe): {sens}")
    print(f"Sharpe {sh:.3f}; vs H650 Sharpe-optimal (1.980): {sh - 1.980:+.3f}; Sortino {so:.3f}")

    finalize_run(
        "651",
        "deployable-sortino",
        "DeployableSortino651",
        "Set13 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable Sortino-optimal: weights fit on the 1-day-LAGGED signal to maximize leveraged "
            f"Sortino, interior-only grid. Config {best_key}. 5-start Sortinos {shs} "
            f"(range {max(shs) - min(shs):.4f}). Sensitivity {sens}. Sharpe {sh:.3f}, Calmar {cal:.3f}, "
            f"Sortino {so:.3f}. vs H650 Sharpe-optimal (1.980): {sh - 1.980:+.3f}."
        ),
        extra={
            "objective": "deployable-lev-sortino-lagged-fit-interior",
            "champion_config": best_key,
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "restart_sortinos": shs,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableSortino651",
    )


if __name__ == "__main__":
    main()
