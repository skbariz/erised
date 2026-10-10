"""H650: GrandDeployable650 -- terminal deployable champion, weights fit on the LAGGED signal.

Optimizes weights directly on the 1-day-lagged regime (aligns optimization with deployment), over an
interior-only grid (threshold {15,16,17} x window {20}, plain + asymmetric). Full battery: 5-start
stability, sensitivity, and the gap vs the same-day H645 and the optimize-then-lag H643.

Run: uv run python scripts/run_h650.py
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


def lev_sharpe_neg(mat, idx, tv, cap):
    def neg(w):
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    return neg


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print("H650: GrandDeployable650 -- terminal deployable strategy\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    print("\nInterior-only candidates, weights FIT on the lagged signal:")
    cands = {}
    for thr in [15.0, 16.0, 17.0]:
        for asym in (False, True):
            tv, cap = lagged_caps(vix, idx, thr, asym)
            w = slsqp(lev_sharpe_neg(mat, idx, tv, cap), w0, n)
            sh = portfolio_metrics(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))[0]
            key = f"t{thr:.0f}w20{'-asym' if asym else ''}"
            cands[key] = (sh, w, tv, cap, thr, asym)
            print(f"  {key}: Sh={sh:.3f}")

    best_key = max(cands, key=lambda k: cands[k][0])
    _, w, tv, cap, thr, asym = cands[best_key]
    print(f"\nDeployable champion config: {best_key}")
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)

    shs = []
    for seed in range(5):
        ws = np.random.RandomState(seed).dirichlet(np.ones(n))
        wr = slsqp(lev_sharpe_neg(mat, idx, tv, cap), ws, n)
        shs.append(
            round(float(portfolio_metrics(apply_leverage(mat @ wr, idx, tv, cap, LOOKBACK))[0]), 4)
        )
    print(f"Restart Sharpes: {shs} (range {max(shs) - min(shs):.4f})")

    sens = {}
    for d in [-1.0, 0.0, 1.0]:
        tvd, capd = lagged_caps(vix, idx, thr + d, asym)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, tvd, capd, LOOKBACK))[0]), 3
        )
    print(f"Sensitivity: {sens}")
    print(
        f"vs H645 same-day (2.740): {sh - 2.740:+.3f}; vs H643 optimize-then-lag (1.918): {sh - 1.918:+.3f}"
    )

    finalize_run(
        "650",
        "grand-deployable-650",
        "GrandDeployable650",
        "Set12 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"TERMINAL DEPLOYABLE champion: weights fit on the 1-day-LAGGED signal, interior-only grid. "
            f"Config {best_key}. 5-start Sharpes {shs} (range {max(shs) - min(shs):.4f}). Sensitivity {sens}. "
            f"Calmar {cal:.3f}, Sortino {so:.3f}. vs H645 same-day (2.740): {sh - 2.740:+.3f}; "
            f"vs H643 optimize-then-lag (1.918): {sh - 1.918:+.3f}. Honest executable terminal strategy."
        ),
        extra={
            "objective": "grand-deployable-lagged-fit-interior",
            "champion_config": best_key,
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "restart_sharpes": shs,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="GrandDeployable650",
    )


if __name__ == "__main__":
    main()
