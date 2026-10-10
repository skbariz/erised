"""H662: DeployableSubperiodRobust662 -- deployable max-min-Sharpe-across-halves (regime-robust) objective.

Same lag-aware, interior-only construction as H650, but the pool weights are fit to maximize the minimum
Sharpe across the two equal calendar halves of the OOS period on the 1-day-lagged signal. A minimax,
regime-robust criterion: the allocation must earn its Sharpe in both the crisis-heavy first half and the
calmer second half, not average a great era against a mediocre one.

Run: uv run python scripts/run_h662.py
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


def _half_sharpes(r, split):
    def sh(x):
        s = x.std()
        return x.mean() / s * np.sqrt(TRADING_DAYS) if s > 0 else 0.0

    return sh(r[:split]), sh(r[split:])


def lev_subperiod_neg(mat, idx, tv, cap, split):
    def neg(w):
        a, b = _half_sharpes(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK), split)
        return -min(a, b)

    return neg


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print(
        "H662: DeployableSubperiodRobust662 -- deployable max-min-half-Sharpe objective\n"
        + "=" * 64
    )
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)
    split = len(idx) // 2
    print(f"Half split at {idx[split].date()} ({split} / {len(idx) - split} days)")

    print("\nInterior-only candidates, weights FIT (max-min half-Sharpe) on the lagged signal:")
    cands = {}
    for thr in [15.0, 16.0, 17.0]:
        for asym in (False, True):
            tv, cap = lagged_caps(vix, idx, thr, asym)
            w = slsqp(lev_subperiod_neg(mat, idx, tv, cap, split), w0, n)
            a, b = _half_sharpes(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK), split)
            key = f"t{thr:.0f}w20{'-asym' if asym else ''}"
            cands[key] = (-min(a, b), w, tv, cap, thr, asym)
            print(f"  {key}: half1={a:.3f} half2={b:.3f} min={min(a, b):.3f}")

    best_key = min(cands, key=lambda k: cands[k][0])  # highest worse-half Sharpe
    _, w, tv, cap, thr, asym = cands[best_key]
    print(f"\nDeployable subperiod-robust champion config: {best_key}")
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)
    a, b = _half_sharpes(r, split)

    mins = []
    for seed in range(5):
        ws = np.random.RandomState(seed).dirichlet(np.ones(n))
        wr = slsqp(lev_subperiod_neg(mat, idx, tv, cap, split), ws, n)
        ha, hb = _half_sharpes(apply_leverage(mat @ wr, idx, tv, cap, LOOKBACK), split)
        mins.append(round(float(min(ha, hb)), 4))
    print(f"Restart min-half-Sharpes: {mins} (range {max(mins) - min(mins):.4f})")

    sens = {}
    for d in [-1.0, 0.0, 1.0]:
        tvd, capd = lagged_caps(vix, idx, thr + d, asym)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, tvd, capd, LOOKBACK))[0]), 3
        )
    print(f"Sensitivity (Sharpe): {sens}")
    print(
        f"Half-Sharpes {a:.3f}/{b:.3f} (min {min(a, b):.3f}); pooled Sharpe {sh:.3f}; "
        f"vs H650 (1.979): {sh - 1.979:+.3f}"
    )

    finalize_run(
        "662",
        "deployable-subperiod-robust",
        "DeployableSubperiodRobust662",
        "Set14 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable subperiod-robust: maximize min(half1 Sharpe, half2 Sharpe) on the 1-day-LAGGED "
            f"signal, interior-only grid. Split {idx[split].date()}. Config {best_key}. Half-Sharpes "
            f"{a:.3f}/{b:.3f} (min {min(a, b):.3f}). 5-start min-half-Sharpes {mins} "
            f"(range {max(mins) - min(mins):.4f}). Sensitivity {sens}. Pooled Sharpe {sh:.3f}, "
            f"Calmar {cal:.3f}, Sortino {so:.3f}. vs H650 (1.979): {sh - 1.979:+.3f}."
        ),
        extra={
            "objective": "deployable-max-min-half-sharpe-lagged-interior",
            "champion_config": best_key,
            "split_date": str(idx[split].date()),
            "half1_sharpe": round(float(a), 4),
            "half2_sharpe": round(float(b), 4),
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "restart_min_half_sharpes": mins,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableSubperiodRobust662",
    )


if __name__ == "__main__":
    main()
