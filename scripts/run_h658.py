"""H658: DeployableMAD658 -- deployable return-to-MAD (mean-absolute-deviation) objective.

Same lag-aware, interior-only construction as H650, but the pool weights are fit to maximize the leveraged
annualized-return / mean-absolute-deviation ratio on the 1-day-lagged signal. An L1-dispersion (outlier-
robust) point on the deployable frontier, distinct from Sharpe's L2 variance denominator.

Run: uv run python scripts/run_h658.py
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


def mad_ratio(r):
    """Annualized return divided by the annualized mean-absolute-deviation (an L1 Sharpe analogue)."""
    ann_mean = r.mean() * TRADING_DAYS
    mad = np.abs(r - r.mean()).mean() * np.sqrt(TRADING_DAYS)
    return ann_mean / mad if mad > 0 else 0.0


def lev_mad_neg(mat, idx, tv, cap):
    def neg(w):
        return -mad_ratio(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))

    return neg


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print("H658: DeployableMAD658 -- deployable return-to-MAD objective\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    print("\nInterior-only candidates, weights FIT (return/MAD) on the lagged signal:")
    cands = {}
    for thr in [15.0, 16.0, 17.0]:
        for asym in (False, True):
            tv, cap = lagged_caps(vix, idx, thr, asym)
            w = slsqp(lev_mad_neg(mat, idx, tv, cap), w0, n)
            r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
            ratio, sh = mad_ratio(r), portfolio_metrics(r)[0]
            key = f"t{thr:.0f}w20{'-asym' if asym else ''}"
            cands[key] = (-ratio, w, tv, cap, thr, asym)
            print(f"  {key}: MAD-ratio={ratio:.3f} Sh={sh:.3f}")

    best_key = min(cands, key=lambda k: cands[k][0])  # highest return/MAD ratio
    _, w, tv, cap, thr, asym = cands[best_key]
    print(f"\nDeployable MAD champion config: {best_key}")
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)
    ratio = mad_ratio(r)

    ratios = []
    for seed in range(5):
        ws = np.random.RandomState(seed).dirichlet(np.ones(n))
        wr = slsqp(lev_mad_neg(mat, idx, tv, cap), ws, n)
        ratios.append(round(float(mad_ratio(apply_leverage(mat @ wr, idx, tv, cap, LOOKBACK))), 4))
    print(f"Restart MAD-ratios: {ratios} (range {max(ratios) - min(ratios):.4f})")

    sens = {}
    for d in [-1.0, 0.0, 1.0]:
        tvd, capd = lagged_caps(vix, idx, thr + d, asym)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, tvd, capd, LOOKBACK))[0]), 3
        )
    print(f"Sensitivity (Sharpe): {sens}")
    print(
        f"MAD-ratio {ratio:.3f} Sharpe {sh:.3f} DD {dd:.2%}; vs H650 Sharpe (1.979): {sh - 1.979:+.3f}"
    )

    finalize_run(
        "658",
        "deployable-mad",
        "DeployableMAD658",
        "Set14 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable return-to-MAD: maximize annualized return / mean-absolute-deviation on the "
            f"1-day-LAGGED signal, interior-only grid. Config {best_key}. 5-start MAD-ratios {ratios} "
            f"(range {max(ratios) - min(ratios):.4f}). Sensitivity {sens}. MAD-ratio {ratio:.3f}, "
            f"Sharpe {sh:.3f}, Calmar {cal:.3f}, Sortino {so:.3f}. vs H650 Sharpe (1.979): {sh - 1.979:+.3f}."
        ),
        extra={
            "objective": "deployable-lev-return-mad-lagged-fit-interior",
            "champion_config": best_key,
            "mad_ratio": round(float(ratio), 4),
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "restart_mad_ratios": ratios,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableMAD658",
    )


if __name__ == "__main__":
    main()
