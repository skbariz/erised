"""H659: DeployableERC659 -- deployable equal-risk-contribution (correlation-aware risk parity) baseline.

A non-optimized allocation: pool weights equalize each sleeve's contribution to portfolio variance
(w_i * (Sigma w)_i equal across sleeves), then the same 1-day-lagged interior-grid leverage as H650 is
applied. Return-blind -- the allocation depends only on the covariance, so it cannot be over-fit. The
correlation-aware sibling of the inverse-vol baseline H653.

Run: uv run python scripts/run_h659.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.optimize import minimize
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


def erc_weights(mat):
    """Equal-risk-contribution weights: minimize the dispersion of risk contributions w_i*(Sigma w)_i."""
    cov = np.cov(mat.T) * TRADING_DAYS
    n = cov.shape[0]
    vols = np.sqrt(np.diag(cov))
    w0 = (1.0 / vols) / (1.0 / vols).sum()  # inverse-vol warm start

    def obj(w):
        rc = w * (cov @ w)
        return float(np.sum((rc[:, None] - rc[None, :]) ** 2))

    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=[(1e-6, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 2000, "ftol": 1e-16},
    )
    return res.x


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print("H659: DeployableERC659 -- deployable equal-risk-contribution baseline\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx = clean.index

    w = erc_weights(mat)
    cov = np.cov(mat.T) * TRADING_DAYS
    rc = w * (cov @ w)
    rc_share = rc / rc.sum()
    print(
        f"Risk-contribution share min/max: {rc_share.min():.4f} / {rc_share.max():.4f} (target 1/n)"
    )

    print("\nERC weights fixed; sweeping the interior leverage grid:")
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
    print(f"\nDeployable ERC champion config: {best_key}")
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
        f"Sharpe {sh:.3f} DD {dd:.2%} CAGR {cg:.2%}; vs inverse-vol H653 (1.303): {sh - 1.303:+.3f}; "
        f"vs H650 (1.979): {sh - 1.979:+.3f}"
    )

    finalize_run(
        "659",
        "deployable-erc",
        "DeployableERC659",
        "Set14 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable equal-risk-contribution baseline (return-blind, correlation-aware risk parity) + "
            f"1-day-LAGGED interior-grid leverage. Config {best_key}. RC share min/max "
            f"{rc_share.min():.4f}/{rc_share.max():.4f} (target {1.0 / len(w):.4f}). Sensitivity {sens}. "
            f"Sharpe {sh:.3f}, Calmar {cal:.3f}, Sortino {so:.3f}. vs inverse-vol H653 (1.303): "
            f"{sh - 1.303:+.3f}; vs H650 (1.979): {sh - 1.979:+.3f}."
        ),
        extra={
            "objective": "deployable-equal-risk-contribution-lagged-interior",
            "champion_config": best_key,
            "rc_share_min": round(float(rc_share.min()), 5),
            "rc_share_max": round(float(rc_share.max()), 5),
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableERC659",
    )


if __name__ == "__main__":
    main()
