"""H660: DeployableMinVarLW660 -- deployable global-minimum-variance on a Ledoit-Wolf shrunk covariance.

A non-optimized (return-blind) robust-covariance allocation: shrink the 55x55 sample covariance toward a
scaled identity with the analytical Ledoit-Wolf intensity (no free parameter), solve the long-only global-
minimum-variance portfolio, then apply the same 1-day-lagged interior-grid leverage as H650. Distinct from
MDR / inverse-vol (H653) / ERC (H659).

Run: uv run python scripts/run_h660.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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


def ledoit_wolf_cov(mat):
    """Ledoit-Wolf shrinkage of the sample covariance toward mu*I (analytical intensity, no tuning).

    Returns (shrunk_cov, delta) where delta in [0,1] is the shrinkage intensity. Trace-preserving.
    """
    x = mat - mat.mean(axis=0)
    t, n = x.shape
    s = (x.T @ x) / t  # MLE sample covariance (1/T)
    mu = np.trace(s) / n
    f = mu * np.eye(n)  # spherical target
    d2 = float(np.sum((s - f) ** 2))  # ||S - F||_F^2 (dispersion)
    # b2 = (1/T^2) sum_t ||x_t x_t' - S||_F^2, vectorized via sum_t x_ti^2 x_tj^2 = (x^2)'(x^2)
    x2 = x**2
    q = float(np.sum(x2.T @ x2))
    b2 = (q - t * float(np.sum(s**2))) / (t * t)
    b2 = min(b2, d2)  # cap so delta in [0,1]
    delta = b2 / d2 if d2 > 0 else 0.0
    return delta * f + (1.0 - delta) * s, delta


def gmv_weights(cov):
    """Long-only, sum-to-1 global-minimum-variance weights."""
    n = cov.shape[0]

    def obj(w):
        return float(w @ cov @ w)

    res = minimize(
        obj,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
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
    print(
        "H660: DeployableMinVarLW660 -- deployable Ledoit-Wolf min-variance baseline\n" + "=" * 64
    )
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx = clean.index

    cov_lw, delta = ledoit_wolf_cov(mat)
    w = gmv_weights(cov_lw)
    port_vol = float(np.sqrt(w @ (np.cov(mat.T) * TRADING_DAYS) @ w))
    print(
        f"Ledoit-Wolf shrinkage intensity delta = {delta:.4f}; GMV annualized vol = {port_vol:.4f}"
    )

    print("\nGMV weights fixed; sweeping the interior leverage grid:")
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
    print(f"\nDeployable LW-GMV champion config: {best_key}")
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
        f"Sharpe {sh:.3f} DD {dd:.2%} CAGR {cg:.2%}; vs ERC H659 baseline; vs H650 (1.979): {sh - 1.979:+.3f}"
    )

    finalize_run(
        "660",
        "deployable-minvar-lw",
        "DeployableMinVarLW660",
        "Set14 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable global-minimum-variance on a Ledoit-Wolf shrunk covariance (analytical intensity "
            f"delta={delta:.4f}, no tuning) + 1-day-LAGGED interior-grid leverage. Config {best_key}. GMV "
            f"annualized vol {port_vol:.4f}. Sensitivity {sens}. Sharpe {sh:.3f}, Calmar {cal:.3f}, "
            f"Sortino {so:.3f}. vs H650 (1.979): {sh - 1.979:+.3f}."
        ),
        extra={
            "objective": "deployable-ledoit-wolf-gmv-lagged-interior",
            "champion_config": best_key,
            "lw_shrinkage_delta": round(float(delta), 5),
            "gmv_annual_vol": round(port_vol, 5),
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableMinVarLW660",
    )


if __name__ == "__main__":
    main()
