"""H657: DeployableOmega657 -- deployable Omega-ratio objective (full-distribution).

Same lag-aware, interior-only construction as H650, but the pool weights are fit to maximize the Omega
ratio at a 0% daily threshold -- Omega(0) = sum(gains) / sum(|losses|) -- on the 1-day-lagged signal.
A fully-distributional point on the deployable frontier: every moment enters, not just the first two.

Run: uv run python scripts/run_h657.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from set8_lev import (
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
THETA = 0.0  # daily Omega threshold


def omega_ratio(r, theta=THETA):
    """Omega ratio: probability-weighted gains-above-threshold / losses-below-threshold."""
    gains = np.maximum(r - theta, 0.0).sum()
    losses = np.maximum(theta - r, 0.0).sum()
    return gains / losses if losses > 0 else 0.0


def lev_omega_neg(mat, idx, tv, cap):
    def neg(w):
        return -omega_ratio(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))

    return neg


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print("H657: DeployableOmega657 -- deployable Omega-ratio objective\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    print("\nInterior-only candidates, weights FIT (Omega) on the lagged signal:")
    cands = {}
    for thr in [15.0, 16.0, 17.0]:
        for asym in (False, True):
            tv, cap = lagged_caps(vix, idx, thr, asym)
            w = slsqp(lev_omega_neg(mat, idx, tv, cap), w0, n)
            r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
            om, sh = omega_ratio(r), portfolio_metrics(r)[0]
            key = f"t{thr:.0f}w20{'-asym' if asym else ''}"
            cands[key] = (-om, w, tv, cap, thr, asym)
            print(f"  {key}: Omega={om:.3f} Sh={sh:.3f}")

    best_key = min(cands, key=lambda k: cands[k][0])  # highest Omega
    _, w, tv, cap, thr, asym = cands[best_key]
    print(f"\nDeployable Omega champion config: {best_key}")
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)
    om = omega_ratio(r)

    oms = []
    for seed in range(5):
        ws = np.random.RandomState(seed).dirichlet(np.ones(n))
        wr = slsqp(lev_omega_neg(mat, idx, tv, cap), ws, n)
        oms.append(round(float(omega_ratio(apply_leverage(mat @ wr, idx, tv, cap, LOOKBACK))), 4))
    print(f"Restart Omegas: {oms} (range {max(oms) - min(oms):.4f})")

    sens = {}
    for d in [-1.0, 0.0, 1.0]:
        tvd, capd = lagged_caps(vix, idx, thr + d, asym)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, tvd, capd, LOOKBACK))[0]), 3
        )
    print(f"Sensitivity (Sharpe): {sens}")
    print(f"Omega {om:.3f} Sharpe {sh:.3f} DD {dd:.2%}; vs H650 Sharpe (1.979): {sh - 1.979:+.3f}")

    finalize_run(
        "657",
        "deployable-omega",
        "DeployableOmega657",
        "Set14 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable Omega ratio: maximize Omega(0)=sum(gains)/sum(|losses|) on the 1-day-LAGGED signal, "
            f"interior-only grid. Config {best_key}. 5-start Omegas {oms} (range {max(oms) - min(oms):.4f}). "
            f"Sensitivity {sens}. Omega {om:.3f}, Sharpe {sh:.3f}, Calmar {cal:.3f}, Sortino {so:.3f}. "
            f"vs H650 Sharpe (1.979): {sh - 1.979:+.3f}."
        ),
        extra={
            "objective": "deployable-lev-omega-lagged-fit-interior",
            "champion_config": best_key,
            "omega_threshold": THETA,
            "omega": round(float(om), 4),
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "restart_omegas": oms,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableOmega657",
    )


if __name__ == "__main__":
    main()
