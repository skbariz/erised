"""H653: DeployableInverseVol653 -- non-optimized inverse-vol deployable baseline.

Pool weights are set by inverse realized volatility (risk-parity weighting on each sleeve's available
history) -- no SLSQP, so the weights cannot be overfit -- with the same 1-day-lagged interior-grid leverage
as H650. Isolates the leverage-mechanism contribution and places a principled heuristic on the
weighting-scheme ladder between equal-weight (H648) and optimized (H650).

Run: uv run python scripts/run_h653.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from set8_lev import (
    apply_leverage,
    finalize_run,
    portfolio_metrics,
    two_state_cap,
    vix_momentum_mask,
)
from set8_pool import build_pool

LOOKBACK = 20


def inverse_vol_weights(clean):
    """Inverse-volatility weights from each column's available-history (NaN-aware) std."""
    vols = clean.std().to_numpy()
    inv = np.where(vols > 0, 1.0 / vols, 0.0)
    s = inv.sum()
    return inv / s if s > 0 else inv


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print("H653: DeployableInverseVol653 -- non-optimized inverse-vol baseline\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx = clean.index

    w = inverse_vol_weights(clean)  # deterministic, un-overfittable
    print(f"\nInverse-vol weights (deterministic): {int((w > 0.005).sum())} sleeves > 0.5%")

    print("\nInterior-only candidates (fixed inverse-vol weights, lagged signal):")
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
    print(f"\nDeployable inverse-vol champion config: {best_key}")
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
        f"Sharpe {sh:.3f}; vs equal-weight H648 (1.383): {sh - 1.383:+.3f}; "
        f"vs optimized H650 (1.980): {sh - 1.980:+.3f}"
    )

    finalize_run(
        "653",
        "deployable-inverse-vol",
        "DeployableInverseVol653",
        "Set13 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable inverse-vol (non-optimized, un-overfittable) baseline on the 1-day-LAGGED signal, "
            f"interior-only grid. Config {best_key}. Deterministic risk-parity weights. Sensitivity {sens}. "
            f"Sharpe {sh:.3f}, Calmar {cal:.3f}, Sortino {so:.3f}. vs equal-weight H648 (1.383): "
            f"{sh - 1.383:+.3f}; vs optimized H650 (1.980): {sh - 1.980:+.3f}. Weighting-scheme ladder rung."
        ),
        extra={
            "objective": "deployable-inverse-vol-no-optimization-interior",
            "champion_config": best_key,
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
            "weights_deterministic": True,
        },
        label="DeployableInverseVol653",
    )


if __name__ == "__main__":
    main()
