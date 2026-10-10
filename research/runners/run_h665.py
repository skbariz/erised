"""H665: DeployableEnsembleCapstone665 -- equal-weight blend of the Set13/14 deployable champion allocations.

The Set14 capstone. At the frontier config t15w20 (1-day-lagged regime), fit the six deployable champion
weight vectors -- Sharpe (H650), Sortino (H651), CAGR-with-floor (H655), return-to-CVaR (H656),
equal-risk-contribution (H659), Ledoit-Wolf GMV (H660) -- and average them into one deployable allocation,
then apply the shared lagged-interior leverage once. A fund-of-objectives: diversifying across objective-
specification error, the allocation analogue of the H610 stream ensemble.

Run: uv run python scripts/run_h665.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from run_h650 import lagged_caps, lev_sharpe_neg
from run_h651 import lev_sortino_neg
from run_h655 import lev_cagr_ddfloor_neg
from run_h656 import lev_cvar_neg
from run_h659 import erc_weights
from run_h660 import gmv_weights, ledoit_wolf_cov
from set8_lev import (
    apply_leverage,
    finalize_run,
    mdr_warm_start,
    portfolio_metrics,
    slsqp,
)
from set8_pool import build_pool

LOOKBACK = 20


def blend_weights(ws):
    """Equal-weight average of weight vectors, renormalized to sum to 1."""
    w = np.mean(np.vstack(ws), axis=0)
    return w / w.sum()


def main():
    print(
        "H665: DeployableEnsembleCapstone665 -- blend of deployable champion allocations\n"
        + "=" * 64
    )
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    thr, asym = 15.0, False  # frontier-champion config
    tv, cap = lagged_caps(vix, idx, thr, asym)

    print("\nFitting the six deployable champion allocations at t15w20...")
    components = {
        "H650-sharpe": slsqp(lev_sharpe_neg(mat, idx, tv, cap), w0, n),
        "H651-sortino": slsqp(lev_sortino_neg(mat, idx, tv, cap), w0, n),
        "H655-cagr-floor": slsqp(lev_cagr_ddfloor_neg(mat, idx, tv, cap), w0, n),
        "H656-cvar": slsqp(lev_cvar_neg(mat, idx, tv, cap), w0, n),
        "H659-erc": erc_weights(mat),
        "H660-lw-gmv": gmv_weights(ledoit_wolf_cov(mat)[0]),
    }
    for k, wc in components.items():
        sh_c = portfolio_metrics(apply_leverage(mat @ wc, idx, tv, cap, LOOKBACK))[0]
        print(f"  {k:18}: component Sharpe {sh_c:.3f}")

    w = blend_weights(list(components.values()))
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)
    print(
        f"\nBlended deployable capstone: Sharpe {sh:.3f} Calmar {cal:.3f} Sortino {so:.3f} DD {dd:.2%}"
    )

    sens = {}
    for d in [-1.0, 0.0, 1.0]:
        tvd, capd = lagged_caps(vix, idx, thr + d, asym)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, tvd, capd, LOOKBACK))[0]), 3
        )
    print(f"Sensitivity (Sharpe): {sens}; vs H650 (1.979): {sh - 1.979:+.3f}")

    finalize_run(
        "665",
        "deployable-ensemble-capstone",
        "DeployableEnsembleCapstone665",
        "Set14 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Set14 capstone: equal-weight blend of six deployable champion allocations (Sharpe/Sortino/"
            f"CAGR-floor/CVaR/ERC/LW-GMV) at config t{thr:.0f}w20, 1-day-LAGGED signal, leverage applied "
            f"once to the blended weights. Sensitivity {sens}. Sharpe {sh:.3f}, Calmar {cal:.3f}, "
            f"Sortino {so:.3f}, CAGR {cg:.2%}, MaxDD {dd:.2%}. vs H650 (1.979): {sh - 1.979:+.3f}. "
            f"Diversifies across objective-specification error (allocation analogue of the H610 ensemble)."
        ),
        extra={
            "objective": "deployable-ensemble-of-objective-allocations-lagged",
            "champion_config": f"t{thr:.0f}w20",
            "components": list(components.keys()),
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableEnsembleCapstone665",
    )


if __name__ == "__main__":
    main()
