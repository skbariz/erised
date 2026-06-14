"""H635: VIXMomDDFloor -- DD floor via post-hoc tv reduction at the VIX-momentum optimum.

Optimize lev-Sharpe weights at tv=6.5% (VIX<16-and-not-rising), then lower target_vol until realized
MaxDD <= 7% (no in-objective penalty -- H590 showed penalty fails). Target Sh>2.0 at MaxDD<=-7%.

Run: uv run python scripts/run_h635.py
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

BASE_TV = 0.065
LOOKBACK = 20
DD_FLOOR = 0.07


def main():
    print("H635: VIXMomDDFloor\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_momentum_mask(vix, idx, 16.0, 20), 2.5, 1.0, idx)

    def neg(w):
        r = apply_leverage(mat @ w, idx, BASE_TV, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)

    chosen = None
    for tv in np.round(np.arange(BASE_TV, 0.029, -0.0025), 4):
        r = apply_leverage(mat @ w, idx, float(tv), cap, LOOKBACK)
        if abs(portfolio_metrics(r)[3]) <= DD_FLOOR:
            chosen = (float(tv), r)
            break
    if chosen is None:
        chosen = (0.030, apply_leverage(mat @ w, idx, 0.030, cap, LOOKBACK))
    tv, r = chosen
    print(f"DD-floor tv={tv:.4f}")
    finalize_run(
        "635",
        "vix-mom-dd-floor",
        "VIXMomDDFloor",
        "Set10 Frontier Objectives",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"DD floor via post-hoc tv reduction at VIX<16-and-not-rising (weights at tv={BASE_TV:.1%}, "
            f"lowered to {tv:.4f} for MaxDD<={DD_FLOOR:.0%}). No in-objective penalty (H590 failed). H598 ref Sh 2.037."
        ),
        extra={
            "objective": "lev-sharpe-then-dd-floor-vix-momentum",
            "vix_threshold": 16.0,
            "base_target_vol": BASE_TV,
            "floored_target_vol": tv,
            "dd_floor": DD_FLOOR,
            "leverage_lookback": LOOKBACK,
        },
        label="VIXMom-DDFloor",
    )


if __name__ == "__main__":
    main()
