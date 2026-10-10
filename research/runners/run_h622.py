"""H622: VIXMomentumAsymmetric -- VIX-momentum signal + asymmetric tv+cap.

Combines the two strongest levers: VIX-momentum regime (H620, 2.472) + asymmetric tv+cap (H592, Calmar 2.264).
Bull (VIX<16 AND not-rising): tv=8%/cap=2.5x; bear: tv=4%/cap=1.0x.

Run: uv run python scripts/run_h622.py
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
    slsqp,
    two_state_cap,
    vix_momentum_mask,
)
from set8_pool import build_pool

LOOKBACK = 20


def main():
    print("H622: VIXMomentumAsymmetric\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    bull = vix_momentum_mask(vix, idx, 16.0, 20)
    tv = two_state_cap(bull, 0.08, 0.04, idx)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    print(f"Bull fraction (VIX<16 & not-rising): {bull.mean():.1%}")

    def neg(w):
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    finalize_run(
        "622",
        "vix-momentum-asymmetric",
        "VIXMomentumAsymmetric",
        "Set10 VIX-Momentum",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "VIX<16 & not-rising bull: tv=8%/cap=2.5x; bear: tv=4%/cap=1.0x. Combines H620 VIX-momentum "
            "(2.472) + H592 asymmetric (Calmar 2.264). Tests whether the two strongest levers compose."
        ),
        extra={
            "objective": "lev-sharpe-vix-momentum-asymmetric",
            "vix_threshold": 16.0,
            "window": 20,
            "bull_tv": 0.08,
            "bull_cap": 2.5,
            "bear_tv": 0.04,
            "bear_cap": 1.0,
            "leverage_lookback": LOOKBACK,
        },
        label="VIXMom-asym",
    )


if __name__ == "__main__":
    main()
