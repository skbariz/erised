"""H611: VIX15Asymmetric -- asymmetric tv+cap at the VIX<15 threshold.

Combines the two Batch4 champions: VIX<15 (H591, Sharpe 2.237) + asymmetric tv+cap (H592, Calmar 2.264).
Bull (VIX<15): tv=8%/cap=2.5x; bear (VIX>=15): tv=4%/cap=1.0x.

Run: uv run python scripts/run_h611.py
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
    slsqp,
    two_state_cap,
    vix_bull_mask,
)
from set8_pool import build_pool

LOOKBACK = 20


def main():
    print("H611: VIX15Asymmetric\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    bull = vix_bull_mask(vix, idx, 15.0)
    tv = two_state_cap(bull, 0.08, 0.04, idx)
    cap = two_state_cap(bull, 2.5, 1.0, idx)

    def neg(w):
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    finalize_run(
        "611",
        "vix15-asymmetric",
        "VIX15Asymmetric",
        "Set9 Frontier Objectives",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "VIX<15 bull: tv=8%/cap=2.5x; bear: tv=4%/cap=1.0x. Combines H591 (VIX<15, 2.237) + "
            "H592 (asymmetric, Calmar 2.264). Tests the two Batch4 champions at their joint best params."
        ),
        extra={
            "objective": "lev-sharpe-vix15-asymmetric",
            "vix_threshold": 15.0,
            "bull_tv": 0.08,
            "bull_cap": 2.5,
            "bear_tv": 0.04,
            "bear_cap": 1.0,
            "leverage_lookback": LOOKBACK,
        },
        label="VIX15-asym",
    )


if __name__ == "__main__":
    main()
