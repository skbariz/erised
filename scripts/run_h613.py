"""H613: VIX15Sortino -- downside-semideviation vol leverage at VIX<15.

Combines the VIX<15 optimum with H583/H595's downside vol targeting. CAGR frontier with the
tighter bull threshold. Bull cap=2.5x, bear cap=1.0x, tv=6.5%.

Run: uv run python scripts/run_h613.py
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

TARGET_VOL = 0.065
LOOKBACK = 20


def main():
    print("H613: VIX15Sortino\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_bull_mask(vix, idx, 15.0), 2.5, 1.0, idx)

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK, downside=True)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK, downside=True)
    finalize_run(
        "613",
        "vix15-sortino",
        "VIX15Sortino",
        "Set9 Frontier Objectives",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "Downside semi-deviation vol leverage at VIX<15 (bull 2.5x/bear 1.0x), tv=6.5%. "
            "H595 (VIX<20 Sortino) ref: Sh 2.200, CAGR 15.82%. H583 CAGR ref 17.60%."
        ),
        extra={
            "objective": "lev-sharpe-downside-vol-vix15",
            "vix_threshold": 15.0,
            "downside_vol": True,
            "leverage_max_bull": 2.5,
            "leverage_max_bear": 1.0,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="VIX15-Sortino",
    )


if __name__ == "__main__":
    main()
