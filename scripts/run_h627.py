"""H627: EWMAVolLeverage -- EWMA vol estimate instead of trailing rolling std.

Sizes leverage off an exponentially-weighted vol (span 20) rather than equal-weighted rolling std,
under the VIX<16-and-not-rising regime. EWMA reacts faster to vol spikes. Bull cap=2.5x/bear 1.0x, tv=6.5%.

Run: uv run python scripts/run_h627.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from set8_lev import (
    BORROW_COST_ANNUAL,
    TRADING_DAYS,
    finalize_run,
    mdr_warm_start,
    slsqp,
    two_state_cap,
    vix_momentum_mask,
)
from set8_pool import build_pool

TARGET_VOL = 0.065
SPAN = 20


def apply_leverage_ewma(r_arr, idx, cap):
    r = pd.Series(r_arr, index=idx)
    rv = r.ewm(span=SPAN, min_periods=SPAN).std().shift(1) * np.sqrt(TRADING_DAYS)
    lev = (TARGET_VOL / rv).clip(upper=cap)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    borrow = (lev - 1.0).clip(lower=0.0) * BORROW_COST_ANNUAL / TRADING_DAYS
    return (lev * r - borrow).to_numpy()


def main():
    print("H627: EWMAVolLeverage\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_momentum_mask(vix, idx, 16.0, 20), 2.5, 1.0, idx)

    def neg(w):
        r = apply_leverage_ewma(mat @ w, idx, cap)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage_ewma(mat @ w, idx, cap)
    finalize_run(
        "627",
        "ewma-vol-leverage",
        "EWMAVolLeverage",
        "Set10 Leverage Mechanics",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"EWMA vol estimate (span {SPAN}) instead of rolling std, VIX<16-and-not-rising regime, tv={TARGET_VOL:.1%}. "
            "EWMA reacts faster to vol spikes. H620 (rolling 20d) ref 2.472."
        ),
        extra={
            "objective": "lev-sharpe-ewma-vol-vix-momentum",
            "ewma_span": SPAN,
            "leverage_target_vol": TARGET_VOL,
        },
        label="EWMA-vol",
    )


if __name__ == "__main__":
    main()
