"""H637: TurnoverVIXMom -- VIX-momentum champion + leverage-rebalance turnover cost.

Charges |lev_t - lev_{t-1}| * 1.5bps on top of borrow under VIX<16-and-not-rising. Capacity/cost realism
for the terminal champion. tv=6.5%, lb=20d.

Run: uv run python scripts/run_h637.py
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
LOOKBACK = 20
TURNOVER_BPS = 0.00015


def lev_with_turnover(r_arr, idx, cap):
    r = pd.Series(r_arr, index=idx)
    rv = r.rolling(LOOKBACK, min_periods=LOOKBACK).std().shift(1) * np.sqrt(TRADING_DAYS)
    lev = (TARGET_VOL / rv).clip(upper=cap)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    borrow = (lev - 1.0).clip(lower=0.0) * BORROW_COST_ANNUAL / TRADING_DAYS
    turnover = lev.diff().abs().fillna(0.0) * TURNOVER_BPS
    return (lev * r - borrow - turnover).to_numpy()


def main():
    print("H637: TurnoverVIXMom\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_momentum_mask(vix, idx, 16.0, 20), 2.5, 1.0, idx)

    def neg(w):
        r = lev_with_turnover(mat @ w, idx, cap)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = lev_with_turnover(mat @ w, idx, cap)
    finalize_run(
        "637",
        "turnover-vix-mom",
        "TurnoverVIXMom",
        "Set10 Realism",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"VIX-momentum champion + leverage-rebalance turnover cost (|d_lev|*{TURNOVER_BPS * 1e4:.1f}bps). "
            "Capacity/cost realism. H620 (no turnover) ref 2.472. H617 (VIX<15) was 2.198 (-0.04)."
        ),
        extra={
            "objective": "lev-sharpe-vix-momentum-turnover",
            "vix_threshold": 16.0,
            "turnover_bps": TURNOVER_BPS * 1e4,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="Turnover-VIXMom",
    )


if __name__ == "__main__":
    main()
