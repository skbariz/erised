"""H617: TurnoverCostVIX15 -- VIX<15 champion with explicit leverage-rebalance cost.

The vol-target leverage multiplier changes daily; rebalancing it incurs cost. Charges
|lev_t - lev_{t-1}| * 1.5bps on top of borrow. Capacity/cost realism for the leverage overlay.

Run: uv run python scripts/run_h617.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
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
    vix_bull_mask,
)
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20
TURNOVER_BPS = 0.00015  # 1.5bps per unit leverage change


def lev_with_turnover(r_arr, idx, cap):
    r = pd.Series(r_arr, index=idx)
    rv = r.rolling(LOOKBACK, min_periods=LOOKBACK).std().shift(1) * np.sqrt(TRADING_DAYS)
    lev = (TARGET_VOL / rv).clip(upper=cap)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    borrow = (lev - 1.0).clip(lower=0.0) * BORROW_COST_ANNUAL / TRADING_DAYS
    turnover = lev.diff().abs().fillna(0.0) * TURNOVER_BPS
    return (lev * r - borrow - turnover).to_numpy()


def main():
    print("H617: TurnoverCostVIX15\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_bull_mask(vix, idx, 15.0), 2.5, 1.0, idx)

    def neg(w):
        r = lev_with_turnover(mat @ w, idx, cap)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = lev_with_turnover(mat @ w, idx, cap)
    finalize_run(
        "617",
        "turnover-cost-vix15",
        "TurnoverCostVIX15",
        "Set9 Realism",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"VIX<15 champion + leverage-rebalance turnover cost (|d_lev|*{TURNOVER_BPS * 1e4:.1f}bps) on top of "
            "borrow. Capacity/cost realism for the daily vol-target overlay. H600/H591 (no turnover cost) ref 2.237."
        ),
        extra={
            "objective": "lev-sharpe-vix15-turnover-cost",
            "vix_threshold": 15.0,
            "turnover_bps": TURNOVER_BPS * 1e4,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="Turnover-VIX15",
    )


if __name__ == "__main__":
    main()
