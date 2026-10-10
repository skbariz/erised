"""H636: RealBorrowVIXMom -- VIX-momentum champion + realistic Fed Funds borrow cost.

Charges leverage at the historical Fed Funds step-function + 50bps spread under VIX<16-and-not-rising.
Net-of-financing Sharpe for the terminal champion. tv=6.5%, lb=20d.

Run: uv run python scripts/run_h636.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
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

TARGET_VOL = 0.065
LOOKBACK = 20
SPREAD = 0.0050
FED_FUNDS_STEPS = [
    ("1993-01-01", "2001-01-01", 0.0550),
    ("2001-01-01", "2004-06-01", 0.0150),
    ("2004-06-01", "2008-12-01", 0.0450),
    ("2008-12-01", "2015-12-01", 0.0010),
    ("2015-12-01", "2019-07-01", 0.0200),
    ("2019-07-01", "2020-03-01", 0.0175),
    ("2020-03-01", "2022-03-01", 0.0010),
    ("2022-03-01", "2099-01-01", 0.0500),
]


def borrow_series(index):
    cost = pd.Series(SPREAD, index=index)
    for start, end, rate in FED_FUNDS_STEPS:
        cost[(index >= start) & (index < end)] = rate + SPREAD
    return cost


def main():
    print("H636: RealBorrowVIXMom\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_momentum_mask(vix, idx, 16.0, 20), 2.5, 1.0, idx)
    borrow = borrow_series(idx)

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK, borrow_annual=borrow)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK, borrow_annual=borrow)
    finalize_run(
        "636",
        "real-borrow-vix-mom",
        "RealBorrowVIXMom",
        "Set10 Realism",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"VIX-momentum champion + Fed Funds step borrow (+{SPREAD * 1e4:.0f}bps). Mean borrow "
            f"{borrow.mean():.3%}/yr. H620 (flat 50bps) ref 2.472. H616 (VIX<15 real borrow) was 2.063."
        ),
        extra={
            "objective": "lev-sharpe-vix-momentum-real-borrow",
            "vix_threshold": 16.0,
            "mean_borrow_annual": round(float(borrow.mean()), 4),
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="RealBorrow-VIXMom",
    )


if __name__ == "__main__":
    main()
