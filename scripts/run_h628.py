"""H628: DrawdownThrottleVIXMom -- regime cap x continuous drawdown throttle.

Layers a path-dependent throttle on the VIX-momentum regime: the effective leverage cap is reduced
continuously by the base portfolio's running drawdown (cap *= clip(1 + dd*k, floor, 1), dd from prior day).
Two-layer de-risking. tv=6.5%, lb=20d.

Run: uv run python scripts/run_h628.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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
K = 3.0
FLOOR = 0.4


def throttle(base, idx):
    cum = np.cumprod(1.0 + base)
    dd = cum / np.maximum.accumulate(cum) - 1.0  # <= 0
    t = np.clip(1.0 + dd * K, FLOOR, 1.0)
    return pd.Series(t, index=idx).shift(1).fillna(1.0)


def main():
    print("H628: DrawdownThrottleVIXMom\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    regime_cap = two_state_cap(vix_momentum_mask(vix, idx, 16.0, 20), 2.5, 1.0, idx)

    def neg(w):
        base = mat @ w
        cap = regime_cap * throttle(base, idx)
        r = apply_leverage(base, idx, TARGET_VOL, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    base = mat @ w
    cap = regime_cap * throttle(base, idx)
    r = apply_leverage(base, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "628",
        "drawdown-throttle-vix-mom",
        "DrawdownThrottleVIXMom",
        "Set10 Leverage Mechanics",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"VIX-momentum regime cap x continuous drawdown throttle (clip(1+dd*{K},{FLOOR},1) on base DD, lagged). "
            "Two-layer de-risking. H620 (regime only) ref 2.472. H557 found ratchet degrades low-DD bases."
        ),
        extra={
            "objective": "lev-sharpe-vix-momentum-dd-throttle",
            "throttle_k": K,
            "throttle_floor": FLOOR,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="DD-throttle",
    )


if __name__ == "__main__":
    main()
