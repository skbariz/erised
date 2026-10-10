"""H629: RatchetVIXMom -- H087-style tiered drawdown ratchet on VIX-momentum leverage.

Step-function de-risk vs H628's continuous throttle: base portfolio drawdown from its running peak
cuts the cap in tiers (dd<-5% -> 0.8x, <-10% -> 0.6x, <-15% -> 0.4x). Layered on VIX-momentum. tv=6.5%.

Run: uv run python scripts/run_h629.py
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


def ratchet(base, idx):
    cum = np.cumprod(1.0 + base)
    dd = cum / np.maximum.accumulate(cum) - 1.0
    tier = np.select([dd < -0.15, dd < -0.10, dd < -0.05], [0.4, 0.6, 0.8], default=1.0)
    return pd.Series(tier, index=idx).shift(1).fillna(1.0)


def main():
    print("H629: RatchetVIXMom\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    regime_cap = two_state_cap(vix_momentum_mask(vix, idx, 16.0, 20), 2.5, 1.0, idx)

    def neg(w):
        base = mat @ w
        cap = regime_cap * ratchet(base, idx)
        r = apply_leverage(base, idx, TARGET_VOL, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    base = mat @ w
    cap = regime_cap * ratchet(base, idx)
    r = apply_leverage(base, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "629",
        "ratchet-vix-mom",
        "RatchetVIXMom",
        "Set10 Leverage Mechanics",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "VIX-momentum regime cap x tiered ratchet (dd<-5/-10/-15% -> 0.8/0.6/0.4x, lagged). "
            "Step-function vs H628 continuous throttle. H620 (regime only) ref 2.472."
        ),
        extra={
            "objective": "lev-sharpe-vix-momentum-ratchet",
            "tiers": "5/10/15pct->0.8/0.6/0.4",
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="Ratchet",
    )


if __name__ == "__main__":
    main()
