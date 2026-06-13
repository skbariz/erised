"""H606: ConcentratedPoolVIX15 -- VIX<15 regime on the top-12 standalone-Sharpe sub-pool.

Tests the H516 small-pool finding (focused pool > full pool for MDR-style optimization) under the
champion's VIX<15 leverage. Bull cap=2.5x, bear cap=1.0x, tv=6.5%.

Run: uv run python scripts/run_h606.py
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
    per_strategy_sharpe,
    slsqp,
    two_state_cap,
    vix_bull_mask,
)
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20
BULL_CAP, BEAR_CAP = 2.5, 1.0
TOP_K = 12


def main():
    print("H606: ConcentratedPoolVIX15\n" + "=" * 64)
    pool = build_pool()
    clean = pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx = clean.index

    sh_by = per_strategy_sharpe(clean)
    top = sorted(sh_by, key=sh_by.get, reverse=True)[:TOP_K]
    print(f"Top-{TOP_K} by standalone Sharpe: {[(t, round(sh_by[t], 2)) for t in top]}")
    sub = clean[top].fillna(0.0).to_numpy(dtype="float64")
    n = sub.shape[1]
    cap = two_state_cap(vix_bull_mask(vix, idx, 15.0), BULL_CAP, BEAR_CAP, idx)

    def neg(w):
        r = apply_leverage(sub @ w, idx, TARGET_VOL, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(sub, n), n)
    r = apply_leverage(sub @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "606",
        "concentrated-pool-vix15",
        "ConcentratedPoolVIX15",
        "Set9 Pool Robustness",
        r,
        idx,
        w,
        top,
        folds,
        spy,
        notes=(
            f"Top-{TOP_K} standalone-Sharpe sub-pool {top} + VIX<15 regime (bull {BULL_CAP}x/bear {BEAR_CAP}x). "
            f"Tests H516 small-pool finding under champion leverage. H600/H591 (full 55-pool) ref 2.237."
        ),
        extra={
            "objective": "lev-sharpe-concentrated-vix15",
            "top_k": TOP_K,
            "selected": top,
            "leverage_max_bull": BULL_CAP,
            "leverage_max_bear": BEAR_CAP,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="Concentrated-12",
    )


if __name__ == "__main__":
    main()
