"""H643: DeployableLaggedChampion -- H622 asymmetric mechanism with the regime signal lagged 1 day.

Decisions use only VIX known at the prior close (realistic next-day execution). The honest "what you'd
actually realize" champion. H638 showed VIX-momentum loses ~0.5 Sharpe under a 1-day lag.

Run: uv run python scripts/run_h643.py
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
    print("H643: DeployableLaggedChampion\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    bull = vix_momentum_mask(vix, idx, 16.0, 20).shift(1).fillna(True)  # 1-day execution lag
    tv = two_state_cap(bull, 0.08, 0.04, idx)
    cap = two_state_cap(bull, 2.5, 1.0, idx)

    def neg(w):
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    finalize_run(
        "643",
        "deployable-lagged-champion",
        "DeployableLaggedChampion",
        "Set11 Defensible",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "H622 asymmetric VIX-momentum mechanism with regime signal lagged 1 day (next-day execution baked in). "
            "Honest deployable Sharpe. H622 (0-lag) ref 2.642; gap = execution cost of the signal."
        ),
        extra={
            "objective": "lev-sharpe-vixmom-asym-1day-lag",
            "vix_threshold": 16.0,
            "window": 20,
            "regime_lag_days": 1,
            "leverage_lookback": LOOKBACK,
        },
        label="Deployable-Lagged",
    )


if __name__ == "__main__":
    main()
