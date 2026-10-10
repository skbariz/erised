"""H618: RegimeLagVIX15 -- lag the VIX regime signal to test execution realism.

The champion sets day-t leverage from day-t VIX. This lags the regime mask by k days (signal known
only with delay). Sweeps k in {1,3,5}; headline = k=1 (realistic next-day execution). Bull cap=2.5x/bear 1.0x.

Run: uv run python scripts/run_h618.py
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
    portfolio_metrics,
    slsqp,
    two_state_cap,
    vix_bull_mask,
)
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20
LAGS = [1, 3, 5]


def main():
    print("H618: RegimeLagVIX15\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    bull0 = vix_bull_mask(vix, idx, 15.0)
    w0 = mdr_warm_start(mat, n)

    sweep, headline = {}, None
    for k in LAGS:
        bull = bull0.shift(k).fillna(True)
        cap = two_state_cap(bull, 2.5, 1.0, idx)

        def neg(w, cap=cap):
            r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
            s = r.std() * np.sqrt(TRADING_DAYS)
            return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

        w = slsqp(neg, w0, n)
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        sh = portfolio_metrics(r)[0]
        sweep[f"lag{k}"] = round(float(sh), 4)
        print(f"  lag={k}d: Sh={sh:.3f}")
        if k == 1:
            headline = (w, r)

    w, r = headline
    finalize_run(
        "618",
        "regime-lag-vix15",
        "RegimeLagVIX15",
        "Set9 Realism",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"VIX<15 regime signal lagged k days (next-day+ execution). Sharpe by lag: {sweep}. "
            f"Headline = 1-day lag (realistic execution). H600/H591 (0-lag) ref 2.237; small degradation => robust."
        ),
        extra={
            "objective": "lev-sharpe-vix15-lagged-regime",
            "vix_threshold": 15.0,
            "lag_sweep": sweep,
            "headline_lag": 1,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="RegimeLag1-VIX15",
    )


if __name__ == "__main__":
    main()
