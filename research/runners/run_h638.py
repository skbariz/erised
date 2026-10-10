"""H638: RegimeLagVIXMom -- lag the VIX-momentum signal to test execution realism.

Lags the VIX<16-and-not-rising mask by k days (signal known only with delay). Sweeps k in {1,3,5};
headline = k=1 (realistic next-day execution). Bull cap=2.5x/bear 1.0x, tv=6.5%.

Run: uv run python scripts/run_h638.py
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
    vix_momentum_mask,
)
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20
LAGS = [1, 3, 5]


def main():
    print("H638: RegimeLagVIXMom\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    bull0 = vix_momentum_mask(vix, idx, 16.0, 20)
    w0 = mdr_warm_start(mat, n)

    sweep, headline = {}, None
    for k in LAGS:
        cap = two_state_cap(bull0.shift(k).fillna(True), 2.5, 1.0, idx)

        def neg(w, cap=cap):
            r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
            s = r.std() * np.sqrt(TRADING_DAYS)
            return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

        w = slsqp(neg, w0, n)
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        sweep[f"lag{k}"] = round(float(portfolio_metrics(r)[0]), 4)
        print(f"  lag={k}d: Sh={sweep[f'lag{k}']}")
        if k == 1:
            headline = (w, r)

    w, r = headline
    finalize_run(
        "638",
        "regime-lag-vix-mom",
        "RegimeLagVIXMom",
        "Set10 Realism",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"VIX-momentum signal lagged k days. Sharpe by lag: {sweep}. Headline = 1-day lag (realistic). "
            f"H620 (0-lag) ref 2.472. H618 (VIX<15, 1d lag) was 1.977."
        ),
        extra={
            "objective": "lev-sharpe-vix-momentum-lagged",
            "vix_threshold": 16.0,
            "lag_sweep": sweep,
            "headline_lag": 1,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="RegimeLag1-VIXMom",
    )


if __name__ == "__main__":
    main()
