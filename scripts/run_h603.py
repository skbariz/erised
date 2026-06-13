"""H603: CompositeRegimeScore -- fuse VIX + trend + cross-asset corr into a regime score.

score = mean of {VIX<15, SPY>200d-MA, SPY-TLT 60d corr<0}; leverage cap scales 1.0 -> 2.5 with score.
Tests whether multi-signal fusion beats the single VIX<15 signal (H600/H591: 2.237).

Run: uv run python scripts/run_h603.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from set8_lev import TRADING_DAYS, apply_leverage, finalize_run, mdr_warm_start, slsqp
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20


def main():
    print("H603: CompositeRegimeScore\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    prices = pool["prices_aa"]
    idx, n = clean.index, mat.shape[1]

    vixr = vix.reindex(idx).ffill().bfill()
    spy_close = spy["adj_close"].reindex(idx).ffill()
    spy_ma = spy_close.rolling(200, min_periods=100).mean()
    spy_ret = spy_close.pct_change()
    tlt = prices["tlt_close"].reindex(idx).ffill()
    tlt_ret = tlt.pct_change()
    corr60 = spy_ret.rolling(60, min_periods=40).corr(tlt_ret)

    vix_bull = (vixr < 15).astype(float)
    ma_bull = (spy_close > spy_ma).astype(float)
    corr_bull = (corr60 < 0).astype(float)
    score = ((vix_bull + ma_bull + corr_bull) / 3.0).fillna(0.5)
    cap = (1.0 + 1.5 * score).clip(1.0, 2.5)
    print(f"Regime score mean={score.mean():.2f}; cap mean={cap.mean():.2f}")

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "603",
        "composite-regime-score",
        "CompositeRegimeScore",
        "Set9 Regime Forms",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "Composite regime score = mean{VIX<15, SPY>200dMA, SPY-TLT 60d corr<0}; "
            f"cap=clip(1+1.5*score,1,2.5), tv={TARGET_VOL:.1%}. Mean cap {cap.mean():.2f}. "
            "H600/H591 (VIX<15 only) ref 2.237."
        ),
        extra={
            "objective": "lev-sharpe-composite-regime",
            "components": "vix15+spy200ma+spytlt_corr",
            "mean_cap": round(float(cap.mean()), 3),
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="Composite",
    )


if __name__ == "__main__":
    main()
