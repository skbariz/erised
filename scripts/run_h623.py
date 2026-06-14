"""H623: VIXRateOfChangeContinuous -- de-risk in proportion to how fast VIX is rising.

Continuous cap = clip(3.0 - K*max(0, VIX 21d %change), 1.0, 3.0). Cleanest test of whether the
*magnitude* of VIX rise carries more than the binary not-rising flag (H620). tv=6.5%, lb=20d.

Run: uv run python scripts/run_h623.py
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
K = 5.0


def main():
    print("H623: VIXRateOfChangeContinuous\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    vixr = vix.reindex(idx).ffill().bfill()
    roc = vixr.pct_change(21).clip(lower=0.0).fillna(0.0)  # only rising VIX de-risks
    cap = (3.0 - K * roc).clip(lower=1.0, upper=3.0)
    print(f"Cap: mean={cap.mean():.2f} min={cap.min():.2f} (K={K})")

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "623",
        "vix-roc-continuous",
        "VIXRateOfChangeContinuous",
        "Set10 VIX-Momentum",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Continuous cap=clip(3.0-{K}*max(0,VIX 21d %chg),1,3): leverage falls with VIX rise speed. "
            f"Mean cap {cap.mean():.2f}, tv={TARGET_VOL:.1%}. Tests magnitude vs H620's binary not-rising flag (2.472)."
        ),
        extra={
            "objective": "lev-sharpe-vix-roc-continuous",
            "k": K,
            "roc_window": 21,
            "mean_cap": round(float(cap.mean()), 3),
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="VIX-RoC",
    )


if __name__ == "__main__":
    main()
