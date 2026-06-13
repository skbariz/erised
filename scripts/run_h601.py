"""H601: ContinuousVIXLeverage -- leverage cap as a smooth function of VIX.

cap = clip(3.5 - VIX/10, 1.0, 3.0): cap=3.0 at VIX<=5, 2.5 at VIX=10, 1.5 at VIX=20, 1.0 at VIX>=25.
Tests whether a continuous VIX response beats the discrete VIX<15 step (H600/H591: 2.237).

Run: uv run python scripts/run_h601.py
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
    print("H601: ContinuousVIXLeverage\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    vixr = vix.reindex(idx).ffill().bfill()
    cap = (3.5 - vixr / 10.0).clip(lower=1.0, upper=3.0)
    print(f"Continuous cap: mean={cap.mean():.2f} min={cap.min():.2f} max={cap.max():.2f}")

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "601",
        "continuous-vix-leverage",
        "ContinuousVIXLeverage",
        "Set9 Regime Forms",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"cap=clip(3.5-VIX/10,1.0,3.0), tv={TARGET_VOL:.1%}/lb={LOOKBACK}d. Continuous VIX response. "
            f"Mean cap {cap.mean():.2f}. H600/H591 (VIX<15 step) ref 2.237."
        ),
        extra={
            "objective": "lev-sharpe-continuous-vix-cap",
            "cap_formula": "clip(3.5-VIX/10,1,3)",
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
            "mean_cap": round(float(cap.mean()), 3),
        },
        label="Continuous-VIX",
    )


if __name__ == "__main__":
    main()
