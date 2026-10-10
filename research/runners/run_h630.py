"""H630: VolOfVolScaling -- scale leverage down when VIX itself is volatile (uncertain regime).

Effective cap = VIX-momentum regime cap x (1 - 0.4 * vol-of-vol percentile), where vol-of-vol is VIX's
own 20d std ranked by expanding percentile. De-risk when the regime signal is itself unstable. tv=6.5%.

Run: uv run python scripts/run_h630.py
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

TARGET_VOL = 0.065
LOOKBACK = 20


def main():
    print("H630: VolOfVolScaling\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    vixr = vix.reindex(idx).ffill().bfill()
    vov = vixr.rolling(20, min_periods=10).std()
    vov_pct = vov.expanding(min_periods=252).rank(pct=True).shift(1).fillna(0.5)
    scale = (1.0 - 0.4 * vov_pct).clip(0.5, 1.0)
    regime_cap = two_state_cap(vix_momentum_mask(vix, idx, 16.0, 20), 2.5, 1.0, idx)
    cap = regime_cap * scale
    print(f"Vol-of-vol scale: mean={scale.mean():.2f} min={scale.min():.2f}")

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "630",
        "vol-of-vol-scaling",
        "VolOfVolScaling",
        "Set10 Leverage Mechanics",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "VIX-momentum regime cap x (1-0.4*vol-of-vol pctile); de-risk when VIX's own 20d vol is high. "
            f"Mean scale {scale.mean():.2f}, tv={TARGET_VOL:.1%}. H620 (no vov scale) ref 2.472."
        ),
        extra={
            "objective": "lev-sharpe-vix-momentum-vol-of-vol",
            "vov_window": 20,
            "vov_weight": 0.4,
            "mean_scale": round(float(scale.mean()), 3),
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="VolOfVol",
    )


if __name__ == "__main__":
    main()
