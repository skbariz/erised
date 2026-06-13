"""H602: VIXPercentileRegime -- bull = VIX below its expanding 40th percentile.

Adaptive regime with no fixed VIX level: bull when VIX < expanding-window 40th pct (shifted, no lookahead).
Robust to secular VIX drift. Bull cap=2.5x, bear cap=1.0x, tv=6.5%.

Run: uv run python scripts/run_h602.py
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
    expanding_percentile_mask,
    finalize_run,
    mdr_warm_start,
    slsqp,
    two_state_cap,
)
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20
BULL_CAP, BEAR_CAP = 2.5, 1.0
QUANTILE = 0.40


def main():
    print("H602: VIXPercentileRegime\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    bull = expanding_percentile_mask(vix, idx, QUANTILE)
    cap = two_state_cap(bull, BULL_CAP, BEAR_CAP, idx)
    print(f"Bull fraction (VIX < expanding {QUANTILE:.0%}-pct): {bull.mean():.1%}")

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "602",
        "vix-percentile-regime",
        "VIXPercentileRegime",
        "Set9 Regime Forms",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Bull = VIX < expanding {QUANTILE:.0%}-pctile (adaptive, no fixed level). "
            f"Bull cap={BULL_CAP}x/bear {BEAR_CAP}x, tv={TARGET_VOL:.1%}. Bull frac {bull.mean():.1%}. "
            f"H600/H591 (fixed VIX<15) ref 2.237."
        ),
        extra={
            "objective": "lev-sharpe-vix-expanding-pctile",
            "quantile": QUANTILE,
            "bull_fraction": round(float(bull.mean()), 4),
            "leverage_max_bull": BULL_CAP,
            "leverage_max_bear": BEAR_CAP,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="VIX-pctile",
    )


if __name__ == "__main__":
    main()
