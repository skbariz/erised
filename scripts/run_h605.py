"""H605: RealizedVolRegime -- regime from the portfolio's OWN realized vol (no external VIX).

bull = portfolio 20d realized vol below its 252d rolling median. Cap=2.5x bull / 1.0x bear, tv=6.5%.
Falsification test: if a VIX-less vol-state proxy nearly matches VIX<15 (2.237), the edge is the
leverage mechanism, not VIX specifically -- and the champion generalizes to VIX-less assets.

Run: uv run python scripts/run_h605.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from set8_lev import TRADING_DAYS, apply_leverage, finalize_run, mdr_warm_start, slsqp
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20
BULL_CAP, BEAR_CAP = 2.5, 1.0


def vol_state_cap(r_base, idx):
    """Bull (cap 2.5) where 20d realized vol < its 252d rolling median, else bear (1.0). Shifted."""
    rb = pd.Series(r_base, index=idx)
    rv = rb.rolling(20, min_periods=20).std()
    med = rv.rolling(252, min_periods=120).median()
    bull = (rv < med).shift(1).fillna(True)
    return pd.Series(np.where(bull.to_numpy(), BULL_CAP, BEAR_CAP), index=idx)


def main():
    print("H605: RealizedVolRegime\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, spy = pool["folds_aa"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    def neg(w):
        base = mat @ w
        cap = vol_state_cap(base, idx)
        r = apply_leverage(base, idx, TARGET_VOL, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    base = mat @ w
    cap = vol_state_cap(base, idx)
    r = apply_leverage(base, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "605",
        "realized-vol-regime",
        "RealizedVolRegime",
        "Set9 Regime Forms",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "Regime from portfolio's OWN 20d realized vol vs 252d rolling median (NO external VIX). "
            f"Bull cap={BULL_CAP}x/bear {BEAR_CAP}x, tv={TARGET_VOL:.1%}. Falsification test for VIX-specificity. "
            "H600/H591 (VIX<15) ref 2.237; near-match => edge is the leverage mechanism, generalizes VIX-less."
        ),
        extra={
            "objective": "lev-sharpe-realized-vol-regime",
            "leverage_max_bull": BULL_CAP,
            "leverage_max_bear": BEAR_CAP,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="RealizedVol-regime",
    )


if __name__ == "__main__":
    main()
