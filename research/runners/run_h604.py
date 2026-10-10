"""H604: VIXMomentumRegime -- bull requires VIX<15 AND VIX not rising.

bull = (VIX < 15) AND (VIX <= its 20d mean). De-risk on rising vol even at low absolute levels,
capturing vol-of-vol. Bull cap=2.5x, bear cap=1.0x, tv=6.5%.

Run: uv run python scripts/run_h604.py
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
)
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20
BULL_CAP, BEAR_CAP = 2.5, 1.0
VIX_LEVEL = 15.0


def main():
    print("H604: VIXMomentumRegime\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    vixr = vix.reindex(idx).ffill().bfill()
    not_rising = vixr <= vixr.rolling(20, min_periods=10).mean()
    bull = ((vixr < VIX_LEVEL) & not_rising).fillna(True)
    cap = two_state_cap(bull, BULL_CAP, BEAR_CAP, idx)
    print(f"Bull fraction (VIX<{VIX_LEVEL} & not rising): {bull.mean():.1%}")

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "604",
        "vix-momentum-regime",
        "VIXMomentumRegime",
        "Set9 Regime Forms",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Bull = VIX<{VIX_LEVEL} AND VIX<=20d-mean (not rising). Bull cap={BULL_CAP}x/bear {BEAR_CAP}x, "
            f"tv={TARGET_VOL:.1%}. Bull frac {bull.mean():.1%}. H600/H591 (VIX<15 level only) ref 2.237."
        ),
        extra={
            "objective": "lev-sharpe-vix-level-and-momentum",
            "vix_level": VIX_LEVEL,
            "bull_fraction": round(float(bull.mean()), 4),
            "leverage_max_bull": BULL_CAP,
            "leverage_max_bear": BEAR_CAP,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="VIX-momentum",
    )


if __name__ == "__main__":
    main()
