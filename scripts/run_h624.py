"""H624: VIXMomentum3State -- 3-state cap on VIX level x direction.

If VIX>=16: cap=1.0 (bear). Else (VIX<16): falling -> cap=3.0, flat -> cap=2.0.
Adds a high-conviction tier when VIX is both low AND actively falling. tv=6.5%, lb=20d.

Run: uv run python scripts/run_h624.py
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


def main():
    print("H624: VIXMomentum3State\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    vixr = vix.reindex(idx).ffill().bfill()
    ma20 = vixr.rolling(20, min_periods=10).mean()
    low = vixr < 16.0
    falling = vixr <= 0.98 * ma20
    cap = pd.Series(np.select([~low, low & falling], [1.0, 3.0], default=2.0), index=idx)
    fr = {
        "strong(low&falling)": float((low & falling).mean()),
        "neutral(low&flat)": float((low & ~falling).mean()),
        "bear(high)": float((~low).mean()),
    }
    print(f"State fractions: {fr}")

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "624",
        "vix-momentum-3state",
        "VIXMomentum3State",
        "Set10 VIX-Momentum",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "3-state: VIX>=16 cap=1.0; VIX<16 & falling cap=3.0; VIX<16 & flat cap=2.0. "
            f"State fractions {fr}. tv={TARGET_VOL:.1%}. H620 (binary VIX<16 not-rising) ref 2.472."
        ),
        extra={
            "objective": "lev-sharpe-vix-3state-direction",
            "vix_level": 16.0,
            "state_fractions": fr,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="VIXMom-3state",
    )


if __name__ == "__main__":
    main()
