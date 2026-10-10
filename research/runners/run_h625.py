"""H625: VIXMomentumDownsideVol -- VIX-momentum + downside semi-deviation vol leverage.

Sizes leverage off downside semi-deviation (rewards strategies with limited downside) under the
VIX<16-and-not-rising regime. CAGR-leaning sizing on the champion signal. Bull cap=2.5x/bear 1.0x, tv=6.5%.

Run: uv run python scripts/run_h625.py
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
    print("H625: VIXMomentumDownsideVol\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_momentum_mask(vix, idx, 16.0, 20), 2.5, 1.0, idx)

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK, downside=True)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK, downside=True)
    finalize_run(
        "625",
        "vix-momentum-downside-vol",
        "VIXMomentumDownsideVol",
        "Set10 VIX-Momentum",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "Downside semi-deviation vol leverage at VIX<16 & not-rising (bull 2.5x/bear 1.0x), tv=6.5%. "
            "CAGR-leaning sizing on the champion signal. H620 (std vol) ref 2.472."
        ),
        extra={
            "objective": "lev-sharpe-downside-vol-vix-momentum",
            "vix_threshold": 16.0,
            "window": 20,
            "downside_vol": True,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="VIXMom-downsideVol",
    )


if __name__ == "__main__":
    main()
