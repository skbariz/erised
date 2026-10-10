"""H612: VIX15Calmar -- leveraged-Calmar objective at the VIX<15 threshold.

Push Calmar past H592 (2.264) by using the tighter VIX<15 bull definition with a lev-Calmar objective.
Bull cap=2.5x, bear cap=1.0x, tv=6.5%.

Run: uv run python scripts/run_h612.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from set8_lev import (
    apply_leverage,
    finalize_run,
    mdr_warm_start,
    portfolio_metrics,
    slsqp,
    two_state_cap,
    vix_bull_mask,
)
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20


def main():
    print("H612: VIX15Calmar\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_bull_mask(vix, idx, 15.0), 2.5, 1.0, idx)

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        return -portfolio_metrics(r)[1]  # -calmar

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "612",
        "vix15-calmar",
        "VIX15Calmar",
        "Set9 Frontier Objectives",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "Leveraged-Calmar objective at VIX<15 (bull 2.5x/bear 1.0x), tv=6.5%. "
            "Targets Calmar > H592 (2.264). Tighter bull threshold than H594 (VIX<20 Calmar, 2.117)."
        ),
        extra={
            "objective": "lev-calmar-vix15",
            "vix_threshold": 15.0,
            "leverage_max_bull": 2.5,
            "leverage_max_bear": 1.0,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="VIX15-Calmar",
    )


if __name__ == "__main__":
    main()
