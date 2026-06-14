"""H631: VIXMomCalmar -- leveraged-Calmar objective at the VIX-momentum optimum.

Push Calmar past H592 (2.264) using the VIX<16-and-not-rising champion regime with a lev-Calmar objective.
Bull cap=2.5x, bear cap=1.0x, tv=6.5%.

Run: uv run python scripts/run_h631.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from set8_lev import (
    apply_leverage,
    finalize_run,
    mdr_warm_start,
    portfolio_metrics,
    slsqp,
    two_state_cap,
    vix_momentum_mask,
)
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20


def main():
    print("H631: VIXMomCalmar\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_momentum_mask(vix, idx, 16.0, 20), 2.5, 1.0, idx)

    def neg(w):
        return -portfolio_metrics(apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK))[1]

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "631",
        "vix-mom-calmar",
        "VIXMomCalmar",
        "Set10 Frontier Objectives",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "Leveraged-Calmar objective at VIX<16-and-not-rising (bull 2.5x/bear 1.0x), tv=6.5%. "
            "Targets Calmar > H592 (2.264) on the champion signal."
        ),
        extra={
            "objective": "lev-calmar-vix-momentum",
            "vix_threshold": 16.0,
            "window": 20,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="VIXMom-Calmar",
    )


if __name__ == "__main__":
    main()
