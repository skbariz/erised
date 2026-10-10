"""H633: VIXMomCAGRMax -- leveraged-CAGR objective at the VIX-momentum optimum (bull cap=3.0x).

Family-office mandate on the champion signal: maximize CAGR with bull cap=3.0x under VIX<16-and-not-rising.
Keep-guard CI-lower>1.4. tv=6.5%, lb=20d.

Run: uv run python scripts/run_h633.py
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
    vix_momentum_mask,
)
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20


def main():
    print("H633: VIXMomCAGRMax\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_momentum_mask(vix, idx, 16.0, 20), 3.0, 1.0, idx)

    def neg(w):
        return -portfolio_metrics(apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK))[2]

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "633",
        "vix-mom-cagr-max",
        "VIXMomCAGRMax",
        "Set10 Frontier Objectives",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "Leveraged-CAGR objective at VIX<16-and-not-rising, bull cap=3.0x/bear 1.0x, tv=6.5%. "
            "Family-office CAGR mandate on the champion signal. H613 (VIX<15 Sortino) CAGR ref 14.19%."
        ),
        extra={
            "objective": "lev-cagr-vix-momentum",
            "vix_threshold": 16.0,
            "window": 20,
            "leverage_max_bull": 3.0,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="VIXMom-CAGR",
    )


if __name__ == "__main__":
    main()
