"""H634: VIXMomMultiObj -- maximize Sharpe x Calmar at the VIX-momentum optimum.

Balanced champion on the best signal: SLSQP objective is the product of leveraged Sharpe and Calmar.
Bull cap=2.5x, bear cap=1.0x, tv=6.5%.

Run: uv run python scripts/run_h634.py
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
    print("H634: VIXMomMultiObj\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_momentum_mask(vix, idx, 16.0, 20), 2.5, 1.0, idx)

    def neg(w):
        sh, cal, _, _, _ = portfolio_metrics(
            apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        )
        return -(sh * cal)

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "634",
        "vix-mom-multi-obj",
        "VIXMomMultiObj",
        "Set10 Frontier Objectives",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "Maximize Sharpe x Calmar at VIX<16-and-not-rising (bull 2.5x/bear 1.0x), tv=6.5%. Balanced "
            "champion on the best signal. H620 (Sharpe-only) 2.472; H592 (Calmar) 2.264."
        ),
        extra={
            "objective": "lev-sharpe-times-calmar-vix-momentum",
            "vix_threshold": 16.0,
            "window": 20,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="VIXMom-MultiObj",
    )


if __name__ == "__main__":
    main()
