"""H626: LeverageLookbackGrid -- vol-target lookback sweep under VIX-momentum.

The vol-estimation lookback has been fixed at 20d since H559. Sweeps {10,15,20,30,63}d under the
VIX<16-and-not-rising champion regime. Sensitivity + optimum. Bull cap=2.5x/bear 1.0x, tv=6.5%.

Run: uv run python scripts/run_h626.py
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
    portfolio_metrics,
    slsqp,
    two_state_cap,
    vix_momentum_mask,
)
from set8_pool import build_pool

TARGET_VOL = 0.065


def main():
    print("H626: LeverageLookbackGrid\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_momentum_mask(vix, idx, 16.0, 20), 2.5, 1.0, idx)
    w0 = mdr_warm_start(mat, n)

    sweep, best = {}, None
    for lb in [10, 15, 20, 30, 63]:

        def neg(w, lb=lb):
            r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, lb)
            s = r.std() * np.sqrt(TRADING_DAYS)
            return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

        w = slsqp(neg, w0, n)
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, lb)
        sh = portfolio_metrics(r)[0]
        sweep[lb] = round(float(sh), 4)
        print(f"  lookback={lb}d: Sh={sh:.3f}")
        if best is None or sh > best[0]:
            best = (sh, w, r, lb)

    sh, w, r, lb = best
    print(f"\nBest lookback: {lb}d -> Sh={sh:.3f} (H620 ref 20d: 2.472)")
    finalize_run(
        "626",
        "leverage-lookback-grid",
        "LeverageLookbackGrid",
        "Set10 Leverage Mechanics",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Vol-target lookback sweep under VIX<16-and-not-rising regime. Sharpe by lookback: {sweep}. "
            f"Best {lb}d. H620 (20d) ref 2.472. Tests the always-20d assumption since H559."
        ),
        extra={
            "objective": "lev-sharpe-vix-momentum-lookback-grid",
            "best_lookback": lb,
            "lookback_sweep": sweep,
            "leverage_target_vol": TARGET_VOL,
        },
        label="LookbackGrid",
    )


if __name__ == "__main__":
    main()
