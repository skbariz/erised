"""H621: VIXMomentumGrid -- optimal (threshold, not-rising window) for the VIX-momentum signal.

Sweeps VIX threshold {15,16,17,18} x not-rising window {10,20,40}, re-optimizing lev-Sharpe at each.
Sensitivity + optimum for the H620 champion (VIX<16, 20d window -> 2.472). Bull cap=2.5x/bear 1.0x, tv=6.5%.

Run: uv run python scripts/run_h621.py
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
    print("H621: VIXMomentumGrid\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    grid, best = {}, None
    for thr in [15.0, 16.0, 17.0, 18.0]:
        for win in [10, 20, 40]:
            cap = two_state_cap(vix_momentum_mask(vix, idx, thr, win), 2.5, 1.0, idx)

            def neg(w, cap=cap):
                r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
                s = r.std() * np.sqrt(TRADING_DAYS)
                return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

            w = slsqp(neg, w0, n)
            r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
            sh = portfolio_metrics(r)[0]
            grid[f"t{thr:.0f}w{win}"] = round(float(sh), 4)
            print(f"  thr={thr:.0f} win={win}: Sh={sh:.3f}")
            if best is None or sh > best[0]:
                best = (sh, w, r, thr, win)

    sh, w, r, thr, win = best
    print(f"\nBest: VIX<{thr:.0f}, not-rising {win}d -> Sh={sh:.3f} (H620 ref t16w20: 2.472)")
    finalize_run(
        "621",
        "vix-momentum-grid",
        "VIXMomentumGrid",
        "Set10 VIX-Momentum",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"VIX-momentum grid (threshold x not-rising window). Sharpe grid: {grid}. "
            f"Best VIX<{thr:.0f}/{win}d. H620 ref (t16/w20) 2.472. Sensitivity for the champion signal."
        ),
        extra={
            "objective": "lev-sharpe-vix-momentum-grid",
            "best_threshold": thr,
            "best_window": win,
            "sharpe_grid": grid,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="VIXMom-grid",
    )


if __name__ == "__main__":
    main()
