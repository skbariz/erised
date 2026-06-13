"""H619: BootstrapRobustVIX15 -- bootstrap-averaged weights at the VIX<15 optimum.

H599 mechanism (resample OOS rows 100x, Sharpe-optimize, average weights) applied to the champion's
VIX<15 leverage. Tests whether the terminal champion's weights are robust, not OOS-fit.

Run: uv run python scripts/run_h619.py
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
    slsqp,
    two_state_cap,
    vix_bull_mask,
)
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20
N_BOOT = 100
H600_MIN_FOLD = 1.07  # H600/H591 worst fold


def main():
    print("H619: BootstrapRobustVIX15\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_bull_mask(vix, idx, 15.0), 2.5, 1.0, idx)
    w0 = mdr_warm_start(mat, n)
    nrows = mat.shape[0]

    print(f"Bootstrap-optimizing weights over {N_BOOT} resamples...")
    weights = []
    for seed in range(N_BOOT):
        rng = np.random.RandomState(seed)
        mb = mat[rng.choice(nrows, nrows, replace=True)]

        def neg(w, mb=mb):
            r = mb @ w
            s = r.std() * np.sqrt(TRADING_DAYS)
            return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

        weights.append(slsqp(neg, w0, n, maxiter=500))
    w = np.mean(weights, axis=0)
    w = w / w.sum()
    wstd = float(np.mean(np.std(weights, axis=0)))
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "619",
        "bootstrap-robust-vix15",
        "BootstrapRobustVIX15",
        "Set9 Realism",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Bootstrap-averaged weights ({N_BOOT} resamples) + VIX<15 leverage. Mean weight-std {wstd:.4f}. "
            f"H600/H591 ref Sh 2.237 (min-fold ~{H600_MIN_FOLD}). H599 (VIX<20 bootstrap) min-fold 1.311. "
            "Tests champion weight robustness."
        ),
        extra={
            "objective": "bootstrap-avg-weights-vix15-leverage",
            "n_bootstrap": N_BOOT,
            "weight_std_across_boot": round(wstd, 4),
            "vix_threshold": 15.0,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="Bootstrap-VIX15",
    )


if __name__ == "__main__":
    main()
