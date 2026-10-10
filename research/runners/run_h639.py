"""H639: BootstrapVIXMom -- bootstrap-averaged weights at the VIX-momentum optimum.

Resamples the OOS matrix 100x, Sharpe-optimizes each, averages weights, then applies VIX-momentum
leverage. Overfitting probe on the terminal champion. Bull cap=2.5x/bear 1.0x, tv=6.5%.

Run: uv run python scripts/run_h639.py
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
N_BOOT = 100


def main():
    print("H639: BootstrapVIXMom\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_momentum_mask(vix, idx, 16.0, 20), 2.5, 1.0, idx)
    w0 = mdr_warm_start(mat, n)
    nrows = mat.shape[0]

    print(f"Bootstrap-optimizing over {N_BOOT} resamples...")
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
        "639",
        "bootstrap-vix-mom",
        "BootstrapVIXMom",
        "Set10 Realism",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Bootstrap-averaged weights ({N_BOOT} resamples) + VIX-momentum leverage. Mean weight-std {wstd:.4f}. "
            f"H620 ref 2.472. H619 (VIX<15 bootstrap) was 2.127. Tests champion weight robustness."
        ),
        extra={
            "objective": "bootstrap-avg-weights-vix-momentum",
            "n_bootstrap": N_BOOT,
            "weight_std_across_boot": round(wstd, 4),
            "vix_threshold": 16.0,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="Bootstrap-VIXMom",
    )


if __name__ == "__main__":
    main()
