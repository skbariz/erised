"""H648: DeployableEqualWeight -- equal-weight pool + lagged asymmetric VIX-momentum (no optimization).

Fully-deployable no-optimization baseline. Tests how much weight optimization buys over a naive
equal-weight book once execution is honest (1-day lag). Gap vs H643 (deployable optimized, 1.918).

Run: uv run python scripts/run_h648.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from set8_lev import apply_leverage, finalize_run, two_state_cap, vix_momentum_mask
from set8_pool import build_pool

LOOKBACK = 20


def main():
    print("H648: DeployableEqualWeight\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    bull = vix_momentum_mask(vix, idx, 16.0, 20).shift(1).fillna(True)
    tv = two_state_cap(bull, 0.08, 0.04, idx)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    w = np.full(n, 1.0 / n)
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    finalize_run(
        "648",
        "deployable-equal-weight",
        "DeployableEqualWeight",
        "Set12 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "Equal-weight 55-pool (1/n) + deployable (1-day-lagged) asymmetric VIX-momentum regime. NO optimization. "
            "Baseline: gap vs H643 (deployable optimized, 1.918) = value of weight optimization once execution is honest."
        ),
        extra={
            "objective": "equal-weight-vixmom-asym-lag",
            "vix_threshold": 16.0,
            "window": 20,
            "regime_lag_days": 1,
            "leverage_lookback": LOOKBACK,
        },
        label="Deployable-EW",
    )


if __name__ == "__main__":
    main()
