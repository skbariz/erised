"""H608: EqualWeightVIX15 -- equal-weight 55-pool + VIX<15 leverage (no optimization).

The no-optimization baseline. Falsification test: if equal-weight + VIX<15 leverage nearly matches the
SLSQP champion (H600/H591: 2.237), then the leverage mechanism -- not the optimized weights -- is the edge.

Run: uv run python scripts/run_h608.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from set8_lev import apply_leverage, finalize_run, two_state_cap, vix_bull_mask
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20
BULL_CAP, BEAR_CAP = 2.5, 1.0


def main():
    print("H608: EqualWeightVIX15\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    w = np.full(n, 1.0 / n)
    cap = two_state_cap(vix_bull_mask(vix, idx, 15.0), BULL_CAP, BEAR_CAP, idx)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "608",
        "equal-weight-vix15",
        "EqualWeightVIX15",
        "Set9 Pool Robustness",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Equal-weight 55-pool (1/n each) + VIX<15 regime leverage (bull {BULL_CAP}x/bear {BEAR_CAP}x), "
            f"tv={TARGET_VOL:.1%}. NO optimization -- falsification baseline. H600/H591 (SLSQP) ref 2.237; "
            "near-match => edge is the leverage mechanism, not the optimized weights."
        ),
        extra={
            "objective": "equal-weight-vix15-leverage",
            "leverage_max_bull": BULL_CAP,
            "leverage_max_bear": BEAR_CAP,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="EqualWeight",
    )


if __name__ == "__main__":
    main()
