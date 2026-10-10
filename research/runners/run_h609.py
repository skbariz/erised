"""H609: SubperiodStableVIX15 -- weights averaged across two sub-periods.

Optimize lev-Sharpe weights separately on 1998-2010 and 2010-2022 sub-samples, average them, then
apply VIX<15 leverage on the full sample. Tests weight stability to regime drift (overfitting guard).

Run: uv run python scripts/run_h609.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
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
BULL_CAP, BEAR_CAP = 2.5, 1.0
SPLIT = "2010-06-01"


def opt_on(mask, mat, idx, cap, w0, n):
    sub = mat[mask.to_numpy()]
    sub_idx = idx[mask.to_numpy()]
    sub_cap = cap[mask.to_numpy()]

    def neg(w):
        r = apply_leverage(sub @ w, sub_idx, TARGET_VOL, sub_cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    return slsqp(neg, w0, n)


def main():
    print("H609: SubperiodStableVIX15\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_bull_mask(vix, idx, 15.0), BULL_CAP, BEAR_CAP, idx)
    w0 = mdr_warm_start(mat, n)

    early = pd.Series(idx < pd.Timestamp(SPLIT), index=idx)
    late = ~early
    w_e = opt_on(early, mat, idx, cap, w0, n)
    w_l = opt_on(late, mat, idx, cap, w0, n)
    w = (w_e + w_l) / 2.0
    w = w / w.sum()
    drift = float(np.mean(np.abs(w_e - w_l)))
    print(f"Mean |w_early - w_late| = {drift:.4f}")

    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "609",
        "subperiod-stable-vix15",
        "SubperiodStableVIX15",
        "Set9 Pool Robustness",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Weights = avg of lev-Sharpe optima on pre-{SPLIT} and post-{SPLIT} sub-samples, VIX<15 leverage. "
            f"Cross-period weight drift {drift:.4f}. Robustness to regime drift. H600/H591 (full-sample) ref 2.237."
        ),
        extra={
            "objective": "lev-sharpe-subperiod-avg-vix15",
            "split": SPLIT,
            "weight_drift": round(drift, 4),
            "leverage_max_bull": BULL_CAP,
            "leverage_max_bear": BEAR_CAP,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="Subperiod-stable",
    )


if __name__ == "__main__":
    main()
