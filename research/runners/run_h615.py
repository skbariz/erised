"""H615: MultiObjectiveVIX15 -- maximize Sharpe x Calmar at VIX<15.

Balanced champion: the SLSQP objective is the product of leveraged Sharpe and Calmar (both rewarded),
which favors portfolios strong on risk-adjusted return AND drawdown. Bull cap=2.5x, bear cap=1.0x, tv=6.5%.

Run: uv run python scripts/run_h615.py
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
    vix_bull_mask,
)
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20


def main():
    print("H615: MultiObjectiveVIX15\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_bull_mask(vix, idx, 15.0), 2.5, 1.0, idx)

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        sh, cal, _, _, _ = portfolio_metrics(r)
        return -(sh * cal)

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    finalize_run(
        "615",
        "multi-objective-vix15",
        "MultiObjectiveVIX15",
        "Set9 Frontier Objectives",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "Maximize Sharpe x Calmar (product) at VIX<15 (bull 2.5x/bear 1.0x), tv=6.5%. Balanced champion "
            "rewarding both risk-adjusted return and drawdown. H600/H591 (Sharpe-only) 2.237; H592 (Calmar) 2.264."
        ),
        extra={
            "objective": "lev-sharpe-times-calmar-vix15",
            "vix_threshold": 15.0,
            "leverage_max_bull": 2.5,
            "leverage_max_bear": 1.0,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="MultiObj-VIX15",
    )


if __name__ == "__main__":
    main()
