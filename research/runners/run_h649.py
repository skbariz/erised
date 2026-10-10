"""H649: DeployableMultiObj -- lagged asymmetric VIX-momentum + Sharpe x Calmar objective.

Balanced deployable champion: rewards risk-adjusted return AND drawdown control, all executable
(1-day regime lag). Asymmetric tv+cap at VIX<16-and-not-rising (lagged).

Run: uv run python scripts/run_h649.py
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

LOOKBACK = 20


def main():
    print("H649: DeployableMultiObj\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    bull = vix_momentum_mask(vix, idx, 16.0, 20).shift(1).fillna(True)
    tv = two_state_cap(bull, 0.08, 0.04, idx)
    cap = two_state_cap(bull, 2.5, 1.0, idx)

    def neg(w):
        sh, cal, _, _, _ = portfolio_metrics(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))
        return -(sh * cal)

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    finalize_run(
        "649",
        "deployable-multi-obj",
        "DeployableMultiObj",
        "Set12 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "Sharpe x Calmar objective + deployable (1-day-lagged) asymmetric VIX-momentum regime. "
            "Balanced executable champion. H643 (deployable plain Sharpe) ref 1.918; H649 adds Calmar reward."
        ),
        extra={
            "objective": "lev-sharpe-times-calmar-vixmom-asym-lag",
            "vix_threshold": 16.0,
            "window": 20,
            "regime_lag_days": 1,
            "leverage_lookback": LOOKBACK,
        },
        label="Deployable-MultiObj",
    )


if __name__ == "__main__":
    main()
