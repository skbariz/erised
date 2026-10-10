"""H646: DeployableCalmar -- H642 (asym + lev-Calmar) with the regime signal lagged 1 day.

The deployable version of the new Calmar record-holder (H642: Calmar 3.093 / MaxDD -3.97%, same-day signal).
Bull (VIX<16 & not-rising, lagged 1d): tv=8%/cap=2.5x; bear: tv=4%/cap=1.0x. lev-Calmar objective.

Run: uv run python scripts/run_h646.py
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
    print("H646: DeployableCalmar\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    bull = vix_momentum_mask(vix, idx, 16.0, 20).shift(1).fillna(True)  # 1-day execution lag
    tv = two_state_cap(bull, 0.08, 0.04, idx)
    cap = two_state_cap(bull, 2.5, 1.0, idx)

    def neg(w):
        return -portfolio_metrics(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))[1]  # -calmar

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    finalize_run(
        "646",
        "deployable-calmar",
        "DeployableCalmar",
        "Set12 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "H642 mechanism (asym tv+cap + lev-Calmar) with regime signal lagged 1 day. Deployable Calmar champion. "
            "H642 (0-lag) ref Calmar 3.093 / MaxDD -3.97%. Gap = execution cost of the Calmar edge."
        ),
        extra={
            "objective": "lev-calmar-vixmom-asym-1day-lag",
            "vix_threshold": 16.0,
            "window": 20,
            "regime_lag_days": 1,
            "leverage_lookback": LOOKBACK,
        },
        label="Deployable-Calmar",
    )


if __name__ == "__main__":
    main()
