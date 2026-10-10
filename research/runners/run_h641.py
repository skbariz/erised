"""H641: DefensibleBlend -- equal-weight blend of 4 defensible Set10 champion streams.

Blends the leveraged streams of H622 (VIX-mom asymmetric), H625 (downside-vol), H630 (vol-of-vol),
and H631 (lev-Calmar) -- all interior/defensible, deliberately excluding the overfit H640/H621 corner.

Run: uv run python scripts/run_h641.py
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

LOOKBACK = 20


def champion_stream(mat, idx, vix, n, w0, kind):
    bull = vix_momentum_mask(vix, idx, 16.0, 20)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv, down = 0.065, False
    if kind == "asym":
        tv = two_state_cap(bull, 0.08, 0.04, idx)
    elif kind == "downside":
        down = True
    elif kind == "volofvol":
        vixr = vix.reindex(idx).ffill().bfill()
        vov = vixr.rolling(20, min_periods=10).std()
        vov_pct = vov.expanding(min_periods=252).rank(pct=True).shift(1).fillna(0.5)
        cap = cap * (1.0 - 0.4 * vov_pct).clip(0.5, 1.0)

    if kind == "calmar":

        def neg(w):
            return -portfolio_metrics(
                apply_leverage(mat @ w, idx, tv, cap, LOOKBACK, downside=down)
            )[1]
    else:

        def neg(w):
            r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK, downside=down)
            s = r.std() * np.sqrt(TRADING_DAYS)
            return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, w0, n)
    return apply_leverage(mat @ w, idx, tv, cap, LOOKBACK, downside=down)


def main():
    print("H641: DefensibleBlend\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    kinds = ["asym", "downside", "volofvol", "calmar"]
    streams = [champion_stream(mat, idx, vix, n, w0, k) for k in kinds]
    r = np.mean(streams, axis=0)
    finalize_run(
        "641",
        "defensible-blend",
        "DefensibleBlend",
        "Set11 Defensible",
        r,
        idx,
        None,
        names,
        folds,
        spy,
        notes=(
            f"Equal-weight blend of 4 defensible leveraged champion streams {kinds} "
            "(H622 asym, H625 downside, H630 vol-of-vol, H631 lev-Calmar). Excludes overfit H640/H621 corner. "
            "Defensible Sharpe champion H622 ref 2.642; Calmar champion H631 ref 2.538/-4.12%."
        ),
        extra={
            "objective": "equal-blend-defensible-streams",
            "components": kinds,
            "leverage_lookback": LOOKBACK,
        },
        label="DefensibleBlend",
    )


if __name__ == "__main__":
    main()
