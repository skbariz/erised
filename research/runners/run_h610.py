"""H610: EnsembleOfChampions -- equal-weight blend of leveraged champion streams.

Re-derives the leveraged return streams of H591 (VIX<15), H581 (VIX<20), H592 (asym VIX), and
H595 (Sortino VIX), then equal-weights the four streams. Diversifies across regime configurations.

Run: uv run python scripts/run_h610.py
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
    vix_bull_mask,
)
from set8_pool import build_pool

LOOKBACK = 20


def champion_stream(mat, idx, vix, n, w0, kind):
    if kind == "vix15":
        cap, tv = two_state_cap(vix_bull_mask(vix, idx, 15.0), 2.5, 1.0, idx), 0.065
    elif kind == "vix20":
        cap, tv = two_state_cap(vix_bull_mask(vix, idx, 20.0), 2.5, 1.0, idx), 0.065
    elif kind == "asym":
        bull = vix_bull_mask(vix, idx, 20.0)
        cap, tv = two_state_cap(bull, 2.5, 1.0, idx), two_state_cap(bull, 0.08, 0.04, idx)
    down = kind == "sortino"
    if kind == "sortino":
        cap, tv = two_state_cap(vix_bull_mask(vix, idx, 20.0), 2.5, 1.0, idx), 0.065

    def neg(w):
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK, downside=down)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, w0, n)
    return apply_leverage(mat @ w, idx, tv, cap, LOOKBACK, downside=down)


def main():
    print("H610: EnsembleOfChampions\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    kinds = ["vix15", "vix20", "asym", "sortino"]
    streams = [champion_stream(mat, idx, vix, n, w0, k) for k in kinds]
    r = np.mean(streams, axis=0)
    finalize_run(
        "610",
        "ensemble-of-champions",
        "EnsembleOfChampions",
        "Set9 Pool Robustness",
        r,
        idx,
        None,
        names,
        folds,
        spy,
        notes=(
            f"Equal-weight blend of 4 leveraged champion streams {kinds} "
            "(H591 VIX<15, H581 VIX<20, H592 asym, H595 Sortino). Diversifies across regime configs. "
            "H600/H591 best single ref 2.237."
        ),
        extra={
            "objective": "equal-blend-leveraged-champion-streams",
            "components": kinds,
            "leverage_lookback": LOOKBACK,
        },
        label="Ensemble-of-champions",
    )


if __name__ == "__main__":
    main()
