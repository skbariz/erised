"""H642: VIXMomAsymCalmar -- asymmetric tv+cap at VIX-momentum, lev-Calmar objective.

Combines the two defensible record-holders: H622 (asymmetric, Calmar 2.172) + H631 (lev-Calmar objective,
Calmar 2.538 / MaxDD -4.12%). Bull (VIX<16 & not-rising): tv=8%/cap=2.5x; bear: tv=4%/cap=1.0x.

Run: uv run python scripts/run_h642.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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
    print("H642: VIXMomAsymCalmar\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    bull = vix_momentum_mask(vix, idx, 16.0, 20)
    tv = two_state_cap(bull, 0.08, 0.04, idx)
    cap = two_state_cap(bull, 2.5, 1.0, idx)

    def neg(w):
        return -portfolio_metrics(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))[1]  # -calmar

    w = slsqp(neg, mdr_warm_start(mat, n), n)
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    finalize_run(
        "642",
        "vixmom-asym-calmar",
        "VIXMomAsymCalmar",
        "Set11 Defensible",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            "Asymmetric tv+cap (bull tv=8%/cap=2.5x, bear tv=4%/cap=1.0x) at VIX<16-and-not-rising, "
            "lev-Calmar objective. Combines H622 (asym) + H631 (Calmar obj). Target Calmar > 2.538 at MaxDD<=-5%."
        ),
        extra={
            "objective": "lev-calmar-vixmom-asymmetric",
            "vix_threshold": 16.0,
            "window": 20,
            "bull_tv": 0.08,
            "bear_tv": 0.04,
            "leverage_lookback": LOOKBACK,
        },
        label="VIXMom-AsymCalmar",
    )


if __name__ == "__main__":
    main()
