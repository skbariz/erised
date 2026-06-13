"""H598: VIXRegimeDDFloor -- DD-constrained without an in-objective penalty.

H590 proved an in-objective MaxDD penalty destroys value (-29.87% DD). Instead: optimize
lev-Sharpe weights at tv=6.5% (VIX<20 regime), then deterministically lower target_vol
(scales leverage linearly) until realized MaxDD <= 7%. Keeps the highest tv satisfying the floor.
Target: Sh > 1.90 at MaxDD <= -7%.

Run: uv run python scripts/run_h598.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from set8_lev import (
    TRADING_DAYS,
    apply_leverage,
    fold_sharpe_list,
    h001_correlation,
    mdr_warm_start,
    portfolio_metrics,
    slsqp,
    top_weights,
    two_state_cap,
    vix_bull_mask,
)
from set8_pool import build_pool
from write_summary import write_summary

from quant.metrics.performance import bootstrap_sharpe_ci

VIX_THRESHOLD = 20.0
BULL_CAP, BEAR_CAP = 2.5, 1.0
BASE_TV = 0.065
LOOKBACK = 20
DD_FLOOR = 0.07


def main():
    print("H598: VIXRegimeDDFloor\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    cap = two_state_cap(vix_bull_mask(vix, idx, VIX_THRESHOLD), BULL_CAP, BEAR_CAP, idx)
    w0 = mdr_warm_start(mat, n)

    def neg(w):
        r = apply_leverage(mat @ w, idx, BASE_TV, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, w0, n)

    # Deterministic DD floor: lower tv (scales leverage) until MaxDD <= DD_FLOOR.
    chosen = None
    for tv in np.round(np.arange(BASE_TV, 0.029, -0.0025), 4):
        r = apply_leverage(mat @ w, idx, float(tv), cap, LOOKBACK)
        sh, cal, cg, dd, _ = portfolio_metrics(r)
        if abs(dd) <= DD_FLOOR:
            chosen = (float(tv), sh, cal, cg, dd, r)
            break
    if chosen is None:  # fall back to lowest tv tried
        tv = 0.030
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        sh, cal, cg, dd, _ = portfolio_metrics(r)
        chosen = (tv, sh, cal, cg, dd, r)

    tv, sh, cal, cg, dd, r = chosen
    print(
        f"DD-floor tv={tv:.4f}: Sh={sh:.3f} Cal={cal:.3f} CAGR={cg:.2%} DD={dd:.2%} (floor -{DD_FLOOR:.0%})"
    )
    best_port = pd.Series(r, index=idx)
    lo, hi = bootstrap_sharpe_ci(best_port, n_boot=5000)
    fs = fold_sharpe_list(best_port, folds)
    corr = h001_correlation(best_port, spy, folds)
    tw = top_weights(names, w)
    print(f"CI [{lo:.3f}, {hi:.3f}]  Folds {min(fs):.2f}..{max(fs):.2f}  H001 corr {corr:+.4f}")
    decision = "KEEP" if sh > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="598",
        name="vix-regime-dd-floor",
        strategy_class="VIXRegimeDDFloor",
        family="Set8 Batch4 VIX-Regime Synthesis",
        sharpe=sh,
        ci_low=lo,
        ci_high=hi,
        cagr=cg,
        max_drawdown=dd,
        n_trades=0,
        h001_corr=corr,
        fold_sharpes=fs,
        decision=decision,
        notes=(
            f"DD-floor via post-hoc tv reduction (NO in-objective penalty -- H590 showed penalty fails). "
            f"Weights optimized at tv={BASE_TV:.1%}, then tv lowered to {tv:.4f} for MaxDD<={DD_FLOOR:.0%}. "
            f"VIX<{VIX_THRESHOLD} regime (bull {BULL_CAP}x/bear {BEAR_CAP}x). Target Sh>1.90 at DD<=-7%. "
            f"Achieved Sh={sh:.3f}, DD={dd:.2%}, Calmar={cal:.3f}."
        ),
        canonical_params={
            "objective": "lev-sharpe-then-dd-floor-tv",
            "vix_threshold": VIX_THRESHOLD,
            "leverage_max_bull": BULL_CAP,
            "leverage_max_bear": BEAR_CAP,
            "base_target_vol": BASE_TV,
            "floored_target_vol": tv,
            "dd_floor": DD_FLOOR,
            "leverage_lookback": LOOKBACK,
            "calmar": round(float(cal), 4),
            "top_weights": tw,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
