"""H592: VIXAsymmetricVolTarget -- combine H581 VIX signal + H586 asymmetric tv+cap.

Bull (VIX<20): target_vol=8%, cap=2.5x. Bear (VIX>=20): target_vol=4%, cap=1.0x.
Tests whether the two best Batch3 ideas (VIX regime 2.119, asymmetric mechanism 2.054) compose.

Run: uv run python scripts/run_h592.py
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
BULL_TV, BULL_CAP = 0.08, 2.5
BEAR_TV, BEAR_CAP = 0.04, 1.0
LOOKBACK = 20


def main():
    print("H592: VIXAsymmetricVolTarget\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    bull = vix_bull_mask(vix, idx, VIX_THRESHOLD)
    tv = two_state_cap(bull, BULL_TV, BEAR_TV, idx)
    cap = two_state_cap(bull, BULL_CAP, BEAR_CAP, idx)
    print(f"Bull fraction (VIX<{VIX_THRESHOLD}): {bull.mean():.1%}")
    w0 = mdr_warm_start(mat, n)

    def neg(w):
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, w0, n)
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, _ = portfolio_metrics(r)
    print(f"Sh={sh:.3f} Cal={cal:.3f} CAGR={cg:.2%} DD={dd:.2%}  (H581 2.119, H586 2.054)")
    best_port = pd.Series(r, index=idx)
    lo, hi = bootstrap_sharpe_ci(best_port, n_boot=5000)
    fs = fold_sharpe_list(best_port, folds)
    corr = h001_correlation(best_port, spy, folds)
    tw = top_weights(names, w)
    print(f"CI [{lo:.3f}, {hi:.3f}]  Folds {min(fs):.2f}..{max(fs):.2f}  H001 corr {corr:+.4f}")
    decision = "KEEP" if sh > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="592",
        name="vix-asymmetric-vol-target",
        strategy_class="VIXAsymmetricVolTarget",
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
            f"VIX<{VIX_THRESHOLD} bull: tv={BULL_TV:.0%}/cap={BULL_CAP}x; bear: tv={BEAR_TV:.0%}/cap={BEAR_CAP}x. "
            f"Combines H581 VIX signal + H586 asymmetric mechanism. Cal={cal:.3f}. "
            f"H581 ref 2.119, H586 ref 2.054. Delta vs H581: {sh - 2.119:+.3f}."
        ),
        canonical_params={
            "objective": "lev-sharpe-vix-asymmetric",
            "vix_threshold": VIX_THRESHOLD,
            "bull_target_vol": BULL_TV,
            "bull_cap": BULL_CAP,
            "bear_target_vol": BEAR_TV,
            "bear_cap": BEAR_CAP,
            "leverage_lookback": LOOKBACK,
            "calmar": round(float(cal), 4),
            "top_weights": tw,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
