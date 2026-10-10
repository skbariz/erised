"""H599: VIXRegimeBootstrapRobust -- bootstrap-averaged ensemble weights.

Overfitting probe: resample the OOS return matrix rows N=100x, optimize Sharpe each time,
average the weight vectors, then apply the VIX<20 regime leverage. Tests whether the champion's
diversification is robust (worst-fold Sharpe >= H581's min) rather than fit to one OOS realization.

Run: uv run python scripts/run_h599.py
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

from erised.metrics.performance import bootstrap_sharpe_ci

VIX_THRESHOLD = 20.0
BULL_CAP, BEAR_CAP = 2.5, 1.0
TARGET_VOL = 0.065
LOOKBACK = 20
N_BOOT = 100
H581_MIN_FOLD = 1.095  # H581 worst fold (2008-2010)


def main():
    print("H599: VIXRegimeBootstrapRobust\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_bull_mask(vix, idx, VIX_THRESHOLD), BULL_CAP, BEAR_CAP, idx)
    w0 = mdr_warm_start(mat, n)
    nrows = mat.shape[0]

    print(f"Bootstrap-optimizing weights over {N_BOOT} resamples...")
    weights = []
    for seed in range(N_BOOT):
        rng = np.random.RandomState(seed)
        bidx = rng.choice(nrows, nrows, replace=True)
        mb = mat[bidx]

        def neg(w, mb=mb):
            r = mb @ w
            s = r.std() * np.sqrt(TRADING_DAYS)
            return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

        weights.append(slsqp(neg, w0, n, maxiter=500))
    w = np.mean(weights, axis=0)
    w = w / w.sum()
    wstd = float(np.mean(np.std(weights, axis=0)))
    print(f"Mean per-strategy weight std across bootstraps: {wstd:.4f}")

    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    sh, cal, cg, dd, _ = portfolio_metrics(r)
    best_port = pd.Series(r, index=idx)
    lo, hi = bootstrap_sharpe_ci(best_port, n_boot=5000)
    fs = fold_sharpe_list(best_port, folds)
    corr = h001_correlation(best_port, spy, folds)
    tw = top_weights(names, w)
    print(
        f"Sh={sh:.3f} Cal={cal:.3f} CAGR={cg:.2%} DD={dd:.2%}  min-fold={min(fs):.3f} (H581 min {H581_MIN_FOLD})"
    )
    print(f"CI [{lo:.3f}, {hi:.3f}]  H001 corr {corr:+.4f}")
    robust = min(fs) >= H581_MIN_FOLD and sh > 1.95
    decision = "KEEP" if sh > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}  | robust (min-fold>=H581 & Sh>1.95): {robust}")

    write_summary(
        hypothesis_id="599",
        name="vix-regime-bootstrap-robust",
        strategy_class="VIXRegimeBootstrapRobust",
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
            f"Bootstrap-averaged weights ({N_BOOT} row-resamples, Sharpe-optimized) + VIX<{VIX_THRESHOLD} "
            f"regime leverage. Mean weight-std across bootstraps {wstd:.4f}. min-fold {min(fs):.3f} vs "
            f"H581 min {H581_MIN_FOLD}. Robust={robust}. H581 ref Sh 2.119. Delta {sh - 2.119:+.3f}."
        ),
        canonical_params={
            "objective": "bootstrap-avg-sharpe-weights-vix-regime-leverage",
            "n_bootstrap": N_BOOT,
            "weight_std_across_boot": round(wstd, 4),
            "min_fold_sharpe": round(float(min(fs)), 3),
            "h581_min_fold": H581_MIN_FOLD,
            "robust": bool(robust),
            "vix_threshold": VIX_THRESHOLD,
            "leverage_max_bull": BULL_CAP,
            "leverage_max_bear": BEAR_CAP,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
            "top_weights": tw,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
