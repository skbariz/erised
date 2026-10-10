"""H596: VIXRegimeLowCorr -- H001-correlation-penalized lev-Sharpe + VIX regime.

Maximizes lev-Sharpe - lambda * corr(port, H001) under the VIX<20 regime; sweeps lambda.
Targets a new Sharpe/H001-corr Pareto point: corr < 0.30 at Sharpe > 1.80 (H581 is +0.411).

Run: uv run python scripts/run_h596.py
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
from erised.strategies.base import MovingAverageCrossover
from erised.validation.walk_forward import walk_forward

VIX_THRESHOLD = 20.0
BULL_CAP, BEAR_CAP = 2.5, 1.0
TARGET_VOL = 0.065
LOOKBACK = 20
LAMBDAS = [0.5, 1.0, 2.0, 3.0]
CORR_TARGET = 0.30


def main():
    print("H596: VIXRegimeLowCorr\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    cap = two_state_cap(vix_bull_mask(vix, idx, VIX_THRESHOLD), BULL_CAP, BEAR_CAP, idx)
    h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    h001r = h001.combined_returns.reindex(idx).fillna(0.0).to_numpy()
    w0 = mdr_warm_start(mat, n)

    sweep = {}
    chosen = None
    for lam in LAMBDAS:

        def neg(w, lam=lam):
            r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
            s = r.std() * np.sqrt(TRADING_DAYS)
            sh = (r.mean() * TRADING_DAYS / s) if s > 0 else 0.0
            c = np.corrcoef(r, h001r)[0, 1]
            return -(sh - lam * c)

        w = slsqp(neg, w0, n)
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        sh, cal, cg, dd, _ = portfolio_metrics(r)
        c = float(np.corrcoef(r, h001r)[0, 1])
        sweep[lam] = {"sharpe": round(float(sh), 3), "corr": round(c, 3)}
        print(f"  lambda={lam}: Sh={sh:.3f} corr={c:+.3f} DD={dd:.2%}")
        # prefer lowest corr that still meets corr<target with max Sharpe; else lowest corr
        cand = (c, -sh, lam, sh, cal, cg, dd, w, r)
        if chosen is None:
            chosen = cand
        else:
            meets, meets_old = c < CORR_TARGET, chosen[0] < CORR_TARGET
            if meets and not meets_old or meets and meets_old and sh > chosen[3] or not meets and not meets_old and c < chosen[0]:
                chosen = cand

    c, _, lam, sh, cal, cg, dd, w, r = chosen
    print(f"\nChosen lambda={lam}: Sh={sh:.3f} corr={c:+.3f} (H581 corr +0.411 @ Sh 2.119)")
    best_port = pd.Series(r, index=idx)
    lo, hi = bootstrap_sharpe_ci(best_port, n_boot=5000)
    fs = fold_sharpe_list(best_port, folds)
    corr = h001_correlation(best_port, spy, folds)
    tw = top_weights(names, w)
    print(f"CI [{lo:.3f}, {hi:.3f}]  Folds {min(fs):.2f}..{max(fs):.2f}  H001 corr {corr:+.4f}")
    decision = "KEEP" if sh > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="596",
        name="vix-regime-low-corr",
        strategy_class="VIXRegimeLowCorr",
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
            f"H001-corr-penalized lev-Sharpe (obj=Sh-lambda*corr) + VIX<{VIX_THRESHOLD} regime. "
            f"Lambda sweep {LAMBDAS}: {sweep}. Chosen lambda={lam} -> corr {corr:+.3f} at Sh {sh:.3f}. "
            f"Target: corr<{CORR_TARGET} at Sh>1.80. H581 ref: corr +0.411 / Sh 2.119."
        ),
        canonical_params={
            "objective": "lev-sharpe-minus-lambda-h001corr-vix-regime",
            "vix_threshold": VIX_THRESHOLD,
            "lambda_chosen": lam,
            "lambda_sweep": sweep,
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
