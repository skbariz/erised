"""H591: VIXThresholdOptimal -- optimal VIX regime threshold (+ H581 sensitivity).

Sweeps the VIX bull/bear threshold and re-optimizes lev-Sharpe weights at each.
H581 fixed threshold=20 (Sharpe 2.119); this finds the optimum and serves as H581's
required sensitivity analysis. Bull cap=2.5x, bear cap=1.0x, tv=6.5%, lb=20d, 55-strat pool.

Run: uv run python scripts/run_h591.py
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

TARGET_VOL = 0.065
BULL_CAP = 2.5
BEAR_CAP = 1.0
LOOKBACK = 20
THRESHOLDS = [15.0, 17.5, 20.0, 22.5, 25.0]


def main():
    print("H591: VIXThresholdOptimal\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    sweep = {}
    best = None
    for thr in THRESHOLDS:
        cap = two_state_cap(vix_bull_mask(vix, idx, thr), BULL_CAP, BEAR_CAP, idx)

        def neg(w, cap=cap):
            r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
            s = r.std() * np.sqrt(TRADING_DAYS)
            return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

        w = slsqp(neg, w0, n)
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        sh, cal, cg, dd, _ = portfolio_metrics(r)
        sweep[thr] = round(float(sh), 4)
        print(f"  VIX<{thr:>4}: Sh={sh:.3f} Cal={cal:.3f} CAGR={cg:.2%} DD={dd:.2%}")
        if best is None or sh > best[0]:
            best = (sh, cal, cg, dd, thr, w, r)

    sh, cal, cg, dd, thr, w, r = best
    print(f"\nBest threshold: VIX<{thr} -> Sh={sh:.3f} (H581 ref @20: 2.119)")
    best_port = pd.Series(r, index=idx)
    lo, hi = bootstrap_sharpe_ci(best_port, n_boot=5000)
    fs = fold_sharpe_list(best_port, folds)
    corr = h001_correlation(best_port, spy, folds)
    tw = top_weights(names, w)
    print(f"CI [{lo:.3f}, {hi:.3f}]  Folds {min(fs):.2f}..{max(fs):.2f}  H001 corr {corr:+.4f}")
    decision = "KEEP" if sh > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="591",
        name="vix-threshold-optimal",
        strategy_class="VIXThresholdOptimal",
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
            f"VIX threshold sweep {THRESHOLDS}; best=VIX<{thr}. Sharpe by threshold: {sweep}. "
            f"Bull cap={BULL_CAP}x/bear cap={BEAR_CAP}x, tv={TARGET_VOL:.1%}/lb={LOOKBACK}d. "
            f"H581 ref (threshold=20): Sh=2.119. Delta vs H581: {sh - 2.119:+.3f}. "
            f"Doubles as H581 sensitivity analysis -- threshold robustness."
        ),
        canonical_params={
            "objective": "lev-sharpe-vix-threshold-sweep",
            "best_threshold": thr,
            "sharpe_by_threshold": sweep,
            "leverage_target_vol": TARGET_VOL,
            "leverage_max_bull": BULL_CAP,
            "leverage_max_bear": BEAR_CAP,
            "leverage_lookback": LOOKBACK,
            "top_weights": tw,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
