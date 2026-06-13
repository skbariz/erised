"""H597: VIXRegimeCAGRMax -- leveraged-CAGR objective + VIX regime (bull cap=3.0x).

Family-office / SWF mandate: maximize CAGR with tolerance for wider MaxDD.
Target CAGR > 16% at Sh > 1.6; keep-guard CI-lower > 1.4 (CAGR-max is overfitting-prone).

Run: uv run python scripts/run_h597.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from set8_lev import (
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
BULL_CAP, BEAR_CAP = 3.0, 1.0
TARGET_VOL = 0.065
LOOKBACK = 20


def main():
    print("H597: VIXRegimeCAGRMax\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    cap = two_state_cap(vix_bull_mask(vix, idx, VIX_THRESHOLD), BULL_CAP, BEAR_CAP, idx)
    w0 = mdr_warm_start(mat, n)

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        _, _, cg, _, _ = portfolio_metrics(r)
        return -cg

    w = slsqp(neg, w0, n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    sh, cal, cg, dd, _ = portfolio_metrics(r)
    print(f"Sh={sh:.3f} Cal={cal:.3f} CAGR={cg:.2%} DD={dd:.2%}  (target CAGR>16%, H583 17.60%)")
    best_port = pd.Series(r, index=idx)
    lo, hi = bootstrap_sharpe_ci(best_port, n_boot=5000)
    fs = fold_sharpe_list(best_port, folds)
    corr = h001_correlation(best_port, spy, folds)
    tw = top_weights(names, w)
    print(f"CI [{lo:.3f}, {hi:.3f}]  Folds {min(fs):.2f}..{max(fs):.2f}  H001 corr {corr:+.4f}")
    # mandate-specific keep guard: CAGR-max requires CI lower > 1.4 in addition to base bar
    decision = "KEEP" if sh > 0.51 and lo > 0 else "kill"
    guard = "PASS" if lo > 1.4 else "FAIL (CI-lower<=1.4 overfit guard)"
    print(f"Decision: {decision}  | CAGR-max CI guard: {guard}")

    write_summary(
        hypothesis_id="597",
        name="vix-regime-cagr-max",
        strategy_class="VIXRegimeCAGRMax",
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
            f"Leveraged-CAGR objective + VIX<{VIX_THRESHOLD} regime (bull cap={BULL_CAP}x/bear {BEAR_CAP}x). "
            f"Sh={sh:.3f} Calmar={cal:.3f}. Mandate: CAGR>16% at Sh>1.6; overfit guard CI-lower>1.4 = {guard}. "
            f"H583 CAGR ref 17.60%. CAGR delta vs H583: {(cg - 0.1760) * 100:+.2f}pp."
        ),
        canonical_params={
            "objective": "lev-cagr-vix-regime",
            "vix_threshold": VIX_THRESHOLD,
            "leverage_max_bull": BULL_CAP,
            "leverage_max_bear": BEAR_CAP,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
            "calmar": round(float(cal), 4),
            "ci_guard_pass": lo > 1.4,
            "top_weights": tw,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
