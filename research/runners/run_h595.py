"""H595: VIXRegimeSortino -- downside-semideviation vol leverage + VIX regime caps.

Ports H583's downside vol estimate onto the VIX<20 regime (bull cap=2.5x, bear cap=1.0x).
Targets the CAGR frontier: beat H583's CAGR (17.60%) while improving MaxDD (-10.81%).

Run: uv run python scripts/run_h595.py
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


def main():
    print("H595: VIXRegimeSortino\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    cap = two_state_cap(vix_bull_mask(vix, idx, VIX_THRESHOLD), BULL_CAP, BEAR_CAP, idx)
    w0 = mdr_warm_start(mat, n)

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK, downside=True)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, w0, n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK, downside=True)
    sh, cal, cg, dd, so = portfolio_metrics(r)
    print(
        f"Sh={sh:.3f} Sortino={so:.3f} Cal={cal:.3f} CAGR={cg:.2%} DD={dd:.2%}  (H583 CAGR 17.60%)"
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
        hypothesis_id="595",
        name="vix-regime-sortino",
        strategy_class="VIXRegimeSortino",
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
            f"Downside semi-deviation vol-target leverage + VIX<{VIX_THRESHOLD} regime "
            f"(bull cap={BULL_CAP}x/bear {BEAR_CAP}x), tv={TARGET_VOL:.1%}/lb={LOOKBACK}d. "
            f"Sortino={so:.3f}, Calmar={cal:.3f}. H583 ref CAGR 17.60% / DD -10.81% / Sh 2.007. "
            f"CAGR delta vs H583: {(cg - 0.1760) * 100:+.2f}pp; DD delta: {(dd + 0.1081) * 100:+.2f}pp."
        ),
        canonical_params={
            "objective": "lev-sharpe-downside-vol-vix-regime",
            "vix_threshold": VIX_THRESHOLD,
            "leverage_max_bull": BULL_CAP,
            "leverage_max_bear": BEAR_CAP,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
            "downside_vol": True,
            "sortino": round(float(so), 4),
            "calmar": round(float(cal), 4),
            "top_weights": tw,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
