"""H600: GrandChampion600 -- terminal strategy of the 600-hypothesis Erised program.

Selects the best VIX-regime lev-Sharpe configuration (threshold sweep + asymmetric variant),
then runs the full champion battery: 5-start weight stability, sensitivity (threshold/tv/cap),
CI, per-fold, H001 corr, and a comparison table vs prior champions.

Run: uv run python scripts/run_h600.py
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

LOOKBACK = 20


def lev_sharpe_neg(mat, idx, tv, cap):
    def neg(w):
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    return neg


def main():
    print("H600: GrandChampion600 -- terminal strategy\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    # --- 1. Select best VIX-regime config ---
    print("\nSelecting champion config:")
    candidates = {}
    for thr in [17.5, 20.0, 22.5]:
        cap = two_state_cap(vix_bull_mask(vix, idx, thr), 2.5, 1.0, idx)
        w = slsqp(lev_sharpe_neg(mat, idx, 0.065, cap), w0, n)
        r = apply_leverage(mat @ w, idx, 0.065, cap, LOOKBACK)
        sh = portfolio_metrics(r)[0]
        candidates[f"vix<{thr}"] = (sh, w, 0.065, cap, thr)
        print(f"  vix<{thr}: Sh={sh:.3f}")
    # asymmetric variant (H592 mechanism)
    bull = vix_bull_mask(vix, idx, 20.0)
    tv_a = two_state_cap(bull, 0.08, 0.04, idx)
    cap_a = two_state_cap(bull, 2.5, 1.0, idx)
    w_a = slsqp(lev_sharpe_neg(mat, idx, tv_a, cap_a), w0, n)
    sh_a = portfolio_metrics(apply_leverage(mat @ w_a, idx, tv_a, cap_a, LOOKBACK))[0]
    candidates["vix<20-asym"] = (sh_a, w_a, tv_a, cap_a, 20.0)
    print(f"  vix<20-asym: Sh={sh_a:.3f}")

    best_key = max(candidates, key=lambda k: candidates[k][0])
    sh0, w, tv, cap, thr = candidates[best_key]
    print(f"\nChampion config: {best_key} (Sh={sh0:.3f})")

    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)
    best_port = pd.Series(r, index=idx)
    lo, hi = bootstrap_sharpe_ci(best_port, n_boot=5000)
    fs = fold_sharpe_list(best_port, folds)
    corr = h001_correlation(best_port, spy, folds)
    tw = top_weights(names, w)
    print(
        f"\nMetrics: Sh={sh:.3f} CI[{lo:.3f},{hi:.3f}] CAGR={cg:.2%} DD={dd:.2%} Cal={cal:.3f} Sortino={so:.3f}"
    )
    print(f"Folds {min(fs):.2f}..{max(fs):.2f}  H001 corr {corr:+.4f}")
    print(f"Top weights: {tw}")

    # --- 2. Weight stability: 5 random restarts ---
    print("\nWeight stability (5 random restarts):")
    shs = []
    for seed in range(5):
        rng = np.random.RandomState(seed)
        ws = rng.dirichlet(np.ones(n))
        wr = slsqp(lev_sharpe_neg(mat, idx, tv, cap), ws, n)
        shr = portfolio_metrics(apply_leverage(mat @ wr, idx, tv, cap, LOOKBACK))[0]
        shs.append(round(float(shr), 4))
    print(f"  Restart Sharpes: {shs} (range {max(shs) - min(shs):.4f})")

    # --- 3. Sensitivity: threshold, tv, cap ---
    print("\nSensitivity:")
    sens = {}
    for dthr in [-2.5, 0.0, 2.5]:
        c = two_state_cap(vix_bull_mask(vix, idx, thr + dthr), 2.5, 1.0, idx)
        s = portfolio_metrics(apply_leverage(mat @ w, idx, 0.065, c, LOOKBACK))[0]
        sens[f"thr{thr + dthr}"] = round(float(s), 3)
    for dtv in [-0.01, 0.0, 0.01]:
        base_tv = 0.065 + dtv
        s = portfolio_metrics(apply_leverage(mat @ w, idx, base_tv, cap, LOOKBACK))[0]
        sens[f"tv{round(base_tv, 3)}"] = round(float(s), 3)
    for dcap in [-0.5, 0.0, 0.5]:
        c = two_state_cap(vix_bull_mask(vix, idx, thr), 2.5 + dcap, 1.0, idx)
        s = portfolio_metrics(apply_leverage(mat @ w, idx, 0.065, c, LOOKBACK))[0]
        sens[f"bullcap{2.5 + dcap}"] = round(float(s), 3)
    print(f"  {sens}")

    # --- 4. Comparison table ---
    print("\nComparison vs prior champions:")
    print(f"  {'H600':10} Sh={sh:.3f} CAGR={cg:.2%} DD={dd:.2%} Cal={cal:.3f} corr={corr:+.3f}")
    print(f"  {'H581':10} Sh=2.119 CAGR=12.88% DD=-7.81% Cal=1.649 corr=+0.411")
    print(f"  {'H570':10} Sh=2.021 CAGR=12.79% DD=-7.50% Cal=1.706 corr=+0.381")
    print(f"  {'H559':10} Sh=1.879 CAGR=13.08% DD=-9.73% Cal=1.344")
    print(f"  {'H300':10} Sh=1.525 CAGR= 5.82% DD=-6.86% Cal=0.848")

    decision = "KEEP" if sh > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    write_summary(
        hypothesis_id="600",
        name="grand-champion-600",
        strategy_class="GrandChampion600",
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
            f"TERMINAL STRATEGY of the 600-hypothesis Erised program. Champion config: {best_key}. "
            f"5-start restart Sharpes {shs} (range {max(shs) - min(shs):.4f}). Sensitivity {sens}. "
            f"Calmar={cal:.3f} Sortino={so:.3f}. vs H581 (2.119): {sh - 2.119:+.3f}. "
            f"Best regime signal (VIX) + lev-Sharpe objective + MDR-warm-started SLSQP weights."
        ),
        canonical_params={
            "objective": "grand-champion-vix-regime-lev-sharpe",
            "champion_config": best_key,
            "vix_threshold": thr,
            "leverage_lookback": LOOKBACK,
            "calmar": round(float(cal), 4),
            "sortino": round(float(so), 4),
            "restart_sharpes": shs,
            "sensitivity": sens,
            "top_weights": tw,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
