"""H620: GrandChampion620 -- terminal strategy of the Set9 (H601-H620) program.

Selects the best VIX-regime config (fine threshold sweep + asymmetric variant), then runs the full
champion battery: 5-start weight stability, sensitivity (threshold/tv/cap), comparison vs H600.

Run: uv run python scripts/run_h620.py
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
    portfolio_metrics,
    slsqp,
    two_state_cap,
    vix_bull_mask,
)
from set8_pool import build_pool

LOOKBACK = 20


def lev_sharpe_neg(mat, idx, tv, cap):
    def neg(w):
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    return neg


def main():
    print("H620: GrandChampion620 -- terminal Set9 strategy\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    # --- 1. Select best config: fine threshold sweep + asymmetric ---
    print("\nSelecting champion config:")
    cands = {}
    for thr in [13.0, 14.0, 15.0, 16.0, 17.0]:
        cap = two_state_cap(vix_bull_mask(vix, idx, thr), 2.5, 1.0, idx)
        w = slsqp(lev_sharpe_neg(mat, idx, 0.065, cap), w0, n)
        sh = portfolio_metrics(apply_leverage(mat @ w, idx, 0.065, cap, LOOKBACK))[0]
        cands[f"vix<{thr}"] = (sh, w, 0.065, cap, thr)
        print(f"  vix<{thr}: Sh={sh:.3f}")
    for thr in [14.0, 15.0]:
        bull = vix_bull_mask(vix, idx, thr)
        tv_a, cap_a = two_state_cap(bull, 0.08, 0.04, idx), two_state_cap(bull, 2.5, 1.0, idx)
        w = slsqp(lev_sharpe_neg(mat, idx, tv_a, cap_a), w0, n)
        sh = portfolio_metrics(apply_leverage(mat @ w, idx, tv_a, cap_a, LOOKBACK))[0]
        cands[f"vix<{thr}-asym"] = (sh, w, tv_a, cap_a, thr)
        print(f"  vix<{thr}-asym: Sh={sh:.3f}")
    # VIX-momentum: bull = VIX<thr AND VIX not rising (H604 mechanism, the Batch-A champion)
    for thr in [14.0, 15.0, 16.0]:
        vixr = vix.reindex(idx).ffill().bfill()
        not_rising = vixr <= vixr.rolling(20, min_periods=10).mean()
        bull = ((vixr < thr) & not_rising).fillna(True)
        cap_m = two_state_cap(bull, 2.5, 1.0, idx)
        w = slsqp(lev_sharpe_neg(mat, idx, 0.065, cap_m), w0, n)
        sh = portfolio_metrics(apply_leverage(mat @ w, idx, 0.065, cap_m, LOOKBACK))[0]
        cands[f"vix<{thr}-mom"] = (sh, w, 0.065, cap_m, thr)
        print(f"  vix<{thr}-mom: Sh={sh:.3f}")

    best_key = max(cands, key=lambda k: cands[k][0])
    _, w, tv, cap, thr = cands[best_key]
    print(f"\nChampion config: {best_key}")

    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)

    # --- 2. Weight stability: 5 random restarts ---
    shs = []
    for seed in range(5):
        ws = np.random.RandomState(seed).dirichlet(np.ones(n))
        wr = slsqp(lev_sharpe_neg(mat, idx, tv, cap), ws, n)
        shs.append(
            round(float(portfolio_metrics(apply_leverage(mat @ wr, idx, tv, cap, LOOKBACK))[0]), 4)
        )
    print(f"Restart Sharpes: {shs} (range {max(shs) - min(shs):.4f})")

    # --- 3. Sensitivity ---
    sens = {}
    for d in [-2.0, 0.0, 2.0]:
        c = two_state_cap(vix_bull_mask(vix, idx, thr + d), 2.5, 1.0, idx)
        sens[f"thr{thr + d}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, 0.065, c, LOOKBACK))[0]), 3
        )
    for d in [-0.01, 0.0, 0.01]:
        sens[f"tv{round(0.065 + d, 3)}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, 0.065 + d, cap, LOOKBACK))[0]), 3
        )
    print(f"Sensitivity: {sens}")
    print(f"vs H604 (2.422): {sh - 2.422:+.3f}")

    finalize_run(
        "620",
        "grand-champion-620",
        "GrandChampion620",
        "Set9 Realism",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"TERMINAL Set9 strategy. Champion config: {best_key}. 5-start Sharpes {shs} "
            f"(range {max(shs) - min(shs):.4f}). Sensitivity {sens}. vs H604 (2.422): {sh - 2.422:+.3f}. "
            f"Sortino {so:.3f}, Calmar {cal:.3f}."
        ),
        extra={
            "objective": "grand-champion-620-vix-regime",
            "champion_config": best_key,
            "vix_threshold": thr,
            "restart_sharpes": shs,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="GrandChampion620",
    )


if __name__ == "__main__":
    main()
