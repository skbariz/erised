"""H640: GrandChampion640 -- terminal strategy of the Set10 (H621-H640) program.

Selects the best VIX-momentum config (threshold x window grid + asymmetric variant), then runs the
full champion battery: 5-start weight stability, sensitivity (threshold/window/tv), comparison vs H620.

Run: uv run python scripts/run_h640.py
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
    vix_momentum_mask,
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
    print("H640: GrandChampion640 -- terminal Set10 strategy\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    # Grid brackets the H621 optimum (t18w10) so the sensitivity analysis can expose whether the
    # peak is stable or a grid-edge overfit artifact (shorter window + higher threshold kept climbing).
    print("\nSelecting champion config:")
    cands = {}
    for thr in [16.0, 17.0, 18.0, 19.0]:
        for win in [5, 10, 20]:
            cap = two_state_cap(vix_momentum_mask(vix, idx, thr, win), 2.5, 1.0, idx)
            w = slsqp(lev_sharpe_neg(mat, idx, 0.065, cap), w0, n)
            sh = portfolio_metrics(apply_leverage(mat @ w, idx, 0.065, cap, LOOKBACK))[0]
            cands[f"t{thr:.0f}w{win}"] = (sh, w, 0.065, cap, thr, win)
    for thr in [18.0]:
        bull = vix_momentum_mask(vix, idx, thr, 10)
        tv_a, cap_a = two_state_cap(bull, 0.08, 0.04, idx), two_state_cap(bull, 2.5, 1.0, idx)
        w = slsqp(lev_sharpe_neg(mat, idx, tv_a, cap_a), w0, n)
        sh = portfolio_metrics(apply_leverage(mat @ w, idx, tv_a, cap_a, LOOKBACK))[0]
        cands[f"t{thr:.0f}w10-asym"] = (sh, w, tv_a, cap_a, thr, 10)
    for k, v in sorted(cands.items(), key=lambda kv: -kv[1][0])[:6]:
        print(f"  {k}: Sh={v[0]:.3f}")

    best_key = max(cands, key=lambda k: cands[k][0])
    _, w, tv, cap, thr, win = cands[best_key]
    print(f"\nChampion config: {best_key}")

    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)

    shs = []
    for seed in range(5):
        ws = np.random.RandomState(seed).dirichlet(np.ones(n))
        wr = slsqp(lev_sharpe_neg(mat, idx, tv, cap), ws, n)
        shs.append(
            round(float(portfolio_metrics(apply_leverage(mat @ wr, idx, tv, cap, LOOKBACK))[0]), 4)
        )
    print(f"Restart Sharpes: {shs} (range {max(shs) - min(shs):.4f})")

    sens = {}
    for d in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        c = two_state_cap(vix_momentum_mask(vix, idx, thr + d, win), 2.5, 1.0, idx)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, 0.065, c, LOOKBACK))[0]), 3
        )
    for ww in [5, 10, 20, 40]:
        c = two_state_cap(vix_momentum_mask(vix, idx, thr, ww), 2.5, 1.0, idx)
        sens[f"win{ww}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, 0.065, c, LOOKBACK))[0]), 3
        )
    print(f"Sensitivity: {sens}")
    print(f"vs H620 (2.472): {sh - 2.472:+.3f}")

    finalize_run(
        "640",
        "grand-champion-640",
        "GrandChampion640",
        "Set10 Realism",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"TERMINAL Set10 strategy. Champion config: {best_key}. 5-start Sharpes {shs} "
            f"(range {max(shs) - min(shs):.4f}). Sensitivity {sens}. vs H620 (2.472): {sh - 2.472:+.3f}. "
            f"Sortino {so:.3f}, Calmar {cal:.3f}."
        ),
        extra={
            "objective": "grand-champion-640-vix-momentum",
            "champion_config": best_key,
            "vix_threshold": thr,
            "window": win,
            "restart_sharpes": shs,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="GrandChampion640",
    )


if __name__ == "__main__":
    main()
