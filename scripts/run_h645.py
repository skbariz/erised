"""H645: DefensibleTerminal645 -- terminal defensible champion (interior configs only).

Selects the best VIX-momentum config from an INTERIOR-only grid (threshold {15,16,17} x window {20},
plain + asymmetric), explicitly excluding the overfit short-window/high-threshold corner that produced
H640's inflated 3.04. Full battery: 5-start stability, sensitivity, and a 1-day-lag robustness check.

Run: uv run python scripts/run_h645.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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
    print("H645: DefensibleTerminal645\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    print("\nInterior-only candidate selection (window=20 fixed, threshold<=17):")
    cands = {}
    for thr in [15.0, 16.0, 17.0]:
        bull = vix_momentum_mask(vix, idx, thr, 20)
        cap = two_state_cap(bull, 2.5, 1.0, idx)
        w = slsqp(lev_sharpe_neg(mat, idx, 0.065, cap), w0, n)
        sh = portfolio_metrics(apply_leverage(mat @ w, idx, 0.065, cap, LOOKBACK))[0]
        cands[f"t{thr:.0f}w20"] = (sh, w, 0.065, cap, thr)
        print(f"  t{thr:.0f}w20: Sh={sh:.3f}")
        tv_a = two_state_cap(bull, 0.08, 0.04, idx)
        wa = slsqp(lev_sharpe_neg(mat, idx, tv_a, cap), w0, n)
        sha = portfolio_metrics(apply_leverage(mat @ wa, idx, tv_a, cap, LOOKBACK))[0]
        cands[f"t{thr:.0f}w20-asym"] = (sha, wa, tv_a, cap, thr)
        print(f"  t{thr:.0f}w20-asym: Sh={sha:.3f}")

    best_key = max(cands, key=lambda k: cands[k][0])
    _, w, tv, cap, thr = cands[best_key]
    print(f"\nDefensible champion config: {best_key}")
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)

    # 5-start weight stability
    shs = []
    for seed in range(5):
        ws = np.random.RandomState(seed).dirichlet(np.ones(n))
        wr = slsqp(lev_sharpe_neg(mat, idx, tv, cap), ws, n)
        shs.append(
            round(float(portfolio_metrics(apply_leverage(mat @ wr, idx, tv, cap, LOOKBACK))[0]), 4)
        )
    print(f"Restart Sharpes: {shs} (range {max(shs) - min(shs):.4f})")

    # threshold/window sensitivity + 1-day lag robustness
    sens = {}
    for d in [-1.0, 0.0, 1.0]:
        c = two_state_cap(vix_momentum_mask(vix, idx, thr + d, 20), 2.5, 1.0, idx)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, 0.065, c, LOOKBACK))[0]), 3
        )
    for ww in [10, 20, 40]:
        c = two_state_cap(vix_momentum_mask(vix, idx, thr, ww), 2.5, 1.0, idx)
        sens[f"win{ww}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, 0.065, c, LOOKBACK))[0]), 3
        )
    lag_cap = two_state_cap(
        vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True), 2.5, 1.0, idx
    )
    if not np.isscalar(tv):
        lag_tv = two_state_cap(
            vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True), 0.08, 0.04, idx
        )
    else:
        lag_tv = tv
    sh_lag = round(
        float(portfolio_metrics(apply_leverage(mat @ w, idx, lag_tv, lag_cap, LOOKBACK))[0]), 3
    )
    sens["lag1d"] = sh_lag
    print(f"Sensitivity: {sens}")
    print(
        f"1-day-lag Sharpe: {sh_lag} (gap {sh - sh_lag:+.3f}); vs H640 overfit (3.04): {sh - 3.04:+.3f}"
    )

    finalize_run(
        "645",
        "defensible-terminal-645",
        "DefensibleTerminal645",
        "Set11 Defensible",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"TERMINAL DEFENSIBLE champion (interior-only grid, window=20, thr<=17). Config {best_key}. "
            f"5-start Sharpes {shs} (range {max(shs) - min(shs):.4f}). Sensitivity {sens}. "
            f"1-day-lag Sharpe {sh_lag} (gap {sh - sh_lag:+.3f}). Calmar {cal:.3f}, Sortino {so:.3f}. "
            f"Excludes overfit H640 corner (3.04). Honest deployable champion."
        ),
        extra={
            "objective": "defensible-terminal-interior-vixmom",
            "champion_config": best_key,
            "vix_threshold": thr,
            "window": 20,
            "restart_sharpes": shs,
            "sensitivity": sens,
            "lag1d_sharpe": sh_lag,
            "leverage_lookback": LOOKBACK,
        },
        label="DefensibleTerminal645",
    )


if __name__ == "__main__":
    main()
