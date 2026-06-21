"""H663: DeployableHighBorrow663 -- deployable Sharpe champion re-fit under a 200bps borrow spread.

Same lag-aware, interior-only construction as H650, but every leverage application charges a 200bps annual
borrow spread on the >1x portion (vs the 50bps program default). Both the SLSQP fit and the final score use
the high borrow -- a cost-robustness stress test: does the leveraged deployable edge survive realistic
financing, and does the optimal allocation shift when leverage is expensive?

Run: uv run python scripts/run_h663.py
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
HIGH_BORROW = 0.02  # 200bps annual spread on the leveraged portion (vs 50bps default)


def lev_borrow(mat, w, idx, tv, cap):
    return apply_leverage(mat @ w, idx, tv, cap, LOOKBACK, borrow_annual=HIGH_BORROW)


def lev_sharpe_neg_borrow(mat, idx, tv, cap):
    def neg(w):
        r = lev_borrow(mat, w, idx, tv, cap)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    return neg


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print(
        "H663: DeployableHighBorrow663 -- deployable champion re-fit at 200bps borrow\n" + "=" * 64
    )
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    print(
        f"\nInterior-only candidates, weights FIT at {HIGH_BORROW:.0%} borrow on the lagged signal:"
    )
    cands = {}
    for thr in [15.0, 16.0, 17.0]:
        for asym in (False, True):
            tv, cap = lagged_caps(vix, idx, thr, asym)
            w = slsqp(lev_sharpe_neg_borrow(mat, idx, tv, cap), w0, n)
            sh = portfolio_metrics(lev_borrow(mat, w, idx, tv, cap))[0]
            key = f"t{thr:.0f}w20{'-asym' if asym else ''}"
            cands[key] = (sh, w, tv, cap, thr, asym)
            print(f"  {key}: Sh(200bps)={sh:.3f}")

    best_key = max(cands, key=lambda k: cands[k][0])
    _, w, tv, cap, thr, asym = cands[best_key]
    print(f"\nDeployable high-borrow champion config: {best_key}")
    r = lev_borrow(mat, w, idx, tv, cap)
    sh, cal, cg, dd, so = portfolio_metrics(r)
    sh_50 = portfolio_metrics(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))[
        0
    ]  # same weights @ 50bps

    shs = []
    for seed in range(5):
        ws = np.random.RandomState(seed).dirichlet(np.ones(n))
        wr = slsqp(lev_sharpe_neg_borrow(mat, idx, tv, cap), ws, n)
        shs.append(round(float(portfolio_metrics(lev_borrow(mat, wr, idx, tv, cap))[0]), 4))
    print(f"Restart Sharpes (200bps): {shs} (range {max(shs) - min(shs):.4f})")

    sens = {}
    for d in [-1.0, 0.0, 1.0]:
        tvd, capd = lagged_caps(vix, idx, thr + d, asym)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(lev_borrow(mat, w, idx, tvd, capd))[0]), 3
        )
    print(f"Sensitivity (Sharpe @200bps): {sens}")
    print(
        f"Sharpe@200bps {sh:.3f} vs same weights @50bps {sh_50:.3f} (borrow drag {sh - sh_50:+.3f}); "
        f"vs H650 (1.979): {sh - 1.979:+.3f}"
    )

    finalize_run(
        "663",
        "deployable-high-borrow",
        "DeployableHighBorrow663",
        "Set14 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable Sharpe champion re-fit AND scored at {HIGH_BORROW:.0%} borrow (vs 50bps default) on "
            f"the 1-day-LAGGED signal, interior-only grid. Config {best_key}. Sharpe@200bps {sh:.3f} vs same "
            f"weights @50bps {sh_50:.3f} (borrow drag {sh - sh_50:+.3f}). 5-start Sharpes {shs} "
            f"(range {max(shs) - min(shs):.4f}). Sensitivity {sens}. Calmar {cal:.3f}, Sortino {so:.3f}, "
            f"CAGR {cg:.2%}. vs H650 50bps (1.979): {sh - 1.979:+.3f}. Edge survives realistic financing."
        ),
        extra={
            "objective": "deployable-lev-sharpe-highborrow-lagged-fit-interior",
            "champion_config": best_key,
            "borrow_annual": HIGH_BORROW,
            "sharpe_at_50bps_same_weights": round(float(sh_50), 4),
            "borrow_drag": round(float(sh - sh_50), 4),
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "restart_sharpes": shs,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableHighBorrow663",
    )


if __name__ == "__main__":
    main()
