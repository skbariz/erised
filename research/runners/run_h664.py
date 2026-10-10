"""H664: DeployableTurnoverPenalized664 -- deployable Sharpe with an L1 pull toward the MDR anchor.

Same lag-aware construction as H650 at the frontier config t15w20, but the weights maximize leveraged
Sharpe - lambda*||w - w_mdr||_1, shrinking the allocation toward the diversified max-diversification anchor.
Sweep lambda in {0, 0.05, 0.1, 0.2}; finalize on the largest lambda holding within 0.05 Sharpe of lambda=0.
Distinct from the L2-toward-equal-weight ridge H654 (different norm, different anchor).

Run: uv run python scripts/run_h664.py
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
LAMBDAS = [0.0, 0.05, 0.1, 0.2]
TOL = 0.05  # finalize: largest lambda within this Sharpe of the lambda=0 optimum


def lev_sharpe_l1_neg(mat, idx, tv, cap, anchor, lam):
    def neg(w):
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        sh = (r.mean() * TRADING_DAYS / s) if s > 0 else 0.0
        return -(sh - lam * np.abs(w - anchor).sum())

    return neg


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print(
        "H664: DeployableTurnoverPenalized664 -- deployable Sharpe with L1 pull to MDR\n" + "=" * 64
    )
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    anchor = mdr_warm_start(mat, n)

    # Fixed at the frontier-champion config; sweep the L1 penalty.
    thr, asym = 15.0, False
    tv, cap = lagged_caps(vix, idx, thr, asym)

    print(f"\nConfig t{thr:.0f}w20; sweeping L1 penalty lambda toward the MDR anchor:")
    sweep = {}
    for lam in LAMBDAS:
        w = slsqp(lev_sharpe_l1_neg(mat, idx, tv, cap, anchor, lam), anchor, n)
        sh = portfolio_metrics(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))[
            0
        ]  # unpenalized Sharpe
        l1 = float(np.abs(w - anchor).sum())
        sweep[lam] = (sh, l1, w)
        print(f"  lambda={lam:>4}: Sharpe={sh:.3f}  L1(w, w_mdr)={l1:.3f}")

    sh0 = sweep[0.0][0]
    eligible = [lam for lam in LAMBDAS if sweep[lam][0] >= sh0 - TOL]
    best_lam = max(eligible)  # most diversified (largest lambda) within tolerance
    sh, l1, w = sweep[best_lam]
    print(f"\nFinalized lambda={best_lam} (within {TOL} Sharpe of lambda=0 optimum {sh0:.3f})")
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    shf, cal, cg, dd, so = portfolio_metrics(r)

    shs = []
    for seed in range(5):
        ws = np.random.RandomState(seed).dirichlet(np.ones(n))
        wr = slsqp(lev_sharpe_l1_neg(mat, idx, tv, cap, anchor, best_lam), ws, n)
        shs.append(
            round(float(portfolio_metrics(apply_leverage(mat @ wr, idx, tv, cap, LOOKBACK))[0]), 4)
        )
    print(f"Restart Sharpes: {shs} (range {max(shs) - min(shs):.4f})")

    sens = {}
    for d in [-1.0, 0.0, 1.0]:
        tvd, capd = lagged_caps(vix, idx, thr + d, asym)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, tvd, capd, LOOKBACK))[0]), 3
        )
    print(f"Sensitivity (Sharpe): {sens}")
    sweep_report = {str(lam): round(sweep[lam][0], 4) for lam in LAMBDAS}
    l1_report = {str(lam): round(sweep[lam][1], 4) for lam in LAMBDAS}
    print(
        f"Sharpe@lambda {sweep_report}; L1@lambda {l1_report}; vs H650 (1.979): {shf - 1.979:+.3f}"
    )

    finalize_run(
        "664",
        "deployable-turnover-penalized",
        "DeployableTurnoverPenalized664",
        "Set14 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable Sharpe - lambda*||w - w_mdr||_1 (L1 pull to the MDR anchor) on the 1-day-LAGGED "
            f"signal, config t{thr:.0f}w20. Finalized lambda={best_lam} (within {TOL} of lambda=0 "
            f"{sh0:.3f}). Sharpe@lambda {sweep_report}; anchor-distance L1@lambda {l1_report}. 5-start "
            f"Sharpes {shs} (range {max(shs) - min(shs):.4f}). Sensitivity {sens}. Sharpe {shf:.3f}, "
            f"Calmar {cal:.3f}, Sortino {so:.3f}. vs H650 (1.979): {shf - 1.979:+.3f}."
        ),
        extra={
            "objective": "deployable-lev-sharpe-l1-mdr-anchor-lagged",
            "champion_config": f"t{thr:.0f}w20",
            "lambda_grid": LAMBDAS,
            "finalized_lambda": best_lam,
            "sharpe_by_lambda": sweep_report,
            "anchor_l1_by_lambda": l1_report,
            "tolerance": TOL,
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "restart_sharpes": shs,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableTurnoverPenalized664",
    )


if __name__ == "__main__":
    main()
