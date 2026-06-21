"""H656: DeployableCVaR656 -- deployable return-to-CVaR (5% expected shortfall) objective.

Same lag-aware, interior-only construction as H650, but the pool weights are fit to maximize the
leveraged annualized-return / CVaR(5%) ratio on the 1-day-lagged signal. A tail-risk-aware point on the
deployable frontier: the denominator is the mean of the worst 5% of days, not the symmetric variance.

Run: uv run python scripts/run_h656.py
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
ALPHA = 0.05  # CVaR tail fraction (worst 5% of days)


def cvar_ratio(r, alpha=ALPHA):
    """Annualized return divided by the annualized 5% expected shortfall (a tail-Sharpe analogue)."""
    ann_mean = r.mean() * TRADING_DAYS
    k = max(1, int(alpha * len(r)))
    es = -np.sort(r)[:k].mean()  # positive daily expected shortfall (mean of worst k days)
    es_ann = es * np.sqrt(TRADING_DAYS)
    return ann_mean / es_ann if es_ann > 0 else 0.0


def lev_cvar_neg(mat, idx, tv, cap):
    def neg(w):
        return -cvar_ratio(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))

    return neg


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print("H656: DeployableCVaR656 -- deployable return-to-CVaR objective\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    print("\nInterior-only candidates, weights FIT (return/CVaR) on the lagged signal:")
    cands = {}
    for thr in [15.0, 16.0, 17.0]:
        for asym in (False, True):
            tv, cap = lagged_caps(vix, idx, thr, asym)
            w = slsqp(lev_cvar_neg(mat, idx, tv, cap), w0, n)
            r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
            ratio, sh = cvar_ratio(r), portfolio_metrics(r)[0]
            key = f"t{thr:.0f}w20{'-asym' if asym else ''}"
            cands[key] = (-ratio, w, tv, cap, thr, asym)
            print(f"  {key}: CVaR-ratio={ratio:.3f} Sh={sh:.3f}")

    best_key = min(cands, key=lambda k: cands[k][0])  # highest return/CVaR ratio
    _, w, tv, cap, thr, asym = cands[best_key]
    print(f"\nDeployable CVaR champion config: {best_key}")
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)
    ratio = cvar_ratio(r)

    ratios = []
    for seed in range(5):
        ws = np.random.RandomState(seed).dirichlet(np.ones(n))
        wr = slsqp(lev_cvar_neg(mat, idx, tv, cap), ws, n)
        rr = apply_leverage(mat @ wr, idx, tv, cap, LOOKBACK)
        ratios.append(round(float(cvar_ratio(rr)), 4))
    print(f"Restart CVaR-ratios: {ratios} (range {max(ratios) - min(ratios):.4f})")

    sens = {}
    for d in [-1.0, 0.0, 1.0]:
        tvd, capd = lagged_caps(vix, idx, thr + d, asym)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, tvd, capd, LOOKBACK))[0]), 3
        )
    print(f"Sensitivity (Sharpe): {sens}")
    print(
        f"CVaR-ratio {ratio:.3f} Sharpe {sh:.3f} DD {dd:.2%}; vs H650 Sharpe (1.979): {sh - 1.979:+.3f}"
    )

    finalize_run(
        "656",
        "deployable-cvar",
        "DeployableCVaR656",
        "Set14 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable return-to-CVaR: maximize annualized return / CVaR({ALPHA:.0%}) on the 1-day-LAGGED "
            f"signal, interior-only grid. Config {best_key}. 5-start CVaR-ratios {ratios} "
            f"(range {max(ratios) - min(ratios):.4f}). Sensitivity {sens}. CVaR-ratio {ratio:.3f}, "
            f"Sharpe {sh:.3f}, Calmar {cal:.3f}, Sortino {so:.3f}. vs H650 Sharpe (1.979): {sh - 1.979:+.3f}."
        ),
        extra={
            "objective": "deployable-lev-return-cvar-lagged-fit-interior",
            "champion_config": best_key,
            "cvar_alpha": ALPHA,
            "cvar_ratio": round(float(ratio), 4),
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "restart_cvar_ratios": ratios,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableCVaR656",
    )


if __name__ == "__main__":
    main()
