"""H669: DeployableRidge669 -- deployable ridge-regularized max-Sharpe allocation.

Deployable counterpart of H650 with the optimized weights L2-shrunk toward equal weight: SLSQP maximizes
Sharpe(pool@w) - lambda*||w - 1/n||^2 (long-only, sum-to-1, MDR warm-start). The ridge penalty pulls the
Sharpe-optimal solution back toward the maximally-robust 1/55 book. Canonical lambda=0.5; the lambda sweep is
reported as the regularization-path sensitivity. The same 1-day-lagged interior-grid leverage as H650.

Run: uv run python scripts/run_h669.py
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
LAMBDA = 0.5


def ridge_weights(mat, lam):
    """Long-only weights maximizing Sharpe(mat@w) - lam*||w - 1/n||^2."""
    n = mat.shape[1]
    eq = np.full(n, 1.0 / n)
    w0 = mdr_warm_start(mat, n)

    def neg_obj(w):
        r = mat @ w
        sd = r.std() * np.sqrt(TRADING_DAYS)
        sh = 0.0 if sd <= 0 else (r.mean() * TRADING_DAYS) / sd
        return -sh + lam * float(np.sum((w - eq) ** 2))

    return slsqp(neg_obj, w0, n)


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print("H669: DeployableRidge669 -- ridge-regularized max-Sharpe allocation\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx = clean.index
    n = mat.shape[1]
    eq = np.full(n, 1.0 / n)

    w = ridge_weights(mat, LAMBDA)
    l2_to_eq = float(np.sqrt(np.sum((w - eq) ** 2)))
    order = np.argsort(w)[::-1]
    top = ", ".join(f"{names[i]} {w[i]:.3f}" for i in order[:5])
    print(
        f"Ridge weights (lambda={LAMBDA}): max {w.max():.3f}, ||w-1/n||={l2_to_eq:.4f}; top5: {top}"
    )

    print("\nRidge weights fixed; sweeping the interior leverage grid:")
    cands = {}
    for thr in [15.0, 16.0, 17.0]:
        for asym in (False, True):
            tv, cap = lagged_caps(vix, idx, thr, asym)
            sh = portfolio_metrics(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))[0]
            key = f"t{thr:.0f}w20{'-asym' if asym else ''}"
            cands[key] = (sh, tv, cap, thr, asym)
            print(f"  {key}: Sh={sh:.3f}")

    best_key = max(cands, key=lambda k: cands[k][0])
    _, tv, cap, thr, asym = cands[best_key]
    print(f"\nDeployable ridge champion config: {best_key}")
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)

    # Regularization path: deployable Sharpe + flatness as lambda varies (at the champion leverage config).
    lam_sweep = {}
    for lam in [0.0, 0.25, 0.5, 1.0, 2.0]:
        wl = ridge_weights(mat, lam)
        shl = portfolio_metrics(apply_leverage(mat @ wl, idx, tv, cap, LOOKBACK))[0]
        lam_sweep[f"lam{lam}"] = (
            round(float(shl), 3),
            round(float(np.sqrt(np.sum((wl - eq) ** 2))), 4),
        )
    print(f"Lambda path (Sharpe, ||w-1/n||): {lam_sweep}")

    sens = {}
    for d in [-1.0, 0.0, 1.0]:
        tvd, capd = lagged_caps(vix, idx, thr + d, asym)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, tvd, capd, LOOKBACK))[0]), 3
        )
    print(f"Sensitivity (Sharpe): {sens}")
    print(f"Sharpe {sh:.3f} DD {dd:.2%} CAGR {cg:.2%}; vs H650 (1.979): {sh - 1.979:+.3f}")

    finalize_run(
        "669",
        "deployable-ridge",
        "DeployableRidge669",
        "Set15 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable ridge-regularized max-Sharpe (SLSQP, MDR warm-start, lambda={LAMBDA} L2-shrink "
            f"toward 1/n) + 1-day-LAGGED interior-grid leverage. Config {best_key}. ||w-1/n||={l2_to_eq:.4f}. "
            f"Lambda path {lam_sweep}. Sensitivity {sens}. Sharpe {sh:.3f}, Calmar {cal:.3f}, "
            f"Sortino {so:.3f}. vs H650 (1.979): {sh - 1.979:+.3f}."
        ),
        extra={
            "objective": "deployable-ridge-shrink-lagged-interior",
            "champion_config": best_key,
            "lambda": LAMBDA,
            "l2_to_equal": round(l2_to_eq, 5),
            "lambda_path": lam_sweep,
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableRidge669",
    )


if __name__ == "__main__":
    main()
