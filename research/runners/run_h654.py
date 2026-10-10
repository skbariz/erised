"""H654: DeployableShrinkage654 -- deployable Sharpe weights ridge-shrunk toward equal weight.

Same lag-aware, interior-only construction as H650, but the objective adds a ridge penalty pulling the
weights toward equal weight: maximize leveraged Sharpe - lambda * ||w - 1/n||^2 on the 1-day-lagged signal,
sweeping lambda in {0,1,5,20}. Finalizes on the largest lambda that holds within 0.05 Sharpe of the
unshrunk optimum -- the deployable, lag-aware counterpart of the same-day H644 shrinkage champion.

Run: uv run python scripts/run_h654.py
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
LAMBDAS = [0.0, 1.0, 5.0, 20.0]


def lev_sharpe_ridge_neg(mat, idx, tv, cap, lam, eq):
    """Negative leveraged Sharpe plus a ridge penalty pulling w toward equal weight `eq`."""

    def neg(w):
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        sh = (r.mean() * TRADING_DAYS / s) if s > 0 else 0.0
        return -sh + lam * float(np.sum((w - eq) ** 2))

    return neg


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print("H654: DeployableShrinkage654 -- ridge-shrunk deployable weights\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    eq = np.full(n, 1.0 / n)
    w0 = mdr_warm_start(mat, n)

    print("\nInterior-only candidates (unshrunk Sharpe-optimal, lagged signal):")
    cands = {}
    for thr in [15.0, 16.0, 17.0]:
        for asym in (False, True):
            tv, cap = lagged_caps(vix, idx, thr, asym)
            w = slsqp(lev_sharpe_ridge_neg(mat, idx, tv, cap, 0.0, eq), w0, n)
            sh = portfolio_metrics(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))[0]
            key = f"t{thr:.0f}w20{'-asym' if asym else ''}"
            cands[key] = (sh, tv, cap, thr, asym)
            print(f"  {key}: Sh={sh:.3f}")

    best_key = max(cands, key=lambda k: cands[k][0])
    _, tv, cap, thr, asym = cands[best_key]
    print(f"\nConfig fixed at {best_key}; sweeping shrinkage lambda:")

    lam_sweep, weights = {}, {}
    for lam in LAMBDAS:
        w = slsqp(lev_sharpe_ridge_neg(mat, idx, tv, cap, lam, eq), w0, n)
        sh = portfolio_metrics(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))[0]
        disp = float(np.sum((w - eq) ** 2))
        lam_sweep[str(lam)] = round(float(sh), 4)
        weights[lam] = w
        print(f"  lambda={lam:>4}: Sh={sh:.3f}  weight-dispersion={disp:.4f}")

    sh0 = lam_sweep["0.0"]
    chosen_lam = max(lam for lam in LAMBDAS if sh0 - lam_sweep[str(lam)] <= 0.05)
    w = weights[chosen_lam]
    print(f"\nChosen lambda (largest within 0.05 of unshrunk): {chosen_lam}")
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)

    shs = []
    for seed in range(5):
        ws = np.random.RandomState(seed).dirichlet(np.ones(n))
        wr = slsqp(lev_sharpe_ridge_neg(mat, idx, tv, cap, chosen_lam, eq), ws, n)
        shs.append(
            round(float(portfolio_metrics(apply_leverage(mat @ wr, idx, tv, cap, LOOKBACK))[0]), 4)
        )
    print(f"Restart Sharpes (lambda={chosen_lam}): {shs} (range {max(shs) - min(shs):.4f})")
    print(f"Sharpe {sh:.3f}; vs H650 unshrunk (1.980): {sh - 1.980:+.3f}")

    finalize_run(
        "654",
        "deployable-shrinkage",
        "DeployableShrinkage654",
        "Set13 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable ridge-shrinkage: lev-Sharpe - lambda*||w-1/n||^2 on the 1-day-LAGGED signal, "
            f"interior config {best_key}. lambda-sweep {lam_sweep}; chosen lambda={chosen_lam}. "
            f"5-start Sharpes {shs} (range {max(shs) - min(shs):.4f}). Sharpe {sh:.3f}, Calmar {cal:.3f}, "
            f"Sortino {so:.3f}. vs H650 unshrunk (1.980): {sh - 1.980:+.3f}. Deployable counterpart of H644."
        ),
        extra={
            "objective": "deployable-lev-sharpe-ridge-shrinkage-lagged-fit-interior",
            "champion_config": best_key,
            "chosen_lambda": chosen_lam,
            "lambda_sweep": lam_sweep,
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "restart_sharpes": shs,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableShrinkage654",
    )


if __name__ == "__main__":
    main()
