"""H644: ShrinkageRobustChampion -- H622 asymmetric mechanism with ridge shrinkage to MDR.

Objective = -lev-Sharpe + mu*||w - w_mdr||^2 under the asymmetric VIX-momentum regime; sweeps mu.
Trades a sliver of in-sample Sharpe for more diversified, less overfit weights.

Run: uv run python scripts/run_h644.py
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
MUS = [0.01, 0.1, 0.5, 2.0]


def main():
    print("H644: ShrinkageRobustChampion\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    bull = vix_momentum_mask(vix, idx, 16.0, 20)
    tv = two_state_cap(bull, 0.08, 0.04, idx)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    w_mdr = mdr_warm_start(mat, n)

    sweep = {}
    for mu in MUS:

        def neg(w, mu=mu):
            r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
            s = r.std() * np.sqrt(TRADING_DAYS)
            sh = (r.mean() * TRADING_DAYS / s) if s > 0 else 0.0
            return -sh + mu * float(np.sum((w - w_mdr) ** 2))

        w = slsqp(neg, w_mdr, n)
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        sh = portfolio_metrics(r)[0]
        sweep[mu] = round(float(sh), 4)
        print(f"  mu={mu}: Sh={sh:.3f} maxw={w.max():.3f}")
    # choose most-shrunk mu within 0.05 of the best Sharpe (robust pick)
    best_sh = max(sweep.values())
    chosen_mu = max([m for m in MUS if sweep[m] >= best_sh - 0.05])

    def neg_c(w):
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        sh = (r.mean() * TRADING_DAYS / s) if s > 0 else 0.0
        return -sh + chosen_mu * float(np.sum((w - w_mdr) ** 2))

    w = slsqp(neg_c, w_mdr, n)
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    print(f"Chosen mu={chosen_mu} (most-shrunk within 0.05 of best Sharpe), maxw={w.max():.3f}")
    finalize_run(
        "644",
        "shrinkage-robust-champion",
        "ShrinkageRobustChampion",
        "Set11 Defensible",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Asymmetric VIX-momentum + ridge shrinkage to MDR. mu sweep {sweep}; chosen mu={chosen_mu} "
            f"(most-shrunk within 0.05 of best). Max weight {w.max():.3f}. H622 (mu=0) ref 2.642."
        ),
        extra={
            "objective": "lev-sharpe-mdr-shrinkage-vixmom-asym",
            "mu_chosen": chosen_mu,
            "mu_sweep": sweep,
            "max_weight": round(float(w.max()), 4),
            "leverage_lookback": LOOKBACK,
        },
        label="Shrinkage-Robust",
    )


if __name__ == "__main__":
    main()
