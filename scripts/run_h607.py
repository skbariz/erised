"""H607: ShrinkageWeightsVIX15 -- lev-Sharpe with ridge shrinkage toward MDR weights.

Objective = neg_lev_sharpe + mu * ||w - w_mdr||^2 under VIX<15 regime. Sweeps mu; reports the best.
Shrinkage anchors the optimizer near the diversified MDR prior, guarding against OOS weight concentration.

Run: uv run python scripts/run_h607.py
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
    vix_bull_mask,
)
from set8_pool import build_pool

TARGET_VOL = 0.065
LOOKBACK = 20
BULL_CAP, BEAR_CAP = 2.5, 1.0
MUS = [0.01, 0.1, 0.5, 2.0]


def main():
    print("H607: ShrinkageWeightsVIX15\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_bull_mask(vix, idx, 15.0), BULL_CAP, BEAR_CAP, idx)
    w_mdr = mdr_warm_start(mat, n)

    sweep = {}
    best = None
    for mu in MUS:

        def neg(w, mu=mu):
            r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
            s = r.std() * np.sqrt(TRADING_DAYS)
            sh = (r.mean() * TRADING_DAYS / s) if s > 0 else 0.0
            return -sh + mu * float(np.sum((w - w_mdr) ** 2))

        w = slsqp(neg, w_mdr, n)
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        sh = portfolio_metrics(r)[0]
        sweep[mu] = round(float(sh), 4)
        print(f"  mu={mu}: Sh={sh:.3f}")
        if best is None or sh > best[0]:
            best = (sh, mu, w, r)

    sh, mu, w, r = best
    finalize_run(
        "607",
        "shrinkage-weights-vix15",
        "ShrinkageWeightsVIX15",
        "Set9 Pool Robustness",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"lev-Sharpe + mu*||w-w_mdr||^2 ridge shrinkage, VIX<15 regime. mu sweep {sweep}; best mu={mu}. "
            f"Shrinkage toward MDR prior. H600/H591 (mu=0) ref 2.237."
        ),
        extra={
            "objective": "lev-sharpe-mdr-shrinkage-vix15",
            "mu_best": mu,
            "mu_sweep": sweep,
            "leverage_max_bull": BULL_CAP,
            "leverage_max_bear": BEAR_CAP,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="Shrinkage",
    )


if __name__ == "__main__":
    main()
