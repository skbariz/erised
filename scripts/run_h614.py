"""H614: VIX15LowCorr -- H001-corr-penalized lev-Sharpe at the VIX<15 threshold.

Like H596 (VIX<20, corr +0.299 @ Sh 2.087) but on the tighter VIX<15 optimum. Sweeps lambda; targets
lower H001 corr at higher Sharpe. obj = lev-Sharpe - lambda*corr(port, H001).

Run: uv run python scripts/run_h614.py
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

from quant.strategies.base import MovingAverageCrossover
from quant.validation.walk_forward import walk_forward

TARGET_VOL = 0.065
LOOKBACK = 20
LAMBDAS = [0.5, 1.0, 2.0, 3.0]
CORR_TARGET = 0.30


def main():
    print("H614: VIX15LowCorr\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    cap = two_state_cap(vix_bull_mask(vix, idx, 15.0), 2.5, 1.0, idx)
    h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    h001r = h001.combined_returns.reindex(idx).fillna(0.0).to_numpy()
    w0 = mdr_warm_start(mat, n)

    sweep, chosen = {}, None
    for lam in LAMBDAS:

        def neg(w, lam=lam):
            r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
            s = r.std() * np.sqrt(TRADING_DAYS)
            sh = (r.mean() * TRADING_DAYS / s) if s > 0 else 0.0
            return -(sh - lam * np.corrcoef(r, h001r)[0, 1])

        w = slsqp(neg, w0, n)
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        sh = portfolio_metrics(r)[0]
        c = float(np.corrcoef(r, h001r)[0, 1])
        sweep[lam] = {"sharpe": round(float(sh), 3), "corr": round(c, 3)}
        print(f"  lambda={lam}: Sh={sh:.3f} corr={c:+.3f}")
        cand = (c, sh, lam, w, r)
        if chosen is None:
            chosen = cand
        else:
            meets, meets_old = c < CORR_TARGET, chosen[0] < CORR_TARGET
            if (
                (meets and not meets_old)
                or (meets and meets_old and sh > chosen[1])
                or (not meets and not meets_old and c < chosen[0])
            ):
                chosen = cand

    c, sh, lam, w, r = chosen
    finalize_run(
        "614",
        "vix15-low-corr",
        "VIX15LowCorr",
        "Set9 Frontier Objectives",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"H001-corr-penalized lev-Sharpe at VIX<15. lambda sweep {sweep}; chosen lambda={lam}. "
            f"Target corr<{CORR_TARGET} at high Sharpe. H596 (VIX<20) ref: corr +0.299 @ Sh 2.087."
        ),
        extra={
            "objective": "lev-sharpe-minus-lambda-corr-vix15",
            "vix_threshold": 15.0,
            "lambda_chosen": lam,
            "lambda_sweep": sweep,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
        },
        label="VIX15-lowcorr",
    )


if __name__ == "__main__":
    main()
