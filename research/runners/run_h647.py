"""H647: DeployableLowCorr -- lagged asymmetric VIX-momentum + H001-corr penalty.

Deployable institutional diversifier: obj = lev-Sharpe - lambda*corr(port, H001) with the asymmetric
VIX-momentum regime lagged 1 day; lambda swept. Honest corr/Sharpe frontier point holdable vs an equity book.

Run: uv run python scripts/run_h647.py
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

from erised.strategies.base import MovingAverageCrossover
from erised.validation.walk_forward import walk_forward

LOOKBACK = 20
LAMBDAS = [0.5, 1.0, 2.0, 3.0]
CORR_TARGET = 0.30


def main():
    print("H647: DeployableLowCorr\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    bull = vix_momentum_mask(vix, idx, 16.0, 20).shift(1).fillna(True)
    tv = two_state_cap(bull, 0.08, 0.04, idx)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    h001r = h001.combined_returns.reindex(idx).fillna(0.0).to_numpy()
    w0 = mdr_warm_start(mat, n)

    sweep, chosen = {}, None
    for lam in LAMBDAS:

        def neg(w, lam=lam):
            r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
            s = r.std() * np.sqrt(TRADING_DAYS)
            sh = (r.mean() * TRADING_DAYS / s) if s > 0 else 0.0
            return -(sh - lam * np.corrcoef(r, h001r)[0, 1])

        w = slsqp(neg, w0, n)
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
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
        "647",
        "deployable-low-corr",
        "DeployableLowCorr",
        "Set12 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable (1-day-lagged) asym VIX-momentum + H001-corr penalty. lambda sweep {sweep}; chosen {lam}. "
            f"Target corr<{CORR_TARGET} at deployable Sharpe>1.5. H632 (same-day VIX-mom) ref corr +0.253."
        ),
        extra={
            "objective": "lev-sharpe-minus-lambda-corr-vixmom-asym-lag",
            "vix_threshold": 16.0,
            "regime_lag_days": 1,
            "lambda_chosen": lam,
            "lambda_sweep": sweep,
            "leverage_lookback": LOOKBACK,
        },
        label="Deployable-LowCorr",
    )


if __name__ == "__main__":
    main()
