"""H655: DeployableCagrDDfloor655 -- deployable growth-with-floor (CAGR max, -10% MaxDD floor).

Same lag-aware, interior-only construction as H650, but the weights are fit to maximize leveraged CAGR
subject to a -10% MaxDD floor (quadratic penalty on drawdown beyond the floor) on the 1-day-lagged signal.
A different point on the deployable frontier than the Sharpe/Sortino/Calmar champions -- the deployable
counterpart of the same-day H635/H566 CAGR-with-floor work.

Run: uv run python scripts/run_h655.py
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
FLOOR = -0.10
PEN = 25.0


def lev_cagr_ddfloor_neg(mat, idx, tv, cap, floor=FLOOR, pen=PEN):
    """Negative leveraged CAGR plus a quadratic penalty on drawdown breaching `floor`."""

    def neg(w):
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        cagr = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
        cum = np.cumprod(1 + r)
        dd = (cum / np.maximum.accumulate(cum) - 1).min()
        breach = max(0.0, floor - dd)  # positive when drawdown is worse than the floor
        return -cagr + pen * breach**2

    return neg


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print("H655: DeployableCagrDDfloor655 -- deployable growth-with-floor strategy\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    print(
        f"\nInterior-only candidates, weights FIT (CAGR, floor {FLOOR:.0%}) on the lagged signal:"
    )
    cands = {}
    for thr in [15.0, 16.0, 17.0]:
        for asym in (False, True):
            tv, cap = lagged_caps(vix, idx, thr, asym)
            obj = lev_cagr_ddfloor_neg(mat, idx, tv, cap)
            w = slsqp(obj, w0, n)
            sh, cal, cg, dd, so = portfolio_metrics(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))
            key = f"t{thr:.0f}w20{'-asym' if asym else ''}"
            cands[key] = (obj(w), w, tv, cap, thr, asym)
            print(f"  {key}: CAGR={cg:.2%} DD={dd:.2%} Sh={sh:.3f}")

    best_key = min(cands, key=lambda k: cands[k][0])  # lowest penalized-neg-CAGR
    _, w, tv, cap, thr, asym = cands[best_key]
    print(f"\nDeployable growth-with-floor champion config: {best_key}")
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)

    shs = []
    for seed in range(5):
        ws = np.random.RandomState(seed).dirichlet(np.ones(n))
        wr = slsqp(lev_cagr_ddfloor_neg(mat, idx, tv, cap), ws, n)
        rr = apply_leverage(mat @ wr, idx, tv, cap, LOOKBACK)
        shs.append(round(float(portfolio_metrics(rr)[2]), 4))  # CAGR per restart
    print(f"Restart CAGRs: {shs} (range {max(shs) - min(shs):.4f})")

    sens = {}
    for d in [-1.0, 0.0, 1.0]:
        tvd, capd = lagged_caps(vix, idx, thr + d, asym)
        sens[f"thr{thr + d:.0f}"] = round(
            float(portfolio_metrics(apply_leverage(mat @ w, idx, tvd, capd, LOOKBACK))[0]), 3
        )
    print(f"Sensitivity (Sharpe): {sens}")
    print(
        f"CAGR {cg:.2%} DD {dd:.2%} Sharpe {sh:.3f}; vs H650 CAGR (10.08%): {cg - 0.1008:+.2%}; "
        f"floor {FLOOR:.0%} {'HELD' if dd >= FLOOR else 'BREACHED'}"
    )

    finalize_run(
        "655",
        "deployable-cagr-ddfloor",
        "DeployableCagrDDfloor655",
        "Set13 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable growth-with-floor: maximize leveraged CAGR s.t. {FLOOR:.0%} MaxDD floor "
            f"(quadratic penalty, pen={PEN}) on the 1-day-LAGGED signal, interior-only grid. Config "
            f"{best_key}. 5-start CAGRs {shs} (range {max(shs) - min(shs):.4f}). Sensitivity {sens}. "
            f"CAGR {cg:.2%}, MaxDD {dd:.2%} (floor {'held' if dd >= FLOOR else 'breached'}), Sharpe {sh:.3f}, "
            f"Calmar {cal:.3f}. vs H650 CAGR (10.08%): {cg - 0.1008:+.2%}."
        ),
        extra={
            "objective": "deployable-lev-cagr-ddfloor-lagged-fit-interior",
            "champion_config": best_key,
            "dd_floor": FLOOR,
            "penalty": PEN,
            "floor_held": bool(dd >= FLOOR),
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "restart_cagrs": shs,
            "sensitivity": sens,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableCagrDDfloor655",
    )


if __name__ == "__main__":
    main()
