"""H587: RegimeConditionalCalmar -- H570 regime caps, optimize leveraged Calmar.

Bull (SPY>200d MA): cap=2.5x. Bear: cap=1.0x. tv=6.5%/lb=20d.
Objective: maximize CAGR_lev / |MaxDD_lev| (Calmar ratio).

H558 used pure Calmar (Calmar 1.802 but Sharpe 1.731).
H570 regime + Calmar: tests if regime conditioning improves Calmar performance.

Run: uv run python scripts/run_h587.py
Expected runtime: ~3-4 hours.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from set8_pool import build_pool
from write_summary import write_summary

from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies import MovingAverageCrossover
from erised.validation.walk_forward import walk_forward

TRADING_DAYS = 252
BORROW_COST_ANNUAL = 0.005
LEV_TARGET_VOL = 0.065
LEV_MAX_BULL = 2.5
LEV_MAX_BEAR = 1.0
LEV_LOOKBACK = 20


def apply_regime_leverage(r_arr, spy_close, r_index):
    r = pd.Series(r_arr, index=r_index)
    spy_ma200 = spy_close.rolling(200, min_periods=100).mean()
    bull = (spy_close > spy_ma200).reindex(r_index).fillna(True)
    cap = pd.Series(np.where(bull.values, LEV_MAX_BULL, LEV_MAX_BEAR), index=r_index)
    rv = r.rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).std().shift(1) * np.sqrt(TRADING_DAYS)
    lev = (LEV_TARGET_VOL / rv).clip(upper=cap)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    borrow = (lev - 1.0).clip(lower=0.0) * BORROW_COST_ANNUAL / TRADING_DAYS
    return (lev * r - borrow).values


def portfolio_metrics(r_arr):
    ann_mean = r_arr.mean() * TRADING_DAYS
    ann_std = r_arr.std() * np.sqrt(TRADING_DAYS)
    sh = ann_mean / ann_std if ann_std > 0 else 0.0
    cagr = (1 + r_arr).prod() ** (TRADING_DAYS / len(r_arr)) - 1
    cum = np.cumprod(1 + r_arr)
    dd = (cum / np.maximum.accumulate(cum) - 1).min()
    calmar = cagr / abs(dd) if dd < 0 else 0.0
    return sh, calmar, cagr, dd


def mdr_warm_start(mat, n):
    cov = np.cov(mat.T) * TRADING_DAYS
    vols = np.sqrt(np.diag(cov))

    def neg_dr(w):
        pv = w @ cov @ w
        return 0.0 if pv <= 0 else -(np.dot(w, vols) / np.sqrt(pv))

    res = minimize(
        neg_dr,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return res.x


def main():
    print("H587: RegimeConditionalCalmar\n" + "=" * 64)

    pool = build_pool()
    mat = pool["mat"]
    strategy_names = pool["strategy_names"]
    clean = pool["clean"]
    folds_aa = pool["folds_aa"]
    spy_close = pool["spy_close"]
    spy = pool["spy"]
    n = mat.shape[1]

    spy_ma200 = spy_close.rolling(200, min_periods=100).mean()
    bull_frac = (spy_close > spy_ma200).reindex(clean.index).fillna(True).mean()
    print(f"Bull regime fraction: {bull_frac:.1%}")

    # MDR warm-start (same as H570)
    w0 = mdr_warm_start(mat, n)

    # Lev-Sharpe reference (same as H570 for comparison)
    def neg_lev_sharpe(w):
        r_lev = apply_regime_leverage(mat @ w, spy_close, clean.index)
        s = r_lev.std() * np.sqrt(TRADING_DAYS)
        return -(r_lev.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    res_sh = minimize(
        neg_lev_sharpe,
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    r_lev_ref = apply_regime_leverage(mat @ res_sh.x, spy_close, clean.index)
    sh_ref, cal_ref, cg_ref, dd_ref = portfolio_metrics(r_lev_ref)
    print(
        f"Lev-Sharpe ref (H570 style): Sh={sh_ref:.3f} Cal={cal_ref:.3f} CAGR={cg_ref:.2%} DD={dd_ref:.2%}"
    )

    # Lev-Calmar objective with H570 regime caps
    def neg_lev_calmar(w):
        r_lev = apply_regime_leverage(mat @ w, spy_close, clean.index)
        cagr = (1 + r_lev).prod() ** (TRADING_DAYS / len(r_lev)) - 1
        cum = np.cumprod(1 + r_lev)
        dd = (cum / np.maximum.accumulate(cum) - 1).min()
        return -cagr / abs(dd) if dd < 0 else 0.0

    # Warm-start from Sharpe-optimized solution
    res_cal = minimize(
        neg_lev_calmar,
        res_sh.x,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    w_opt = res_cal.x
    r_lev = apply_regime_leverage(mat @ w_opt, spy_close, clean.index)
    best_port = pd.Series(r_lev, index=clean.index)
    sh, cal, cg, dd = portfolio_metrics(r_lev)
    print(f"Regime Calmar: Sh={sh:.3f} Cal={cal:.3f} CAGR={cg:.2%} MaxDD={dd:.2%}")
    print(f"Delta vs H570: Sh {sh - 2.021:+.3f}, Cal {cal - 1.706:+.3f}")

    lo, hi = bootstrap_sharpe_ci(best_port, n_boot=5000)
    print(f"CI [{lo:.3f}, {hi:.3f}]")
    top_w = {strategy_names[i]: float(v) for i, v in enumerate(w_opt) if v > 0.005}
    print(f"Top weights: {top_w}")

    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (best_port.index >= fold.test_start) & (best_port.index < fold.test_end)
        fr = best_port[mask]
        fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs))
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = best_port.corr(wh.combined_returns.reindex(best_port.index))
    print(f"H001 correlation: {corr:+.4f}")

    decision = "KEEP" if sh > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="587",
        name="regime-conditional-calmar",
        strategy_class="RegimeConditionalCalmar",
        family="Set8 Batch3 Regime",
        sharpe=sh,
        ci_low=lo,
        ci_high=hi,
        cagr=cg,
        max_drawdown=dd,
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            f"H570 regime caps (bull={LEV_MAX_BULL}x/bear={LEV_MAX_BEAR}x, 200d MA) "
            f"+ lev-Calmar objective (vs H570 lev-Sharpe). "
            f"tv={LEV_TARGET_VOL:.1%}/lb={LEV_LOOKBACK}d. Bull fraction: {bull_frac:.1%}. "
            f"Lev-Sharpe ref: Sh={sh_ref:.3f} Cal={cal_ref:.3f}. "
            f"H570 reference: Sh=2.021 Cal=1.706. "
            f"H558 reference (flat-cap Calmar): Cal=1.802 Sh=1.731. "
            f"Delta vs H570: Sh {sh - 2.021:+.3f}, Cal {cal - 1.706:+.3f}."
        ),
        canonical_params={
            "objective": "lev-calmar-regime-conditional",
            "leverage_target_vol": LEV_TARGET_VOL,
            "leverage_max_bull": LEV_MAX_BULL,
            "leverage_max_bear": LEV_MAX_BEAR,
            "leverage_lookback": LEV_LOOKBACK,
            "ma_window": 200,
            "bull_regime_fraction": round(float(bull_frac), 4),
            "lev_sharpe_ref": round(sh_ref, 4),
            "top_weights": top_w,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
