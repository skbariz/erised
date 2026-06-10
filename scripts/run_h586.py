"""H586: AsymmetricVolTargetRegime -- BOTH vol target AND cap vary with regime.

Bull (SPY>200d MA): tv=8.0%, cap=2.5x
Bear:              tv=4.0%, cap=1.0x

H570 only adjusts cap (tv fixed at 6.5%). H586 also reduces vol target in bear
for stronger de-risking; raises it in bull for more CAGR.

Run: uv run python scripts/run_h586.py
Expected runtime: ~3-4 hours.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from set8_pool import build_pool
from write_summary import write_summary

from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover
from quant.validation.walk_forward import walk_forward

TRADING_DAYS = 252
BORROW_COST_ANNUAL = 0.005
LEV_TV_BULL = 0.080
LEV_TV_BEAR = 0.040
LEV_MAX_BULL = 2.5
LEV_MAX_BEAR = 1.0
LEV_LOOKBACK = 20


def apply_asym_vol_regime_leverage(r_arr, spy_close, r_index):
    r = pd.Series(r_arr, index=r_index)
    spy_ma200 = spy_close.rolling(200, min_periods=100).mean()
    bull = (spy_close > spy_ma200).reindex(r_index).fillna(True)

    tv = pd.Series(np.where(bull.values, LEV_TV_BULL, LEV_TV_BEAR), index=r_index)
    cap = pd.Series(np.where(bull.values, LEV_MAX_BULL, LEV_MAX_BEAR), index=r_index)

    rv = r.rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).std().shift(1) * np.sqrt(TRADING_DAYS)
    lev = (tv / rv).clip(upper=cap)
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
    print("H586: AsymmetricVolTargetRegime\n" + "=" * 64)

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
    print(
        f"Bull: tv={LEV_TV_BULL:.0%}/cap={LEV_MAX_BULL}x | Bear: tv={LEV_TV_BEAR:.0%}/cap={LEV_MAX_BEAR}x"
    )

    w0 = mdr_warm_start(mat, n)

    def neg_lev_sharpe(w):
        r_lev = apply_asym_vol_regime_leverage(mat @ w, spy_close, clean.index)
        s = r_lev.std() * np.sqrt(TRADING_DAYS)
        return -(r_lev.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    res = minimize(
        neg_lev_sharpe,
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    w_opt = res.x
    r_lev = apply_asym_vol_regime_leverage(mat @ w_opt, spy_close, clean.index)
    best_port = pd.Series(r_lev, index=clean.index)
    sh, cal, cg, dd = portfolio_metrics(r_lev)
    print(f"Asym vol regime: Sh={sh:.3f} Cal={cal:.3f} CAGR={cg:.2%} MaxDD={dd:.2%}")

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
        hypothesis_id="586",
        name="asymmetric-vol-target-regime",
        strategy_class="AsymmetricVolTargetRegime",
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
            f"Both tv AND cap vary with regime. Bull (MA): tv={LEV_TV_BULL:.0%}/cap={LEV_MAX_BULL}x. "
            f"Bear: tv={LEV_TV_BEAR:.0%}/cap={LEV_MAX_BEAR}x. lb={LEV_LOOKBACK}d. "
            f"Bull fraction: {bull_frac:.1%}. "
            f"H570 reference (cap-only asym, tv=6.5%): Sh=2.021 DD=-7.50%. "
            f"Delta vs H570: {sh - 2.021:+.3f} Sh, {(dd + 0.075) * 100:+.2f}pp MaxDD."
        ),
        canonical_params={
            "objective": "lev-sharpe-asym-vol-regime",
            "leverage_tv_bull": LEV_TV_BULL,
            "leverage_tv_bear": LEV_TV_BEAR,
            "leverage_max_bull": LEV_MAX_BULL,
            "leverage_max_bear": LEV_MAX_BEAR,
            "leverage_lookback": LEV_LOOKBACK,
            "ma_window": 200,
            "bull_regime_fraction": round(float(bull_frac), 4),
            "top_weights": top_w,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
