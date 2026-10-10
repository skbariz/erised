"""H588: RealBorrowRegimeConditional -- H570 regime caps + H569 realistic borrow cost.

Bull (SPY>200d MA): cap=2.5x. Bear: cap=1.0x. tv=6.5%/lb=20d.
Borrow cost: Fed Funds step-function + 50bps spread (H569 mechanism).

Quantifies: does H570's +0.143 regime advantage survive realistic financing?
H569 result: flat-regime + real cost = Sh 1.731 (vs H559 flat cost 1.879, delta -0.148).
H588 result should be ~H570 (2.021) minus ~0.148 drag ≈ 1.87.

Run: uv run python scripts/run_h588.py
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
LEV_TARGET_VOL = 0.065
LEV_MAX_BULL = 2.5
LEV_MAX_BEAR = 1.0
LEV_LOOKBACK = 20
SPREAD_BPS = 0.0050  # 50bps above Fed Funds

# Fed Funds rate step-function (approximate historical periods)
# Key eras: ~5% (1993-2000), ~1% (2001-2004), ~5% (2005-2008), ~0.1% (2009-2015),
# ~1.5% (2016-2018), ~2.5% (2018-2019), ~0.1% (2020-2021), ~5% (2022-)
FED_FUNDS_STEPS = [
    ("1993-01-01", "2001-01-01", 0.0550),
    ("2001-01-01", "2004-06-01", 0.0150),
    ("2004-06-01", "2008-12-01", 0.0450),
    ("2008-12-01", "2015-12-01", 0.0010),
    ("2015-12-01", "2019-07-01", 0.0200),
    ("2019-07-01", "2020-03-01", 0.0175),
    ("2020-03-01", "2022-03-01", 0.0010),
    ("2022-03-01", "2099-01-01", 0.0500),
]


def get_borrow_cost_series(index):
    """Build daily borrow cost series from Fed Funds step-function + spread."""
    cost = pd.Series(SPREAD_BPS, index=index)  # base spread
    for start, end, rate in FED_FUNDS_STEPS:
        mask = (index >= start) & (index < end)
        cost[mask] = rate + SPREAD_BPS
    return cost


def apply_regime_real_cost_leverage(r_arr, spy_close, r_index, borrow_series):
    r = pd.Series(r_arr, index=r_index)
    spy_ma200 = spy_close.rolling(200, min_periods=100).mean()
    bull = (spy_close > spy_ma200).reindex(r_index).fillna(True)
    cap = pd.Series(np.where(bull.values, LEV_MAX_BULL, LEV_MAX_BEAR), index=r_index)
    rv = r.rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).std().shift(1) * np.sqrt(TRADING_DAYS)
    lev = (LEV_TARGET_VOL / rv).clip(upper=cap)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    borrow = (lev - 1.0).clip(lower=0.0) * borrow_series / TRADING_DAYS
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
    print("H588: RealBorrowRegimeConditional\n" + "=" * 64)

    pool = build_pool()
    mat = pool["mat"]
    strategy_names = pool["strategy_names"]
    clean = pool["clean"]
    folds_aa = pool["folds_aa"]
    spy_close = pool["spy_close"]
    spy = pool["spy"]
    n = mat.shape[1]

    borrow_series = get_borrow_cost_series(clean.index)
    mean_borrow = borrow_series.mean()
    print(f"Mean borrow rate: {mean_borrow:.3%}/yr")

    spy_ma200 = spy_close.rolling(200, min_periods=100).mean()
    bull_frac = (spy_close > spy_ma200).reindex(clean.index).fillna(True).mean()
    print(f"Bull regime fraction: {bull_frac:.1%}")

    w0 = mdr_warm_start(mat, n)

    def neg_lev_sharpe(w):
        r_lev = apply_regime_real_cost_leverage(mat @ w, spy_close, clean.index, borrow_series)
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
    r_lev = apply_regime_real_cost_leverage(mat @ w_opt, spy_close, clean.index, borrow_series)
    best_port = pd.Series(r_lev, index=clean.index)
    sh, cal, cg, dd = portfolio_metrics(r_lev)
    print(f"Real-cost regime: Sh={sh:.3f} Cal={cal:.3f} CAGR={cg:.2%} MaxDD={dd:.2%}")
    print(f"Delta vs H570 (flat cost): {sh - 2.021:+.3f} Sh")
    print(f"Delta vs H569 (flat regime): {sh - 1.731:+.3f} Sh")

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
        hypothesis_id="588",
        name="real-borrow-regime-conditional",
        strategy_class="RealBorrowRegimeConditional",
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
            f"+ realistic Fed Funds borrow cost (+{SPREAD_BPS * 10000:.0f}bps spread). "
            f"Mean borrow: {mean_borrow:.3%}/yr. Bull fraction: {bull_frac:.1%}. "
            f"H570 ref (flat cost): Sh=2.021. H569 ref (flat regime): Sh=1.731. "
            f"Delta vs H570: {sh - 2.021:+.3f}. Delta vs H569: {sh - 1.731:+.3f}."
        ),
        canonical_params={
            "objective": "lev-sharpe-regime-real-cost",
            "leverage_target_vol": LEV_TARGET_VOL,
            "leverage_max_bull": LEV_MAX_BULL,
            "leverage_max_bear": LEV_MAX_BEAR,
            "leverage_lookback": LEV_LOOKBACK,
            "ma_window": 200,
            "borrow_spread_bps": int(SPREAD_BPS * 10000),
            "mean_borrow_annual": round(float(mean_borrow), 4),
            "bull_regime_fraction": round(float(bull_frac), 4),
            "top_weights": top_w,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
