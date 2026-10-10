"""H582: ThreeStateRegimeLeverage -- three-state regime using SPY MA + VIX.

Strong bull (SPY>200d MA AND VIX<20): cap=3.0x
Neutral (only one signal confirms): cap=1.5x
Bear (SPY<200d MA AND VIX>=25): cap=0.75x

Run: uv run python scripts/run_h582.py
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
from erised.strategies.base import MovingAverageCrossover
from erised.validation.walk_forward import walk_forward

TRADING_DAYS = 252
BORROW_COST_ANNUAL = 0.005
LEV_TARGET_VOL = 0.065
LEV_MAX_STRONG = 3.0
LEV_MAX_NEUTRAL = 1.5
LEV_MAX_BEAR = 0.75
LEV_LOOKBACK = 20
VIX_BULL_THRESH = 20.0
VIX_BEAR_THRESH = 25.0


def apply_three_state_leverage(r_arr, spy_close, vix_series, r_index):
    r = pd.Series(r_arr, index=r_index)
    spy_ma200 = spy_close.rolling(200, min_periods=100).mean()
    ma_bull = (spy_close > spy_ma200).reindex(r_index).fillna(True)
    vix = vix_series.reindex(r_index).ffill()
    vix_bull = vix < VIX_BULL_THRESH
    vix_bear = vix >= VIX_BEAR_THRESH

    # Three states
    strong_bull = ma_bull & vix_bull
    bear = (~ma_bull) & vix_bear
    # neutral = everything else

    cap_vals = np.where(
        strong_bull.values, LEV_MAX_STRONG, np.where(bear.values, LEV_MAX_BEAR, LEV_MAX_NEUTRAL)
    )
    cap = pd.Series(cap_vals, index=r_index)

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
    print("H582: ThreeStateRegimeLeverage\n" + "=" * 64)

    pool = build_pool()
    mat = pool["mat"]
    strategy_names = pool["strategy_names"]
    clean = pool["clean"]
    folds_aa = pool["folds_aa"]
    spy_close = pool["spy_close"]
    vix_series = pool["vix_series"]
    spy = pool["spy"]
    n = mat.shape[1]

    spy_ma200 = spy_close.rolling(200, min_periods=100).mean()
    ma_bull = (spy_close > spy_ma200).reindex(clean.index).fillna(True)
    vix = vix_series.reindex(clean.index).ffill()
    strong_bull_frac = (ma_bull & (vix < VIX_BULL_THRESH)).mean()
    bear_frac = ((~ma_bull) & (vix >= VIX_BEAR_THRESH)).mean()
    neutral_frac = 1.0 - strong_bull_frac - bear_frac
    print(
        f"State fractions — strong bull: {strong_bull_frac:.1%}, "
        f"neutral: {neutral_frac:.1%}, bear: {bear_frac:.1%}"
    )

    w0 = mdr_warm_start(mat, n)

    def neg_lev_sharpe(w):
        r_lev = apply_three_state_leverage(mat @ w, spy_close, vix_series, clean.index)
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
    r_lev = apply_three_state_leverage(mat @ w_opt, spy_close, vix_series, clean.index)
    best_port = pd.Series(r_lev, index=clean.index)
    sh, cal, cg, dd = portfolio_metrics(r_lev)
    print(f"Three-state: Sh={sh:.3f} Cal={cal:.3f} CAGR={cg:.2%} MaxDD={dd:.2%}")

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
        hypothesis_id="582",
        name="three-state-regime-leverage",
        strategy_class="ThreeStateRegimeLeverage",
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
            f"Strong bull (MA AND VIX<{VIX_BULL_THRESH}): cap={LEV_MAX_STRONG}x. "
            f"Bear (no-MA AND VIX>={VIX_BEAR_THRESH}): cap={LEV_MAX_BEAR}x. Neutral: cap={LEV_MAX_NEUTRAL}x. "
            f"State fractions: strong_bull={strong_bull_frac:.1%}, neutral={neutral_frac:.1%}, bear={bear_frac:.1%}. "
            f"H570 reference (binary MA): Sh=2.021 DD=-7.50%. "
            f"Delta vs H570: {sh - 2.021:+.3f} Sh, {(dd + 0.075) * 100:+.2f}pp MaxDD."
        ),
        canonical_params={
            "objective": "lev-sharpe-three-state",
            "leverage_target_vol": LEV_TARGET_VOL,
            "leverage_max_strong_bull": LEV_MAX_STRONG,
            "leverage_max_neutral": LEV_MAX_NEUTRAL,
            "leverage_max_bear": LEV_MAX_BEAR,
            "leverage_lookback": LEV_LOOKBACK,
            "vix_bull_threshold": VIX_BULL_THRESH,
            "vix_bear_threshold": VIX_BEAR_THRESH,
            "strong_bull_fraction": round(float(strong_bull_frac), 4),
            "bear_fraction": round(float(bear_frac), 4),
            "top_weights": top_w,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
