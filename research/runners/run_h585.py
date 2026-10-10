"""H585: SPYTLTCorrRegimeConditional -- SPY-TLT 60d correlation as regime signal.

Positive corr (crisis/bear): cap=1.0x. Negative corr (normal): cap=2.5x.
Always-computable signal (no data gaps), captures flight-to-safety dynamics.
H097 showed this correlation signal produces Sharpe 0.992 at portfolio level.

Run: uv run python scripts/run_h585.py
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

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import MovingAverageCrossover
from erised.validation.walk_forward import walk_forward

TRADING_DAYS = 252
BORROW_COST_ANNUAL = 0.005
LEV_TARGET_VOL = 0.065
LEV_MAX_BULL = 2.5
LEV_MAX_BEAR = 1.0
LEV_LOOKBACK = 20
CORR_WINDOW = 60


def apply_corr_regime_leverage(r_arr, spy_rets, tlt_rets, r_index):
    r = pd.Series(r_arr, index=r_index)
    corr = spy_rets.rolling(CORR_WINDOW, min_periods=30).corr(tlt_rets)
    # Positive correlation = crisis/bear = de-lever
    bull = corr.reindex(r_index).fillna(-0.1) <= 0
    bull_frac = bull.mean()

    rv = r.rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).std().shift(1) * np.sqrt(TRADING_DAYS)
    cap = pd.Series(np.where(bull.values, LEV_MAX_BULL, LEV_MAX_BEAR), index=r_index)
    lev = (LEV_TARGET_VOL / rv).clip(upper=cap)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    borrow = (lev - 1.0).clip(lower=0.0) * BORROW_COST_ANNUAL / TRADING_DAYS
    return (lev * r - borrow).values, float(bull_frac)


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
    print("H585: SPYTLTCorrRegimeConditional\n" + "=" * 64)

    pool = build_pool()
    mat = pool["mat"]
    strategy_names = pool["strategy_names"]
    clean = pool["clean"]
    folds_aa = pool["folds_aa"]
    spy = pool["spy"]
    n = mat.shape[1]

    spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    tlt = load_research("TLT")
    tlt_rets = tlt["adj_close"].pct_change().fillna(0.0)

    # Compute regime stats
    corr = spy_rets.rolling(CORR_WINDOW, min_periods=30).corr(tlt_rets)
    corr_aligned = corr.reindex(clean.index).fillna(-0.1)
    bull_frac = (corr_aligned <= 0).mean()
    print(f"Bull regime fraction (SPY-TLT corr <= 0): {bull_frac:.1%}")
    print(f"Mean correlation: {corr_aligned.mean():.3f}")

    w0 = mdr_warm_start(mat, n)

    def neg_lev_sharpe(w):
        r_lev, _ = apply_corr_regime_leverage(mat @ w, spy_rets, tlt_rets, clean.index)
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
    r_lev, _ = apply_corr_regime_leverage(mat @ w_opt, spy_rets, tlt_rets, clean.index)
    best_port = pd.Series(r_lev, index=clean.index)
    sh, cal, cg, dd = portfolio_metrics(r_lev)
    print(f"Corr regime: Sh={sh:.3f} Cal={cal:.3f} CAGR={cg:.2%} MaxDD={dd:.2%}")

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
    corr_h001 = best_port.corr(wh.combined_returns.reindex(best_port.index))
    print(f"H001 correlation: {corr_h001:+.4f}")

    decision = "KEEP" if sh > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="585",
        name="spy-tlt-corr-regime-conditional",
        strategy_class="SPYTLTCorrRegimeConditional",
        family="Set8 Batch3 Regime",
        sharpe=sh,
        ci_low=lo,
        ci_high=hi,
        cagr=cg,
        max_drawdown=dd,
        n_trades=0,
        h001_corr=float(corr_h001),
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            f"SPY-TLT {CORR_WINDOW}d rolling correlation as regime: "
            f"corr<=0 (normal) cap={LEV_MAX_BULL}x; corr>0 (crisis) cap={LEV_MAX_BEAR}x. "
            f"tv={LEV_TARGET_VOL:.1%}/lb={LEV_LOOKBACK}d. "
            f"Bull (normal) fraction: {bull_frac:.1%}. "
            f"H570 reference (MA regime): Sh=2.021 DD=-7.50%. "
            f"Delta vs H570: {sh - 2.021:+.3f} Sh."
        ),
        canonical_params={
            "objective": "lev-sharpe-corr-regime",
            "corr_window": CORR_WINDOW,
            "leverage_target_vol": LEV_TARGET_VOL,
            "leverage_max_bull": LEV_MAX_BULL,
            "leverage_max_bear": LEV_MAX_BEAR,
            "leverage_lookback": LEV_LOOKBACK,
            "bull_regime_fraction": round(float(bull_frac), 4),
            "top_weights": top_w,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
