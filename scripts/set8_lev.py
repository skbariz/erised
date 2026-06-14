"""Shared leverage / optimization machinery for Set8 Batch4 runners (H591-H600).

Keeps each Batch4 runner thin: it supplies only its objective + leverage spec,
and imports the common pieces from here. All consume the cached pool from set8_pool.

Import: from set8_lev import (...)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant.metrics.performance import sharpe
from quant.strategies.base import MovingAverageCrossover
from quant.validation.walk_forward import walk_forward

TRADING_DAYS = 252
BORROW_COST_ANNUAL = 0.005  # flat 50bps spread (H559/H570 convention)


def rolling_vol(r: pd.Series, lookback: int, downside: bool) -> pd.Series:
    """Annualized trailing vol, shifted 1 day (no lookahead). Downside = semi-deviation."""
    if downside:
        neg2 = np.minimum(r.to_numpy(), 0.0) ** 2
        ms = pd.Series(neg2, index=r.index).rolling(lookback, min_periods=lookback).mean()
        rv = np.sqrt(ms)
    else:
        rv = r.rolling(lookback, min_periods=lookback).std()
    return rv.shift(1) * np.sqrt(TRADING_DAYS)


def apply_leverage(
    r_arr,
    index,
    target_vol,
    cap,
    lookback: int = 20,
    downside: bool = False,
    borrow_annual: float = BORROW_COST_ANNUAL,
):
    """Vol-target leverage with a (possibly regime-varying) cap.

    target_vol and cap may each be a scalar or a Series aligned to `index`.
    Returns the leveraged net-of-borrow return array.
    """
    r = pd.Series(r_arr, index=index)
    rv = rolling_vol(r, lookback, downside)
    tv = target_vol if np.isscalar(target_vol) else pd.Series(target_vol, index=index)
    lev = (tv / rv).clip(upper=cap)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    borrow = (lev - 1.0).clip(lower=0.0) * borrow_annual / TRADING_DAYS
    return (lev * r - borrow).to_numpy()


def vix_bull_mask(vix_series, index, threshold):
    """Boolean Series: True where VIX < threshold (bull), aligned to index."""
    return (vix_series.reindex(index).ffill() < threshold).fillna(True)


def two_state_cap(mask, bull_cap, bear_cap, index):
    """Series of leverage caps: bull_cap where mask True, else bear_cap."""
    return pd.Series(np.where(np.asarray(mask), bull_cap, bear_cap), index=index)


def vix_momentum_mask(vix, index, threshold, window=20):
    """Bull = VIX < threshold AND VIX not rising (VIX <= its `window`-day mean). H620 champion signal."""
    vixr = vix.reindex(index).ffill().bfill()
    not_rising = vixr <= vixr.rolling(window, min_periods=max(5, window // 2)).mean()
    return ((vixr < threshold) & not_rising).fillna(True)


def mdr_warm_start(mat, n):
    """Max-diversification-ratio weights as a warm start for the leveraged objective."""
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


def slsqp(neg_obj, w0, n, maxiter: int = 2000):
    """Long-only, sum-to-1 SLSQP minimization of neg_obj(w)."""
    res = minimize(
        neg_obj,
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": maxiter, "ftol": 1e-12},
    )
    return res.x


def portfolio_metrics(r_arr):
    """Return (sharpe, calmar, cagr, max_drawdown, sortino) for a daily return array."""
    ann_mean = r_arr.mean() * TRADING_DAYS
    ann_std = r_arr.std() * np.sqrt(TRADING_DAYS)
    sh = ann_mean / ann_std if ann_std > 0 else 0.0
    cagr = (1 + r_arr).prod() ** (TRADING_DAYS / len(r_arr)) - 1
    cum = np.cumprod(1 + r_arr)
    dd = (cum / np.maximum.accumulate(cum) - 1).min()
    calmar = cagr / abs(dd) if dd < 0 else 0.0
    ds = r_arr[r_arr < 0]
    ds_std = np.sqrt((ds**2).mean()) * np.sqrt(TRADING_DAYS) if len(ds) > 0 else 1.0
    sortino = ann_mean / ds_std if ds_std > 0 else 0.0
    return sh, calmar, cagr, dd, sortino


def fold_sharpe_list(best_port, folds):
    """Per-fold OOS Sharpe (0.0 for folds with < 10 days)."""
    out = []
    for fold in folds:
        mask = (best_port.index >= fold.test_start) & (best_port.index < fold.test_end)
        fr = best_port[mask]
        out.append(float(sharpe(fr)) if len(fr) > 10 else 0.0)
    return out


def h001_correlation(best_port, spy, folds):
    """Correlation of the strategy's combined returns with H001 MA(50,200)."""
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    return float(best_port.corr(wh.combined_returns.reindex(best_port.index)))


def top_weights(strategy_names, w, threshold: float = 0.005):
    return {strategy_names[i]: float(v) for i, v in enumerate(w) if v > threshold}


def expanding_percentile_mask(series, index, q, min_periods=252):
    """Bull mask: True where `series` is below its expanding q-quantile (no lookahead via shift)."""
    s = series.reindex(index).ffill().bfill()
    thr = s.expanding(min_periods=min_periods).quantile(q).shift(1)
    return (s < thr).fillna(True)


def per_strategy_sharpe(clean):
    """Standalone annualized Sharpe of each pool column (NaN-safe)."""
    out = {}
    for c in clean.columns:
        r = clean[c].dropna()
        out[c] = float(sharpe(r)) if len(r) > 10 else 0.0
    return out


def finalize_run(hid, name, cls, family, r, idx, w, names, folds, spy, notes, extra, label):
    """Compute metrics + CI + folds + H001 corr, decide KEEP/kill, write summary, print. Returns dict."""
    from write_summary import write_summary

    from quant.metrics.performance import bootstrap_sharpe_ci

    sh, cal, cg, dd, so = portfolio_metrics(r)
    port = pd.Series(r, index=idx)
    lo, hi = bootstrap_sharpe_ci(port, n_boot=5000)
    fs = fold_sharpe_list(port, folds)
    corr = h001_correlation(port, spy, folds)
    tw = top_weights(names, w) if w is not None else {}
    decision = "KEEP" if sh > 0.51 and lo > 0 else "kill"
    print(
        f"{label}: Sh={sh:.3f} Cal={cal:.3f} Sortino={so:.3f} CAGR={cg:.2%} DD={dd:.2%} "
        f"CI[{lo:.2f},{hi:.2f}] folds {min(fs):.2f}..{max(fs):.2f} corr={corr:+.3f} -> {decision}"
    )
    params = {
        "calmar": round(float(cal), 4),
        "sortino": round(float(so), 4),
        "top_weights": tw,
        "n_strategies": len(names),
    }
    params.update(extra)
    write_summary(
        hypothesis_id=hid,
        name=name,
        strategy_class=cls,
        family=family,
        sharpe=sh,
        ci_low=lo,
        ci_high=hi,
        cagr=cg,
        max_drawdown=dd,
        n_trades=0,
        h001_corr=corr,
        fold_sharpes=fs,
        decision=decision,
        notes=notes,
        canonical_params=params,
    )
    return {"sharpe": sh, "calmar": cal, "cagr": cg, "dd": dd, "sortino": so, "decision": decision}
