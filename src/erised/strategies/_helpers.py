"""Shared signal and weighting helpers. Private: not part of the public API."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _calendar_month_ranks(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Rank each date from the start and end of its calendar month.

    Uses a fixed business-day calendar anchored at index.min() + 50 years so
    that truncating the price series at any point T does not change the
    rank_end value at T. This makes calendar strategies pass assert_no_lookahead
    (which checks that positions at T are stable under price-data extension).
    """
    start = index.min()
    cal_end = start + pd.DateOffset(years=50)
    all_bdays = pd.bdate_range(start, cal_end)
    ym = all_bdays.year * 100 + all_bdays.month
    ym_series = pd.Series(ym, index=all_bdays, name="ym")
    rank_start = ym_series.groupby(ym_series).cumcount()
    rank_end = ym_series.groupby(ym_series).cumcount(ascending=False)
    return rank_start.reindex(index), rank_end.reindex(index)


def _calendar_quarter_rank_end(index: pd.DatetimeIndex) -> pd.Series:
    """Rank each date from the END of its calendar quarter.

    Same fixed-calendar projection as _calendar_month_ranks.
    """
    start = index.min()
    cal_end = start + pd.DateOffset(years=50)
    all_bdays = pd.bdate_range(start, cal_end)
    yq = all_bdays.year * 10 + all_bdays.quarter
    yq_series = pd.Series(yq, index=all_bdays, name="yq")
    rank_end = yq_series.groupby(yq_series).cumcount(ascending=False)
    return rank_end.reindex(index)


def _h021_weights(
    spy_r: pd.Series, tlt_r: pd.Series, vol_lookback: int
) -> tuple[pd.Series, pd.Series]:
    """Shared H021 inverse-vol SPY/TLT weights."""
    spy_vol = spy_r.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1) * np.sqrt(252)
    tlt_vol = tlt_r.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1) * np.sqrt(252)
    inv_spy = 1.0 / spy_vol.replace(0, np.nan)
    inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
    total = inv_spy + inv_tlt
    return (inv_spy / total).fillna(0.0), (inv_tlt / total).fillna(0.0)


def _vrp_series(spy_r: pd.Series, vix: pd.Series) -> pd.Series:
    """VRP = annualized implied var (VIX^2) minus annualized realized var (21d)."""
    implied_var = (vix.shift(1) / 100.0) ** 2
    realized_var = spy_r.rolling(21, min_periods=10).var().shift(1) * 252
    return (implied_var - realized_var).fillna(0.0)


def _h021_base_weights(prices: pd.DataFrame, vol_lookback: int = 60) -> tuple[pd.Series, pd.Series]:
    """Compute H021 inverse-vol base weights. Returns (w_spy, w_tlt)."""
    spy_r = prices["adj_close"].pct_change()
    tlt_r = prices["tlt_close"].pct_change()
    spy_vol = spy_r.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1) * np.sqrt(252)
    tlt_vol = tlt_r.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1) * np.sqrt(252)
    inv_spy = 1.0 / spy_vol.replace(0, np.nan)
    inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
    total = inv_spy + inv_tlt
    w_spy = (inv_spy / total).fillna(0.0)
    w_tlt = (inv_tlt / total).fillna(0.0)
    return w_spy, w_tlt
