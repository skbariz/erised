"""Synthetic daily bars for tests, examples, and pipeline checks.

The series are geometric random walks with positive drift and occasional fat-tail
shocks. Beyond the drift there is nothing to predict, so backtest results on them
say nothing about a strategy; use them to exercise plumbing only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_bars(
    start: str = "1993-01-29",
    end: str = "2022-12-30",
    initial_price: float = 44.0,
    daily_drift: float = 0.0004,
    daily_vol: float = 0.011,
    seed: int = 42,
) -> pd.DataFrame:
    """Return business-day OHLCV + adj_close bars that pass ``validate_bars``."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    log_returns = rng.normal(daily_drift, daily_vol, size=n)
    shocks = rng.choice([0, 1], size=n, p=[0.995, 0.005])
    log_returns += rng.normal(0, 0.04, size=n) * shocks

    close = initial_price * np.exp(np.cumsum(log_returns))

    intraday_range = np.abs(rng.normal(0, daily_vol * 0.7, n))
    high = close * (1 + intraday_range)
    low = close * (1 - intraday_range)
    open_ = close * (1 + rng.normal(0, daily_vol * 0.3, n))
    high = np.maximum.reduce([high, low, open_, close])
    low = np.minimum.reduce([high, low, open_, close])

    volume = rng.integers(1_000_000, 50_000_000, size=n)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "date"
    return df
