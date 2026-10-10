"""Synthetic daily bars for tests, examples, and pipeline checks.

The series are geometric random walks with positive drift and occasional fat-tail
shocks. Beyond the drift there is nothing to predict, so backtest results on them
say nothing about a strategy; use them to exercise plumbing only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import _full_path

SYNTHETIC_END = "2026-05-20"

# Per-symbol (start, initial price, daily drift, daily vol, seed). Start dates match real
# inception dates so the research/holdout split and data-availability gaps behave as with
# real data.
SYNTHETIC_PARAMS: dict[str, tuple[str, float, float, float, int]] = {
    "SPY": ("1993-01-29", 44.0, 0.00040, 0.011, 42),
    "TLT": ("2002-07-30", 90.0, 0.00010, 0.007, 43),
    "GLD": ("2004-11-18", 45.0, 0.00020, 0.009, 44),
    "SHY": ("2002-07-31", 77.0, 0.00005, 0.001, 45),
    "IEF": ("2002-07-31", 82.0, 0.00010, 0.004, 46),
    "LQD": ("2002-07-31", 110.0, 0.00010, 0.004, 47),
    "HYG": ("2007-04-11", 95.0, 0.00020, 0.006, 48),
    "DBC": ("2006-02-07", 25.0, 0.00010, 0.010, 49),
    "VNQ": ("2004-09-30", 30.0, 0.00030, 0.012, 50),
    "IVE": ("2000-05-23", 60.0, 0.00030, 0.010, 51),
    "IVW": ("2000-05-23", 65.0, 0.00030, 0.012, 52),
    "IWM": ("2000-05-23", 55.0, 0.00030, 0.013, 53),
    "QQQ": ("1999-03-11", 50.0, 0.00040, 0.015, 54),
    "XLP": ("1998-12-17", 15.0, 0.00020, 0.007, 55),
    "XLU": ("1998-12-17", 13.0, 0.00020, 0.007, 56),
    "VIX": ("1993-01-29", 20.0, 0.00000, 0.050, 57),
}


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


def write_synthetic_universe(end: str = SYNTHETIC_END) -> list[str]:
    """Write a synthetic ``<SYMBOL>.parquet`` for every universe symbol; return the paths."""
    paths = []
    for symbol, (start, price, drift, vol, seed) in SYNTHETIC_PARAMS.items():
        out = _full_path(symbol)
        out.parent.mkdir(parents=True, exist_ok=True)
        synthetic_bars(start, end, price, drift, vol, seed).to_parquet(out, compression="snappy")
        paths.append(str(out))
    return paths
