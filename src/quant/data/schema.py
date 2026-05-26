"""Data schema and constants for the quant platform.

All bar data follows this schema. Enforced at ingestion and load time.
"""

from __future__ import annotations

import pandas as pd

# Canonical column order. Lowercase, no spaces.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Required columns in a price DataFrame
REQUIRED_COLUMNS = OHLCV_COLUMNS

# The index of every price DataFrame is a tz-naive pd.DatetimeIndex
# representing the SESSION DATE (not a timestamp). For daily bars this
# is the trading day. We do not store the close timestamp — that's a
# market-microstructure concern that doesn't belong in daily data.

# Adjusted vs. unadjusted: we store BOTH.
# - close, open, high, low, volume = AS REPORTED (unadjusted)
# - adj_close = split- and dividend-adjusted close
# Strategies should use adj_close for return calculations and
# unadjusted prices for any realism check on absolute price levels.
ADJ_CLOSE_COLUMN = "adj_close"

ALL_COLUMNS = OHLCV_COLUMNS + [ADJ_CLOSE_COLUMN]


def validate_bars(df: pd.DataFrame, symbol: str = "?") -> None:
    """Raise if the DataFrame does not conform to our schema."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"[{symbol}] index must be a DatetimeIndex, got {type(df.index)}")
    if df.index.tz is not None:
        raise ValueError(f"[{symbol}] index must be tz-naive, got tz={df.index.tz}")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"[{symbol}] index must be sorted ascending")
    if df.index.has_duplicates:
        dups = df.index[df.index.duplicated()].tolist()
        raise ValueError(f"[{symbol}] index has duplicates: {dups[:5]}")

    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[{symbol}] missing required columns: {missing}")

    # Sanity: high >= max(open, close, low), low <= min(open, close, high)
    bad_hi = (df["high"] < df[["open", "close", "low"]].max(axis=1)).sum()
    bad_lo = (df["low"] > df[["open", "close", "high"]].min(axis=1)).sum()
    if bad_hi or bad_lo:
        raise ValueError(
            f"[{symbol}] OHLC inconsistencies: {bad_hi} high violations, {bad_lo} low violations"
        )

    # Sanity: no negative or zero prices
    for col in ["open", "high", "low", "close", "adj_close"]:
        if (df[col] <= 0).any():
            raise ValueError(f"[{symbol}] non-positive values in column {col}")

    # Volume can be zero on rare days but never negative
    if (df["volume"] < 0).any():
        raise ValueError(f"[{symbol}] negative volume")
