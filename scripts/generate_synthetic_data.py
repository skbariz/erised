"""Generate synthetic SPY-like data for offline testing.

This is for verifying the pipeline plumbing only. Real data comes from
scripts/fetch_data.py which requires network access to yfinance/stooq.

Run: python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.data.loader import _full_path  # noqa: E402


def generate_spy_like(start: str = "1993-01-29", end: str = "2026-05-20") -> pd.DataFrame:
    """Produce a plausible-looking SPY history.

    Drift and vol roughly match historical SPY. NOT for backtesting strategies —
    only for testing that the pipeline runs.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start=start, end=end)  # business days
    n = len(dates)

    # Daily log returns: drift ~ 0.04% mean (~10% annual), vol ~ 1% daily
    daily_drift = 0.0004
    daily_vol = 0.011
    log_returns = rng.normal(daily_drift, daily_vol, size=n)

    # Add occasional fat-tail shocks
    shocks = rng.choice([0, 1], size=n, p=[0.995, 0.005])
    shock_mag = rng.normal(0, 0.04, size=n) * shocks
    log_returns = log_returns + shock_mag

    close = 44.0 * np.exp(np.cumsum(log_returns))  # SPY started near $44

    # OHL: simulate intraday range
    intraday_range = np.abs(rng.normal(0, 0.008, n))
    high = close * (1 + intraday_range)
    low = close * (1 - intraday_range)
    open_ = close * (1 + rng.normal(0, 0.003, n))
    # Clamp so OHL is consistent
    high = np.maximum.reduce([high, low, open_, close])
    low = np.minimum.reduce([high, low, open_, close])

    # Volume: ~100M shares/day with variation
    volume = rng.integers(50_000_000, 200_000_000, size=n)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,  # synthetic data has no dividends/splits
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def main() -> None:
    df = generate_spy_like()
    out = _full_path("SPY")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="snappy")
    print(f"Wrote {len(df)} synthetic rows to {out}")
    print(f"Range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Close: ${df['close'].iloc[0]:.2f} -> ${df['close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
