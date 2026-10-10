"""Cross-sectional stock data loading.

Individual stock parquet files live in data/stocks/{TICKER}.parquet.
Point-in-time S&P 500 membership lives in data/sp500_members.parquet.
GICS sector mapping lives in data/sp500_sectors.parquet.

Run scripts/fetch_sp500_members.py then scripts/fetch_stocks.py to populate.
"""

from __future__ import annotations

import pandas as pd

from ..config import DATA_DIR

STOCKS_DIR = DATA_DIR / "stocks"
MEMBERS_FILE = DATA_DIR / "sp500_members.parquet"
SECTORS_FILE = DATA_DIR / "sp500_sectors.parquet"

RESEARCH_END = pd.Timestamp("2022-12-31")


def load_stock_adj_close(ticker: str, research_only: bool = True) -> pd.Series:
    path = STOCKS_DIR / f"{ticker.upper()}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data for {ticker}. Run scripts/fetch_stocks.py first.")
    s = pd.read_parquet(path)["adj_close"]
    s.index = pd.to_datetime(s.index)
    s.name = ticker.upper()
    if research_only:
        s = s.loc[:RESEARCH_END]
    return s


def load_stocks_panel(tickers: list[str], research_only: bool = True) -> pd.DataFrame:
    """Load adj_close panel. Returns DataFrame with rows=dates, cols=tickers."""
    import contextlib

    series: dict[str, pd.Series] = {}
    for t in tickers:
        with contextlib.suppress(FileNotFoundError):
            series[t] = load_stock_adj_close(t, research_only=research_only)
    if not series:
        raise FileNotFoundError("No stock data found. Run scripts/fetch_stocks.py first.")
    panel = pd.DataFrame(series)
    panel.index = pd.to_datetime(panel.index)
    panel.sort_index(inplace=True)
    return panel


def load_sp500_members(research_only: bool = True) -> pd.DataFrame:
    """Load wide membership table: rows=dates, cols=tickers, values=bool.

    True means the stock was a member of the S&P 500 on that date.
    """
    if not MEMBERS_FILE.exists():
        raise FileNotFoundError(
            f"No membership file at {MEMBERS_FILE}. Run scripts/fetch_sp500_members.py first."
        )
    df = pd.read_parquet(MEMBERS_FILE)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    if research_only:
        df = df.loc[:RESEARCH_END]
    return df


def load_sp500_sectors() -> pd.Series:
    """Return Series: index=ticker, values=GICS sector string."""
    if not SECTORS_FILE.exists():
        raise FileNotFoundError(
            f"No sectors file at {SECTORS_FILE}. Run scripts/fetch_sp500_members.py first."
        )
    df = pd.read_parquet(SECTORS_FILE)
    return df.iloc[:, 0]


def get_all_historical_tickers() -> list[str]:
    """Return every ticker that ever appeared in the membership table."""
    members = load_sp500_members(research_only=False)
    return members.columns.tolist()
