"""Point-in-time fundamental data loader for cross-sectional strategies.

Data source: SEC EDGAR XBRL via fetch_fundamentals.py.
Coverage: ~2009+ (XBRL mandatory for large-cap filers from 2009).

Key principle: values become available on their `filed_date`, NOT on their
`period_end` date. Using period_end introduces 45-75 days of lookahead bias.

Usage:
    from quant.data.fundamentals import load_fundamentals_panel, load_pit_field

    # Build a daily panel of book equity (point-in-time)
    book_equity = load_fundamentals_panel(tickers, "book_equity")
    # Returns DataFrame: rows=business_dates, cols=tickers, values=most recently
    # filed value as of each date (NaN before first filing available).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[4] / "data"
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"

VALID_FIELDS = [
    "total_assets",
    "book_equity",
    "net_income",
    "gross_profit",
    "revenue",
    "operating_cf",
    "current_assets",
    "current_liabilities",
    "long_term_debt",
    "shares_outstanding",
]


@lru_cache(maxsize=512)
def _load_ticker_fundamentals(ticker: str) -> pd.DataFrame | None:
    """Load raw filing-level fundamentals for a single ticker (cached)."""
    path = FUNDAMENTALS_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    # Ensure datetime types
    for col in ("period_end", "filed_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def load_pit_field(
    ticker: str,
    field: str,
    date_index: pd.DatetimeIndex,
    annual_only: bool = False,
) -> pd.Series:
    """Build a point-in-time daily series for one ticker and one fundamental field.

    Value on date t = the most recently filed value with filed_date <= t.
    Returns NaN before the first filing date.

    annual_only: if True, only use 10-K filings (annual). If False, also use 10-Q.
    """
    df = _load_ticker_fundamentals(ticker)
    if df is None or field not in df.columns:
        return pd.Series(np.nan, index=date_index, name=ticker)

    df = df[["filed_date", "form", field]].dropna(subset=[field])
    if annual_only:
        df = df[df["form"] == "10-K"]
    if df.empty:
        return pd.Series(np.nan, index=date_index, name=ticker)

    # Keep the most recently filed value for each filed_date
    df = df.sort_values("filed_date").drop_duplicates("filed_date", keep="last")
    df = df.set_index("filed_date")[field].sort_index()

    # Build daily series: as-of join — value = last filing with filed_date <= t
    result = df.reindex(df.index.union(date_index)).ffill().reindex(date_index)
    result.name = ticker
    return result


def load_fundamentals_panel(
    tickers: list[str],
    field: str,
    date_index: pd.DatetimeIndex,
    annual_only: bool = False,
) -> pd.DataFrame:
    """Build a point-in-time panel for `field` across all tickers.

    Returns DataFrame with rows=date_index, cols=tickers.
    NaN where no filing data is available (pre-2009 or missing ticker).
    """
    if field not in VALID_FIELDS:
        raise ValueError(f"Unknown field '{field}'. Valid: {VALID_FIELDS}")

    series_list = []
    for ticker in tickers:
        s = load_pit_field(ticker, field, date_index, annual_only=annual_only)
        series_list.append(s)

    return pd.DataFrame(series_list).T


def compute_book_to_market(
    book_equity: pd.DataFrame,
    prices_panel: pd.DataFrame,
    shares_outstanding: pd.DataFrame,
) -> pd.DataFrame:
    """Compute B/M = book_equity / (price × shares_outstanding).

    All inputs are point-in-time panels (rows=dates, cols=tickers).
    Returns B/M ratio panel; NaN where data is missing.
    """
    market_cap = prices_panel * shares_outstanding
    bm = book_equity / market_cap
    # Remove negative book equity (undefined B/M)
    return bm.where(book_equity > 0)


def compute_earnings_yield(
    net_income: pd.DataFrame,
    prices_panel: pd.DataFrame,
    shares_outstanding: pd.DataFrame,
) -> pd.DataFrame:
    """Compute E/P = net_income / (price × shares_outstanding).

    Returns earnings yield; NaN for negative earnings (or missing data).
    """
    market_cap = prices_panel * shares_outstanding
    ep = net_income / market_cap
    return ep.where(net_income > 0)


def compute_piotroski_fscore(
    tickers: list[str],
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute a simplified 8-point Piotroski F-score for each stock and date.

    Signals (1 point each):
      Profitability (4):
        F1: ROA > 0 (net_income / total_assets > 0)
        F2: Operating CF > 0
        F3: Change in ROA > 0 (ROA improved vs prior year)
        F4: Accruals < 0 (operating_cf/total_assets > ROA — cash earnings > accrual earnings)
      Leverage/Liquidity (2):
        F5: Change in long_term_debt/total_assets < 0 (leverage decreased)
        F6: Change in current_ratio > 0 (liquidity improved)
      Operating Efficiency (2):
        F7: Change in gross_profit/revenue > 0 (gross margin improved)
        F8: Change in revenue/total_assets > 0 (asset turnover improved)

    Returns integer panel (0-8); NaN where insufficient history.
    """
    # Load all required fields (annual filings only for year-over-year changes)
    fields = [
        "total_assets",
        "net_income",
        "operating_cf",
        "long_term_debt",
        "current_assets",
        "current_liabilities",
        "gross_profit",
        "revenue",
    ]
    panels: dict[str, pd.DataFrame] = {}
    for f in fields:
        panels[f] = load_fundamentals_panel(tickers, f, date_index, annual_only=True)

    ta = panels["total_assets"]
    ni = panels["net_income"]
    ocf = panels["operating_cf"]
    ltd = panels["long_term_debt"].fillna(0.0)
    ca = panels["current_assets"]
    cl = panels["current_liabilities"]
    gp = panels["gross_profit"]
    rev = panels["revenue"]

    # Current ratios and derived metrics
    roa = ni / ta
    lev = ltd / ta
    curr_ratio = ca / cl.replace(0, np.nan)
    gm = gp / rev.replace(0, np.nan)
    ato = rev / ta

    # Year-ago values (shift ~252 trading days back)
    lag = 252
    roa_lag = roa.shift(lag)
    lev_lag = lev.shift(lag)
    curr_lag = curr_ratio.shift(lag)
    gm_lag = gm.shift(lag)
    ato_lag = ato.shift(lag)

    # Individual F-score components (1 if condition met, 0 otherwise, NaN if data missing)
    f1 = (roa > 0).astype(float).where(roa.notna())
    f2 = (ocf > 0).astype(float).where(ocf.notna())
    f3 = (roa > roa_lag).astype(float).where(roa.notna() & roa_lag.notna())
    f4 = (ocf / ta > roa).astype(float).where(ocf.notna() & ta.notna() & roa.notna())
    f5 = (lev < lev_lag).astype(float).where(lev.notna() & lev_lag.notna())
    f6 = (curr_ratio > curr_lag).astype(float).where(curr_ratio.notna() & curr_lag.notna())
    f7 = (gm > gm_lag).astype(float).where(gm.notna() & gm_lag.notna())
    f8 = (ato > ato_lag).astype(float).where(ato.notna() & ato_lag.notna())

    # Sum (NaN components are treated as 0 for the sum, but total is NaN if all are NaN)
    components = [f1, f2, f3, f4, f5, f6, f7, f8]
    fscore = sum(c.fillna(0) for c in components)
    # Mark as NaN if more than 4 components are missing
    missing_count = sum(c.isna().astype(int) for c in components)
    return fscore.where(missing_count <= 4)
