"""The ETF and index universe used by single-asset and multi-asset strategies.

Each entry maps a local symbol to the first date requested from the data vendor. Start
dates sit just before each fund's inception so the full history is captured.
"""

from __future__ import annotations

ETF_UNIVERSE: dict[str, str] = {
    "SPY": "1993-01-01",  # S&P 500; inception 1993-01-22
    "TLT": "2002-01-01",  # 20+ year Treasuries; inception 2002-07-26
    "GLD": "2004-01-01",  # Gold; inception 2004-11-18
    "SHY": "2002-01-01",  # 1-3 year Treasuries; inception 2002-07-26
    "IEF": "2002-01-01",  # 7-10 year Treasuries; inception 2002-07-30
    "LQD": "2002-01-01",  # Investment-grade corporates; inception 2002-07-02
    "HYG": "2007-01-01",  # High-yield corporates; inception 2007-04-11
    "DBC": "2006-01-01",  # Commodities; inception 2006-02-06
    "VNQ": "2004-01-01",  # REITs; inception 2004-09-23
    "IVE": "2000-01-01",  # S&P 500 Value; inception 2000-05-22
    "IVW": "2000-01-01",  # S&P 500 Growth; inception 2000-05-22
    "IWM": "2000-01-01",  # Russell 2000; inception 2000-05-22
    "QQQ": "1999-01-01",  # Nasdaq-100; inception 1999-03-10
    "XLP": "1998-01-01",  # Consumer Staples; inception 1998-12-16
    "XLU": "1998-01-01",  # Utilities; inception 1998-12-16
    "VIX": "1993-01-01",  # CBOE Volatility Index (regime signal only, not tradable)
}

# Local symbol -> Yahoo Finance ticker, where they differ.
YAHOO_TICKERS: dict[str, str] = {"VIX": "^VIX"}
