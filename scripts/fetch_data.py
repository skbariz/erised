"""Fetch and store the full SPY history.

Run: uv run python scripts/fetch_data.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Make src/ importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.data.ingest import ingest
from quant.data.loader import _full_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# Per-symbol start dates (limited by ETF inception; yfinance clips automatically)
SYMBOLS = {
    "SPY": "1993-01-01",  # inception 1993-01-22
    "TLT": "2002-01-01",  # inception 2002-07-26
    "GLD": "2004-01-01",  # inception 2004-11-18
    # Batch 10 additions
    "SHY": "2002-01-01",  # 1-3yr Treasuries; inception 2002-07-26 — yield curve short end
    "IEF": "2002-01-01",  # 7-10yr Treasuries; inception 2002-07-30 — yield curve mid
    "LQD": "2002-01-01",  # IG Corporate bonds; inception 2002-07-02 — credit quality
    "HYG": "2007-01-01",  # High Yield bonds; inception 2007-04-11 — credit risk
    "DBC": "2006-01-01",  # Commodities index; inception 2006-02-06 — inflation hedge
    "VNQ": "2004-01-01",  # REIT index; inception 2004-09-23 — real estate
    # Batch 12 additions — equity factor ETFs
    "IVE": "2000-01-01",  # iShares S&P 500 Value; inception 2000-05-22
    "IVW": "2000-01-01",  # iShares S&P 500 Growth; inception 2000-05-22
    "IWM": "2000-01-01",  # iShares Russell 2000 (small cap); inception 2000-05-22
    "QQQ": "1999-01-01",  # Invesco Nasdaq-100 (growth/tech); inception 1999-03-10
    "XLP": "1998-01-01",  # SPDR Consumer Staples; inception 1998-12-16
    "XLU": "1998-01-01",  # SPDR Utilities; inception 1998-12-16
}


def main() -> None:
    for symbol, start in SYMBOLS.items():
        report = ingest(symbol, _full_path(symbol), start=start)
        print(f"\n=== {symbol} ===")
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
