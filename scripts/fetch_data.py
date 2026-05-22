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
    "SPY": "1993-01-01",   # inception 1993-01-22
    "TLT": "2002-01-01",   # inception 2002-07-26
    "GLD": "2004-01-01",   # inception 2004-11-18
}


def main() -> None:
    for symbol, start in SYMBOLS.items():
        report = ingest(symbol, _full_path(symbol), start=start)
        print(f"\n=== {symbol} ===")
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
