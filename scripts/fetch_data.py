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


SYMBOLS = ["SPY"]
START = "1993-01-01"


def main() -> None:
    for symbol in SYMBOLS:
        report = ingest(symbol, _full_path(symbol), start=START)
        print(f"\n=== {symbol} ===")
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
