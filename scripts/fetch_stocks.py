"""Download daily OHLCV for all historical S&P 500 members via yfinance.

Reads data/sp500_tickers.txt (created by fetch_sp500_members.py).
Outputs one parquet per ticker to data/stocks/{TICKER}.parquet.

Run: uv run python scripts/fetch_sp500_members.py   (first)
     uv run python scripts/fetch_stocks.py

This script is resumable: tickers already downloaded are skipped.
Expect ~30-60 minutes for a full download (~700-900 historical tickers).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erised.config import DATA_DIR  # noqa: E402
from erised.data.stocks import is_valid_ticker  # noqa: E402

STOCKS_DIR = DATA_DIR / "stocks"
TICKERS_FILE = DATA_DIR / "sp500_tickers.txt"

FETCH_START = "1990-01-01"  # Extra history for warm-up before 1993
FETCH_END = "2026-12-31"  # Full series including holdout (holdout cut done separately)

RATE_LIMIT_SLEEP = 0.25  # seconds between downloads


def _download_ticker(ticker: str) -> pd.DataFrame | None:
    try:
        raw = yf.download(
            ticker,
            start=FETCH_START,
            end=FETCH_END,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            return None
        # yfinance returns MultiIndex columns when auto_adjust=True
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower().replace(" ", "_") for c in raw.columns]
        else:
            raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        # Rename 'close' to 'adj_close' (auto_adjust means Close IS adjusted)
        if "close" in raw.columns and "adj_close" not in raw.columns:
            raw = raw.rename(columns={"close": "adj_close"})
        raw.index = pd.to_datetime(raw.index)
        return raw[["adj_close"]].dropna()
    except Exception as e:
        log.warning(f"{ticker}: download failed — {e}")
        return None


def main() -> None:
    STOCKS_DIR.mkdir(parents=True, exist_ok=True)

    if not TICKERS_FILE.exists():
        log.error(f"Tickers file not found: {TICKERS_FILE}. Run fetch_sp500_members.py first.")
        sys.exit(1)

    tickers = [t.strip() for t in TICKERS_FILE.read_text().splitlines() if t.strip()]
    for bad in [t for t in tickers if not is_valid_ticker(t)]:
        log.warning(f"Skipping invalid ticker symbol in {TICKERS_FILE.name}: {bad!r}")
    tickers = [t for t in tickers if is_valid_ticker(t)]
    log.info(f"Total tickers to download: {len(tickers)}")

    skipped = downloaded = failed = 0
    for i, ticker in enumerate(tickers):
        out_path = STOCKS_DIR / f"{ticker}.parquet"
        if out_path.exists():
            skipped += 1
            continue

        df = _download_ticker(ticker)
        if df is not None and not df.empty:
            df.to_parquet(out_path)
            downloaded += 1
            log.info(f"[{i + 1}/{len(tickers)}] {ticker}: {len(df)} rows saved")
        else:
            failed += 1
            log.warning(f"[{i + 1}/{len(tickers)}] {ticker}: no data")

        time.sleep(RATE_LIMIT_SLEEP)

    log.info(f"Done. Downloaded: {downloaded}, Skipped (existing): {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
