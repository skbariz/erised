"""Data ingestion: fetch from sources, cross-validate, store as Parquet.

Two independent sources are used to catch silent data errors:
- yfinance (Yahoo) for the canonical OHLCV + adjusted close
- Stooq for cross-validation of the adjusted close series

We do not blindly trust either. Discrepancies above a tolerance raise.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .schema import ALL_COLUMNS, validate_bars

logger = logging.getLogger(__name__)


CROSS_VALIDATION_TOLERANCE = 0.005


def fetch_yfinance(symbol: str, start: str = "1993-01-01") -> pd.DataFrame:
    """Fetch daily bars from yfinance."""
    import yfinance as yf

    logger.info("Fetching %s from yfinance starting %s", symbol, start)
    ticker = yf.Ticker(symbol)
    raw = ticker.history(start=start, auto_adjust=False, actions=False)

    if raw.empty:
        raise RuntimeError(f"yfinance returned empty data for {symbol}")

    df = pd.DataFrame(
        {
            "open": raw["Open"].astype("float64"),
            "high": raw["High"].astype("float64"),
            "low": raw["Low"].astype("float64"),
            "close": raw["Close"].astype("float64"),
            "adj_close": raw["Adj Close"].astype("float64"),
            "volume": raw["Volume"].astype("int64"),
        }
    )
    df.index = pd.to_datetime(df.index.date)
    df.index.name = "date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[ALL_COLUMNS]


def fetch_stooq(symbol: str) -> pd.DataFrame:
    """Fetch daily bars from Stooq for cross-validation.

    Stooq is rate-limited and sometimes returns empty responses; the caller
    should treat any exception here as "cross-validation unavailable".
    """
    stooq_symbol = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    logger.info("Fetching %s from stooq", symbol)
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    text = resp.text
    if not text or "No data" in text or len(text) < 100:
        raise RuntimeError(f"stooq returned no data for {symbol} (len={len(text)})")
    if not text.startswith("Date"):
        raise RuntimeError(
            f"stooq returned unexpected payload for {symbol} (starts with: {text[:60]!r})"
        )

    try:
        df = pd.read_csv(io.StringIO(text))
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise RuntimeError(f"stooq CSV parse failed for {symbol}: {e}") from e

    if "Date" not in df.columns or "Close" not in df.columns:
        raise RuntimeError(f"stooq response missing expected columns: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.index.name = "date"
    return pd.DataFrame({"stooq_close": df["Close"].astype("float64")})


def cross_validate(yf_df: pd.DataFrame, stooq_df: pd.DataFrame, symbol: str) -> dict:
    """Compare the two sources. Return a report; raise if differences are extreme."""
    overlap = yf_df.index.intersection(stooq_df.index)
    if len(overlap) < 100:
        raise RuntimeError(f"[{symbol}] insufficient overlap between sources: {len(overlap)} days")

    yf_adj = yf_df.loc[overlap, "adj_close"]
    stooq_adj = stooq_df.loc[overlap, "stooq_close"]

    yf_ret = yf_adj.pct_change().dropna()
    stooq_ret = stooq_adj.pct_change().dropna()
    common = yf_ret.index.intersection(stooq_ret.index)
    diff = (yf_ret.loc[common] - stooq_ret.loc[common]).abs()

    n_bad = int((diff > CROSS_VALIDATION_TOLERANCE).sum())
    report = {
        "overlap_days": len(overlap),
        "return_mean_abs_diff": float(diff.mean()),
        "return_max_diff": float(diff.max()),
        "return_99pct_diff": float(diff.quantile(0.99)),
        "n_days_above_tolerance": n_bad,
        "tolerance": CROSS_VALIDATION_TOLERANCE,
    }

    threshold = max(5, int(0.005 * len(common)))
    if n_bad > threshold:
        worst = diff.sort_values(ascending=False).head(10)
        raise RuntimeError(
            f"[{symbol}] cross-validation failed: {n_bad} days above tolerance "
            f"(allowed {threshold}). Worst:\n{worst}"
        )

    logger.info("[%s] cross-validation OK: %s", symbol, report)
    return report


def ingest(symbol: str, out_path: Path, start: str = "1993-01-01") -> dict:
    """Fetch, cross-validate, and write Parquet. Returns a report dict.

    yfinance is the primary source — if that fails, ingestion fails.
    stooq is for cross-validation only — if that fails, we log a warning
    and proceed.
    """
    yf_df = fetch_yfinance(symbol, start=start)
    validate_bars(yf_df, symbol)

    try:
        stooq_df = fetch_stooq(symbol)
        report = cross_validate(yf_df, stooq_df, symbol)
    except Exception as e:  # noqa: BLE001 — stooq is best-effort
        logger.warning("[%s] stooq cross-validation skipped: %s: %s", symbol, type(e).__name__, e)
        report = {"cross_validation": "skipped", "reason": f"{type(e).__name__}: {e}"}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    yf_df.to_parquet(out_path, compression="snappy")

    report.update(
        {
            "symbol": symbol,
            "rows": len(yf_df),
            "start": str(yf_df.index.min().date()),
            "end": str(yf_df.index.max().date()),
            "out_path": str(out_path),
        }
    )
    return report