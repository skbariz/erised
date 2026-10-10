"""Download quarterly fundamentals for historical S&P 500 members from SEC EDGAR XBRL.

Uses the EDGAR company facts API (free, no key required):
  https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json

Coverage: XBRL mandatory for large accelerated filers from 2009, accelerated filers from 2010.
Pre-2009 data will be absent — strategies using fundamentals will have zero signal for
folds 0-4 (test periods 1998-2008) and valid signal for folds 5-11 (2008-2022).

Output: one Parquet file per ticker at <data dir>/fundamentals/{TICKER}.parquet
  Columns: period_end, filed_date, form, total_assets, book_equity, net_income,
           gross_profit, revenue, operating_cf, current_assets, current_liabilities,
           long_term_debt, shares_outstanding

CLI: SEC_EDGAR_USER_AGENT="my-project you@example.com" erised data fundamentals
Resumable: tickers already downloaded are skipped. Expect 10-20 minutes for ~1,100
tickers at the SEC's 10 requests/second limit.
"""

from __future__ import annotations

import logging
import os
import time

import pandas as pd
import requests

from ...config import DATA_DIR
from ..stocks import read_ticker_list

log = logging.getLogger(__name__)

FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"

# SEC EDGAR endpoints
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# SEC requires a descriptive user-agent with contact info, e.g. "my-project you@example.com".
# https://www.sec.gov/os/accessing-edgar-data
USER_AGENT_ENV = "SEC_EDGAR_USER_AGENT"

# Rate limit: SEC allows 10 requests/second; we use 0.12s sleep = ~8 req/s
RATE_LIMIT_SLEEP = 0.12

# XBRL concept names to try for each fundamental field (first match wins)
CONCEPTS: dict[str, list[str]] = {
    "total_assets": ["Assets"],
    "book_equity": [
        "StockholdersEquity",
        "StockholdersEquityAttributableToParent",
        "EquityAttributableToOwnersOfParent",
    ],
    "net_income": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "gross_profit": ["GrossProfit"],
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent", "LongTermNotesPayable"],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "SharesOutstanding",
    ],
}


def _build_cik_map(headers: dict[str, str]) -> dict[str, str]:
    """Fetch ticker→CIK mapping from SEC EDGAR. Returns {TICKER: CIK_padded_10digits}."""
    log.info("Fetching CIK map from SEC EDGAR...")
    resp = requests.get(EDGAR_TICKERS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Format: {idx: {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."}}
    mapping: dict[str, str] = {}
    for entry in data.values():
        ticker = entry["ticker"].upper()
        cik = str(entry["cik_str"]).zfill(10)
        mapping[ticker] = cik
    log.info(f"CIK map: {len(mapping)} tickers")
    return mapping


def _extract_concept(
    facts_us_gaap: dict,
    concept_tags: list[str],
    field_name: str,
) -> pd.DataFrame:
    """Extract a financial concept from EDGAR XBRL facts, returning per-filing rows.

    Prefers 10-K and 10-Q forms only. For each (period_end, form) pair, keeps
    the most recently filed value (in case of amendments).

    Returns DataFrame with: period_end, filed_date, form, <field_name>
    """
    for tag in concept_tags:
        if tag not in facts_us_gaap:
            continue
        units = facts_us_gaap[tag].get("units", {})
        # Financial data is in USD (or shares for share counts)
        for unit_key in ("USD", "shares"):
            if unit_key not in units:
                continue
            rows = []
            for item in units[unit_key]:
                form = item.get("form", "")
                if form not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                    continue
                end = item.get("end")
                filed = item.get("filed")
                val = item.get("val")
                if end is None or filed is None or val is None:
                    continue
                rows.append(
                    {
                        "period_end": pd.Timestamp(end),
                        "filed_date": pd.Timestamp(filed),
                        "form": form.replace("/A", ""),  # treat amendments same as originals
                        field_name: float(val),
                    }
                )
            if not rows:
                continue
            df = pd.DataFrame(rows)
            # Keep most-recently-filed row per (period_end, form)
            df = df.sort_values("filed_date").groupby(["period_end", "form"], as_index=False).last()
            return df[["period_end", "filed_date", "form", field_name]]
    # None of the tags found
    return pd.DataFrame(columns=["period_end", "filed_date", "form", field_name])


def _fetch_ticker_fundamentals(
    ticker: str, cik: str, headers: dict[str, str]
) -> pd.DataFrame | None:
    """Fetch and parse EDGAR XBRL facts for one ticker. Returns per-filing DataFrame."""
    url = EDGAR_FACTS_URL.format(cik=cik)
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 404:
            log.debug(f"{ticker}: no EDGAR XBRL data (404)")
            return None
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning(f"{ticker}: fetch error — {e}")
        return None

    facts_us_gaap = data.get("facts", {}).get("us-gaap", {})
    if not facts_us_gaap:
        log.debug(f"{ticker}: no us-gaap facts")
        return None

    # Extract each concept
    concept_dfs: list[pd.DataFrame] = []
    for field, tags in CONCEPTS.items():
        df = _extract_concept(facts_us_gaap, tags, field)
        concept_dfs.append(df)

    if not any(len(d) > 0 for d in concept_dfs):
        return None

    # Merge all concepts on (period_end, form), aligning filed_dates
    # Strategy: for each concept, merge on period_end + form using outer join,
    # then take the latest filed_date across concepts for each filing period
    base = concept_dfs[0]
    for df in concept_dfs[1:]:
        if len(df) == 0:
            continue
        base = base.merge(
            df,
            on=["period_end", "form"],
            how="outer",
            suffixes=("", "_r"),
        )
        # Keep the later of the two filed dates
        if "filed_date_r" in base.columns:
            base["filed_date"] = base[["filed_date", "filed_date_r"]].max(axis=1)
            base.drop(columns=["filed_date_r"], inplace=True)

    # Keep only rows that have at least total_assets or book_equity
    has_data = (
        base["total_assets"].notna() | base.get("book_equity", pd.Series(dtype=float)).notna()
    )
    base = base[has_data].copy()
    return base.sort_values(["period_end", "filed_date"]).reset_index(drop=True)


def download_fundamentals(user_agent: str | None = None) -> dict[str, int]:
    """Download fundamentals for every ticker in the S&P 500 ticker list.

    ``user_agent`` defaults to the ``SEC_EDGAR_USER_AGENT`` environment variable.
    Returns counts of saved, missing-CIK, and failed tickers.
    """
    user_agent = user_agent or os.environ.get(USER_AGENT_ENV, "")
    if not user_agent:
        raise RuntimeError(
            f"Set {USER_AGENT_ENV} to a contact string such as 'my-project you@example.com' "
            "(required by the SEC EDGAR fair-access policy)."
        )
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)

    tickers = read_ticker_list()
    log.info(f"Tickers to process: {len(tickers)}")

    cik_map = _build_cik_map(headers)
    time.sleep(RATE_LIMIT_SLEEP)

    done = {f.stem for f in FUNDAMENTALS_DIR.glob("*.parquet")}
    log.info(f"Already downloaded: {len(done)} tickers (skipping)")

    missing_cik = 0
    errors = 0
    saved = 0

    for i, ticker in enumerate(tickers):
        if ticker in done:
            continue

        cik = cik_map.get(ticker)
        if cik is None:
            # Try common ticker variations
            alt = ticker.replace(".", "-")
            cik = cik_map.get(alt)
        if cik is None:
            log.debug(f"{ticker}: no CIK in EDGAR map")
            missing_cik += 1
            # Write empty sentinel so we don't retry on next run
            pd.DataFrame().to_parquet(FUNDAMENTALS_DIR / f"{ticker}.parquet")
            continue

        df = _fetch_ticker_fundamentals(ticker, cik, headers)
        time.sleep(RATE_LIMIT_SLEEP)

        if df is None or len(df) == 0:
            # Write empty sentinel
            pd.DataFrame().to_parquet(FUNDAMENTALS_DIR / f"{ticker}.parquet")
        else:
            df.to_parquet(FUNDAMENTALS_DIR / f"{ticker}.parquet", index=False)
            saved += 1

        if (i + 1) % 100 == 0:
            log.info(
                f"Progress: {i + 1}/{len(tickers)} tickers processed "
                f"({saved} saved, {missing_cik} no-CIK, {errors} errors)"
            )

    log.info(
        f"Done. Saved: {saved}, no-CIK: {missing_cik}, errors: {errors}, "
        f"total processed: {len(tickers) - len(done)}"
    )
    return {"saved": saved, "missing_cik": missing_cik, "errors": errors}
