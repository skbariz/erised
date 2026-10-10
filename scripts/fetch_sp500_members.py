"""Fetch S&P 500 point-in-time membership and GICS sectors from Wikipedia.

Outputs:
  data/sp500_members.parquet  -- wide bool DataFrame: dates x tickers
  data/sp500_sectors.parquet  -- ticker -> GICS sector mapping
  data/sp500_tickers.txt      -- all unique historical tickers (one per line)

Run: uv run python scripts/fetch_sp500_members.py

Note: Wikipedia historical changes go back reliably to ~2000. Pre-2000 folds
may have minor survivorship-bias due to incomplete records before that date.
Ticker symbols use current conventions; historical splits/renames handled
by yfinance on download.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erised.config import DATA_DIR  # noqa: E402

RESEARCH_END = pd.Timestamp("2022-12-31")
RESEARCH_START = pd.Timestamp("1993-01-01")

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _fetch_wiki_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    import io
    import urllib.request

    log.info("Fetching Wikipedia S&P 500 page...")
    req = urllib.request.Request(
        WIKI_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"},
    )
    with urllib.request.urlopen(req) as resp:
        html = resp.read()
    tables = pd.read_html(io.BytesIO(html), header=0)
    current = tables[0]
    changes = tables[1]
    return current, changes


def _parse_current(current: pd.DataFrame) -> pd.DataFrame:
    """Extract ticker and GICS sector from the current members table."""
    col_map = {}
    for col in current.columns:
        cl = str(col).lower()
        if "symbol" in cl or "ticker" in cl:
            col_map["ticker"] = col
        elif "gics sector" in cl or "sector" in cl:
            col_map["sector"] = col
    if "ticker" not in col_map:
        raise ValueError(f"Cannot find ticker column in: {list(current.columns)}")
    df = current[[col_map["ticker"]]].copy()
    df.columns = ["ticker"]
    if "sector" in col_map:
        df["sector"] = current[col_map["sector"]].values
    else:
        df["sector"] = "Unknown"
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.strip()
    return df.dropna(subset=["ticker"]).reset_index(drop=True)


def _parse_changes(changes: pd.DataFrame) -> list[tuple[pd.Timestamp, str, str]]:
    """Parse additions/removals table into list of (date, ticker, action) events.

    action is 'add' or 'remove'.
    """
    events: list[tuple[pd.Timestamp, str, str]] = []

    # Flatten multi-level columns if present
    if isinstance(changes.columns, pd.MultiIndex):
        changes.columns = [" ".join(str(c) for c in col).strip() for col in changes.columns]

    cols = {str(c).lower(): c for c in changes.columns}

    # Find date column
    date_col = None
    for k in cols:
        if "date" in k:
            date_col = cols[k]
            break
    if date_col is None:
        log.warning("No date column found in changes table; skipping historical reconstruction.")
        return events

    # Find added/removed ticker columns
    added_col = removed_col = None
    for k, v in cols.items():
        if "added" in k and ("ticker" in k or "symbol" in k):
            added_col = v
        if "removed" in k and ("ticker" in k or "symbol" in k):
            removed_col = v

    if added_col is None and removed_col is None:
        # Try simpler heuristic
        for k, v in cols.items():
            if "added" in k:
                added_col = v
            if "removed" in k:
                removed_col = v

    for _, row in changes.iterrows():
        try:
            dt = pd.Timestamp(str(row[date_col]).strip())
        except Exception:
            continue
        if pd.isna(dt) or dt < RESEARCH_START:
            continue
        if added_col is not None:
            ticker = str(row[added_col]).strip().replace(".", "-")
            if ticker and ticker.lower() not in ("nan", "none", ""):
                events.append((dt, ticker, "add"))
        if removed_col is not None:
            ticker = str(row[removed_col]).strip().replace(".", "-")
            if ticker and ticker.lower() not in ("nan", "none", ""):
                events.append((dt, ticker, "remove"))
    return events


def build_membership(
    current_tickers: list[str],
    events: list[tuple[pd.Timestamp, str, str]],
) -> pd.DataFrame:
    """Reconstruct daily point-in-time membership.

    Start from today's members and apply changes backwards in time.
    Returns wide bool DataFrame: rows=business days, cols=tickers.
    """
    bdays = pd.bdate_range(RESEARCH_START, RESEARCH_END)

    # All tickers ever seen
    all_tickers = set(current_tickers)
    for _, ticker, _ in events:
        all_tickers.add(ticker)
    all_tickers_sorted = sorted(all_tickers)

    # Start with current membership as of RESEARCH_END
    membership: dict[str, bool] = dict.fromkeys(current_tickers, True)
    for t in all_tickers_sorted:
        if t not in membership:
            membership[t] = False

    # Build event map sorted descending (we walk backwards from end to start)
    sorted_events = sorted(events, key=lambda x: x[0], reverse=True)

    # Build daily snapshots by walking backwards
    # Group events by date
    from collections import defaultdict

    events_by_date: dict[pd.Timestamp, list[tuple[str, str]]] = defaultdict(list)
    for dt, ticker, action in sorted_events:
        events_by_date[dt].append((ticker, action))

    rows = {}
    event_dates = sorted(events_by_date.keys(), reverse=True)
    event_idx = 0

    current_state = dict(membership)

    for day in reversed(bdays):
        day_ts = pd.Timestamp(day)
        # Apply events that happen ON this day (when walking backwards:
        # an 'add' event means the stock was NOT in the index before this date)
        while event_idx < len(event_dates) and event_dates[event_idx] > day_ts:
            for ticker, action in events_by_date[event_dates[event_idx]]:
                if action == "add":
                    current_state[ticker] = False
                elif action == "remove":
                    current_state[ticker] = True
            event_idx += 1
        rows[day_ts] = dict(current_state)

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.sort_index(inplace=True)
    df = df.reindex(pd.bdate_range(RESEARCH_START, RESEARCH_END))
    # Forward-fill any gaps (weekends/holidays have no changes)
    df.ffill(inplace=True)
    df.fillna(False, inplace=True)
    return df.astype(bool)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    current, changes = _fetch_wiki_tables()
    current_info = _parse_current(current)
    log.info(f"Current S&P 500 members: {len(current_info)}")

    events = _parse_changes(changes)
    log.info(f"Historical change events parsed: {len(events)}")

    # Save sector map
    sectors = current_info.set_index("ticker")["sector"]
    sectors_path = DATA_DIR / "sp500_sectors.parquet"
    sectors.to_frame().to_parquet(sectors_path)
    log.info(f"Sectors saved: {sectors_path}")

    # Build membership
    log.info("Building point-in-time membership table...")
    members = build_membership(current_info["ticker"].tolist(), events)
    log.info(f"Membership table: {members.shape[0]} days x {members.shape[1]} tickers")

    members_path = DATA_DIR / "sp500_members.parquet"
    members.to_parquet(members_path)
    log.info(f"Membership saved: {members_path}")

    # Save ticker list
    tickers_path = DATA_DIR / "sp500_tickers.txt"
    tickers_path.write_text("\n".join(sorted(members.columns)), encoding="utf-8")
    log.info(f"Tickers saved: {tickers_path}  ({len(members.columns)} total)")


if __name__ == "__main__":
    main()
