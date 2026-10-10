"""Seal the holdout: split each full series into research and holdout files, once.

For every symbol, ``<SYMBOL>.parquet`` is split into:
- ``<SYMBOL>_research.parquet``: everything before ``HOLDOUT_START``
- ``<SYMBOL>_holdout.parquet``: everything from ``HOLDOUT_START`` onward

Each holdout file's SHA-256 is written to ``HOLDOUT_HASH.txt`` in the data directory.
Existing holdouts are never overwritten unless ``force=True``.

CLI: ``erised data seal [--force]``
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR
from .loader import _holdout_path, _research_path, file_hash, load_full
from .universe import ETF_UNIVERSE

logger = logging.getLogger(__name__)


# The split date. Everything ON OR AFTER this date is sealed.
# Choosing 2023-01-01 gives us:
# - ~30 years of research data (1993-2022)
# - ~3+ years of holdout (2023-present)
# This holdout includes meaningful regime variety:
#   - 2023: AI rally, regional banking crisis
#   - 2024: continued bull market
#   - 2025-2026: whatever happens
HOLDOUT_START = date(2023, 1, 1)


def seal_one(symbol: str, force: bool = False) -> dict:
    holdout_path = _holdout_path(symbol)
    research_path = _research_path(symbol)

    if holdout_path.exists() and not force:
        existing_hash = file_hash(holdout_path)
        logger.warning(
            "[%s] holdout already sealed at %s (sha256=%s). Skipping. Use --force to override.",
            symbol,
            holdout_path,
            existing_hash[:16],
        )
        return {"symbol": symbol, "status": "already_sealed", "hash": existing_hash}

    df = load_full(symbol)
    split_ts = pd.Timestamp(HOLDOUT_START)

    research_df = df.loc[df.index < split_ts]
    holdout_df = df.loc[df.index >= split_ts]

    if len(research_df) < 252 * 10:
        raise RuntimeError(
            f"[{symbol}] research set too small: {len(research_df)} rows. Need at least 10 years."
        )
    if len(holdout_df) < 252:
        raise RuntimeError(
            f"[{symbol}] holdout too small: {len(holdout_df)} rows. Need at least 1 year."
        )

    research_df.to_parquet(research_path, compression="snappy")
    holdout_df.to_parquet(holdout_path, compression="snappy")

    h = file_hash(holdout_path)
    logger.info(
        "[%s] sealed. research: %d rows (%s to %s), holdout: %d rows (%s to %s), sha256=%s",
        symbol,
        len(research_df),
        research_df.index.min().date(),
        research_df.index.max().date(),
        len(holdout_df),
        holdout_df.index.min().date(),
        holdout_df.index.max().date(),
        h[:16],
    )
    return {
        "symbol": symbol,
        "status": "sealed",
        "research_rows": len(research_df),
        "holdout_rows": len(holdout_df),
        "research_start": str(research_df.index.min().date()),
        "research_end": str(research_df.index.max().date()),
        "holdout_start": str(holdout_df.index.min().date()),
        "holdout_end": str(holdout_df.index.max().date()),
        "holdout_sha256": h,
    }


def seal_holdout(symbols: list[str] | None = None, force: bool = False) -> list[dict]:
    """Seal every symbol (default: the ETF universe) and write the hash record."""
    reports = [seal_one(s, force=force) for s in (symbols or list(ETF_UNIVERSE))]
    hash_file: Path = DATA_DIR / "HOLDOUT_HASH.txt"
    lines = [
        "# Holdout integrity record.",
        f"# Holdout start: {HOLDOUT_START}",
        "",
    ]
    for r in reports:
        if r["status"] == "sealed":
            lines.append(
                f"{r['symbol']}  sha256={r['holdout_sha256']}  rows={r['holdout_rows']}  range={r['holdout_start']}..{r['holdout_end']}"
            )
        elif r["status"] == "already_sealed":
            lines.append(f"{r['symbol']}  sha256={r['hash']}  (already sealed, not re-written)")
    hash_file.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", hash_file)
    return reports
