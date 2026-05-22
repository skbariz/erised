"""Seal the holdout. Run ONCE, ever.

This script splits the full data file into:
- SPY_research.parquet : everything before HOLDOUT_START
- SPY_holdout.parquet  : everything from HOLDOUT_START onward

The holdout file's SHA-256 is recorded in data/HOLDOUT_HASH.txt.
If we re-run this script, it refuses to overwrite an existing holdout
unless --force is passed (which we should never do during the sprint).

The hash file is committed to git. If the holdout changes, the diff
is visible in PRs.

Run: uv run python scripts/seal_holdout.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.data.loader import (  # noqa: E402
    _full_path,
    _holdout_path,
    _research_path,
    file_hash,
    load_full,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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

SYMBOLS = ["SPY", "TLT", "GLD"]


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Re-seal even if holdout exists")
    args = p.parse_args()

    reports = [seal_one(s, force=args.force) for s in SYMBOLS]

    # Write the hash record, committed to git.
    hash_file = _full_path("SPY").parent / "HOLDOUT_HASH.txt"
    lines = [
        "# Holdout integrity record. Committed to git.",
        f"# Holdout start: {HOLDOUT_START}",
        "",
    ]
    for r in reports:
        if r["status"] == "sealed":
            lines.append(f"{r['symbol']}  sha256={r['holdout_sha256']}  rows={r['holdout_rows']}  range={r['holdout_start']}..{r['holdout_end']}")
        elif r["status"] == "already_sealed":
            lines.append(f"{r['symbol']}  sha256={r['hash']}  (already sealed, not re-written)")
    hash_file.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", hash_file)


if __name__ == "__main__":
    main()
