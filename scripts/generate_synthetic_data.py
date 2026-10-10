"""Generate synthetic price data for all pipeline symbols for offline testing.

This is for verifying the pipeline plumbing only. Real data comes from
scripts/fetch_data.py which requires network access to yfinance/stooq.

Run: python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erised.data.loader import _full_path  # noqa: E402
from erised.data.synthetic import synthetic_bars  # noqa: E402

END_DATE = "2026-05-20"

# Per-symbol config: (start_date, initial_price, daily_drift, daily_vol, rng_seed)
# Start dates match real inception dates so downstream seal/research checks pass.
# Drift and vol are rough proxies only — NOT for backtesting.
SYMBOL_CONFIGS: dict[str, tuple[str, float, float, float, int]] = {
    "SPY": ("1993-01-29", 44.0, 0.00040, 0.011, 42),
    "TLT": ("2002-07-30", 90.0, 0.00010, 0.007, 43),
    "GLD": ("2004-11-18", 45.0, 0.00020, 0.009, 44),
    "SHY": ("2002-07-31", 77.0, 0.00005, 0.001, 45),
    "IEF": ("2002-07-31", 82.0, 0.00010, 0.004, 46),
    "LQD": ("2002-07-31", 110.0, 0.00010, 0.004, 47),
    "HYG": ("2007-04-11", 95.0, 0.00020, 0.006, 48),
    "DBC": ("2006-02-07", 25.0, 0.00010, 0.010, 49),
    "VNQ": ("2004-09-30", 30.0, 0.00030, 0.012, 50),
    "IVE": ("2000-05-23", 60.0, 0.00030, 0.010, 51),
    "IVW": ("2000-05-23", 65.0, 0.00030, 0.012, 52),
    "IWM": ("2000-05-23", 55.0, 0.00030, 0.013, 53),
    "QQQ": ("1999-03-11", 50.0, 0.00040, 0.015, 54),
    "XLP": ("1998-12-17", 15.0, 0.00020, 0.007, 55),
    "XLU": ("1998-12-17", 13.0, 0.00020, 0.007, 56),
}


def main() -> None:
    out_dir = _full_path("SPY").parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for symbol, (start, price, drift, vol, seed) in SYMBOL_CONFIGS.items():
        df = synthetic_bars(start, END_DATE, price, drift, vol, seed)
        out = _full_path(symbol)
        df.to_parquet(out, compression="snappy")
        print(
            f"Wrote {len(df):5d} synthetic rows to {out.name}"
            f"  range {df.index.min().date()} to {df.index.max().date()}"
            f"  close ${df['close'].iloc[0]:.2f} to ${df['close'].iloc[-1]:.2f}"
        )


if __name__ == "__main__":
    main()
