"""Ticker symbols become file names, so path-like symbols must be rejected."""

from __future__ import annotations

import pandas as pd
import pytest

from erised.data.fundamentals import load_pit_field
from erised.data.stocks import is_valid_ticker, load_stock_adj_close, validate_ticker

PATH_LIKE = [
    "/etc/passwd",
    "../secrets",
    "a/b",
    "..",
    "C:\\x",
    "AAPL\n",
    "",
    "BRK.B",
    "TOOLONGTICKER",
]


@pytest.mark.parametrize("ticker", ["AAPL", "BRK-B", "BF-B", "A", "GOOGL", "T"])
def test_plain_symbols_are_valid(ticker):
    assert is_valid_ticker(ticker)
    assert validate_ticker(ticker.lower()) == ticker


@pytest.mark.parametrize("ticker", PATH_LIKE)
def test_path_like_symbols_are_rejected(ticker):
    assert not is_valid_ticker(ticker)


@pytest.mark.parametrize("ticker", ["/etc/passwd", "../secrets", "a/b"])
def test_loaders_refuse_path_like_symbols(ticker):
    with pytest.raises(ValueError, match="Invalid ticker"):
        load_stock_adj_close(ticker)
    with pytest.raises(ValueError, match="Invalid ticker"):
        load_pit_field(ticker, "book_equity", pd.DatetimeIndex([]))
