"""PaperBroker position lookup must not mistake API failures for a flat book."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("alpaca")

from alpaca.common.exceptions import APIError  # noqa: E402

from erised.execution.paper_trader import PaperBroker  # noqa: E402


def _api_error(status: int) -> APIError:
    http_error = SimpleNamespace(response=SimpleNamespace(status_code=status), request=None)
    return APIError(json.dumps({"code": status, "message": "x"}), http_error)


def _broker(raises: Exception | None = None, position=None) -> PaperBroker:
    def get_open_position(_symbol):
        if raises:
            raise raises
        return position

    broker = object.__new__(PaperBroker)
    broker._client = SimpleNamespace(get_open_position=get_open_position)
    return broker


def test_missing_position_returns_none():
    assert _broker(raises=_api_error(404)).get_position("SPY") is None


@pytest.mark.parametrize("status", [401, 429, 500])
def test_other_api_errors_propagate(status):
    with pytest.raises(APIError):
        _broker(raises=_api_error(status)).get_position("SPY")


def test_open_position_is_parsed():
    pos = SimpleNamespace(symbol="SPY", qty="10", avg_entry_price="400.5")
    result = _broker(position=pos).get_position("SPY")
    assert (result.symbol, result.qty, result.avg_entry_price) == ("SPY", 10.0, 400.5)


def test_missing_credentials_raise(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="must be set"):
        PaperBroker()
