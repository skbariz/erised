"""Alpaca paper trading scaffold.

The same Strategy classes that backtest will run live. This module
handles the bridge: at each scheduled tick, it fetches current prices,
asks the strategy for a target position, and reconciles with current
holdings via Alpaca's API.

Credentials are read from the ALPACA_API_KEY and ALPACA_SECRET_KEY
environment variables (see .env.example). PaperBroker defaults to Alpaca's
paper-trading endpoint; pass paper=False only if you intend to trade real money.
run_once() is a dry run unless dry_run=False and a broker is supplied.
Requires the optional dependency: pip install "erised[live]".
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from ..strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float


class PaperBroker:
    """Thin wrapper over alpaca-py. Implemented lazily (no SDK import at
    module load time) so the rest of the package works without alpaca-py
    installed."""

    def __init__(self, api_key: str | None = None, secret: str | None = None, paper: bool = True):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret = secret or os.getenv("ALPACA_SECRET_KEY")
        self.paper = paper
        if not self.api_key or not self.secret:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        # Lazy import
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(self.api_key, self.secret, paper=paper)

    def get_account(self) -> Account:
        a = self._client.get_account()
        return Account(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
        )

    def get_position(self, symbol: str) -> Position | None:
        """Return the open position, or None if there is none.

        Only a 404 means "no position". Any other error is re-raised: treating a
        transient failure as flat would make submit_target_position re-buy the
        whole target.
        """
        from alpaca.common.exceptions import APIError

        try:
            p = self._client.get_open_position(symbol)
        except APIError as e:
            if e.status_code == 404:
                return None
            raise
        return Position(symbol=p.symbol, qty=float(p.qty), avg_entry_price=float(p.avg_entry_price))

    def submit_target_position(
        self, symbol: str, target_fraction: float, latest_price: float
    ) -> dict:
        """Bring the position to target_fraction * equity worth of `symbol`."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        acct = self.get_account()
        target_dollars = target_fraction * acct.equity
        target_shares = int(target_dollars / latest_price)
        current = self.get_position(symbol)
        current_shares = int(current.qty) if current else 0
        delta = target_shares - current_shares
        if delta == 0:
            return {"status": "no_change", "current": current_shares, "target": target_shares}
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        order = MarketOrderRequest(
            symbol=symbol,
            qty=abs(delta),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        result = self._client.submit_order(order_data=order)
        return {
            "status": "submitted",
            "order_id": str(result.id),
            "current": current_shares,
            "target": target_shares,
            "delta": delta,
        }


def run_once(
    strategy: Strategy,
    prices: pd.DataFrame,
    symbol: str,
    broker: PaperBroker | None = None,
    dry_run: bool = True,
) -> dict:
    """Compute target position from latest prices and (optionally) submit to broker.

    `prices` should be the historical price series up to and including the
    most recent close, in the same schema as research data.
    """
    positions = strategy.generate_positions(prices)
    target = float(positions.iloc[-1])
    latest_price = float(prices["close"].iloc[-1])
    decision = {
        "timestamp": datetime.now(UTC).isoformat(),
        "symbol": symbol,
        "strategy": strategy.name,
        "target_position": target,
        "latest_price": latest_price,
        "dry_run": dry_run,
    }
    logger.info("Decision: %s", decision)
    if dry_run or broker is None:
        return decision
    fill = broker.submit_target_position(symbol, target, latest_price)
    decision["fill"] = fill
    return decision
