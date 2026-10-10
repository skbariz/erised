"""Strategy interface.

A strategy is a function that, given the history of prices up to and
including time t, outputs a TARGET POSITION for the *next* bar.

Position semantics:
  +1.0 = fully long
   0.0 = flat
  -1.0 = fully short
Values between -1 and +1 indicate scaled exposure.

The "next bar" convention prevents lookahead: today's close is observed,
the position is set for tomorrow, and the return is realized on tomorrow's
return. The backtester enforces this — strategies cannot violate it."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class for all strategies.

    Override `generate_positions` to produce a Series of target positions
    indexed by date. The returned Series will be SHIFTED by one bar by
    the backtester to enforce no-lookahead trading.
    """

    name: str = "unnamed"

    @abstractmethod
    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        """Given a prices DataFrame, return target positions in [-1, 1].

        Implementations MUST only use data available at each timestamp.
        The framework cannot detect lookahead inside vectorized code;
        the discipline is on the strategy author.
        """
        ...


class MultiAssetStrategy(ABC):
    """Base class for strategies that allocate across multiple assets.

    Override `generate_weights` to produce a DataFrame of portfolio weights.
    Column names must match asset identifiers used by run_multiasset_backtest:
    'SPY' -> prices['adj_close'], 'TLT' -> prices['tlt_close'], etc.

    Weights at time t are shifted one bar by the framework before being applied,
    so today's weight is applied to tomorrow's return. No-lookahead discipline
    is the same as for Strategy: use only data available at each timestamp.
    """

    name: str = "unnamed"

    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of portfolio weights, one column per asset.

        Values should be in [0, 1] (long-only) or [-1, 1] (long/short).
        Rows that sum to 0 represent a cash position.
        """
        ...


class BuyAndHold(Strategy):
    """The benchmark. Always long 1.0."""

    name = "buy_and_hold"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=prices.index, name="position")
