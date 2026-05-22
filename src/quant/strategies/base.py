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
return. The backtester enforces this — strategies cannot violate it.
"""

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


class BuyAndHold(Strategy):
    """The benchmark. Always long 1.0."""

    name = "buy_and_hold"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=prices.index, name="position")


class MovingAverageCrossover(Strategy):
    """Classic dual-MA trend-following.

    Long when fast MA > slow MA, flat otherwise. By default uses
    50/200, which is the textbook "golden cross" / "death cross".
    """

    def __init__(self, fast: int = 50, slow: int = 200, allow_short: bool = False):
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short
        self.name = f"ma_cross_{fast}_{slow}{'_ls' if allow_short else ''}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        fast_ma = px.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = px.rolling(self.slow, min_periods=self.slow).mean()
        long_signal = (fast_ma > slow_ma).astype(float)
        if self.allow_short:
            short_signal = (fast_ma < slow_ma).astype(float)
            pos = long_signal - short_signal
        else:
            pos = long_signal
        # Before we have enough history, stay flat.
        pos = pos.where(slow_ma.notna(), 0.0)
        pos.name = "position"
        return pos


class MeanReversionStrategy(Strategy):
    """Short-term mean reversion after large down moves. (Hypothesis 002)

    Enters long when today's return is more than `threshold` standard
    deviations below the recent rolling mean. Holds for `hold_days` bars,
    with consecutive signals extending the hold window via rolling max.

    The rolling mean and std are shifted by one bar so today's return is
    compared against the *past* distribution — not a distribution that
    includes itself. This is the critical lookahead-avoidance discipline.
    """

    def __init__(self, lookback: int = 20, threshold: float = 2.0, hold_days: int = 5):
        if lookback < 5:
            raise ValueError("lookback must be >= 5")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if hold_days < 1:
            raise ValueError("hold_days must be >= 1")
        self.lookback = lookback
        self.threshold = threshold
        self.hold_days = hold_days
        self.name = f"mean_rev_{lookback}_{threshold}_{hold_days}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        r = px.pct_change()

        # Rolling stats shifted one bar: at time t, these reflect the
        # distribution of returns through t-1, not through t.
        rolling_mean = r.rolling(self.lookback, min_periods=self.lookback).mean().shift(1)
        rolling_std = r.rolling(self.lookback, min_periods=self.lookback).std(ddof=1).shift(1)

        # z-score of today's return against yesterday's distribution.
        z = (r - rolling_mean) / rolling_std

        # Entry: z < -threshold (unusually large down move).
        # Where std is NaN or zero (warm-up / degenerate), stay flat.
        entry = (z < -self.threshold).astype(float)
        entry = entry.where(rolling_std.notna() & (rolling_std > 0), 0.0)

        # Remain long for hold_days bars after any entry signal.
        # rolling max naturally handles overlapping signals.
        pos = entry.rolling(self.hold_days, min_periods=1).max()
        pos.name = "position"
        return pos


class ShortHoldMeanReversion(MeanReversionStrategy):
    """Hypothesis 003: 2-sigma z-score entry, 2-day hold.

    Tests whether H002's 5-day hold overstays the reversion peak.
    Canonical config locks lookback=20, threshold=2.0, hold_days=2.
    """

    name = "short_hold_mean_rev"

    def __init__(self, lookback: int = 20, threshold: float = 2.0, hold_days: int = 2):
        super().__init__(lookback=lookback, threshold=threshold, hold_days=hold_days)
