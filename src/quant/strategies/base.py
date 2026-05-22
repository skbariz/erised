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


class RSIReversionStrategy(Strategy):
    """Hypothesis 004: RSI oversold mean reversion.

    Entry when RSI(period) < entry_rsi. Remains long while RSI <= exit_rsi
    AND within hold_days bars of the entry signal.
    """

    def __init__(
        self,
        period: int = 14,
        entry_rsi: float = 30.0,
        exit_rsi: float = 50.0,
        hold_days: int = 5,
    ):
        if entry_rsi >= exit_rsi:
            raise ValueError("entry_rsi must be < exit_rsi")
        self.period = period
        self.entry_rsi = entry_rsi
        self.exit_rsi = exit_rsi
        self.hold_days = hold_days
        self.name = f"rsi_rev_{period}_{int(entry_rsi)}_{int(exit_rsi)}_{hold_days}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        r = prices["adj_close"].pct_change()
        gain = r.clip(lower=0)
        loss = (-r).clip(lower=0)

        # Wilder's smoothing: alpha = 1/period
        alpha = 1.0 / self.period
        avg_gain = gain.ewm(alpha=alpha, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, min_periods=self.period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100.0 - (100.0 / (1.0 + rs))

        entry = (rsi < self.entry_rsi).astype(float)
        entry = entry.where(rsi.notna(), 0.0)

        # Long while: entry fired recently AND RSI hasn't exceeded exit level
        time_active = entry.rolling(self.hold_days, min_periods=1).max()
        not_exited = (rsi <= self.exit_rsi).astype(float)
        not_exited = not_exited.where(rsi.notna(), 0.0)

        pos = (time_active * not_exited).clip(0.0, 1.0)
        pos.name = "position"
        return pos


class ConsecutiveDownDays(Strategy):
    """Hypothesis 005: entry after n_down consecutive negative closes.

    The rolling-minimum trick: rolling(n_down).min() on the binary
    is_down series equals 1 iff every bar in the window was a down day.
    Lookahead-free and fully vectorized.
    """

    def __init__(self, n_down: int = 3, hold_days: int = 3):
        if n_down < 2:
            raise ValueError("n_down must be >= 2")
        self.n_down = n_down
        self.hold_days = hold_days
        self.name = f"consec_down_{n_down}_{hold_days}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        r = prices["adj_close"].pct_change()
        is_down = (r < 0).astype(float)

        # All n_down bars in the rolling window are down iff min == 1
        consec_min = is_down.rolling(self.n_down, min_periods=self.n_down).min()
        entry = (consec_min >= 1.0).astype(float)

        pos = entry.rolling(self.hold_days, min_periods=1).max()
        pos.name = "position"
        return pos
