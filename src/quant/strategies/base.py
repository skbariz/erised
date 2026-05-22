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


class VIXFilteredReversionStrategy(Strategy):
    """Hypothesis 006: mean reversion only in elevated-VIX regimes.

    Same z-score entry as MeanReversionStrategy but gated by VIX > vix_threshold.
    Hypothesis: reversions succeed in genuine panic; not in low-vol drift.
    Requires a 'vix_close' column in the prices DataFrame.
    """

    def __init__(
        self,
        lookback: int = 20,
        threshold: float = 2.0,
        hold_days: int = 2,
        vix_threshold: float = 20.0,
    ):
        self.lookback = lookback
        self.threshold = threshold
        self.hold_days = hold_days
        self.vix_threshold = vix_threshold
        self.name = f"vix_rev_{lookback}_{threshold}_{hold_days}_{int(vix_threshold)}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VIXFilteredReversionStrategy requires 'vix_close' column")

        r = prices["adj_close"].pct_change()
        rolling_mean = r.rolling(self.lookback, min_periods=self.lookback).mean().shift(1)
        rolling_std = r.rolling(self.lookback, min_periods=self.lookback).std(ddof=1).shift(1)
        z = (r - rolling_mean) / rolling_std

        vix_elevated = (prices["vix_close"] > self.vix_threshold)

        entry = ((z < -self.threshold) & vix_elevated).astype(float)
        entry = entry.where(rolling_std.notna() & (rolling_std > 0), 0.0)

        pos = entry.rolling(self.hold_days, min_periods=1).max()
        pos.name = "position"
        return pos


class TimeSeriesMomentum(Strategy):
    """Hypothesis 007: 12-1 month time-series momentum (skip-month convention).

    Long when the return from 12 months ago to 1 month ago is positive.
    The 1-month skip avoids the short-term reversal at lag 1 (Jegadeesh & Titman).
    """

    def __init__(self, long_months: int = 12, skip_months: int = 1):
        self.long_days = int(long_months * 21)
        self.skip_days = int(skip_months * 21)
        self.name = f"ts_mom_{long_months}m_{skip_months}skip"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        # Return from long_days ago to skip_days ago (both purely past data)
        mom = px.shift(self.skip_days) / px.shift(self.long_days) - 1

        pos = (mom > 0).astype(float)
        pos = pos.where(mom.notna(), 0.0)
        pos.name = "position"
        return pos


class AbsoluteMomentumStrategy(Strategy):
    """Hypothesis 008: Antonacci's absolute momentum.

    Long when the trailing lookback_months return is positive; flat (cash) when
    negative. The simplest crash-protection filter — exits sustained downtrends.
    """

    def __init__(self, lookback_months: int = 12):
        self.lookback_days = int(lookback_months * 21)
        self.name = f"abs_mom_{lookback_months}m"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        mom = px / px.shift(self.lookback_days) - 1

        pos = (mom > 0).astype(float)
        pos = pos.where(mom.notna(), 0.0)
        pos.name = "position"
        return pos


class MACDCrossover(Strategy):
    """Hypothesis 009: MACD signal-line crossover.

    Long when MACD (EMA_fast - EMA_slow) is above its signal line (EMA of MACD).
    All EMAs are computed from data through t — no lookahead.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.name = f"macd_{fast}_{slow}_{signal}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        ema_fast = px.ewm(span=self.fast, adjust=False).mean()
        ema_slow = px.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()

        pos = (macd_line > signal_line).astype(float)
        # Zero out the warm-up window where EMAs are unreliable
        warmup = self.slow + self.signal
        if len(pos) > warmup:
            pos.iloc[:warmup] = 0.0
        pos.name = "position"
        return pos
