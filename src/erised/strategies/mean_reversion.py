"""Short-horizon mean-reversion and contrarian entry strategies on SPY."""

from __future__ import annotations

import pandas as pd

from .base import Strategy


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


class LongTermReversal(Strategy):
    """H038: Long when SPY's 5-year rolling return is in the bottom tercile.

    DeBondt & Thaler (1985): long-horizon losers become medium-horizon winners.
    Applied to SPY as a single asset, this fires when cumulative multi-year
    returns are historically poor — i.e., at or near the end of bear markets.
    These are exactly the periods when H001 is flat or short, producing
    structurally low/negative H001 correlation.

    lookback_days : rolling window to compute the return (default 5yr ≈ 1260d).
    percentile    : rank threshold; long when return is below this percentile
                    of all rolling returns seen so far (default 33 = bottom third).
    history_min   : minimum observations needed to estimate the percentile
                    before producing any signal (default 252).
    """

    def __init__(
        self,
        lookback_days: int = 1260,
        percentile: float = 33.0,
        history_min: int = 252,
    ):
        self.lookback_days = lookback_days
        self.percentile = percentile
        self.history_min = history_min
        self.name = f"long_term_reversal_{lookback_days}d_p{int(percentile)}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        # N-year return shifted 1 bar so today's close is not in its own signal
        rolling_ret = (px / px.shift(self.lookback_days) - 1).shift(1)
        # Expanding quantile: percentile of all past return observations so far
        threshold = rolling_ret.expanding(min_periods=self.history_min).quantile(
            self.percentile / 100.0
        )
        valid = rolling_ret.notna() & threshold.notna()
        pos = (rolling_ret <= threshold).astype(float)
        pos = pos.where(valid, 0.0)
        pos.name = "position"
        return pos


class DeepDrawdownCrisisBuyer(Strategy):
    """H039: Long when SPY drawdown from rolling peak exceeds entry_dd; exit at exit_dd.

    Systematic crisis buying: enter when the market is in historically deep
    distress and exit only once prices have substantially recovered. This
    fires when H001 is definitively flat/short (price well below 200MA),
    targeting structurally negative H001 correlation.

    entry_dd : drawdown threshold to enter long (default −0.20 = −20%).
    exit_dd  : drawdown threshold to exit (default −0.05 = −5%).
    lookback : rolling window to compute the peak (default 252 trading days = 1yr).
    """

    def __init__(
        self,
        entry_dd: float = -0.20,
        exit_dd: float = -0.05,
        lookback: int = 252,
    ):
        if entry_dd >= exit_dd:
            raise ValueError("entry_dd must be more negative than exit_dd")
        self.entry_dd = entry_dd
        self.exit_dd = exit_dd
        self.lookback = lookback
        self.name = f"deep_dd_buyer_{int(abs(entry_dd) * 100)}pct_{int(abs(exit_dd) * 100)}pct"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        # Rolling peak over the lookback window, shifted 1 bar to avoid lookahead
        rolling_peak = px.rolling(self.lookback, min_periods=1).max().shift(1)
        drawdown = px / rolling_peak - 1

        in_crisis = drawdown <= self.entry_dd
        recovered = drawdown >= self.exit_dd

        # Build position state: enter on crisis, exit on recovery
        pos = pd.Series(0.0, index=px.index, name="position")
        active = False
        for i in range(len(pos)):
            if pd.isna(drawdown.iloc[i]) or pd.isna(rolling_peak.iloc[i]):
                continue
            if not active and in_crisis.iloc[i]:
                active = True
            elif active and recovered.iloc[i]:
                active = False
            pos.iloc[i] = 1.0 if active else 0.0

        pos.name = "position"
        return pos


class ShortTermWeeklyReversal(Strategy):
    """H040: Long when SPY's 5-day return is in the bottom decile; hold 5 days.

    Lehmann (1990): weekly price reversals driven by institutional liquidity
    demand. After a very bad week, market-makers have built up inventory
    and push prices back. For SPY as a liquid index fund, the reversal is
    modest but positive-expectation. Fires in declining markets where H001
    is most likely to be bearish, producing low/negative H001 correlation.

    hold_days   : how long to hold after entry signal (default 5).
    history_min : bars needed to estimate the percentile (default 252).
    threshold   : bottom-percentile threshold for entry (default 10 = decile).
    """

    def __init__(
        self,
        hold_days: int = 5,
        history_min: int = 252,
        threshold: float = 10.0,
    ):
        self.hold_days = hold_days
        self.history_min = history_min
        self.threshold = threshold
        self.name = f"short_term_reversal_{hold_days}d_p{int(threshold)}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        ret_5d = px / px.shift(self.hold_days) - 1

        # Expanding percentile threshold: bottom N% of all past 5-day returns
        lower_bound = ret_5d.expanding(min_periods=self.history_min).quantile(
            self.threshold / 100.0
        )
        valid = ret_5d.notna() & lower_bound.notna()
        entry = (ret_5d <= lower_bound).astype(float)
        entry = entry.where(valid, 0.0)

        # Hold for hold_days bars after entry signal
        pos = entry.rolling(self.hold_days, min_periods=1).max()
        pos.name = "position"
        return pos


class PostCrashRecovery(Strategy):
    """H041: Long SPY starting delay_days after VIX spikes above threshold.

    Captures the recovery bounce that follows capitulation events (2003,
    2009, 2020). Unlike H028 (enters at the spike itself and catches downside
    continuation), this waits delay_days after the panic peak for the acute
    phase to pass before going long, then holds for hold_days.

    vix_threshold : VIX level defining a capitulation event (default 40).
    delay_days    : bars to wait after the spike before entering (default 30).
    hold_days     : bars to hold after entry (default 60).
    Requires 'vix_close' column in the prices DataFrame.
    """

    def __init__(
        self,
        vix_threshold: float = 40.0,
        delay_days: int = 30,
        hold_days: int = 60,
    ):
        self.vix_threshold = vix_threshold
        self.delay_days = delay_days
        self.hold_days = hold_days
        self.name = f"post_crash_{int(vix_threshold)}vix_{delay_days}d_delay_{hold_days}d_hold"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("PostCrashRecovery requires 'vix_close' column")

        vix = prices["vix_close"].shift(1)
        spike = (vix > self.vix_threshold).astype(float)

        # Mark the bar delay_days AFTER a spike as an entry signal.
        # Shift the spike forward by delay_days so we enter later.
        delayed_entry = spike.shift(self.delay_days).fillna(0.0)

        # Hold for hold_days bars after the delayed entry
        pos = delayed_entry.rolling(self.hold_days, min_periods=1).max()
        pos.name = "position"
        return pos
