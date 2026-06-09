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

import numpy as np
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


class LongShortMACrossover(MovingAverageCrossover):
    """H013: MA(50,200) long/short — short below the death cross instead of flat.

    Identical signal to H001 (MovingAverageCrossover) but sets allow_short=True,
    going −1 when fast MA < slow MA rather than 0. Captures bear-market returns
    that H001 abandons by going to cash.
    """

    def __init__(self, fast: int = 50, slow: int = 200):
        super().__init__(fast=fast, slow=slow, allow_short=True)


class TrendVIXGate(Strategy):
    """H017: MA(50,200) long/short, forced flat when VIX exceeds vix_extreme.

    Hypothesis: in crash conditions (VIX > 40) trend signals are whipsawed by
    panic volatility. Sitting flat avoids the worst drawdown periods while
    keeping the long/short signal active in normal regimes.
    Requires a 'vix_close' column in the prices DataFrame.
    """

    def __init__(self, fast: int = 50, slow: int = 200, vix_extreme: float = 40.0):
        self.fast = fast
        self.slow = slow
        self.vix_extreme = vix_extreme
        self.name = f"trend_vix_gate_{fast}_{slow}_{int(vix_extreme)}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("TrendVIXGate requires 'vix_close' column")

        px = prices["adj_close"]
        fast_ma = px.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = px.rolling(self.slow, min_periods=self.slow).mean()

        pos = pd.Series(np.where(fast_ma > slow_ma, 1.0, -1.0), index=px.index)
        pos = pos.where(slow_ma.notna(), 0.0)

        # Force flat when VIX signals crash territory
        in_crash = prices["vix_close"] > self.vix_extreme
        pos = pos.where(~in_crash, 0.0)

        pos.name = "position"
        return pos


class VolScaledTrend(Strategy):
    """H016: MA trend direction scaled by volatility targeting.

    Position = sign(MA signal) × min(target_vol / 20d_realized_vol, 1.0).
    Sizes up in calm markets and down in turbulent ones, converting H001's
    binary 0/+1 into a continuous signal. Tests whether dynamic position sizing
    improves risk-adjusted returns over constant full-notional exposure.
    """

    def __init__(
        self,
        fast: int = 50,
        slow: int = 200,
        target_vol: float = 0.10,
        vol_lookback: int = 20,
    ):
        self.fast = fast
        self.slow = slow
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.name = f"vol_scaled_trend_{fast}_{slow}_{int(target_vol * 100)}pct"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        r = px.pct_change()

        fast_ma = px.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = px.rolling(self.slow, min_periods=self.slow).mean()

        direction = pd.Series(np.where(fast_ma > slow_ma, 1.0, -1.0), index=px.index)
        direction = direction.where(slow_ma.notna(), 0.0)

        # Shift realized vol by 1 bar so today's close is not used in today's signal
        realized_vol = r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        scale = (self.target_vol / realized_vol).clip(upper=1.0)
        scale = scale.where(realized_vol.notna() & (realized_vol > 0), 0.0)

        pos = (direction * scale).clip(-1.0, 1.0)
        pos = pos.where(slow_ma.notna(), 0.0)
        pos.name = "position"
        return pos


class DualMALongShort(MovingAverageCrossover):
    """H015: MA(20,100) long/short — faster signal than H013's 50/200 pair.

    Tests whether catching trend turns earlier improves Sharpe or erodes gains
    through excess turnover. allow_short=True: −1 below the 100-day MA.
    """

    def __init__(self, fast: int = 20, slow: int = 100):
        super().__init__(fast=fast, slow=slow, allow_short=True)


class LongShortTSMomentum(Strategy):
    """H014: 12-1 month time-series momentum, long/short.

    Identical lookback logic to H007 (TimeSeriesMomentum) but goes short (−1)
    when the 12-1 month return is negative rather than going flat. Tests whether
    the momentum signal has predictive power on the short side.
    """

    def __init__(self, long_months: int = 12, skip_months: int = 1):
        self.long_days = int(long_months * 21)
        self.skip_days = int(skip_months * 21)
        self.name = f"ls_ts_mom_{long_months}m_{skip_months}skip"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        mom = px.shift(self.skip_days) / px.shift(self.long_days) - 1
        pos = mom.apply(lambda x: 0.0 if pd.isna(x) else float(np.sign(x)))
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

        vix_elevated = prices["vix_close"] > self.vix_threshold

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


class FastMACrossover(MovingAverageCrossover):
    """Hypothesis 010: MA(10/50) — faster trend-following.

    Tests whether catching trend turns earlier than H001's MA(50/200)
    improves Sharpe, or whether the extra turnover erodes gains.
    """

    name = "fast_ma_cross_10_50"

    def __init__(self, fast: int = 10, slow: int = 50):
        super().__init__(fast=fast, slow=slow)


class ADXFilteredMACrossover(Strategy):
    """Hypothesis 011: MA(50,200) crossover confirmed by ADX trend strength.

    Takes H001 signals only when ADX(period) >= adx_threshold, filtering
    out false crossovers in ranging, low-momentum markets.
    """

    def __init__(
        self,
        fast: int = 50,
        slow: int = 200,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
    ):
        self.fast = fast
        self.slow = slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.name = f"adx_ma_{fast}_{slow}_{adx_period}_{int(adx_threshold)}"

    @staticmethod
    def _adx(prices: pd.DataFrame, period: int) -> pd.Series:
        hi = prices["high"]
        lo = prices["low"]
        cl = prices["adj_close"]

        tr = pd.concat([hi - lo, (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()], axis=1).max(
            axis=1
        )

        up = hi - hi.shift(1)
        dn = lo.shift(1) - lo
        plus_dm = ((up > dn) & (up > 0)) * up
        minus_dm = ((dn > up) & (dn > 0)) * dn

        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr

        di_sum = plus_di + minus_di
        dx = (100 * (plus_di - minus_di).abs() / di_sum).where(di_sum > 0, 0.0)
        adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        return adx

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        fast_ma = px.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = px.rolling(self.slow, min_periods=self.slow).mean()

        ma_signal = (fast_ma > slow_ma).astype(float)
        ma_signal = ma_signal.where(slow_ma.notna(), 0.0)

        adx = self._adx(prices, self.adx_period)
        trend_on = (adx >= self.adx_threshold).astype(float)
        trend_on = trend_on.where(adx.notna(), 0.0)

        pos = ma_signal * trend_on
        pos.name = "position"
        return pos


class VolatilityRegime(Strategy):
    """Hypothesis 012: long in low realized-volatility regimes.

    Long when 20-day realized vol is below its rolling historical median
    (using only past vol data). Flat in turbulent regimes. Tests the
    low-volatility anomaly: calm markets produce better risk-adjusted returns.
    """

    def __init__(self, vol_lookback: int = 20, regime_lookback: int = 252):
        self.vol_lookback = vol_lookback
        self.regime_lookback = regime_lookback
        self.name = f"vol_regime_{vol_lookback}_{regime_lookback}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        r = prices["adj_close"].pct_change()
        realized_vol = r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std()

        # Rolling median of PAST vols (shift by 1 so today's vol isn't in its own baseline)
        rolling_median = (
            realized_vol.shift(1)
            .rolling(self.regime_lookback, min_periods=self.regime_lookback // 2)
            .median()
        )

        pos = (realized_vol < rolling_median).astype(float)
        pos = pos.where(rolling_median.notna(), 0.0)
        pos.name = "position"
        return pos


# ---------------------------------------------------------------------------
# Calendar helpers — used by H024 and H026.
# ---------------------------------------------------------------------------


def _calendar_month_ranks(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Rank each date from the start and end of its calendar month.

    Uses a fixed business-day calendar anchored at index.min() + 50 years so
    that truncating the price series at any point T does not change the
    rank_end value at T. This makes calendar strategies pass assert_no_lookahead
    (which checks that positions at T are stable under price-data extension).
    """
    start = index.min()
    cal_end = start + pd.DateOffset(years=50)
    all_bdays = pd.bdate_range(start, cal_end)
    ym = all_bdays.year * 100 + all_bdays.month
    ym_series = pd.Series(ym, index=all_bdays, name="ym")
    rank_start = ym_series.groupby(ym_series).cumcount()
    rank_end = ym_series.groupby(ym_series).cumcount(ascending=False)
    return rank_start.reindex(index), rank_end.reindex(index)


def _calendar_quarter_rank_end(index: pd.DatetimeIndex) -> pd.Series:
    """Rank each date from the END of its calendar quarter.

    Same fixed-calendar projection as _calendar_month_ranks.
    """
    start = index.min()
    cal_end = start + pd.DateOffset(years=50)
    all_bdays = pd.bdate_range(start, cal_end)
    yq = all_bdays.year * 10 + all_bdays.quarter
    yq_series = pd.Series(yq, index=all_bdays, name="yq")
    rank_end = yq_series.groupby(yq_series).cumcount(ascending=False)
    return rank_end.reindex(index)


class SellInMay(Strategy):
    """H023: Sell in May — long Nov-Apr, flat May-Oct.

    Bouman & Jacobsen (2002) documented this anomaly in 37 countries over 40+
    years. Institutional tax-loss selling in December, year-end rebalancing, and
    reduced summer participation drive the pattern. Signal is purely calendar-
    based; correlation with H001 (price-momentum) is structurally low.
    """

    def __init__(self, favorable_months: tuple = (11, 12, 1, 2, 3, 4)):
        self.favorable_months = frozenset(favorable_months)
        self.name = "sell_in_may"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        is_favorable = prices.index.month.isin(self.favorable_months)
        pos = pd.Series(is_favorable.astype(float), index=prices.index, name="position")
        return pos


class TurnOfMonth(Strategy):
    """H024: Long the last + first N trading days of each calendar month.

    Ariel (1987) showed that virtually all stock market gains are concentrated
    near month boundaries. Institutional cash flows (pension contributions,
    dividend reinvestment) concentrate at month turns.

    Uses a fixed business-day calendar projection so that within-month
    rank_end values are stable under price-series truncation (allowing
    assert_no_lookahead to pass for this calendar-based strategy).
    """

    def __init__(self, n_days: int = 3):
        self.n_days = n_days
        self.name = f"turn_of_month_{n_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        rank_start, rank_end = _calendar_month_ranks(prices.index)
        in_window = (rank_start < self.n_days) | (rank_end < self.n_days)
        pos = in_window.astype(float)
        pos.name = "position"
        return pos


class JanuaryBarometer(Strategy):
    """H025: January return predicts the rest of the year.

    'As goes January, so goes the year.' Long SPY from February through
    December if January's return is positive; flat otherwise. Eliminates
    years that begin with a loss (which historically often precede full-year
    bear markets). Signal updates annually at end of January; flat in January.
    """

    name = "january_barometer"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        pos = pd.Series(0.0, index=px.index, name="position")

        for year in sorted(px.index.year.unique()):
            jan_px = px[(px.index.year == year) & (px.index.month == 1)]
            if len(jan_px) < 2:
                continue
            jan_ret = jan_px.iloc[-1] / jan_px.iloc[0] - 1
            # Signal active Feb–Dec of the same year
            rest_of_year = (px.index.year == year) & (px.index.month > 1)
            pos[rest_of_year] = 1.0 if jan_ret > 0 else 0.0

        return pos


class QuarterEndDrift(Strategy):
    """H026: Long the last N trading days of each calendar quarter.

    Lakonishok et al. (1991) documented institutional window dressing near
    quarter-end: fund managers buy winners to improve reported holdings.
    This mechanical buying creates a predictable short-term drift in the
    5 trading days before March/June/September/December quarter ends.

    Only ~20 trading days active per year → structural low H001 correlation.
    """

    def __init__(self, n_days: int = 5):
        self.n_days = n_days
        self.name = f"quarter_end_drift_{n_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        rank_end = _calendar_quarter_rank_end(prices.index)
        pos = (rank_end < self.n_days).astype(float)
        pos.name = "position"
        return pos


class SeasonalityTrendCombo(Strategy):
    """H027: MA(50,200) trend signal gated to seasonally favorable months.

    Applies H001's golden/death-cross trend signal only during Nov–Apr
    (the historically stronger half-year). Forces flat during May–Oct
    regardless of the trend signal. Tests whether seasonal filtering
    improves the trend signal's Sharpe by removing the noisier summer period.
    """

    def __init__(
        self,
        fast: int = 50,
        slow: int = 200,
        favorable_months: tuple = (11, 12, 1, 2, 3, 4),
    ):
        self.fast = fast
        self.slow = slow
        self.favorable_months = frozenset(favorable_months)
        self.name = f"seasonality_trend_{fast}_{slow}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        fast_ma = px.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = px.rolling(self.slow, min_periods=self.slow).mean()

        trend = (fast_ma > slow_ma).astype(float)
        trend = trend.where(slow_ma.notna(), 0.0)

        season = pd.Series(
            prices.index.month.isin(self.favorable_months).astype(float),
            index=prices.index,
        )

        pos = trend * season
        pos.name = "position"
        return pos


class SPYTLTDualMomentum(MultiAssetStrategy):
    """H018: Antonacci GEM-style dual momentum across SPY and TLT.

    Hold SPY when its 12m return is positive. If SPY is negative, hold TLT when
    TLT's 12m return is positive. If both are negative, go to cash. The
    absolute-momentum filter (cash when all assets are negative) is the key
    distinguisher from cross-sectional-only rotation.
    """

    def __init__(self, lookback_months: int = 12, skip_months: int = 1):
        self.lookback_days = int(lookback_months * 21)
        self.skip_days = int(skip_months * 21)
        self.name = f"spy_tlt_dual_mom_{lookback_months}m"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy = prices["adj_close"]
        tlt = prices["tlt_close"]

        spy_mom = spy.shift(self.skip_days) / spy.shift(self.lookback_days) - 1
        tlt_mom = tlt.shift(self.skip_days) / tlt.shift(self.lookback_days) - 1

        w_spy = pd.Series(0.0, index=prices.index)
        w_tlt = pd.Series(0.0, index=prices.index)

        spy_positive = spy_mom > 0
        tlt_positive = tlt_mom > 0
        both_valid = spy_mom.notna() & tlt_mom.notna()

        w_spy = w_spy.where(~(both_valid & spy_positive), 1.0)
        w_tlt = w_tlt.where(~(both_valid & ~spy_positive & tlt_positive), 1.0)

        weights = pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)
        return weights


class SPYGLDRotation(MultiAssetStrategy):
    """H019: SPY/GLD rotation — always allocated, no cash.

    Hold SPY when its 12m momentum is positive; hold GLD when SPY momentum is
    negative. Stays fully invested at all times. GLD historically appreciates
    during equity bear markets, particularly in inflationary regimes.
    """

    def __init__(self, lookback_months: int = 12, skip_months: int = 1):
        self.lookback_days = int(lookback_months * 21)
        self.skip_days = int(skip_months * 21)
        self.name = f"spy_gld_rotation_{lookback_months}m"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy = prices["adj_close"]
        spy_mom = spy.shift(self.skip_days) / spy.shift(self.lookback_days) - 1

        w_spy = pd.Series(0.0, index=prices.index)
        w_gld = pd.Series(0.0, index=prices.index)

        valid = spy_mom.notna()
        spy_positive = valid & (spy_mom > 0)
        spy_negative = valid & (spy_mom <= 0)

        w_spy = w_spy.where(~spy_positive, 1.0)
        w_gld = w_gld.where(~spy_negative, 1.0)

        weights = pd.DataFrame({"SPY": w_spy, "GLD": w_gld}, index=prices.index)
        return weights


class ThreeAssetMomentum(MultiAssetStrategy):
    """H020: Best-of-3 absolute momentum across SPY, TLT, GLD.

    Hold the single best 12m performer. Go to cash if the winner's momentum
    is negative (universal absolute-momentum filter). Broadest opportunity
    set in Batch 3 — captures regimes where equities AND bonds decline
    but commodities rise (e.g. inflationary bear markets).
    """

    def __init__(self, lookback_months: int = 12, skip_months: int = 1):
        self.lookback_days = int(lookback_months * 21)
        self.skip_days = int(skip_months * 21)
        self.name = f"three_asset_mom_{lookback_months}m"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        assets = {
            "SPY": prices["adj_close"],
            "TLT": prices["tlt_close"],
            "GLD": prices["gld_close"],
        }
        moms = pd.DataFrame(
            {
                name: px.shift(self.skip_days) / px.shift(self.lookback_days) - 1
                for name, px in assets.items()
            }
        )

        weights = pd.DataFrame(0.0, index=prices.index, columns=list(assets.keys()))
        valid_rows = moms.notna().all(axis=1)

        if valid_rows.any():
            best = pd.Series(np.nan, index=prices.index, dtype=object)
            best_mom = pd.Series(np.nan, index=prices.index)
            best[valid_rows] = moms[valid_rows].idxmax(axis=1)
            best_mom[valid_rows] = moms[valid_rows].max(axis=1)

            for asset in assets:
                weights[asset] = np.where(valid_rows & (best == asset) & (best_mom > 0), 1.0, 0.0)

        return weights


class RiskParity(MultiAssetStrategy):
    """H021: SPY/TLT equal-risk contribution weighting.

    Weights each asset by the inverse of its 60-day realized volatility,
    normalized to sum to 1. Both assets are held simultaneously at all times.
    Diversification benefit: SPY and TLT have historically low/negative
    correlation, so equal-risk allocation approximates minimum-variance.
    """

    def __init__(self, vol_lookback: int = 60):
        self.vol_lookback = vol_lookback
        self.name = f"risk_parity_{vol_lookback}d"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        # Annualized vol; shift 1 bar to prevent lookahead
        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt

        w_spy = (inv_spy / total).fillna(0.0)
        w_tlt = (inv_tlt / total).fillna(0.0)

        weights = pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)
        return weights


class DefensiveEquityHedge(MultiAssetStrategy):
    """H022: Trend-following with a bond overlay in downtrends.

    100% SPY when MA(fast, slow) is bullish. Shifts to 50% SPY + 50% TLT when
    the trend turns bearish — maintains equity participation but hedges with
    duration. Avoids the full whipsaw cost of going to cash (H001) or going
    short (H013) during slow downtrends.
    """

    def __init__(self, fast: int = 50, slow: int = 200):
        self.fast = fast
        self.slow = slow
        self.name = f"defensive_hedge_{fast}_{slow}"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        px = prices["adj_close"]
        fast_ma = px.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = px.rolling(self.slow, min_periods=self.slow).mean()

        trend_on = fast_ma > slow_ma
        valid = slow_ma.notna()

        w_spy = pd.Series(0.0, index=prices.index)
        w_tlt = pd.Series(0.0, index=prices.index)

        w_spy = w_spy.where(~(valid & trend_on), 1.0)
        w_spy = w_spy.where(~(valid & ~trend_on), 0.5)
        w_tlt = w_tlt.where(~(valid & ~trend_on), 0.5)

        weights = pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)
        return weights


# ---------------------------------------------------------------------------
# Batch 5 — H028–H032: VIX-Based Strategies
# All require a 'vix_close' column in the prices DataFrame.
# ---------------------------------------------------------------------------


class VIXSpikeEntry(Strategy):
    """H028: Contrarian long after VIX spikes above a threshold.

    When VIX crosses above vix_threshold, enter long SPY and hold for
    hold_days trading days. Fear spikes historically mark local bottoms —
    this captures the mean-reversion bounce in equities.

    Requires a 'vix_close' column in the prices DataFrame.
    """

    def __init__(self, vix_threshold: float = 30.0, hold_days: int = 20):
        self.vix_threshold = vix_threshold
        self.hold_days = hold_days
        self.name = f"vix_spike_{int(vix_threshold)}_{hold_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VIXSpikeEntry requires 'vix_close' column")
        vix = prices["vix_close"].shift(1)
        spike = (vix > self.vix_threshold).astype(float)
        # Active during hold_days window after a spike (rolling max of spike flag)
        pos = spike.rolling(self.hold_days, min_periods=1).max()
        pos.name = "position"
        return pos


class VIXRegimeFilter(Strategy):
    """H029: Long SPY only when VIX is below threshold (low-fear regime).

    VIX below vix_threshold signals a calm, risk-on environment historically
    associated with positive equity returns and lower drawdown risk.

    Requires a 'vix_close' column in the prices DataFrame.
    """

    def __init__(self, vix_threshold: float = 20.0):
        self.vix_threshold = vix_threshold
        self.name = f"vix_regime_{int(vix_threshold)}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VIXRegimeFilter requires 'vix_close' column")
        vix = prices["vix_close"].shift(1)
        pos = (vix < self.vix_threshold).astype(float)
        pos.name = "position"
        return pos


class VIXTrendSignal(Strategy):
    """H030: Trade equity direction via VIX trend (rising fear → short).

    When VIX is above its rolling MA (trending up), short SPY: rising fear
    implies falling equity prices. When VIX is below its MA (trending down),
    long SPY. Uses 1-bar lag to avoid lookahead.

    Requires a 'vix_close' column in the prices DataFrame.
    """

    def __init__(self, ma_period: int = 20):
        self.ma_period = ma_period
        self.name = f"vix_trend_{ma_period}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VIXTrendSignal requires 'vix_close' column")
        vix = prices["vix_close"]
        vix_ma = vix.rolling(self.ma_period, min_periods=self.ma_period).mean()
        valid = vix_ma.notna()
        pos = pd.Series(0.0, index=prices.index, name="position")
        pos = pos.where(~(valid & (vix.shift(1) < vix_ma.shift(1))), 1.0)
        pos = pos.where(~(valid & (vix.shift(1) >= vix_ma.shift(1))), -1.0)
        return pos


class RealizedVsImpliedVol(Strategy):
    """H031: Long when implied vol (VIX) exceeds realized vol (vol risk premium).

    The volatility risk premium (VRP = VIX − realized_vol) is historically
    positive on average. Long SPY in VRP-positive regimes (implied > realized),
    flat otherwise. Normalized vol: both in annualized percentage terms.

    Requires a 'vix_close' column in the prices DataFrame.
    """

    def __init__(self, realized_window: int = 20):
        self.realized_window = realized_window
        self.name = f"vrp_{realized_window}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("RealizedVsImpliedVol requires 'vix_close' column")
        rets = prices["adj_close"].pct_change()
        realized_vol = (
            rets.rolling(self.realized_window, min_periods=self.realized_window).std().shift(1)
            * np.sqrt(252)
            * 100
        )
        vix = prices["vix_close"].shift(1)
        valid = realized_vol.notna() & vix.notna()
        pos = pd.Series(0.0, index=prices.index, name="position")
        pos[valid & (vix > realized_vol)] = 1.0
        return pos


class VIXMeanReversion(Strategy):
    """H032: Contrarian long when VIX is at a statistically extreme high.

    Computes a rolling z-score of VIX relative to its own history. A high
    z-score (> n_sigma) signals extreme fear that tends to revert — the
    reversion coincides with equity rallies. Unlike H028 (absolute threshold),
    this normalizes by the current volatility regime.

    Requires a 'vix_close' column in the prices DataFrame.
    """

    def __init__(self, lookback: int = 60, n_sigma: float = 2.0):
        self.lookback = lookback
        self.n_sigma = n_sigma
        self.name = f"vix_mr_{lookback}d_{n_sigma}s"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VIXMeanReversion requires 'vix_close' column")
        vix = prices["vix_close"]
        vix_ma = vix.rolling(self.lookback, min_periods=self.lookback).mean().shift(1)
        vix_std = vix.rolling(self.lookback, min_periods=self.lookback).std().shift(1)
        vix_z = (vix.shift(1) - vix_ma) / vix_std.replace(0, np.nan)
        pos = (vix_z > self.n_sigma).astype(float)
        pos.name = "position"
        return pos


# ---------------------------------------------------------------------------
# Batch 6 — H033–H037: Cross-Asset Macro Timing
# All require 'tlt_close' and/or 'gld_close' columns in the prices DataFrame.
# ---------------------------------------------------------------------------


class BondTrendEquityTimer(Strategy):
    """H033: Long SPY when TLT is below its MA (bonds weak = risk-on).

    Flight-to-safety bond rallies occur when investors fear equity losses.
    TLT trending above its 200MA signals bond demand = risk-off environment.
    TLT below its 200MA = no flight to safety = risk-on.

    Decorrelation thesis: bond trend and equity trend diverge in crises —
    exactly the regime where H001 is most likely to be wrong or late.
    Requires 'tlt_close' column in the prices DataFrame.
    """

    def __init__(self, ma_period: int = 200):
        self.ma_period = ma_period
        self.name = f"bond_trend_equity_{ma_period}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "tlt_close" not in prices.columns:
            raise ValueError("BondTrendEquityTimer requires 'tlt_close' column")
        tlt = prices["tlt_close"]
        tlt_ma = tlt.rolling(self.ma_period, min_periods=self.ma_period).mean()
        pos = (tlt < tlt_ma).astype(float)
        pos = pos.where(tlt_ma.notna(), 0.0)
        pos.name = "position"
        return pos


class SPYTLTCorrelationRegime(Strategy):
    """H034: Long SPY only when rolling SPY-TLT correlation is negative.

    In normal markets, equities and bonds have negative return correlation
    (flight-to-safety effect). During crisis contagion, both assets fall
    together and correlation flips positive. Flat when correlation is
    positive avoids the regime where diversification fails entirely.

    Decorrelation thesis: this signal uses cross-asset covariance structure,
    not price levels. It turns off at exactly the time H001 gets hurt most.
    Requires 'tlt_close' column in the prices DataFrame.
    """

    def __init__(self, corr_window: int = 60):
        self.corr_window = corr_window
        self.name = f"spy_tlt_corr_regime_{corr_window}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "tlt_close" not in prices.columns:
            raise ValueError("SPYTLTCorrelationRegime requires 'tlt_close' column")
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        rolling_corr = spy_r.rolling(self.corr_window, min_periods=self.corr_window).corr(tlt_r)
        pos = (rolling_corr < 0).astype(float)
        pos = pos.where(rolling_corr.notna(), 0.0)
        pos.name = "position"
        return pos


class GoldFearGauge(Strategy):
    """H035: Long SPY when GLD is below its MA (no panic bid for gold).

    Gold outperformance signals fear and inflation anxiety that historically
    precede equity weakness. GLD trending above its MA = investors seeking
    the gold safe haven = risk-off environment. GLD below MA = no fear
    premium in the gold market = risk-on signal for equities.

    Requires 'gld_close' column in the prices DataFrame.
    """

    def __init__(self, ma_period: int = 50):
        self.ma_period = ma_period
        self.name = f"gold_fear_gauge_{ma_period}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "gld_close" not in prices.columns:
            raise ValueError("GoldFearGauge requires 'gld_close' column")
        gld = prices["gld_close"]
        gld_ma = gld.rolling(self.ma_period, min_periods=self.ma_period).mean()
        pos = (gld < gld_ma).astype(float)
        pos = pos.where(gld_ma.notna(), 0.0)
        pos.name = "position"
        return pos


class EquityVsSafeHavenMomentum(Strategy):
    """H036: Long SPY when SPY 3m return > average of TLT and GLD 3m returns.

    Risk-on regime: equity outperforms safe havens. Risk-off: safe havens
    outperform equity. This relative momentum signal captures regime shifts
    earlier than absolute momentum because safe-haven outperformance often
    precedes equity declines by days to weeks.

    Decorrelation thesis: the signal is relative (SPY vs TLT/GLD), not
    absolute SPY trend. In bull markets both agree; in transitions, this
    signal flips before H001's MA crossover fires.
    Requires 'tlt_close' and 'gld_close' columns in the prices DataFrame.
    """

    def __init__(self, lookback_days: int = 63):
        self.lookback_days = lookback_days
        self.name = f"equity_vs_safe_haven_{lookback_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "tlt_close" not in prices.columns or "gld_close" not in prices.columns:
            raise ValueError(
                "EquityVsSafeHavenMomentum requires 'tlt_close' and 'gld_close' columns"
            )
        spy_mom = prices["adj_close"] / prices["adj_close"].shift(self.lookback_days) - 1
        tlt_mom = prices["tlt_close"] / prices["tlt_close"].shift(self.lookback_days) - 1
        gld_mom = prices["gld_close"] / prices["gld_close"].shift(self.lookback_days) - 1
        safe_haven_avg = (tlt_mom + gld_mom) / 2
        valid = spy_mom.notna() & tlt_mom.notna() & gld_mom.notna()
        pos = (spy_mom > safe_haven_avg).astype(float)
        pos = pos.where(valid, 0.0)
        pos.name = "position"
        return pos


class DualSafeHavenConfirmation(Strategy):
    """H037: Long SPY only when BOTH TLT and GLD are below their 200MAs.

    Maximum risk-on confirmation: no flight to either bonds or gold.
    When either safe haven is trending up, equity exposure is reduced
    to zero. The dual-filter is intentionally restrictive — it targets
    only the highest-quality risk-on windows where both safe havens
    are dormant, structurally avoiding the crises that hurt all trend
    strategies simultaneously.

    Requires 'tlt_close' and 'gld_close' columns in the prices DataFrame.
    """

    def __init__(self, ma_period: int = 200):
        self.ma_period = ma_period
        self.name = f"dual_safe_haven_{ma_period}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "tlt_close" not in prices.columns or "gld_close" not in prices.columns:
            raise ValueError(
                "DualSafeHavenConfirmation requires 'tlt_close' and 'gld_close' columns"
            )
        tlt = prices["tlt_close"]
        gld = prices["gld_close"]
        tlt_ma = tlt.rolling(self.ma_period, min_periods=self.ma_period).mean()
        gld_ma = gld.rolling(self.ma_period, min_periods=self.ma_period).mean()
        valid = tlt_ma.notna() & gld_ma.notna()
        pos = ((tlt < tlt_ma) & (gld < gld_ma)).astype(float)
        pos = pos.where(valid, 0.0)
        pos.name = "position"
        return pos


# ---------------------------------------------------------------------------
# Batch 7 — H038–H042: Contrarian & Mean Reversion
# Structurally opposite to H001 — fire when H001 is dormant or wrong.
# Target H001 corr: −0.30 to +0.15.
# ---------------------------------------------------------------------------


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


class VolOfVolCalm(Strategy):
    """H042: Long when VIX's own rolling std dev is in the bottom tercile.

    Second-order VIX signal: stable fear (low vol-of-vol) indicates a calm,
    predictable risk environment, historically associated with steady equity
    gains. Spikes in VIX's own volatility often precede drawdowns before the
    VIX level itself rises — this signal turns off earlier than H001 in
    deteriorating regimes. Targeting H001 corr 0.10–0.30.

    vov_window  : rolling window for VIX std dev (default 20 days).
    history_min : min observations to estimate the percentile (default 252).
    percentile  : upper percentile of calm VIX-vol to accept (default 33).
    Requires 'vix_close' column in the prices DataFrame.
    """

    def __init__(
        self,
        vov_window: int = 20,
        history_min: int = 252,
        percentile: float = 33.0,
    ):
        self.vov_window = vov_window
        self.history_min = history_min
        self.percentile = percentile
        self.name = f"vol_of_vol_calm_{vov_window}d_p{int(percentile)}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VolOfVolCalm requires 'vix_close' column")

        vix = prices["vix_close"]
        # Rolling std of VIX (vol-of-vol), shifted to avoid lookahead
        vov = vix.rolling(self.vov_window, min_periods=self.vov_window).std().shift(1)

        # Expanding percentile: is current VoV in the calmest bottom tercile?
        upper_bound = vov.expanding(min_periods=self.history_min).quantile(self.percentile / 100.0)
        valid = vov.notna() & upper_bound.notna()
        pos = (vov <= upper_bound).astype(float)
        pos = pos.where(valid, 0.0)
        pos.name = "position"
        return pos


# ---------------------------------------------------------------------------
# Batch 8 — H043–H047: Calendar Refinements + Cross-Asset Hybrids
# ---------------------------------------------------------------------------


class SellInMayBondConfirm(Strategy):
    """H043: Sell-in-May seasonality gated by bond-trend macro confirmation.

    H023 (Nov–Apr long) produced H001 corr 0.505 — decent but drifts with
    trend in bull winters. Adding a TLT confirmation removes entries where the
    macro environment (bonds in uptrend = risk-off) contradicts the seasonal
    signal, targeting lower H001 corr and reduced winter draw-down exposure.

    Enters the Nov–Apr seasonal window ONLY when TLT < tlt_ma_period MA
    (bonds in downtrend = risk-on). Flat in May–Oct unconditionally.
    Requires 'tlt_close' column.
    """

    def __init__(
        self,
        favorable_months: tuple = (11, 12, 1, 2, 3, 4),
        tlt_ma_period: int = 200,
    ):
        self.favorable_months = frozenset(favorable_months)
        self.tlt_ma_period = tlt_ma_period
        self.name = f"sell_in_may_bond_confirm_{tlt_ma_period}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "tlt_close" not in prices.columns:
            raise ValueError("SellInMayBondConfirm requires 'tlt_close' column")

        tlt = prices["tlt_close"]
        tlt_ma = tlt.rolling(self.tlt_ma_period, min_periods=self.tlt_ma_period).mean().shift(1)

        in_season = pd.Series(
            prices.index.month.isin(self.favorable_months).astype(float), index=prices.index
        )
        bond_riskoff = (tlt.shift(1) > tlt_ma).astype(float)

        # Long only when in season AND bonds NOT in uptrend (risk-on macro)
        pos = in_season * (1.0 - bond_riskoff)
        pos = pos.where(tlt_ma.notna(), 0.0)
        pos.name = "position"
        return pos


class JanuaryBarometerGoldFilter(Strategy):
    """H044: January Barometer with GLD trend override.

    H025 is long Feb–Dec when January return is positive, flat otherwise.
    GLD trending up in January signals inflation/fear (rising gold precedes
    equity weakness) — gold in uptrend overrides the positive January signal.
    Adds a second independent decorrelating filter without changing the
    calendar structure. Requires 'gld_close' column.
    """

    def __init__(self, gld_ma_period: int = 50):
        self.gld_ma_period = gld_ma_period
        self.name = f"january_barometer_gold_filter_{gld_ma_period}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "gld_close" not in prices.columns:
            raise ValueError("JanuaryBarometerGoldFilter requires 'gld_close' column")

        px = prices["adj_close"]
        gld = prices["gld_close"]
        pos = pd.Series(0.0, index=px.index, name="position")

        for year in sorted(px.index.year.unique()):
            jan_px = px[(px.index.year == year) & (px.index.month == 1)]
            if len(jan_px) < 2:
                continue
            jan_ret = jan_px.iloc[-1] / jan_px.iloc[0] - 1
            if jan_ret <= 0:
                continue

            # Check GLD trend at end of January — use last day of Jan
            jan_gld = gld[(gld.index.year == year) & (gld.index.month == 1)]
            if len(jan_gld) == 0:
                continue
            # gld_hist covers all history up to end of January for the MA calculation
            gld_hist = gld[gld.index <= jan_gld.index[-1]]
            if len(gld_hist) < self.gld_ma_period:
                continue
            gld_ma_val = gld_hist.iloc[-self.gld_ma_period :].mean()
            if pd.isna(gld_ma_val) or pd.isna(jan_gld.iloc[-1]):
                # GLD not yet available — no filter, allow the signal
                gld_trending_up = False
            else:
                gld_trending_up = jan_gld.iloc[-1] > gld_ma_val

            if gld_trending_up:
                continue  # Gold fear gauge overrides the positive January signal

            rest_of_year = (px.index.year == year) & (px.index.month > 1)
            pos[rest_of_year] = 1.0

        return pos


class TurnOfMonthLowVol(Strategy):
    """H045: Turn-of-Month window gated by VIX calm regime.

    H024 (3-day turn-of-month, Sharpe 0.41) was killed on Sharpe < 0.51. The
    window-dressing / cash-flow mechanism works in normal markets but is
    overwhelmed by fear during high-VIX periods. Restricting entries to VIX <
    vix_threshold should remove the panic-driven negative turns and improve the
    Sharpe of the remaining trades. Requires 'vix_close' column.
    """

    def __init__(self, n_days: int = 3, vix_threshold: float = 25.0):
        self.n_days = n_days
        self.vix_threshold = vix_threshold
        self.name = f"turn_of_month_low_vol_{n_days}d_vix{int(vix_threshold)}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("TurnOfMonthLowVol requires 'vix_close' column")

        rank_start, rank_end = _calendar_month_ranks(prices.index)
        in_window = (rank_start < self.n_days) | (rank_end < self.n_days)

        vix = prices["vix_close"].shift(1)
        low_vol = vix < self.vix_threshold

        pos = (in_window & low_vol).astype(float)
        pos.name = "position"
        return pos


class HalloweenMomentumConfirm(Strategy):
    """H046: Sell-in-May seasonal window gated by SPY 6-month momentum.

    H023 is long Nov–Apr regardless of the trend backdrop. Adding 6-month
    price momentum as a confirmation filter skips unfavorable winters where
    a bearish trend fights the seasonal tailwind (e.g., 2008 winter).
    Targeting H001 corr 0.30–0.50 with improved Sharpe vs H023.
    """

    def __init__(
        self,
        favorable_months: tuple = (11, 12, 1, 2, 3, 4),
        momentum_days: int = 126,
    ):
        self.favorable_months = frozenset(favorable_months)
        self.momentum_days = momentum_days
        self.name = f"halloween_momentum_{momentum_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        mom = (px / px.shift(self.momentum_days) - 1).shift(1)

        in_season = pd.Series(
            prices.index.month.isin(self.favorable_months).astype(float), index=prices.index
        )
        pos_mom = (mom > 0).astype(float)
        pos_mom = pos_mom.where(mom.notna(), 0.0)

        pos = in_season * pos_mom
        pos.name = "position"
        return pos


class PreEarningsDrift(Strategy):
    """H047: Long the 5 trading days before each quarterly earnings season open.

    Institutional pre-positioning ahead of earnings season drives inflows in
    the first week of January, April, July, and October. This is structurally
    different from H026 (quarter-end window-dressing): that strategy catches
    the last 5 days of Mar/Jun/Sep/Dec; this catches the first 5 days of
    Jan/Apr/Jul/Oct.

    Only ~20 trading days active per year → expected low H001 correlation.
    """

    def __init__(self, n_days: int = 5, earnings_months: tuple = (1, 4, 7, 10)):
        self.n_days = n_days
        self.earnings_months = frozenset(earnings_months)
        self.name = f"pre_earnings_drift_{n_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        rank_start, _ = _calendar_month_ranks(prices.index)
        in_earnings_month = pd.Series(
            prices.index.month.isin(self.earnings_months).astype(float), index=prices.index
        )
        in_early_window = (rank_start < self.n_days).astype(float)
        pos = in_earnings_month * in_early_window
        pos.name = "position"
        return pos


# ---------------------------------------------------------------------------
# Batch 10 — H051–H055: New Asset Class Expansion
# Extends the asset universe beyond SPY/TLT/GLD to include commodities,
# credit markets, and yield-curve instruments.
# ---------------------------------------------------------------------------


class ThreeAssetRiskParity(MultiAssetStrategy):
    """H051: Vol-equalized risk parity across SPY, TLT, and GLD.

    Extends H021 (SPY+TLT) by adding GLD as a third asset. Each asset
    receives weight = (1/realized_vol) / sum(1/realized_vol), normalized
    to sum to 1.0. Daily rebalancing. Falls back to SPY+TLT when GLD
    data is unavailable (folds 0-2, pre-2004).

    Thesis: GLD diversifies a distinct regime — inflation/stagflation —
    where SPY and TLT both suffer but gold holds. SPY+TLT handles
    deflation crises; adding GLD covers inflationary crises.

    Requires: tlt_close, gld_close columns.
    """

    name = "three_asset_risk_parity"

    def __init__(self, vol_lookback: int = 60):
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = (
            prices["tlt_close"].pct_change()
            if "tlt_close" in prices.columns
            else pd.Series(np.nan, index=prices.index)
        )
        gld_r = (
            prices["gld_close"].pct_change()
            if "gld_close" in prices.columns
            else pd.Series(np.nan, index=prices.index)
        )

        def _inv_vol(r: pd.Series) -> pd.Series:
            v = r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
                1
            ) * np.sqrt(252)
            return (1.0 / v.replace(0, np.nan)).fillna(0.0)

        inv_spy = _inv_vol(spy_r)
        inv_tlt = _inv_vol(tlt_r)
        inv_gld = _inv_vol(gld_r)

        # Zero out assets whose price series is missing
        if "tlt_close" not in prices.columns:
            inv_tlt = pd.Series(0.0, index=prices.index)
        if "gld_close" not in prices.columns:
            inv_gld = pd.Series(0.0, index=prices.index)

        # When GLD NaN (pre-2004), treat as 0 weight → falls back to SPY+TLT parity
        inv_gld = inv_gld.where(
            prices.get("gld_close", pd.Series(np.nan, index=prices.index)).notna(), 0.0
        )

        total = inv_spy + inv_tlt + inv_gld
        valid = total > 0

        w_spy = (inv_spy / total).where(valid, 0.0)
        w_tlt = (inv_tlt / total).where(valid, 0.0)
        w_gld = (inv_gld / total).where(valid, 0.0)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt, "GLD": w_gld}, index=prices.index)


class CommodityMomentumOverlay(Strategy):
    """H052: Long SPY when DBC is below its MA (no commodity inflation regime).

    Rising commodity prices signal an inflationary environment that historically
    compresses equity multiples and triggers bond selloffs simultaneously.
    When DBC trends above its MA, the inflation regime is active — reducing
    equity exposure avoids the dual equity+bond drawdown of inflationary bears.

    Long SPY when DBC < ma_period MA (no inflation pressure); flat otherwise.
    When DBC data unavailable, defaults to long SPY.

    Requires: dbc_close column.
    """

    def __init__(self, ma_period: int = 200):
        self.ma_period = ma_period
        self.name = f"commodity_momentum_{ma_period}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "dbc_close" not in prices.columns:
            return pd.Series(1.0, index=prices.index, name="position")
        dbc = prices["dbc_close"]
        dbc_ma = dbc.rolling(self.ma_period, min_periods=self.ma_period).mean().shift(1)
        no_inflation = dbc.shift(1) < dbc_ma
        pos = no_inflation.astype(float)
        pos = pos.where(dbc_ma.notna(), 0.0)
        pos.name = "position"
        return pos


class CreditSpreadRegime(Strategy):
    """H053: Long SPY when HYG outperforms LQD (healthy credit appetite).

    Credit spread widening (HYG underperforming LQD) signals rising corporate
    default risk — a leading indicator of equity stress. When high-yield bonds
    are gaining relative to investment-grade, credit risk appetite is positive
    and equity exposure is favorable.

    Signal: long SPY when HYG momentum_days return > LQD momentum_days return.
    Falls back to flat when HYG data unavailable (pre-2007).

    Requires: hyg_close, lqd_close columns.
    """

    def __init__(self, momentum_days: int = 63):
        self.momentum_days = momentum_days
        self.name = f"credit_spread_regime_{momentum_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "hyg_close" not in prices.columns or "lqd_close" not in prices.columns:
            return pd.Series(0.0, index=prices.index, name="position")
        hyg = prices["hyg_close"]
        lqd = prices["lqd_close"]
        hyg_mom = hyg / hyg.shift(self.momentum_days) - 1
        lqd_mom = lqd / lqd.shift(self.momentum_days) - 1
        valid = hyg_mom.notna() & lqd_mom.notna()
        pos = (hyg_mom >= lqd_mom).astype(float)
        pos = pos.where(valid, 0.0)
        pos.name = "position"
        return pos


class YieldCurveProxy(Strategy):
    """H054: Long SPY when the yield curve is not inverted (SHY not outperforming TLT).

    The yield curve slope (short vs. long rates) is the classic recession
    predictor. Proxied via ETF relative performance: when SHY (1-3yr Treasury)
    outperforms TLT (20yr Treasury) on a 3-month basis, short rates are rising
    relative to long rates — a flattening/inverting yield curve signalling
    recession risk.

    Long SPY when TLT 3m return >= SHY 3m return (curve positive or steepening).
    Flat when SHY outperforms (curve inverted). Defaults to long SPY when data
    unavailable.

    Requires: shy_close column. tlt_close column used if available.
    """

    def __init__(self, mom_days: int = 63):
        self.mom_days = mom_days
        self.name = f"yield_curve_proxy_{mom_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "shy_close" not in prices.columns:
            return pd.Series(1.0, index=prices.index, name="position")
        shy = prices["shy_close"]
        tlt = prices["tlt_close"] if "tlt_close" in prices.columns else shy
        shy_mom = shy / shy.shift(self.mom_days) - 1
        tlt_mom = tlt / tlt.shift(self.mom_days) - 1
        valid = shy_mom.notna() & tlt_mom.notna()
        # Curve NOT inverted when TLT return >= SHY return (long duration not lagging short)
        curve_ok = tlt_mom >= shy_mom
        pos = curve_ok.astype(float)
        pos = pos.where(valid, 0.0)
        pos.name = "position"
        return pos


class FiveAssetRotation(MultiAssetStrategy):
    """H055: Cross-sectional momentum rotation across SPY, TLT, GLD, DBC, VNQ.

    Ranks all available assets by 12-1m momentum. Holds the top-ranked asset
    if its momentum is positive; goes to cash if the top performer has negative
    momentum (absolute momentum filter). Broadens H020 (3-asset) to 5 assets,
    adding commodities (DBC) and real estate (VNQ).

    Requires: tlt_close, gld_close, dbc_close, vnq_close columns.
    Assets with unavailable data are excluded from ranking for that period.
    """

    name = "five_asset_rotation"

    def __init__(self, long_days: int = 252, skip_days: int = 21):
        self.long_days = long_days
        self.skip_days = skip_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {
            "SPY": "adj_close",
            "TLT": "tlt_close",
            "GLD": "gld_close",
            "DBC": "dbc_close",
            "VNQ": "vnq_close",
        }
        # Only include assets whose column exists
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        moms = pd.DataFrame(index=prices.index)
        for asset, col in available.items():
            px = prices[col]
            moms[asset] = px.shift(self.skip_days) / px.shift(self.long_days) - 1

        weights = pd.DataFrame(0.0, index=prices.index, columns=list(available.keys()))
        valid_rows = moms.notna().any(axis=1)

        for i in range(len(prices)):
            if not valid_rows.iloc[i]:
                continue
            day_mom = moms.iloc[i].dropna()
            if day_mom.empty:
                continue
            best = day_mom.idxmax()
            if day_mom[best] > 0:
                weights.at[weights.index[i], best] = 1.0

        return weights


# ---------------------------------------------------------------------------
# Batch 11 — H056–H060: Yield Curve & Credit Macro Timing
# ---------------------------------------------------------------------------


class YieldCurveRotation(MultiAssetStrategy):
    """H056: Rotate SPY→TLT on yield curve inversion (SHY outperforms TLT).

    Normal curve (TLT outperforms SHY): long SPY.
    Inverted curve (SHY outperforms TLT): long TLT (flight-to-safety).
    Before SHY/TLT data: flat.
    """

    name = "yield_curve_rotation"

    def __init__(self, mom_days: int = 63) -> None:
        self.mom_days = mom_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt = prices.get("tlt_close", pd.Series(np.nan, index=prices.index))

        shy_r = shy.shift(1) / shy.shift(1 + self.mom_days) - 1
        tlt_r = tlt.shift(1) / tlt.shift(1 + self.mom_days) - 1

        valid = shy_r.notna() & tlt_r.notna()
        inverted = valid & (shy_r > tlt_r)

        w_spy = pd.Series(0.0, index=prices.index)
        w_tlt = pd.Series(0.0, index=prices.index)
        w_spy[valid & ~inverted] = 1.0
        w_tlt[inverted] = 1.0

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class BondCreditDualConfirm(Strategy):
    """H057: Long SPY when TLT below 200d MA AND HYG outperforms LQD.

    Falls back to TLT-only filter when HYG/LQD data unavailable (pre-2007).
    Both signals together block equity exposure during 2008-style crises.
    """

    def __init__(self, tlt_ma: int = 200, credit_days: int = 63) -> None:
        self.tlt_ma = tlt_ma
        self.credit_days = credit_days

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        tlt = prices.get("tlt_close", pd.Series(np.nan, index=prices.index))
        hyg = prices.get("hyg_close", pd.Series(np.nan, index=prices.index))
        lqd = prices.get("lqd_close", pd.Series(np.nan, index=prices.index))

        tlt_ma = tlt.shift(1).rolling(self.tlt_ma).mean()
        tlt_signal = tlt.shift(1) < tlt_ma

        hyg_r = hyg.shift(1) / hyg.shift(1 + self.credit_days) - 1
        lqd_r = lqd.shift(1) / lqd.shift(1 + self.credit_days) - 1
        credit_ok = hyg_r >= lqd_r

        has_credit = hyg_r.notna() & lqd_r.notna()

        pos = pd.Series(0.0, index=prices.index)
        pos[tlt_signal & (~has_credit)] = 1.0
        pos[tlt_signal & has_credit & credit_ok] = 1.0
        return pos


class InvertedCurveTreasuryLadder(MultiAssetStrategy):
    """H058: 75% TLT + 25% IEF when yield curve inverted for 21+ days.

    Sustained inversion (> sustain_days) signals a genuine recession risk.
    Brief inversions (<= sustain_days) are noise — hold SPY.
    Normal curve: 100% SPY.
    """

    name = "inverted_curve_treasury_ladder"

    def __init__(self, mom_days: int = 63, sustain_days: int = 21) -> None:
        self.mom_days = mom_days
        self.sustain_days = sustain_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt = prices.get("tlt_close", pd.Series(np.nan, index=prices.index))
        ief = prices.get("ief_close", pd.Series(np.nan, index=prices.index))

        shy_r = shy.shift(1) / shy.shift(1 + self.mom_days) - 1
        tlt_r = tlt.shift(1) / tlt.shift(1 + self.mom_days) - 1

        valid = shy_r.notna() & tlt_r.notna()
        raw_inv = valid & (shy_r > tlt_r)

        # Count consecutive inversion days
        inv_arr = raw_inv.to_numpy()
        consec = np.zeros(len(inv_arr), dtype=int)
        for i in range(1, len(inv_arr)):
            consec[i] = (consec[i - 1] + 1) if inv_arr[i] else 0
        sustained = pd.Series(consec >= self.sustain_days, index=prices.index)

        has_ief = ief.notna()

        w_spy = pd.Series(0.0, index=prices.index)
        w_tlt = pd.Series(0.0, index=prices.index)
        w_ief = pd.Series(0.0, index=prices.index)

        w_spy[valid & ~sustained] = 1.0
        w_tlt[sustained & has_ief] = 0.75
        w_ief[sustained & has_ief] = 0.25
        w_tlt[sustained & ~has_ief] = 1.0

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt, "IEF": w_ief}, index=prices.index)


class MacroFearComposite(Strategy):
    """H059: Position scaled by 4-signal macro fear composite.

    Fear votes: VIX > vix_thresh, GLD above gld_ma, HYG < LQD momentum,
    yield curve inverted (SHY > TLT on mom_days). Position = max(0, 1 - 2*score)
    where score = fear_count / 4. Two or more fear signals → flat.
    """

    def __init__(
        self,
        vix_thresh: float = 25.0,
        gld_ma: int = 50,
        mom_days: int = 63,
    ) -> None:
        self.vix_thresh = vix_thresh
        self.gld_ma = gld_ma
        self.mom_days = mom_days

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        vix = prices.get("vix_close", pd.Series(np.nan, index=prices.index))
        gld = prices.get("gld_close", pd.Series(np.nan, index=prices.index))
        hyg = prices.get("hyg_close", pd.Series(np.nan, index=prices.index))
        lqd = prices.get("lqd_close", pd.Series(np.nan, index=prices.index))
        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt = prices.get("tlt_close", pd.Series(np.nan, index=prices.index))

        fear = pd.DataFrame(0.0, index=prices.index, columns=["vix", "gld", "credit", "curve"])

        vix_sig = vix.shift(1)
        fear["vix"] = (vix_sig > self.vix_thresh).where(vix_sig.notna(), np.nan).fillna(0.0)

        gld_ma = gld.shift(1).rolling(self.gld_ma).mean()
        fear["gld"] = (gld.shift(1) > gld_ma).where(gld_ma.notna(), np.nan).fillna(0.0)

        hyg_r = hyg.shift(1) / hyg.shift(1 + self.mom_days) - 1
        lqd_r = lqd.shift(1) / lqd.shift(1 + self.mom_days) - 1
        fear["credit"] = (hyg_r < lqd_r).where(hyg_r.notna() & lqd_r.notna(), np.nan).fillna(0.0)

        shy_r = shy.shift(1) / shy.shift(1 + self.mom_days) - 1
        tlt_r = tlt.shift(1) / tlt.shift(1 + self.mom_days) - 1
        fear["curve"] = (shy_r > tlt_r).where(shy_r.notna() & tlt_r.notna(), np.nan).fillna(0.0)

        return (1.0 - 2.0 * fear.mean(axis=1)).clip(lower=0.0)


class CarryMomentumSPYTLT(MultiAssetStrategy):
    """H060: SPY/TLT/flat with yield curve + SPY trend gate.

    Curve steep (TLT > SHY on mom_days) AND SPY above spy_ma → 100% SPY.
    Curve inverted (SHY > TLT on mom_days) → 100% TLT (defensive carry).
    Otherwise (mixed signals or pre-data) → flat.
    """

    name = "carry_momentum_spy_tlt"

    def __init__(self, mom_days: int = 63, spy_ma: int = 200) -> None:
        self.mom_days = mom_days
        self.spy_ma = spy_ma

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt = prices.get("tlt_close", pd.Series(np.nan, index=prices.index))

        shy_r = shy.shift(1) / shy.shift(1 + self.mom_days) - 1
        tlt_r = tlt.shift(1) / tlt.shift(1 + self.mom_days) - 1
        spy_ma = prices["adj_close"].shift(1).rolling(self.spy_ma).mean()

        valid = shy_r.notna() & tlt_r.notna()
        inverted = valid & (shy_r > tlt_r)
        spy_trend_up = prices["adj_close"].shift(1) > spy_ma

        w_spy = pd.Series(0.0, index=prices.index)
        w_tlt = pd.Series(0.0, index=prices.index)

        w_spy[valid & ~inverted & spy_trend_up] = 1.0
        w_tlt[inverted] = 1.0

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


# ---------------------------------------------------------------------------
# Batch 12 — H061–H065: Equity Factor Rotation
# ---------------------------------------------------------------------------


class ValueGrowthRotation(MultiAssetStrategy):
    """H061: Rotate IVW/IVE/SPY based on yield curve + SPY trend.

    Normal curve (TLT > SHY mom) AND SPY uptrend → IVW (growth).
    Inverted curve (SHY > TLT mom) → IVE (value/defensive).
    Otherwise (no data or mixed) → SPY (baseline).
    """

    name = "value_growth_rotation"

    def __init__(self, mom_days: int = 63, spy_ma: int = 200) -> None:
        self.mom_days = mom_days
        self.spy_ma = spy_ma

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt = prices.get("tlt_close", pd.Series(np.nan, index=prices.index))
        ive = prices.get("ive_close", pd.Series(np.nan, index=prices.index))
        ivw = prices.get("ivw_close", pd.Series(np.nan, index=prices.index))

        shy_r = shy.shift(1) / shy.shift(1 + self.mom_days) - 1
        tlt_r = tlt.shift(1) / tlt.shift(1 + self.mom_days) - 1
        spy_ma = prices["adj_close"].shift(1).rolling(self.spy_ma).mean()

        curve_valid = shy_r.notna() & tlt_r.notna()
        inverted = curve_valid & (shy_r > tlt_r)
        spy_up = prices["adj_close"].shift(1) > spy_ma

        has_ivw = ivw.notna()
        has_ive = ive.notna()

        w_spy = pd.Series(1.0, index=prices.index)
        w_ivw = pd.Series(0.0, index=prices.index)
        w_ive = pd.Series(0.0, index=prices.index)

        # Normal curve + SPY uptrend → IVW (if available, else SPY)
        growth_regime = curve_valid & ~inverted & spy_up
        w_spy[growth_regime & has_ivw] = 0.0
        w_ivw[growth_regime & has_ivw] = 1.0

        # Inverted curve → IVE (if available, else SPY)
        w_spy[inverted & has_ive] = 0.0
        w_ive[inverted & has_ive] = 1.0

        return pd.DataFrame({"SPY": w_spy, "IVW": w_ivw, "IVE": w_ive}, index=prices.index)


class SmallCapCyclical(MultiAssetStrategy):
    """H062: Long IWM when credit healthy AND IWM outperforms SPY on momentum.

    Both conditions met → IWM; otherwise → SPY.
    Falls back to SPY when HYG/LQD data unavailable.
    """

    name = "small_cap_cyclical"

    def __init__(self, credit_days: int = 63, mom_days: int = 126) -> None:
        self.credit_days = credit_days
        self.mom_days = mom_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        hyg = prices.get("hyg_close", pd.Series(np.nan, index=prices.index))
        lqd = prices.get("lqd_close", pd.Series(np.nan, index=prices.index))
        iwm = prices.get("iwm_close", pd.Series(np.nan, index=prices.index))

        hyg_r = hyg.shift(1) / hyg.shift(1 + self.credit_days) - 1
        lqd_r = lqd.shift(1) / lqd.shift(1 + self.credit_days) - 1
        spy_mom = prices["adj_close"].shift(1) / prices["adj_close"].shift(1 + self.mom_days) - 1
        iwm_mom = iwm.shift(1) / iwm.shift(1 + self.mom_days) - 1

        credit_ok = (hyg_r >= lqd_r) & hyg_r.notna() & lqd_r.notna()
        size_ok = (iwm_mom >= spy_mom) & iwm_mom.notna()

        w_spy = pd.Series(1.0, index=prices.index)
        w_iwm = pd.Series(0.0, index=prices.index)

        small_cap_regime = credit_ok & size_ok
        w_spy[small_cap_regime] = 0.0
        w_iwm[small_cap_regime] = 1.0

        return pd.DataFrame({"SPY": w_spy, "IWM": w_iwm}, index=prices.index)


class DefensiveSectorOverlay(MultiAssetStrategy):
    """H063: SPY/XLP rotation based on SPY trend + VIX stress level.

    SPY above MA + VIX < vix_thresh → 100% SPY.
    SPY above MA + VIX >= vix_thresh → 50% SPY + 50% XLP.
    SPY below MA → 100% XLP.
    Before VIX/XLP data: 100% SPY.
    """

    name = "defensive_sector_overlay"

    def __init__(self, spy_ma: int = 200, vix_thresh: float = 25.0) -> None:
        self.spy_ma = spy_ma
        self.vix_thresh = vix_thresh

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        vix = prices.get("vix_close", pd.Series(np.nan, index=prices.index))
        xlp = prices.get("xlp_close", pd.Series(np.nan, index=prices.index))

        spy_ma = prices["adj_close"].shift(1).rolling(self.spy_ma).mean()
        spy_up = prices["adj_close"].shift(1) > spy_ma

        vix_calm = vix.shift(1) < self.vix_thresh
        has_vix = vix.shift(1).notna()
        has_xlp = xlp.notna()

        w_spy = pd.Series(1.0, index=prices.index)
        w_xlp = pd.Series(0.0, index=prices.index)

        # SPY uptrend + VIX calm → 100% SPY (default, no change)
        # SPY uptrend + VIX stressed → 50/50
        stress = spy_ma.notna() & spy_up & has_vix & ~vix_calm & has_xlp
        w_spy[stress] = 0.5
        w_xlp[stress] = 0.5

        # SPY downtrend → 100% XLP
        bearish = spy_ma.notna() & ~spy_up & has_xlp
        w_spy[bearish] = 0.0
        w_xlp[bearish] = 1.0

        return pd.DataFrame({"SPY": w_spy, "XLP": w_xlp}, index=prices.index)


class TechDefensiveMomentum(MultiAssetStrategy):
    """H064: 12-1m cross-sectional momentum between QQQ (tech) and XLU (defensive).

    Long QQQ if QQQ momentum > XLU momentum and QQQ > 0.
    Long XLU if XLU momentum > QQQ momentum and XLU > 0.
    Both negative → long SPY (neutral).
    """

    name = "tech_defensive_momentum"

    def __init__(self, long_days: int = 252, skip_days: int = 21) -> None:
        self.long_days = long_days
        self.skip_days = skip_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        qqq = prices.get("qqq_close", pd.Series(np.nan, index=prices.index))
        xlu = prices.get("xlu_close", pd.Series(np.nan, index=prices.index))

        qqq_mom = qqq.shift(1 + self.skip_days) / qqq.shift(1 + self.long_days) - 1
        xlu_mom = xlu.shift(1 + self.skip_days) / xlu.shift(1 + self.long_days) - 1
        spy_mom = (
            prices["adj_close"].shift(1 + self.skip_days)
            / prices["adj_close"].shift(1 + self.long_days)
            - 1
        )

        w_spy = pd.Series(0.0, index=prices.index)
        w_qqq = pd.Series(0.0, index=prices.index)
        w_xlu = pd.Series(0.0, index=prices.index)

        both_valid = qqq_mom.notna() & xlu_mom.notna()
        qqq_leads = both_valid & (qqq_mom >= xlu_mom) & (qqq_mom > 0)
        xlu_leads = both_valid & (xlu_mom > qqq_mom) & (xlu_mom > 0)
        both_neg = both_valid & (qqq_mom <= 0) & (xlu_mom <= 0)

        w_qqq[qqq_leads] = 1.0
        w_xlu[xlu_leads] = 1.0
        # Both negative: fallback to SPY if SPY positive, else flat
        w_spy[both_neg & (spy_mom > 0)] = 1.0

        return pd.DataFrame({"SPY": w_spy, "QQQ": w_qqq, "XLU": w_xlu}, index=prices.index)


class EqualWeightDefensiveFactor(MultiAssetStrategy):
    """H065: Static 25/25/25/25 equal-weight SPY+XLP+XLU+IVE.

    Pure factor quality tilt — no timing. Renormalizes when IVE
    unavailable (pre-2000): equal-weight among available assets.
    """

    name = "equal_weight_defensive_factor"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {
            "SPY": "adj_close",
            "XLP": "xlp_close",
            "XLU": "xlu_close",
            "IVE": "ive_close",
        }
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        weights = pd.DataFrame(0.0, index=prices.index, columns=list(available.keys()))
        for i, idx in enumerate(prices.index):
            active = [
                asset
                for asset, col in available.items()
                if col == "adj_close" or not np.isnan(prices[col].iloc[i])
            ]
            if active:
                w = 1.0 / len(active)
                for asset in active:
                    weights.at[idx, asset] = w
        return weights


# ---------------------------------------------------------------------------
# Batch 13 — H066–H070: Volatility Regime Advanced
# ---------------------------------------------------------------------------


class VixRiskParityEnhanced(MultiAssetStrategy):
    """H066: H021 RiskParity with VIX spike override.

    Normal: inverse-vol weights for SPY and TLT (identical to H021).
    Crisis: when VIX > vix_spike, override to 80% TLT / 20% SPY.
    Recovery: resume normal risk parity when VIX < vix_resume (hysteresis).
    """

    name = "vix_risk_parity_enhanced"

    def __init__(
        self,
        vol_lookback: int = 60,
        vix_spike: float = 35.0,
        vix_resume: float = 30.0,
        crisis_tlt: float = 0.80,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vix_spike = vix_spike
        self.vix_resume = vix_resume
        self.crisis_tlt = crisis_tlt

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy_rp = (inv_spy / total).fillna(0.0)
        w_tlt_rp = (inv_tlt / total).fillna(0.0)

        vix = prices["vix_close"].shift(1)
        w_spy = w_spy_rp.copy()
        w_tlt = w_tlt_rp.copy()

        # Hysteresis loop: crisis when VIX > spike, resume when VIX < resume
        in_crisis = False
        for i in range(len(prices)):
            v = vix.iloc[i]
            if np.isnan(v):
                continue
            if not in_crisis and v > self.vix_spike:
                in_crisis = True
            elif in_crisis and v < self.vix_resume:
                in_crisis = False
            if in_crisis:
                w_spy.iloc[i] = 1.0 - self.crisis_tlt
                w_tlt.iloc[i] = self.crisis_tlt

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class VixPercentileRegime(Strategy):
    """H067: Expanding-quantile adaptive VIX threshold.

    Long SPY when the current VIX is below its expanding (33rd percentile)
    percentile of all historical VIX readings to date. Adapts to the
    prevailing volatility regime unlike H029's fixed 20 threshold.
    """

    def __init__(self, quantile: float = 0.33) -> None:
        self.quantile = quantile
        self.name = f"vix_percentile_{int(quantile * 100)}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VixPercentileRegime requires 'vix_close' column")
        vix = prices["vix_close"]
        # Expanding quantile of past VIX values; shift 1 to avoid lookahead
        vix_thresh = vix.expanding(min_periods=63).quantile(self.quantile).shift(1)
        pos = pd.Series(0.0, index=prices.index, name="position")
        valid = vix_thresh.notna()
        pos[valid & (vix.shift(1) < vix_thresh)] = 1.0
        return pos


class VrpCalendarHybrid(Strategy):
    """H068: Volatility Risk Premium (H031) gated by Nov-Apr calendar (H043).

    Long SPY only when BOTH the VRP signal fires (VIX > realized vol) AND
    the date falls in the seasonally favorable Nov-Apr window. The calendar
    gate forces the strategy flat May-Oct, reducing H001 correlation.
    """

    def __init__(
        self,
        realized_window: int = 20,
        favorable_months: tuple = (11, 12, 1, 2, 3, 4),
    ) -> None:
        self.realized_window = realized_window
        self.favorable_months = frozenset(favorable_months)
        self.name = f"vrp_calendar_{realized_window}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VrpCalendarHybrid requires 'vix_close' column")
        rets = prices["adj_close"].pct_change()
        realized_vol = (
            rets.rolling(self.realized_window, min_periods=self.realized_window).std().shift(1)
            * np.sqrt(252)
            * 100
        )
        vix = prices["vix_close"].shift(1)
        in_season = pd.Series(
            prices.index.month.isin(self.favorable_months).astype(float), index=prices.index
        )
        vrp_signal = (vix > realized_vol) & realized_vol.notna() & vix.notna()
        pos = pd.Series(0.0, index=prices.index, name="position")
        pos[vrp_signal & (in_season == 1.0)] = 1.0
        return pos


class VixTermStructureProxy(Strategy):
    """H069: VIX term structure proxy via VIX/VIX_63d_MA ratio.

    When VIX is depressed vs. its 63d average (ratio < 0.85), the term
    structure is contango-like (calm) → long SPY.
    When VIX is elevated vs. its average (ratio > 1.15), the term structure
    resembles backwardation (crisis) → flat.
    Between thresholds, hold the prior position (hysteresis).
    """

    def __init__(
        self,
        ma_days: int = 63,
        contango_thresh: float = 0.85,
        backwardation_thresh: float = 1.15,
    ) -> None:
        self.ma_days = ma_days
        self.contango_thresh = contango_thresh
        self.backwardation_thresh = backwardation_thresh
        self.name = (
            f"vix_term_{ma_days}d_{int(contango_thresh * 100)}_{int(backwardation_thresh * 100)}"
        )

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VixTermStructureProxy requires 'vix_close' column")
        vix = prices["vix_close"]
        vix_ma = vix.rolling(self.ma_days, min_periods=self.ma_days).mean().shift(1)
        vix_lagged = vix.shift(1)
        ratio = vix_lagged / vix_ma.replace(0, np.nan)

        pos_arr = np.zeros(len(prices))
        current = 0.0
        for i in range(len(prices)):
            r = ratio.iloc[i]
            if np.isnan(r):
                pos_arr[i] = 0.0
                continue
            if r < self.contango_thresh:
                current = 1.0
            elif r > self.backwardation_thresh:
                current = 0.0
            # else: hold prior
            pos_arr[i] = current

        return pd.Series(pos_arr, index=prices.index, name="position")


class MultiAssetDynamicVolTarget(MultiAssetStrategy):
    """H070: H021 risk-parity allocation scaled by portfolio-level vol targeting.

    Two-layer risk management: H021 sets the SPY/TLT mix via inverse-vol
    weights (asset-level risk parity). A second layer then scales the TOTAL
    portfolio exposure down when 20d realized portfolio vol exceeds vol_target.
    Exposure is capped at 1.0 (no leverage).
    """

    name = "multi_asset_dynamic_vol_target"

    def __init__(
        self,
        vol_lookback: int = 60,
        port_vol_lookback: int = 20,
        vol_target: float = 0.10,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.port_vol_lookback = port_vol_lookback
        self.vol_target = vol_target

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        # H021-style inverse-vol weights
        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy = (inv_spy / total).fillna(0.0)
        w_tlt = (inv_tlt / total).fillna(0.0)

        # Portfolio realized returns using prior-day weights (shift 1 to avoid lookahead)
        port_r = w_spy.shift(1) * spy_r + w_tlt.shift(1) * tlt_r
        port_vol = port_r.rolling(
            self.port_vol_lookback, min_periods=self.port_vol_lookback
        ).std().shift(1) * np.sqrt(252)

        # Scale total exposure; cap at 1.0
        scale = (self.vol_target / port_vol).clip(upper=1.0)
        scale = scale.where(port_vol.notna() & (port_vol > 0), 1.0)

        w_spy_scaled = w_spy * scale
        w_tlt_scaled = w_tlt * scale

        return pd.DataFrame({"SPY": w_spy_scaled, "TLT": w_tlt_scaled}, index=prices.index)


# ---------------------------------------------------------------------------
# Batch 14 — H071–H075: Advanced Calendar Hybrids
# ---------------------------------------------------------------------------


class TripleConfirmCalendar(Strategy):
    """H071: Sell-in-May + bond confirm + January Barometer (three filters).

    Nov-Jan: long when TLT < tlt_ma_period MA (H043 filter).
    Feb-Apr: long when TLT < tlt_ma_period MA AND January return was positive.
    May-Oct: always flat.

    The January Barometer gates the Feb-Apr window only — it is resolved
    at the end of January, so no lookahead when applied from February.
    """

    def __init__(
        self,
        tlt_ma_period: int = 200,
        favorable_months: tuple = (11, 12, 1, 2, 3, 4),
    ) -> None:
        self.tlt_ma_period = tlt_ma_period
        self.favorable_months = frozenset(favorable_months)
        self.feb_apr = frozenset([2, 3, 4])
        self.name = f"triple_confirm_calendar_{tlt_ma_period}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "tlt_close" not in prices.columns:
            raise ValueError("TripleConfirmCalendar requires 'tlt_close' column")
        tlt = prices["tlt_close"]
        tlt_ma = tlt.rolling(self.tlt_ma_period, min_periods=self.tlt_ma_period).mean().shift(1)
        bond_ok = (tlt.shift(1) < tlt_ma) & tlt_ma.notna()

        # January return per year, resolved at month end → safe to use from Feb
        px = prices["adj_close"]
        jan_positive = pd.Series(False, index=prices.index)
        for year in sorted(px.index.year.unique()):
            jan_px = px[(px.index.year == year) & (px.index.month == 1)]
            if len(jan_px) < 2:
                continue
            jan_ret = jan_px.iloc[-1] / jan_px.iloc[0] - 1
            feb_apr_mask = (prices.index.year == year) & prices.index.month.isin(self.feb_apr)
            jan_positive[feb_apr_mask] = jan_ret > 0

        in_season = prices.index.month.isin(self.favorable_months)
        in_feb_apr = prices.index.month.isin(self.feb_apr)

        nov_jan_long = pd.Series(in_season & ~in_feb_apr, index=prices.index) & bond_ok
        feb_apr_long = (
            pd.Series(in_season & in_feb_apr, index=prices.index) & bond_ok & jan_positive
        )

        pos = (nov_jan_long | feb_apr_long).astype(float)
        pos.name = "position"
        return pos


class HalloweenCreditConfirm(Strategy):
    """H072: Halloween (Nov-Apr) + SPY momentum + credit spread confirm.

    Long SPY in Nov-Apr ONLY when: (1) 6m SPY momentum positive (H046 filter)
    AND (2) HYG 60d return > LQD 60d return (credit healthy). Flat otherwise.
    Falls back to flat (not just momentum) when HYG/LQD unavailable — the
    extra filter is not applied if credit data is missing.
    """

    def __init__(
        self,
        favorable_months: tuple = (11, 12, 1, 2, 3, 4),
        mom_days: int = 126,
        credit_days: int = 60,
    ) -> None:
        self.favorable_months = frozenset(favorable_months)
        self.mom_days = mom_days
        self.credit_days = credit_days
        self.name = f"halloween_credit_{mom_days}d_{credit_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        mom = (px / px.shift(self.mom_days) - 1).shift(1)
        in_season = pd.Series(
            prices.index.month.isin(self.favorable_months).astype(float), index=prices.index
        )
        mom_ok = (mom > 0) & mom.notna()

        hyg = prices.get("hyg_close", pd.Series(np.nan, index=prices.index))
        lqd = prices.get("lqd_close", pd.Series(np.nan, index=prices.index))
        hyg_r = hyg.shift(1) / hyg.shift(1 + self.credit_days) - 1
        lqd_r = lqd.shift(1) / lqd.shift(1 + self.credit_days) - 1
        credit_ok = (hyg_r > lqd_r) & hyg_r.notna() & lqd_r.notna()

        pos = pd.Series(0.0, index=prices.index, name="position")
        pos[(in_season == 1.0) & mom_ok & credit_ok] = 1.0
        return pos


class SellInMayYieldCurveConfirm(Strategy):
    """H073: Sell-in-May + bond trend + yield-curve-not-inverted (three filters).

    Long Nov-Apr when: (1) TLT < tlt_ma MA (H043 bond filter) AND
    (2) SHY curve_days return < TLT curve_days return (curve not inverted).
    Always flat May-Oct. Falls back to flat pre-2002 when TLT/SHY unavailable.
    """

    def __init__(
        self,
        tlt_ma: int = 200,
        curve_days: int = 63,
        favorable_months: tuple = (11, 12, 1, 2, 3, 4),
    ) -> None:
        self.tlt_ma = tlt_ma
        self.curve_days = curve_days
        self.favorable_months = frozenset(favorable_months)
        self.name = f"sell_in_may_yield_curve_{tlt_ma}d_{curve_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "tlt_close" not in prices.columns:
            raise ValueError("SellInMayYieldCurveConfirm requires 'tlt_close' column")
        tlt = prices["tlt_close"]
        tlt_ma_val = tlt.rolling(self.tlt_ma, min_periods=self.tlt_ma).mean().shift(1)
        bond_ok = (tlt.shift(1) < tlt_ma_val) & tlt_ma_val.notna()

        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt_r = tlt.shift(1) / tlt.shift(1 + self.curve_days) - 1
        shy_r = shy.shift(1) / shy.shift(1 + self.curve_days) - 1
        not_inverted = (shy_r < tlt_r) & shy_r.notna() & tlt_r.notna()

        in_season = pd.Series(
            prices.index.month.isin(self.favorable_months).astype(float), index=prices.index
        )
        pos = pd.Series(0.0, index=prices.index, name="position")
        pos[(in_season == 1.0) & bond_ok & not_inverted] = 1.0
        return pos


class PreHolidayDrift(Strategy):
    """H074: Long SPY on the N trading days before each US market holiday.

    Pre-holiday effect: short-covering and reduced order flow generate
    positive drift before the market closes. Uses the US Federal Holiday
    calendar as a proxy for NYSE holidays.
    """

    def __init__(self, pre_days: int = 1) -> None:
        self.pre_days = pre_days
        self.name = f"pre_holiday_drift_{pre_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        from pandas.tseries.holiday import USFederalHolidayCalendar

        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=prices.index.min(), end=prices.index.max())
        trading_days = prices.index
        pre_holiday_set: set = set()
        for h in holidays:
            prev = trading_days[trading_days < h]
            for d in range(1, self.pre_days + 1):
                if len(prev) >= d:
                    pre_holiday_set.add(prev[-d])

        return pd.Series(
            prices.index.isin(pre_holiday_set).astype(float), index=prices.index, name="position"
        )


class EarningsSeasonDrift(Strategy):
    """H075: Long SPY during first N trading days of each earnings month.

    Earnings months: Jan, Apr, Jul, Oct. Index-level flow effect from
    positive earnings surprises attracting retail buying. Structurally
    different from H047 (pre-earnings 5 days) — this tests the full
    first half of each earnings month (15 trading days ≈ 3 calendar weeks).
    """

    def __init__(
        self,
        active_days: int = 15,
        earnings_months: tuple = (1, 4, 7, 10),
    ) -> None:
        self.active_days = active_days
        self.earnings_months = frozenset(earnings_months)
        self.name = f"earnings_season_drift_{active_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        rank_start, _ = _calendar_month_ranks(prices.index)
        in_earnings = pd.Series(
            prices.index.month.isin(self.earnings_months).astype(float), index=prices.index
        )
        in_window = (rank_start < self.active_days).astype(float)
        pos = in_earnings * in_window
        pos.name = "position"
        return pos


# ---------------------------------------------------------------------------
# Batch 15 — H076–H080: Multi-Asset Risk Parity Extensions
# ---------------------------------------------------------------------------


class FourAssetRiskParity(MultiAssetStrategy):
    """H076: SPY+TLT+GLD+DBC inverse-vol risk parity (full macro quadrant).

    SPY (growth), TLT (deflation), GLD (stagflation/fear), DBC (inflation).
    Weights each available asset by inverse realized vol, normalized to 1.
    Gracefully degrades to fewer assets when data unavailable.
    """

    name = "four_asset_risk_parity"

    def __init__(self, vol_lookback: int = 60) -> None:
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {
            "SPY": "adj_close",
            "TLT": "tlt_close",
            "GLD": "gld_close",
            "DBC": "dbc_close",
        }
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        vols: dict[str, pd.Series] = {}
        for asset, col in available.items():
            r = prices[col].pct_change()
            vols[asset] = r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
                1
            ) * np.sqrt(252)

        inv_vols = {a: (1.0 / v.replace(0, np.nan)) for a, v in vols.items()}
        total = sum(inv_vols.values())
        weights = pd.DataFrame(index=prices.index, columns=list(available.keys()), dtype=float)
        for asset in available:
            weights[asset] = (inv_vols[asset] / total).fillna(0.0)

        return weights


class RiskParityMomentumFilter(MultiAssetStrategy):
    """H077: H021 risk parity with per-asset 6m momentum filter.

    Each asset's weight is zeroed if its 6m return is negative; the freed
    risk budget is reallocated to remaining positive-momentum assets.
    If all assets have negative 6m momentum, goes to cash (all weights zero).
    """

    name = "risk_parity_momentum_filter"

    def __init__(self, vol_lookback: int = 60, mom_days: int = 126) -> None:
        self.vol_lookback = vol_lookback
        self.mom_days = mom_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        spy_mom = prices["adj_close"].shift(1) / prices["adj_close"].shift(1 + self.mom_days) - 1
        tlt_mom = prices["tlt_close"].shift(1) / prices["tlt_close"].shift(1 + self.mom_days) - 1

        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)

        # Zero out assets with negative or unavailable momentum
        spy_ok = (spy_mom > 0) & spy_mom.notna()
        tlt_ok = (tlt_mom > 0) & tlt_mom.notna()
        inv_spy_filtered = inv_spy.where(spy_ok, 0.0)
        inv_tlt_filtered = inv_tlt.where(tlt_ok, 0.0)

        total = inv_spy_filtered + inv_tlt_filtered
        w_spy = (inv_spy_filtered / total.replace(0, np.nan)).fillna(0.0)
        w_tlt = (inv_tlt_filtered / total.replace(0, np.nan)).fillna(0.0)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class SeasonalRiskParity(MultiAssetStrategy):
    """H078: H021 risk parity Nov-Apr; 100% TLT May-Oct.

    Combines H021's asset allocation with H043's seasonal insight.
    In the favorable seasonal window (Nov-Apr), standard inverse-vol
    weighting between SPY and TLT. In the unfavorable window (May-Oct),
    full defensive rotation into TLT — no cash drag, no equity exposure.
    """

    name = "seasonal_risk_parity"

    def __init__(
        self,
        vol_lookback: int = 60,
        favorable_months: tuple = (11, 12, 1, 2, 3, 4),
    ) -> None:
        self.vol_lookback = vol_lookback
        self.favorable_months = frozenset(favorable_months)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy_rp = (inv_spy / total).fillna(0.0)
        w_tlt_rp = (inv_tlt / total).fillna(0.0)

        in_season = pd.Series(prices.index.month.isin(self.favorable_months), index=prices.index)
        # May-Oct: 100% TLT (when TLT data available)
        tlt_avail = tlt_vol.notna()
        w_spy = w_spy_rp.where(in_season, 0.0)
        w_tlt = w_tlt_rp.where(in_season, tlt_avail.astype(float))

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class CrisisProtocolRiskParity(MultiAssetStrategy):
    """H079: H021 risk parity with VIX+drawdown emergency override.

    Normal: inverse-vol weights (identical to H021).
    Crisis: when SPY drawdown from 20d high < crisis_dd AND VIX > vix_spike,
    override to crisis_tlt% TLT + (1-crisis_tlt)% SPY.
    Resume normal when drawdown recovers past resume_dd AND VIX < vix_resume.
    Uses hysteresis to prevent rapid toggling.
    """

    name = "crisis_protocol_risk_parity"

    def __init__(
        self,
        vol_lookback: int = 60,
        drawdown_window: int = 20,
        crisis_dd: float = -0.10,
        resume_dd: float = -0.07,
        vix_spike: float = 35.0,
        vix_resume: float = 30.0,
        crisis_tlt: float = 0.90,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.drawdown_window = drawdown_window
        self.crisis_dd = crisis_dd
        self.resume_dd = resume_dd
        self.vix_spike = vix_spike
        self.vix_resume = vix_resume
        self.crisis_tlt = crisis_tlt

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy_rp = (inv_spy / total).fillna(0.0)
        w_tlt_rp = (inv_tlt / total).fillna(0.0)

        # SPY drawdown from rolling high (shift 1 to avoid lookahead)
        spy_px = prices["adj_close"].shift(1)
        rolling_high = spy_px.rolling(self.drawdown_window, min_periods=1).max()
        drawdown = (spy_px / rolling_high - 1).fillna(0.0)
        vix = prices["vix_close"].shift(1)

        w_spy = w_spy_rp.copy()
        w_tlt = w_tlt_rp.copy()

        in_crisis = False
        for i in range(len(prices)):
            dd = drawdown.iloc[i]
            v = vix.iloc[i] if not np.isnan(vix.iloc[i]) else 0.0
            if not in_crisis and dd < self.crisis_dd and v > self.vix_spike:
                in_crisis = True
            elif in_crisis and dd > self.resume_dd and v < self.vix_resume:
                in_crisis = False
            if in_crisis:
                w_spy.iloc[i] = 1.0 - self.crisis_tlt
                w_tlt.iloc[i] = self.crisis_tlt

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class RiskParityYieldCurveOverlay(MultiAssetStrategy):
    """H080: H021 risk parity with yield-curve inversion overlay.

    Normal: H021 standard inverse-vol weights.
    Curve inverted (SHY curve_days return > TLT curve_days return):
    multiply SPY weight by spy_scale (0.50) and redistribute freed weight
    to TLT. Adds macro recession awareness on top of vol-based allocation.
    """

    name = "risk_parity_yield_curve_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
        curve_days: int = 63,
        spy_scale: float = 0.50,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.curve_days = curve_days
        self.spy_scale = spy_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy = (inv_spy / total).fillna(0.0)
        w_tlt = (inv_tlt / total).fillna(0.0)

        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt_px = prices["tlt_close"]
        shy_ret = shy.shift(1) / shy.shift(1 + self.curve_days) - 1
        tlt_ret = tlt_px.shift(1) / tlt_px.shift(1 + self.curve_days) - 1
        inverted = (shy_ret > tlt_ret) & shy_ret.notna() & tlt_ret.notna()

        # When inverted: scale SPY weight down, redistribute to TLT
        spy_cut = w_spy * (1.0 - self.spy_scale)
        w_spy_adj = w_spy.where(~inverted, w_spy - spy_cut)
        w_tlt_adj = w_tlt.where(~inverted, w_tlt + spy_cut)

        return pd.DataFrame({"SPY": w_spy_adj, "TLT": w_tlt_adj}, index=prices.index)


class DualFrequencyMomentum(Strategy):
    """H081: Long SPY only when both 12-1m AND 1m momentum are positive.

    12-1m (trend): return from 252 days ago to 21 days ago — confirms durable uptrend.
    1m (short-term): return over last 21 days — avoids entry during pullbacks in uptrend.
    Both must be positive to hold; flat when either is negative.
    """

    name = "dual_frequency_momentum"

    def __init__(self, mom_long: int = 252, mom_skip: int = 21, mom_short: int = 21) -> None:
        self.mom_long = mom_long
        self.mom_skip = mom_skip
        self.mom_short = mom_short

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        spy = prices["adj_close"]
        # 12-1m: return from mom_long days ago to mom_skip days ago (skip recent month)
        mom_long_sig = spy.shift(self.mom_skip) / spy.shift(self.mom_long) - 1
        # 1m: return over last mom_short days (current short-term trend)
        mom_short_sig = spy / spy.shift(self.mom_short) - 1
        pos = pd.Series(0.0, index=prices.index, name="position")
        valid = mom_long_sig.notna() & mom_short_sig.notna()
        pos[valid & (mom_long_sig > 0) & (mom_short_sig > 0)] = 1.0
        return pos


class RiskAdjustedMomentum(MultiAssetStrategy):
    """H082: Long the SPY/TLT/GLD asset with highest (12-1m return / vol) score.

    Risk-adjusts cross-sectional momentum to avoid picking high-vol winners that
    have outsized drawdowns. Normalizes each asset's 12-1m return by its realized
    vol over the same period. Flat if all scores non-positive.
    """

    name = "risk_adjusted_momentum"

    def __init__(self, mom_days: int = 252, skip_days: int = 21, vol_days: int = 252) -> None:
        self.mom_days = mom_days
        self.skip_days = skip_days
        self.vol_days = vol_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {"SPY": "adj_close", "TLT": "tlt_close", "GLD": "gld_close"}
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        scores: dict[str, pd.Series] = {}
        for asset, col in available.items():
            px = prices[col]
            ret = px.pct_change()
            mom = px.shift(1 + self.skip_days) / px.shift(1 + self.mom_days) - 1
            vol = ret.rolling(self.vol_days, min_periods=self.vol_days).std().shift(1) * np.sqrt(
                252
            )
            scores[asset] = (mom / vol.replace(0, np.nan)).shift(0)

        score_df = pd.DataFrame(scores, index=prices.index)
        weights = pd.DataFrame(0.0, index=prices.index, columns=list(available.keys()))
        for i in range(len(prices)):
            row = score_df.iloc[i]
            valid = row.dropna()
            if valid.empty or valid.max() <= 0:
                continue
            winner = valid.idxmax()
            weights.loc[prices.index[i], winner] = 1.0

        return weights


class ContrarianMultiAssetRotation(MultiAssetStrategy):
    """H083: Long the WORST 12-1m performer among SPY/TLT/GLD if its 3m trend reversed.

    De Bondt-Thaler long-term reversal applied to ETFs: the worst-performing asset
    over the prior year is a long candidate, but only when its 3m return turns positive
    (reversal has started — avoids falling knives).
    """

    name = "contrarian_multi_asset_rotation"

    def __init__(self, mom_days: int = 252, skip_days: int = 21, confirm_days: int = 63) -> None:
        self.mom_days = mom_days
        self.skip_days = skip_days
        self.confirm_days = confirm_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {"SPY": "adj_close", "TLT": "tlt_close", "GLD": "gld_close"}
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        moms: dict[str, pd.Series] = {}
        confirms: dict[str, pd.Series] = {}
        for asset, col in available.items():
            px = prices[col]
            moms[asset] = px.shift(1 + self.skip_days) / px.shift(1 + self.mom_days) - 1
            confirms[asset] = px.shift(1) / px.shift(1 + self.confirm_days) - 1

        mom_df = pd.DataFrame(moms, index=prices.index)
        confirm_df = pd.DataFrame(confirms, index=prices.index)
        weights = pd.DataFrame(0.0, index=prices.index, columns=list(available.keys()))

        for i in range(len(prices)):
            m_row = mom_df.iloc[i].dropna()
            c_row = confirm_df.iloc[i].dropna()
            common = m_row.index.intersection(c_row.index)
            if len(common) < 2:
                continue
            loser = m_row[common].idxmin()
            if c_row[loser] > 0:
                weights.loc[prices.index[i], loser] = 1.0

        return weights


class MomentumCarryBlend(MultiAssetStrategy):
    """H084: 50% cross-sectional momentum + 50% carry (term premium proxy).

    Momentum sub-signal: 100% in best 12-1m return asset (SPY/TLT/GLD) if positive.
    Carry sub-signal: 100% TLT if TLT 12m return > SHY 12m return (positive term premium),
    else 100% SPY. Average the two sub-portfolios 50/50.
    """

    name = "momentum_carry_blend"

    def __init__(
        self, mom_days: int = 252, skip_days: int = 21, carry_days: int = 252, blend: float = 0.5
    ) -> None:
        self.mom_days = mom_days
        self.skip_days = skip_days
        self.carry_days = carry_days
        self.blend = blend

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {"SPY": "adj_close", "TLT": "tlt_close", "GLD": "gld_close"}
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        # Momentum sub-signal: top 12-1m return asset (if positive)
        moms: dict[str, pd.Series] = {}
        for asset, col in available.items():
            px = prices[col]
            moms[asset] = px.shift(1 + self.skip_days) / px.shift(1 + self.mom_days) - 1
        mom_df = pd.DataFrame(moms, index=prices.index)

        # Carry sub-signal: TLT if term premium positive (TLT return > SHY return)
        tlt_ret = prices["tlt_close"].shift(1) / prices["tlt_close"].shift(1 + self.carry_days) - 1
        shy_px = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        shy_ret = shy_px.shift(1) / shy_px.shift(1 + self.carry_days) - 1
        term_premium = (tlt_ret > shy_ret) & tlt_ret.notna() & shy_ret.notna()

        cols = list(available.keys())
        w_mom = pd.DataFrame(0.0, index=prices.index, columns=cols)
        w_carry = pd.DataFrame(0.0, index=prices.index, columns=cols)

        idx_val = prices.index
        for i in range(len(prices)):
            row = mom_df.iloc[i].dropna()
            if not row.empty and row.max() > 0:
                w_mom.loc[idx_val[i], row.idxmax()] = 1.0
            # Carry: TLT if term premium, else SPY (when both available)
            if (
                "TLT" in available
                and "SPY" in available
                and pd.notna(tlt_ret.iloc[i])
                and pd.notna(shy_ret.iloc[i])
            ):
                if term_premium.iloc[i]:
                    w_carry.loc[idx_val[i], "TLT"] = 1.0
                else:
                    w_carry.loc[idx_val[i], "SPY"] = 1.0

        return self.blend * w_mom + (1.0 - self.blend) * w_carry


class MomentumMacroGate(MultiAssetStrategy):
    """H085: Cross-sectional momentum gated by yield curve inversion.

    Normally: long the top 12-1m return asset among available SPY/TLT/GLD/DBC/VNQ.
    During yield curve inversion (SHY 3m return > TLT 3m return): force 100% TLT
    regardless of momentum ranking (macro override).
    """

    name = "momentum_macro_gate"

    def __init__(self, mom_days: int = 252, skip_days: int = 21, curve_days: int = 63) -> None:
        self.mom_days = mom_days
        self.skip_days = skip_days
        self.curve_days = curve_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {
            "SPY": "adj_close",
            "TLT": "tlt_close",
            "GLD": "gld_close",
            "DBC": "dbc_close",
            "VNQ": "vnq_close",
        }
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        moms: dict[str, pd.Series] = {}
        for asset, col in available.items():
            px = prices[col]
            moms[asset] = px.shift(1 + self.skip_days) / px.shift(1 + self.mom_days) - 1
        mom_df = pd.DataFrame(moms, index=prices.index)

        tlt_px = prices["tlt_close"]
        shy_px = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt_curve = tlt_px.shift(1) / tlt_px.shift(1 + self.curve_days) - 1
        shy_curve = shy_px.shift(1) / shy_px.shift(1 + self.curve_days) - 1
        inverted = (shy_curve > tlt_curve) & shy_curve.notna() & tlt_curve.notna()

        cols = list(available.keys())
        weights = pd.DataFrame(0.0, index=prices.index, columns=cols)

        idx_val = prices.index
        for i in range(len(prices)):
            if inverted.iloc[i] and "TLT" in available:
                weights.loc[idx_val[i], "TLT"] = 1.0
                continue
            row = mom_df.iloc[i].dropna()
            if row.empty or row.max() <= 0:
                continue
            weights.loc[idx_val[i], row.idxmax()] = 1.0

        return weights


# ---------------------------------------------------------------------------
# Batch 17 — H086–H090: Drawdown Control Strategies
# ---------------------------------------------------------------------------


class LowVolTargetPortfolio(MultiAssetStrategy):
    """H086: H021 inverse-vol allocation with stricter 8% portfolio vol target.

    Identical mechanism to H070 (MultiAssetDynamicVolTarget) but defaults to
    vol_target=8% instead of 10%. Lower target means earlier de-risking and
    smaller positions in high-vol regimes, trading CAGR for Max DD improvement.
    """

    name = "low_vol_target_portfolio"

    def __init__(
        self,
        vol_lookback: int = 60,
        port_vol_lookback: int = 20,
        vol_target: float = 0.08,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.port_vol_lookback = port_vol_lookback
        self.vol_target = vol_target

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy = (inv_spy / total).fillna(0.0)
        w_tlt = (inv_tlt / total).fillna(0.0)

        port_r = w_spy.shift(1) * spy_r + w_tlt.shift(1) * tlt_r
        port_vol = port_r.rolling(
            self.port_vol_lookback, min_periods=self.port_vol_lookback
        ).std().shift(1) * np.sqrt(252)
        scale = (self.vol_target / port_vol).clip(upper=1.0)
        scale = scale.where(port_vol.notna() & (port_vol > 0), 1.0)

        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class RatchetDrawdownProtection(MultiAssetStrategy):
    """H087: H021 weights scaled proportionally by portfolio drawdown from peak.

    scale = max(0, 1 − portfolio_drawdown / drawdown_limit)
    At 0% drawdown: full H021 allocation (scale=1.0).
    At drawdown_limit: flat (scale=0.0).
    Uses H021 unscaled portfolio returns to track the drawdown (no circularity).
    """

    name = "ratchet_drawdown_protection"

    def __init__(self, vol_lookback: int = 60, drawdown_limit: float = 0.15) -> None:
        self.vol_lookback = vol_lookback
        self.drawdown_limit = drawdown_limit

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy = (inv_spy / total).fillna(0.0)
        w_tlt = (inv_tlt / total).fillna(0.0)

        # Simulate H021 portfolio cumulative return (using lagged weights → no lookahead)
        port_r = w_spy.shift(1) * spy_r + w_tlt.shift(1) * tlt_r
        port_cum = (1.0 + port_r.fillna(0.0)).cumprod()
        rolling_peak = port_cum.expanding(min_periods=1).max()
        dd = (port_cum / rolling_peak - 1.0).fillna(0.0)

        # Ratchet scale based on yesterday's drawdown
        scale = (1.0 + dd.shift(1).fillna(0.0) / self.drawdown_limit).clip(lower=0.0, upper=1.0)

        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class H001H021DirectBlend(MultiAssetStrategy):
    """H088: 50/50 blend of H001 (MA trend) and H021 (risk parity) daily.

    H001 portion contributes pure SPY exposure (long or flat).
    H021 portion contributes SPY+TLT via inverse-vol weights.
    Combined: w_spy = 0.5*h001 + 0.5*w021_spy; w_tlt = 0.5*w021_tlt.
    Inherits H001 bull participation and H021 crisis diversification.
    """

    name = "h001_h021_direct_blend"

    def __init__(
        self,
        vol_lookback: int = 60,
        fast: int = 50,
        slow: int = 200,
        blend: float = 0.5,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.fast = fast
        self.slow = slow
        self.blend = blend

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy = prices["adj_close"]
        spy_r = spy.pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        # H001: MA50/200 signal (shift 1 to avoid lookahead — MA uses current-day close)
        fast_ma = spy.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = spy.rolling(self.slow, min_periods=self.slow).mean()
        h001_pos = (fast_ma > slow_ma).astype(float).shift(1).fillna(0.0)

        # H021: inverse-vol weights
        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy_h021 = (inv_spy / total).fillna(0.0)
        w_tlt_h021 = (inv_tlt / total).fillna(0.0)

        w_spy = self.blend * h001_pos + (1.0 - self.blend) * w_spy_h021
        w_tlt = (1.0 - self.blend) * w_tlt_h021

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class H043H021RegimeBlend(MultiAssetStrategy):
    """H089: 50/50 blend of H043 (Sell-in-May bond confirm) and H021 (risk parity).

    H043 is long SPY in Nov-Apr when TLT is NOT in uptrend, flat otherwise.
    H021 runs continuously with inverse-vol SPY+TLT weights.
    During May-Oct (H043 flat): portfolio is 50% H021 → 50% cash (natural de-risk).
    Expected MaxDD between H043's -10% and H021's -18%.
    """

    name = "h043_h021_regime_blend"

    def __init__(
        self,
        vol_lookback: int = 60,
        tlt_ma_period: int = 200,
        favorable_months: tuple = (11, 12, 1, 2, 3, 4),
        blend: float = 0.5,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.tlt_ma_period = tlt_ma_period
        self.favorable_months = frozenset(favorable_months)
        self.blend = blend

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        tlt = prices["tlt_close"]
        spy_r = prices["adj_close"].pct_change()
        tlt_r = tlt.pct_change()

        # H043 signal: Nov-Apr AND TLT NOT in uptrend (risk-on filter)
        tlt_ma = tlt.rolling(self.tlt_ma_period, min_periods=self.tlt_ma_period).mean().shift(1)
        in_season = pd.Series(
            prices.index.month.isin(self.favorable_months).astype(float), index=prices.index
        )
        bond_riskoff = (tlt.shift(1) > tlt_ma).astype(float)
        h043_pos = in_season * (1.0 - bond_riskoff)
        h043_pos = h043_pos.where(tlt_ma.notna(), 0.0)

        # H021: inverse-vol weights
        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy_h021 = (inv_spy / total).fillna(0.0)
        w_tlt_h021 = (inv_tlt / total).fillna(0.0)

        w_spy = self.blend * h043_pos + (1.0 - self.blend) * w_spy_h021
        w_tlt = (1.0 - self.blend) * w_tlt_h021

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class MonthlyRebalancingRiskParity(MultiAssetStrategy):
    """H090: H021 risk parity with monthly (not daily) weight recalculation.

    Computes H021 inverse-vol weights on the first trading day of each calendar
    month and holds them constant until the next month-start. Reduces turnover
    ~20× vs daily H021 (from ~4845 to ~240 rebalances/year) at the cost of
    slower adaptation to volatility regime changes.
    """

    name = "monthly_rebalancing_risk_parity"

    def __init__(self, vol_lookback: int = 60) -> None:
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy_daily = (inv_spy / total).fillna(0.0)
        w_tlt_daily = (inv_tlt / total).fillna(0.0)

        # Identify first trading day of each calendar month
        idx = prices.index
        month_starts = set()
        seen: set = set()
        for d in idx:
            key = (d.year, d.month)
            if key not in seen:
                seen.add(key)
                month_starts.add(d)

        # Keep weights only at month-starts, forward-fill all other days
        w_spy_m = w_spy_daily.copy().where(pd.Series(idx.isin(month_starts), index=idx), np.nan)
        w_tlt_m = w_tlt_daily.copy().where(pd.Series(idx.isin(month_starts), index=idx), np.nan)
        w_spy_m = w_spy_m.ffill().fillna(0.0)
        w_tlt_m = w_tlt_m.ffill().fillna(0.0)

        return pd.DataFrame({"SPY": w_spy_m, "TLT": w_tlt_m}, index=prices.index)


# ---------------------------------------------------------------------------
# Batch 19 — H096–H098: Advanced Signal Construction
# ---------------------------------------------------------------------------


class PCARiskRegime(MultiAssetStrategy):
    """H096: SPY-TLT-GLD principal component regime.

    Fits rolling PCA to identify the dominant risk factor across SPY, TLT, GLD.
    PC1 (forced positive SPY loading) captures risk-on/risk-off sentiment.
    Signal = projected 21-day return onto PC1; positive → long SPY, negative → long TLT.
    """

    name = "pca_risk_regime"

    def __init__(self, pca_lookback: int = 252, signal_days: int = 21) -> None:
        self.pca_lookback = pca_lookback
        self.signal_days = signal_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        has_gld = "gld_close" in prices.columns
        gld_r = (
            prices["gld_close"].pct_change() if has_gld else pd.Series(np.nan, index=prices.index)
        )

        spy_cum = prices["adj_close"].pct_change(self.signal_days)
        tlt_cum = prices["tlt_close"].pct_change(self.signal_days)
        gld_cum = (
            prices["gld_close"].pct_change(self.signal_days)
            if has_gld
            else pd.Series(np.nan, index=prices.index)
        )

        n = len(prices)
        score_arr = np.full(n, np.nan)

        for i in range(self.pca_lookback + self.signal_days, n):
            w_spy = spy_r.values[i - self.pca_lookback : i]
            w_tlt = tlt_r.values[i - self.pca_lookback : i]
            w_gld = gld_r.values[i - self.pca_lookback : i]

            n_gld_valid = int(np.sum(~np.isnan(w_gld)))
            if n_gld_valid > self.pca_lookback // 2:
                ret_mat = np.column_stack([w_spy, w_tlt, w_gld])
                sig = np.array(
                    [spy_cum.values[i - 1], tlt_cum.values[i - 1], gld_cum.values[i - 1]]
                )
            else:
                ret_mat = np.column_stack([w_spy, w_tlt])
                sig = np.array([spy_cum.values[i - 1], tlt_cum.values[i - 1]])

            valid = ~np.isnan(ret_mat).any(axis=1)
            clean = ret_mat[valid]
            if len(clean) < 60 or np.isnan(sig).any():
                continue

            cov = np.cov(clean.T)
            if cov.ndim < 2:
                cov = np.array([[float(cov)]])
            eigvals, eigvecs = np.linalg.eigh(cov)
            pc1 = eigvecs[:, -1]
            if pc1[0] < 0:
                pc1 = -pc1

            score_arr[i] = float(np.dot(sig, pc1))

        scores = pd.Series(score_arr, index=prices.index).shift(1)
        tlt_ok = prices["tlt_close"].notna()
        valid_sig = scores.notna()

        w_spy_out = pd.Series(0.0, index=prices.index)
        w_tlt_out = pd.Series(0.0, index=prices.index)
        w_spy_out[valid_sig & (scores > 0)] = 1.0
        w_tlt_out[valid_sig & (scores <= 0) & tlt_ok] = 1.0

        return pd.DataFrame({"SPY": w_spy_out, "TLT": w_tlt_out}, index=prices.index)


class CorrelationRiskParityHybrid(MultiAssetStrategy):
    """H097: H034 correlation regime overlaid on H021 inverse-vol weights.

    Normal: H021 inverse-vol weights.
    Crisis: when rolling 60d SPY-TLT correlation turns positive → override to
            crisis_spy / crisis_tlt split (default 10% / 90%).
    """

    name = "correlation_risk_parity_hybrid"

    def __init__(
        self,
        vol_lookback: int = 60,
        corr_lookback: int = 60,
        crisis_tlt: float = 0.90,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.corr_lookback = corr_lookback
        self.crisis_tlt = crisis_tlt

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback).std()
        tlt_vol = tlt_r.rolling(self.vol_lookback).std()

        inv_spy = spy_vol.replace(0, np.nan).rdiv(1)
        inv_tlt = tlt_vol.replace(0, np.nan).rdiv(1)
        total = inv_spy + inv_tlt

        w_spy_h021 = (inv_spy / total).fillna(0.5)
        w_tlt_h021 = (inv_tlt / total).fillna(0.5)

        # H034 signal: positive rolling correlation = crisis contagion
        corr = spy_r.rolling(self.corr_lookback).corr(tlt_r).shift(1)
        crisis = corr > 0

        tlt_ok = prices["tlt_close"].notna()
        crisis = crisis & tlt_ok

        w_spy = w_spy_h021.where(~crisis, 1.0 - self.crisis_tlt)
        w_tlt = w_tlt_h021.where(~crisis, self.crisis_tlt)

        w_spy = w_spy.where(tlt_ok, 0.0)
        w_tlt = w_tlt.where(tlt_ok, 0.0)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class ReversionEntryRiskParity(MultiAssetStrategy):
    """H098: H021 with contrarian heavy-TLT entry on SPY drawdown events.

    Normal: H021 inverse-vol weights.
    Entry: SPY drawdown from peak exceeds dd_trigger →
           shift to crisis_tlt/crisis_spy for up to crisis_days bars.
    Exit early: SPY recovers recovery% from crisis trough.
    """

    name = "reversion_entry_risk_parity"

    def __init__(
        self,
        vol_lookback: int = 60,
        dd_trigger: float = 0.10,
        crisis_tlt: float = 0.80,
        crisis_days: int = 20,
        recovery: float = 0.05,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.dd_trigger = dd_trigger
        self.crisis_tlt = crisis_tlt
        self.crisis_days = crisis_days
        self.recovery = recovery

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy = prices["adj_close"]
        tlt = prices["tlt_close"]
        spy_r = spy.pct_change()
        tlt_r = tlt.pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback).std()
        tlt_vol = tlt_r.rolling(self.vol_lookback).std()
        inv_spy = spy_vol.replace(0, np.nan).rdiv(1)
        inv_tlt = tlt_vol.replace(0, np.nan).rdiv(1)
        total = inv_spy + inv_tlt
        w_spy_h021 = (inv_spy / total).fillna(0.5)
        w_tlt_h021 = (inv_tlt / total).fillna(0.5)

        spy_peak = spy.expanding(min_periods=1).max()
        spy_dd = (spy / spy_peak - 1.0).fillna(0.0)

        n = len(prices)
        w_spy_out = pd.Series(0.0, index=prices.index)
        w_tlt_out = pd.Series(0.0, index=prices.index)

        in_crisis = False
        days_left = 0
        crisis_trough = np.inf

        for i in range(n):
            tlt_avail = not pd.isna(tlt.iloc[i])
            dd_lag = spy_dd.iloc[i - 1] if i > 0 else 0.0
            spy_lag = spy.iloc[i - 1] if i > 0 else spy.iloc[i]

            if not tlt_avail:
                in_crisis = False
                days_left = 0
                continue

            if not in_crisis:
                if dd_lag < -self.dd_trigger:
                    in_crisis = True
                    days_left = self.crisis_days
                    crisis_trough = spy_lag
            else:
                if spy_lag < crisis_trough:
                    crisis_trough = spy_lag
                if spy_lag > crisis_trough * (1.0 + self.recovery):
                    in_crisis = False
                    days_left = 0
                else:
                    days_left -= 1
                    if days_left <= 0:
                        in_crisis = False

            if in_crisis:
                w_spy_out.iloc[i] = 1.0 - self.crisis_tlt
                w_tlt_out.iloc[i] = self.crisis_tlt
            else:
                w_spy_out.iloc[i] = w_spy_h021.iloc[i]
                w_tlt_out.iloc[i] = w_tlt_h021.iloc[i]

        return pd.DataFrame({"SPY": w_spy_out, "TLT": w_tlt_out}, index=prices.index)


# ---------------------------------------------------------------------------
# H201: SectorETFMomentum
# ---------------------------------------------------------------------------


class SectorETFMomentum(MultiAssetStrategy):
    """H201: Rotate into top-N equity ETFs by 6m momentum, risk-parity weighted.

    ETF universe: XLP, XLU, IVE, IVW, QQQ, IWM, VNQ.
    Signal: 126-day trailing total return (6 months), shifted 1 bar.
    Portfolio: top-N by 6m return, inverse-vol (vol_lookback-day) risk-parity.
    Falls back to SPY when fewer ETFs have sufficient history.

    Prices DataFrame must contain {symbol.lower()}_close columns for each ETF.
    """

    name = "sector_etf_momentum"

    _EQUITY_ETFS = ["XLP", "XLU", "IVE", "IVW", "QQQ", "IWM", "VNQ"]

    def __init__(
        self,
        mom_window: int = 126,
        vol_lookback: int = 60,
        top_n: int = 3,
    ) -> None:
        self.mom_window = mom_window
        self.vol_lookback = vol_lookback
        self.top_n = top_n

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        col_map = {a: f"{a.lower()}_close" for a in self._EQUITY_ETFS}
        available = {a: col for a, col in col_map.items() if col in prices.columns}

        if not available:
            return pd.DataFrame({"SPY": pd.Series(1.0, index=prices.index)})

        # 6m momentum and rolling vol for each available ETF
        mom_df = pd.DataFrame(
            {a: prices[col].pct_change(self.mom_window) for a, col in available.items()}
        )
        ret_df = pd.DataFrame({a: prices[col].pct_change() for a, col in available.items()})
        vol_df = ret_df.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std()
        # Fill NaN vol with 1.0 so inv-vol is equal-weight when history is short
        inv_vol_df = vol_df.replace(0, np.nan).rdiv(1).fillna(1.0)

        # Rank ETFs by 6m momentum each day (1 = best); NaN = insufficient history
        mom_rank = mom_df.rank(axis=1, ascending=False, na_option="keep")
        top_mask = (mom_rank <= self.top_n) & mom_rank.notna()

        # ETF weights: inv-vol within top-N, normalized to sum to 1
        raw = inv_vol_df.where(top_mask, 0.0)
        total = raw.sum(axis=1).replace(0, np.nan)
        etf_w = raw.div(total, axis=0).fillna(0.0)

        # SPY fallback when no ETF has enough history
        no_etf = top_mask.sum(axis=1) == 0
        etf_w["SPY"] = no_etf.astype(float)

        return etf_w.shift(1).fillna(0.0)


# ---------------------------------------------------------------------------
# H205: SectorMomentumRiskParity
# ---------------------------------------------------------------------------


class SectorMomentumRiskParity(MultiAssetStrategy):
    """H205: H021 variant — top equity ETF by 6m momentum paired with TLT.

    Selects the single best-performing equity ETF (XLP, XLU, IVE, IVW, QQQ,
    IWM, VNQ) by 126-day trailing return, then risk-parity weights it against
    TLT. Hypothesis: concentrating the equity leg in the strongest sector while
    keeping the TLT hedge should improve on H021's fixed SPY exposure.

    Falls back to SPY when no ETF has 6m history. Falls back to 100% winner
    equity ETF when TLT is unavailable.
    """

    name = "sector_momentum_risk_parity"

    _EQUITY_ETFS = ["XLP", "XLU", "IVE", "IVW", "QQQ", "IWM", "VNQ"]

    def __init__(
        self,
        mom_window: int = 126,
        vol_lookback: int = 60,
    ) -> None:
        self.mom_window = mom_window
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        col_map = {a: f"{a.lower()}_close" for a in self._EQUITY_ETFS}
        available = {a: col for a, col in col_map.items() if col in prices.columns}

        tlt_col = "tlt_close"
        has_tlt = tlt_col in prices.columns

        # 6m momentum for equity ETFs
        mom_df = pd.DataFrame(
            {a: prices[col].pct_change(self.mom_window) for a, col in available.items()}
        )

        # 60d vol for equity ETFs, TLT, SPY
        ret_df = pd.DataFrame({a: prices[col].pct_change() for a, col in available.items()})
        vol_etf = ret_df.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std()

        tlt_r = prices[tlt_col].pct_change() if has_tlt else pd.Series(np.nan, index=prices.index)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std()
        spy_r = prices["adj_close"].pct_change()
        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std()

        # Top-1 equity ETF by 6m momentum (all-NaN rows handled explicitly)
        winner_valid = mom_df.notna().any(axis=1)
        winner = pd.Series("SPY", index=prices.index, dtype=object)
        if winner_valid.any():
            winner.loc[winner_valid] = mom_df.loc[winner_valid].idxmax(axis=1, skipna=True)

        # Build output weight DataFrame covering all possible assets
        all_assets = list(
            dict.fromkeys(list(available.keys()) + (["TLT"] if has_tlt else []) + ["SPY"])
        )
        wdf = pd.DataFrame(0.0, index=prices.index, columns=all_assets)

        for date in prices.index:
            w_asset = winner.loc[date]

            # Equity leg vol
            if w_asset == "SPY":
                e_v = spy_vol.loc[date]
            elif w_asset in vol_etf.columns:
                e_v = vol_etf.loc[date, w_asset]
            else:
                e_v = np.nan
            inv_e = (1.0 / e_v) if (not np.isnan(e_v) and e_v > 0) else 1.0

            # TLT leg
            t_v = tlt_vol.loc[date] if has_tlt else np.nan
            tlt_ok = has_tlt and not np.isnan(prices[tlt_col].loc[date])

            if tlt_ok and not np.isnan(t_v) and t_v > 0:
                inv_t = 1.0 / t_v
                tot = inv_e + inv_t
                wdf.loc[date, w_asset] = inv_e / tot
                wdf.loc[date, "TLT"] = inv_t / tot
            else:
                wdf.loc[date, w_asset] = 1.0

        return wdf.shift(1).fillna(0.0)


# ---------------------------------------------------------------------------
# H301-H315: Set 4 — VRP, Funding Stress, and Macro Timing
# ---------------------------------------------------------------------------


def _h021_weights(
    spy_r: pd.Series, tlt_r: pd.Series, vol_lookback: int
) -> tuple[pd.Series, pd.Series]:
    """Shared H021 inverse-vol SPY/TLT weights."""
    spy_vol = spy_r.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1) * np.sqrt(252)
    tlt_vol = tlt_r.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1) * np.sqrt(252)
    inv_spy = 1.0 / spy_vol.replace(0, np.nan)
    inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
    total = inv_spy + inv_tlt
    return (inv_spy / total).fillna(0.0), (inv_tlt / total).fillna(0.0)


def _vrp_series(spy_r: pd.Series, vix: pd.Series) -> pd.Series:
    """VRP = annualized implied var (VIX^2) minus annualized realized var (21d)."""
    implied_var = (vix.shift(1) / 100.0) ** 2
    realized_var = spy_r.rolling(21, min_periods=10).var().shift(1) * 252
    return (implied_var - realized_var).fillna(0.0)


class SimpleVRPOverlay(MultiAssetStrategy):
    """H301: H021 with VRP z-score bounded exposure scaler.

    VRP = (VIX/100)^2 - realized_var_21d * 252  (both annualized).
    Z-score of VRP over vrp_window bars, clipped to [-2, 2].
    scale = 1.0 + vrp_sensitivity * vrp_z ∈ [0.5, 1.5] for vrp_sensitivity=0.25.
    Positive VRP → increase exposure; negative VRP → reduce.
    """

    name = "simple_vrp_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
        vrp_window: int = 252,
        vrp_sensitivity: float = 0.25,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vrp_window = vrp_window
        self.vrp_sensitivity = vrp_sensitivity

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_mean = vrp.rolling(self.vrp_window, min_periods=63).mean()
        vrp_std = vrp.rolling(self.vrp_window, min_periods=63).std().replace(0, np.nan)
        vrp_z = ((vrp - vrp_mean) / vrp_std).clip(-2.0, 2.0).fillna(0.0)

        scale = (1.0 + self.vrp_sensitivity * vrp_z).clip(0.5, 1.5)
        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class VRPRiskParityScaled(MultiAssetStrategy):
    """H302: H021 with 3-tier VRP percentile exposure multiplier.

    VRP > rolling 70th percentile → high_scale (1.25×).
    VRP < rolling 30th percentile → low_scale (0.75×).
    Middle tier → 1.0× (standard H021).
    """

    name = "vrp_risk_parity_scaled"

    def __init__(
        self,
        vol_lookback: int = 60,
        vrp_window: int = 252,
        high_pctile: float = 0.70,
        low_pctile: float = 0.30,
        high_scale: float = 1.25,
        low_scale: float = 0.75,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vrp_window = vrp_window
        self.high_pctile = high_pctile
        self.low_pctile = low_pctile
        self.high_scale = high_scale
        self.low_scale = low_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vrp = _vrp_series(spy_r, prices["vix_close"])
        hi = vrp.rolling(self.vrp_window, min_periods=63).quantile(self.high_pctile)
        lo = vrp.rolling(self.vrp_window, min_periods=63).quantile(self.low_pctile)

        scale_arr = np.where(vrp > hi, self.high_scale, np.where(vrp < lo, self.low_scale, 1.0))
        scale = pd.Series(np.where(hi.isna() | lo.isna(), 1.0, scale_arr), index=prices.index)
        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class VRPTrendConfirm(MultiAssetStrategy):
    """H303: H021 conditional on VRP + SPY 200d MA dual confirmation.

    Both favorable (VRP > 0 AND SPY > MA): full_scale (1.0).
    Exactly one favorable: partial_scale (0.85).
    Neither favorable: min_scale (0.60).
    """

    name = "vrp_trend_confirm"

    def __init__(
        self,
        vol_lookback: int = 60,
        ma_window: int = 200,
        full_scale: float = 1.0,
        partial_scale: float = 0.85,
        min_scale: float = 0.60,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.ma_window = ma_window
        self.full_scale = full_scale
        self.partial_scale = partial_scale
        self.min_scale = min_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_ok = (vrp > 0).fillna(False)

        spy_ma = prices["adj_close"].rolling(self.ma_window, min_periods=self.ma_window // 2).mean()
        trend_ok = (prices["adj_close"].shift(1) > spy_ma.shift(1)).fillna(False)

        n_ok = vrp_ok.astype(float) + trend_ok.astype(float)
        scale_arr = np.where(
            n_ok >= 2, self.full_scale, np.where(n_ok == 1, self.partial_scale, self.min_scale)
        )
        scale = pd.Series(scale_arr, index=prices.index)
        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class VRPVolTarget(MultiAssetStrategy):
    """H304: H086 vol-target with VRP z-score amplifier.

    vol_scale = min(1, vol_target / port_vol)  [H086 mechanism].
    vrp_factor = clip(1 + vrp_amplifier * vrp_z, 0.8, 1.2).
    final_scale = min(max_scale, vol_scale * vrp_factor).
    """

    name = "vrp_vol_target"

    def __init__(
        self,
        vol_lookback: int = 60,
        port_vol_lookback: int = 20,
        vol_target: float = 0.08,
        vrp_window: int = 252,
        vrp_amplifier: float = 0.15,
        max_scale: float = 1.25,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.port_vol_lookback = port_vol_lookback
        self.vol_target = vol_target
        self.vrp_window = vrp_window
        self.vrp_amplifier = vrp_amplifier
        self.max_scale = max_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        port_r = w_spy.shift(1) * spy_r + w_tlt.shift(1) * tlt_r
        port_vol = port_r.rolling(
            self.port_vol_lookback, min_periods=self.port_vol_lookback
        ).std().shift(1) * np.sqrt(252)
        vol_scale = (self.vol_target / port_vol).clip(upper=1.0)
        vol_scale = vol_scale.where(port_vol.notna() & (port_vol > 0), 1.0)

        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_mean = vrp.rolling(self.vrp_window, min_periods=63).mean()
        vrp_std = vrp.rolling(self.vrp_window, min_periods=63).std().replace(0, np.nan)
        vrp_z = ((vrp - vrp_mean) / vrp_std).clip(-2.0, 2.0).fillna(0.0)

        vrp_factor = (1.0 + self.vrp_amplifier * vrp_z).clip(0.8, 1.2)
        scale = (vol_scale * vrp_factor).clip(upper=self.max_scale)
        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class VRPSeasonalHybrid(MultiAssetStrategy):
    """H305: H078 seasonal framework + VRP overlay in Nov-Apr active window.

    May-Oct: 100% TLT (same as H078).
    Nov-Apr + VRP > 0: H021 at favorable_scale (1.0).
    Nov-Apr + VRP ≤ 0: H021 weights * unfavorable_scale (0.75); remainder cash.
    """

    name = "vrp_seasonal_hybrid"

    def __init__(
        self,
        vol_lookback: int = 60,
        unfavorable_scale: float = 0.75,
        winter_months: tuple = (11, 12, 1, 2, 3, 4),
    ) -> None:
        self.vol_lookback = vol_lookback
        self.unfavorable_scale = unfavorable_scale
        self.winter_months = frozenset(winter_months)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy_rp, w_tlt_rp = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_positive = (vrp > 0).fillna(False)

        in_season = pd.Series(prices.index.month.isin(self.winter_months), index=prices.index)
        tlt_avail = prices["tlt_close"].notna()

        # Start with H021 weights
        w_spy = w_spy_rp.copy()
        w_tlt = w_tlt_rp.copy()

        # May-Oct: 100% TLT (if available) or SPY
        summer = ~in_season
        w_spy = w_spy.where(~summer, 0.0)
        w_tlt = w_tlt.where(~summer, tlt_avail.astype(float))

        # Nov-Apr, VRP unfavorable: scale down
        winter_unfav = in_season & ~vrp_positive
        w_spy = w_spy.where(~winter_unfav, w_spy_rp * self.unfavorable_scale)
        w_tlt = w_tlt.where(~winter_unfav, w_tlt_rp * self.unfavorable_scale)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


# ---------------------------------------------------------------------------
# Batch 2: Funding Stress (H306-H310)
# ---------------------------------------------------------------------------


class FundingStressMonitor(MultiAssetStrategy):
    """H306: H021 scaled by 3-signal composite funding stress index.

    stress = mean(vix_pctile_252d, credit_stress, vol_accel)
      credit_stress = 1 if HYG 63d return < LQD 63d return, else 0 (0 if unavailable)
      vol_accel = 1 if SPY 21d vol > 1.5× SPY 63d vol, else 0
      vix_pctile = rolling 252d percentile of VIX level ∈ [0, 1]
    scale = max(min_scale, 1 - stress_sensitivity * stress)
    """

    name = "funding_stress_monitor"

    def __init__(
        self,
        vol_lookback: int = 60,
        stress_sensitivity: float = 0.50,
        min_scale: float = 0.40,
        vix_pctile_window: int = 252,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.stress_sensitivity = stress_sensitivity
        self.min_scale = min_scale
        self.vix_pctile_window = vix_pctile_window

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vix = prices["vix_close"].shift(1)
        vix_hi = vix.rolling(self.vix_pctile_window, min_periods=63).quantile(0.75)
        vix_lo = vix.rolling(self.vix_pctile_window, min_periods=63).quantile(0.25)
        vix_range = (vix_hi - vix_lo).replace(0, np.nan)
        vix_pctile = ((vix - vix_lo) / vix_range).clip(0.0, 1.0).fillna(0.5)

        spy_vol_21 = spy_r.rolling(21, min_periods=10).std().shift(1)
        spy_vol_63 = spy_r.rolling(63, min_periods=30).std().shift(1)
        vol_accel = (spy_vol_21 > spy_vol_63 * 1.5).fillna(False).astype(float)

        n_signals = 2.0
        credit_stress = pd.Series(0.0, index=prices.index)
        if "hyg_close" in prices.columns and "lqd_close" in prices.columns:
            hyg_ret = prices["hyg_close"].pct_change(63).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(63).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            credit_ok = hyg_avail & hyg_ret.notna() & lqd_ret.notna()
            credit_stress = (hyg_ret < lqd_ret).where(credit_ok, False).astype(float)
            n_signals = 3.0

        stress = (vix_pctile + credit_stress + vol_accel) / n_signals
        scale = (1.0 - self.stress_sensitivity * stress).clip(lower=self.min_scale)

        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class LiquiditySpiralProtection(MultiAssetStrategy):
    """H307: H021 with triple-confirm liquidity spiral crisis override.

    Crisis: VIX > vix_spike AND HYG 63d < LQD 63d AND SPY 21d vol > 2× 63d vol.
    Crisis mode: crisis_tlt% TLT / (1-crisis_tlt)% SPY.
    Exit: after resume_days consecutive non-crisis bars.
    Pre-HYG era: 2-of-2 (VIX + vol acceleration) trigger.
    """

    name = "liquidity_spiral_protection"

    def __init__(
        self,
        vol_lookback: int = 60,
        vix_spike: float = 35.0,
        crisis_tlt: float = 0.90,
        resume_days: int = 10,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vix_spike = vix_spike
        self.crisis_tlt = crisis_tlt
        self.resume_days = resume_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vix = prices["vix_close"].shift(1).fillna(0.0)
        spy_vol_21 = spy_r.rolling(21, min_periods=10).std().shift(1)
        spy_vol_63 = spy_r.rolling(63, min_periods=30).std().shift(1)
        vol_spike = (spy_vol_21 > spy_vol_63 * 2.0).fillna(False)

        has_credit = "hyg_close" in prices.columns and "lqd_close" in prices.columns
        if has_credit:
            hyg_ret = prices["hyg_close"].pct_change(63).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(63).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            credit_stress = (hyg_ret < lqd_ret).where(hyg_avail, False).fillna(False)
            crisis_flag = (vix > self.vix_spike) & credit_stress & vol_spike
            # Pre-credit era: use 2-of-2 VIX + vol
            crisis_flag = crisis_flag.where(hyg_avail, (vix > self.vix_spike) & vol_spike)
        else:
            crisis_flag = (vix > self.vix_spike) & vol_spike

        w_spy_out = w_spy.copy()
        w_tlt_out = w_tlt.copy()
        in_crisis = False
        calm_count = 0

        for i in range(len(prices)):
            if crisis_flag.iloc[i]:
                in_crisis = True
                calm_count = 0
            elif in_crisis:
                calm_count += 1
                if calm_count >= self.resume_days:
                    in_crisis = False
                    calm_count = 0
            if in_crisis:
                w_spy_out.iloc[i] = 1.0 - self.crisis_tlt
                w_tlt_out.iloc[i] = self.crisis_tlt

        return pd.DataFrame({"SPY": w_spy_out, "TLT": w_tlt_out}, index=prices.index)


class CreditRegimeRiskParity(MultiAssetStrategy):
    """H308: H021 with HYG vs LQD credit regime signal.

    HYG 63d return > LQD 63d return → credit risk-on → H021 full.
    LQD outperforming → credit risk-off → SPY weight * spy_scale; excess to TLT.
    Pre-HYG era (pre-2007): H021 unmodified.
    """

    name = "credit_regime_risk_parity"

    def __init__(
        self,
        vol_lookback: int = 60,
        credit_lookback: int = 63,
        spy_scale: float = 0.60,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.credit_lookback = credit_lookback
        self.spy_scale = spy_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        if "hyg_close" not in prices.columns or "lqd_close" not in prices.columns:
            return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)

        hyg_ret = prices["hyg_close"].pct_change(self.credit_lookback).shift(1)
        lqd_ret = prices["lqd_close"].pct_change(self.credit_lookback).shift(1)
        hyg_avail = prices["hyg_close"].notna()

        credit_risk_off = (hyg_ret < lqd_ret) & hyg_avail & hyg_ret.notna()

        freed = w_spy * (1.0 - self.spy_scale)
        new_spy = w_spy.where(~credit_risk_off, w_spy * self.spy_scale)
        new_tlt = w_tlt.where(~credit_risk_off, (w_tlt + freed).clip(upper=1.0))

        return pd.DataFrame({"SPY": new_spy, "TLT": new_tlt}, index=prices.index)


class MultiSignalStressAdjusted(MultiAssetStrategy):
    """H309: H021 with 5-signal stress monitor and tiered exposure.

    Signals: VIX>30, SPY 5% DD from 20d high, SHY>TLT 63d, HYG<LQD 63d, SPY vol accel.
    Exposure tiers: 0=100%, 1=85%, 2=70%, 3=55%, 4+=40%.
    Pre-HYG era (4 signals): 0=100%, 1=87%, 2=73%, 3+=40%.
    """

    name = "multi_signal_stress_adjusted"

    def __init__(
        self,
        vol_lookback: int = 60,
        vix_threshold: float = 30.0,
        dd_threshold: float = 0.05,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vix_threshold = vix_threshold
        self.dd_threshold = dd_threshold

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vix = prices["vix_close"].shift(1).fillna(0.0)
        s1 = (vix > self.vix_threshold).astype(float)

        spy_px = prices["adj_close"].shift(1)
        rolling_hi = spy_px.rolling(20, min_periods=5).max()
        dd = (spy_px / rolling_hi - 1.0).fillna(0.0)
        s2 = (dd < -self.dd_threshold).astype(float)

        has_shy = "shy_close" in prices.columns
        s3 = pd.Series(0.0, index=prices.index)
        if has_shy:
            shy_ret = prices["shy_close"].pct_change(63).shift(1)
            tlt_ret_63 = prices["tlt_close"].pct_change(63).shift(1)
            s3 = (shy_ret > tlt_ret_63).fillna(False).astype(float)

        has_credit = "hyg_close" in prices.columns and "lqd_close" in prices.columns
        s4 = pd.Series(0.0, index=prices.index)
        if has_credit:
            hyg_ret = prices["hyg_close"].pct_change(63).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(63).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            s4 = ((hyg_ret < lqd_ret) & hyg_avail).fillna(False).astype(float)

        spy_vol_21 = spy_r.rolling(21, min_periods=10).std().shift(1)
        spy_vol_63 = spy_r.rolling(63, min_periods=30).std().shift(1)
        s5 = (spy_vol_21 > spy_vol_63 * 1.5).fillna(False).astype(float)

        total = s1 + s2 + s3 + s4 + s5
        tiers_5 = [1.0, 0.85, 0.70, 0.55, 0.40]
        scale_arr = np.select(
            [total == 0, total == 1, total == 2, total == 3],
            tiers_5[:4],
            default=tiers_5[4],
        )
        scale = pd.Series(scale_arr, index=prices.index)
        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class FundingStressVolTarget(MultiAssetStrategy):
    """H310: H086 vol-target compounded with binary funding stress reduction.

    vol_scale = min(1, vol_target / port_vol)  [H086].
    stress_flag = VIX > 30 AND HYG 63d < LQD 63d (fallback: VIX > 30 only).
    final_scale = vol_scale * (stress_reduction if stress_flag else 1.0).
    """

    name = "funding_stress_vol_target"

    def __init__(
        self,
        vol_lookback: int = 60,
        port_vol_lookback: int = 20,
        vol_target: float = 0.08,
        vix_threshold: float = 30.0,
        stress_reduction: float = 0.65,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.port_vol_lookback = port_vol_lookback
        self.vol_target = vol_target
        self.vix_threshold = vix_threshold
        self.stress_reduction = stress_reduction

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        port_r = w_spy.shift(1) * spy_r + w_tlt.shift(1) * tlt_r
        port_vol = port_r.rolling(
            self.port_vol_lookback, min_periods=self.port_vol_lookback
        ).std().shift(1) * np.sqrt(252)
        vol_scale = (self.vol_target / port_vol).clip(upper=1.0)
        vol_scale = vol_scale.where(port_vol.notna() & (port_vol > 0), 1.0)

        vix = prices["vix_close"].shift(1).fillna(0.0)
        vix_stress = vix > self.vix_threshold

        if "hyg_close" in prices.columns and "lqd_close" in prices.columns:
            hyg_ret = prices["hyg_close"].pct_change(63).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(63).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            credit_off = (hyg_ret < lqd_ret) & hyg_avail & hyg_ret.notna()
            stress_flag = (vix_stress & credit_off).where(hyg_avail, vix_stress)
        else:
            stress_flag = vix_stress

        stress_scale = stress_flag.map({True: self.stress_reduction, False: 1.0})
        scale = vol_scale * stress_scale

        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


# ---------------------------------------------------------------------------
# Batch 3: Macro Timing (H311-H315)
# ---------------------------------------------------------------------------


class TermSpreadSPYTimer(MultiAssetStrategy):
    """H311: H021 with TLT vs SHY term spread SPY overlay.

    Proxy term spread: TLT lookback-day return minus SHY lookback-day return.
    Positive (normal curve) → H021 full.
    Negative (inverted curve proxy) → SPY weight * spy_scale; freed weight to TLT.
    """

    name = "term_spread_spy_timer"

    def __init__(
        self,
        vol_lookback: int = 60,
        spread_lookback: int = 252,
        spy_scale: float = 0.50,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.spread_lookback = spread_lookback
        self.spy_scale = spy_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        tlt_ret = prices["tlt_close"].pct_change(self.spread_lookback).shift(1)
        shy_ret = prices["shy_close"].pct_change(self.spread_lookback).shift(1)
        inverted = (shy_ret > tlt_ret) & tlt_ret.notna() & shy_ret.notna()

        freed = w_spy * (1.0 - self.spy_scale)
        new_spy = w_spy.where(~inverted, w_spy * self.spy_scale)
        new_tlt = w_tlt.where(~inverted, (w_tlt + freed).clip(upper=1.0))
        return pd.DataFrame({"SPY": new_spy, "TLT": new_tlt}, index=prices.index)


class DefaultSpreadSPYTimer(MultiAssetStrategy):
    """H312: H021 with HYG vs LQD default spread signal.

    HYG 63d return > LQD 63d return → credit risk-on → H021 full.
    LQD outperforming → credit risk-off → SPY weight * spy_scale; freed to TLT.
    Pre-HYG era: H021 unmodified.
    """

    name = "default_spread_spy_timer"

    def __init__(
        self,
        vol_lookback: int = 60,
        credit_lookback: int = 63,
        spy_scale: float = 0.60,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.credit_lookback = credit_lookback
        self.spy_scale = spy_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        if "hyg_close" not in prices.columns or "lqd_close" not in prices.columns:
            return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)

        hyg_ret = prices["hyg_close"].pct_change(self.credit_lookback).shift(1)
        lqd_ret = prices["lqd_close"].pct_change(self.credit_lookback).shift(1)
        hyg_avail = prices["hyg_close"].notna()
        credit_off = (hyg_ret < lqd_ret) & hyg_avail & hyg_ret.notna()

        freed = w_spy * (1.0 - self.spy_scale)
        new_spy = w_spy.where(~credit_off, w_spy * self.spy_scale)
        new_tlt = w_tlt.where(~credit_off, (w_tlt + freed).clip(upper=1.0))
        return pd.DataFrame({"SPY": new_spy, "TLT": new_tlt}, index=prices.index)


class EarningsYieldOverlay(MultiAssetStrategy):
    """H313: H021 with SPY short-vs-long-run CAGR differential valuation signal.

    Signal: SPY 252d return minus SPY 1260d annualized return (5yr CAGR proxy).
    When short-run excess is high (market extended): reduce SPY weight by spy_cut.
    When short-run below long-run (market lagging): neutral to H021.
    Uses z-score over expanding window (5yr min) to calibrate thresholds.
    """

    name = "earnings_yield_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
        short_window: int = 252,
        long_window: int = 1260,
        zscore_min: int = 1260,
        spy_cut: float = 0.15,
        zscore_threshold: float = 1.0,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.short_window = short_window
        self.long_window = long_window
        self.zscore_min = zscore_min
        self.spy_cut = spy_cut
        self.zscore_threshold = zscore_threshold

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        spy_px = prices["adj_close"]
        ret_1yr = spy_px.pct_change(self.short_window).shift(1)
        ret_5yr_ann = (
            (spy_px / spy_px.shift(self.long_window)) ** (252.0 / self.long_window) - 1.0
        ).shift(1)
        excess = (ret_1yr - ret_5yr_ann).fillna(0.0)

        ex_mean = excess.expanding(min_periods=self.zscore_min).mean()
        ex_std = excess.expanding(min_periods=self.zscore_min).std().replace(0, np.nan)
        ex_z = ((excess - ex_mean) / ex_std).fillna(0.0)

        extended = ex_z > self.zscore_threshold
        freed = w_spy * self.spy_cut
        new_spy = w_spy.where(~extended, w_spy * (1.0 - self.spy_cut))
        new_tlt = w_tlt.where(~extended, (w_tlt + freed).clip(upper=1.0))
        return pd.DataFrame({"SPY": new_spy, "TLT": new_tlt}, index=prices.index)


class MacroCombinationOverlay(MultiAssetStrategy):
    """H314: Ensemble of 3 macro signals for H021 exposure adjustment.

    Signal scores (+1 bullish, -1 bearish):
      VRP signal: VRP > 0 = +1, ≤ 0 = -1.
      Term spread: TLT 252d > SHY 252d = +1, else -1.
      Credit spread: HYG 63d > LQD 63d = +1, else -1 (0 if unavailable).
    Sum ≥ 2: 1.20× scale; sum = 1: 1.05×; sum = 0: 0.90×; sum ≤ -1: 0.75×.
    """

    name = "macro_combination_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
    ) -> None:
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vrp = _vrp_series(spy_r, prices["vix_close"])
        s_vrp = np.where(vrp > 0, 1.0, -1.0)

        tlt_ret = prices["tlt_close"].pct_change(252).shift(1)
        shy_ret = prices["shy_close"].pct_change(252).shift(1)
        s_term = np.where(
            tlt_ret.notna() & shy_ret.notna(),
            np.where(tlt_ret > shy_ret, 1.0, -1.0),
            0.0,
        )

        s_credit = pd.Series(0.0, index=prices.index)
        if "hyg_close" in prices.columns and "lqd_close" in prices.columns:
            hyg_ret = prices["hyg_close"].pct_change(63).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(63).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            s_credit = pd.Series(
                np.where(
                    hyg_avail & hyg_ret.notna(),
                    np.where(hyg_ret > lqd_ret, 1.0, -1.0),
                    0.0,
                ),
                index=prices.index,
            )

        total = (
            pd.Series(s_vrp, index=prices.index) + pd.Series(s_term, index=prices.index) + s_credit
        )

        scale_arr = np.select(
            [total >= 2, total == 1, total == 0],
            [1.20, 1.05, 0.90],
            default=0.75,
        )
        scale = pd.Series(scale_arr, index=prices.index)
        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class ConstrainedMacroOverlay(MultiAssetStrategy):
    """H315: H314 macro combination with Campbell-Thompson constraints.

    Applies three economic constraints to H314's macro signal:
      Floor: SPY weight ≥ spy_floor * H021 SPY weight (never fully defensive).
      Cap: SPY weight ≤ spy_cap * H021 SPY weight (no excess leverage).
      Sign constraint: leverage (scale > 1) only when all 3 signals bullish or VRP very positive.
    """

    name = "constrained_macro_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
        spy_floor: float = 0.50,
        spy_cap: float = 1.30,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.spy_floor = spy_floor
        self.spy_cap = spy_cap

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vrp = _vrp_series(spy_r, prices["vix_close"])
        s_vrp = np.where(vrp > 0, 1.0, -1.0)

        tlt_ret = prices["tlt_close"].pct_change(252).shift(1)
        shy_ret = prices["shy_close"].pct_change(252).shift(1)
        s_term = np.where(
            tlt_ret.notna() & shy_ret.notna(),
            np.where(tlt_ret > shy_ret, 1.0, -1.0),
            0.0,
        )

        s_credit = pd.Series(0.0, index=prices.index)
        if "hyg_close" in prices.columns and "lqd_close" in prices.columns:
            hyg_ret = prices["hyg_close"].pct_change(63).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(63).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            s_credit = pd.Series(
                np.where(
                    hyg_avail & hyg_ret.notna(),
                    np.where(hyg_ret > lqd_ret, 1.0, -1.0),
                    0.0,
                ),
                index=prices.index,
            )

        total = (
            pd.Series(s_vrp, index=prices.index) + pd.Series(s_term, index=prices.index) + s_credit
        )

        # Unconstrained H314 scale
        raw_scale = np.select(
            [total >= 2, total == 1, total == 0],
            [1.20, 1.05, 0.90],
            default=0.75,
        )
        raw_scale = pd.Series(raw_scale, index=prices.index)

        # Sign constraint: only leverage if all 3 signals bullish (total ≥ 3)
        constrained_scale = raw_scale.where(total >= 3, raw_scale.clip(upper=1.0))

        # SPY floor and cap
        spy_scaled = (w_spy * constrained_scale).clip(
            lower=w_spy * self.spy_floor,
            upper=w_spy * self.spy_cap,
        )
        tlt_scaled = w_tlt * constrained_scale

        return pd.DataFrame({"SPY": spy_scaled, "TLT": tlt_scaled}, index=prices.index)


# -- Set 4 Batch 4: Slow-Moving Macro State Variables (H316-H320) ---------------


class QuarterlyMacroStateOverlay(MultiAssetStrategy):
    """H316: H021 with quarterly-resampled composite macro signal.

    Three signals scored +1 / -1 / 0:
      Term spread : TLT 252d return vs SHY 252d return (always available).
      Equity momentum : SPY 252d return > 0 -> +1, else -1 (always available).
      Growth/value : IVW 252d vs IVE 252d return spread (post-2000 only -> 0 before).

    Composite sum in [-3, +3] resampled to last business day of each quarter,
    then forward-filled daily so macro state updates at most 4x per year.

    Sum >= 2 : H021 x high_scale
    Sum in [0, 1] : H021 x 1.00
    Sum <= -1 : H021 x low_scale
    """

    name = "quarterly_macro_state_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
        high_scale: float = 1.20,
        low_scale: float = 0.80,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.high_scale = high_scale
        self.low_scale = low_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # --- Term spread signal ---
        tlt_ret = prices["tlt_close"].pct_change(252).shift(1)
        shy_ret = prices["shy_close"].pct_change(252).shift(1)
        s_term = pd.Series(
            np.where(
                tlt_ret.notna() & shy_ret.notna(),
                np.where(tlt_ret > shy_ret, 1.0, -1.0),
                0.0,
            ),
            index=prices.index,
        )

        # --- Equity momentum signal ---
        spy_ret252 = prices["adj_close"].pct_change(252).shift(1)
        s_equity = pd.Series(
            np.where(spy_ret252.notna(), np.where(spy_ret252 > 0, 1.0, -1.0), 0.0),
            index=prices.index,
        )

        # --- Growth/value signal (post-2000 IVW/IVE only) ---
        s_gv = pd.Series(0.0, index=prices.index)
        if "ivw_close" in prices.columns and "ive_close" in prices.columns:
            ivw_ret = prices["ivw_close"].pct_change(252).shift(1)
            ive_ret = prices["ive_close"].pct_change(252).shift(1)
            both_avail = prices["ivw_close"].notna() & prices["ive_close"].notna()
            s_gv = pd.Series(
                np.where(
                    both_avail & ivw_ret.notna() & ive_ret.notna(),
                    np.where(ivw_ret > ive_ret, 1.0, -1.0),
                    0.0,
                ),
                index=prices.index,
            )

        composite_daily = s_term + s_equity + s_gv

        # Resample to quarter-end, then forward-fill so signal is slow-moving
        composite_q = composite_daily.resample("QE").last()
        composite_ff = composite_q.reindex(prices.index).ffill()

        scale = pd.Series(1.0, index=prices.index)
        scale = scale.where(composite_ff < 2, self.high_scale)
        scale = scale.where(composite_ff > -1, self.low_scale)
        # middle band [0, 1] keeps scale at 1.0 (already initialised)
        scale = scale.where(~((composite_ff >= 0) & (composite_ff < 2)), 1.0)
        # apply: sum >= 2 -> high_scale, sum <= -1 -> low_scale, else 1.0
        scale = np.select(
            [composite_ff >= 2, composite_ff <= -1],
            [self.high_scale, self.low_scale],
            default=1.0,
        )
        scale = pd.Series(scale, index=prices.index)

        return pd.DataFrame(
            {"SPY": (w_spy * scale).clip(upper=1.0), "TLT": (w_tlt * scale).clip(upper=1.0)},
            index=prices.index,
        )


class ValuationSpreadMarketState(MultiAssetStrategy):
    """H317: H021 scaled by IVW vs IVE 252-day return spread.

    IVW (Growth ETF) outperforming IVE (Value ETF) -> growth/risk-on regime
    -> H021 x growth_scale.
    IVE outperforming IVW -> value/defensive regime -> H021 x value_scale.
    Pre-2000 (no IVW/IVE data): pure H021 (scale = 1.0).
    """

    name = "valuation_spread_market_state"

    def __init__(
        self,
        vol_lookback: int = 60,
        ivw_lookback: int = 252,
        growth_scale: float = 1.10,
        value_scale: float = 0.90,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.ivw_lookback = ivw_lookback
        self.growth_scale = growth_scale
        self.value_scale = value_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        scale = pd.Series(1.0, index=prices.index)

        if "ivw_close" in prices.columns and "ive_close" in prices.columns:
            ivw_ret = prices["ivw_close"].pct_change(self.ivw_lookback).shift(1)
            ive_ret = prices["ive_close"].pct_change(self.ivw_lookback).shift(1)
            both_avail = (
                prices["ivw_close"].notna()
                & prices["ive_close"].notna()
                & ivw_ret.notna()
                & ive_ret.notna()
            )
            growth_on = both_avail & (ivw_ret > ive_ret)
            value_on = both_avail & (ive_ret >= ivw_ret)
            scale = pd.Series(
                np.select(
                    [growth_on, value_on],
                    [self.growth_scale, self.value_scale],
                    default=1.0,
                ),
                index=prices.index,
            )

        return pd.DataFrame(
            {"SPY": (w_spy * scale).clip(upper=1.0), "TLT": (w_tlt * scale).clip(upper=1.0)},
            index=prices.index,
        )


class RegimeClassifier(MultiAssetStrategy):
    """H318: H021 with 4-state macro regime based on SPY momentum and VIX level.

    States (all signals lagged 1 bar):
      Growth   : SPY 252d annualized > 0 AND VIX < vix_low  -> 1.15x
      Recovery : SPY 252d annualized > 0 AND vix_low <= VIX <= vix_high -> 1.00x
      Stress   : SPY 252d annualized < 0 OR VIX > vix_high  -> 0.70x
      Caution  : all else                                    -> 0.90x
    """

    name = "regime_classifier"

    def __init__(
        self,
        vol_lookback: int = 60,
        vix_low: float = 20.0,
        vix_high: float = 30.0,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vix_low = vix_low
        self.vix_high = vix_high

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        spy_ann = prices["adj_close"].pct_change(252).shift(1)
        vix = prices["vix_close"].shift(1)

        spy_up = spy_ann > 0
        vix_low_mask = vix < self.vix_low
        vix_mid_mask = (vix >= self.vix_low) & (vix <= self.vix_high)
        vix_high_mask = vix > self.vix_high

        growth = spy_up & vix_low_mask
        recovery = spy_up & vix_mid_mask
        stress = (~spy_up) | vix_high_mask
        # caution is the residual (spy_up but vix above mid -- already covered by stress if vix_high)
        # Actually: caution = spy_up AND vix in mid range not already recovery, OR spy_down AND low vix
        # Simpler: use np.select priority order
        scale = pd.Series(
            np.select(
                [stress, growth, recovery],
                [0.70, 1.15, 1.00],
                default=0.90,  # Caution
            ),
            index=prices.index,
        )

        # Fill NaN vix/spy periods with neutral
        no_data = spy_ann.isna() | vix.isna()
        scale = scale.where(~no_data, 1.0)

        return pd.DataFrame(
            {"SPY": (w_spy * scale).clip(upper=1.0), "TLT": (w_tlt * scale).clip(upper=1.0)},
            index=prices.index,
        )


class RiskBudgetExpansion(MultiAssetStrategy):
    """H319: H086-style vol-target with macro-state-dependent target level.

    Favorable (VRP > 0 AND SPY 252d > 0) : vol_target = vol_target_favorable (10%).
    Unfavorable (VRP < 0 OR VIX > 60th rolling percentile) : vol_target = vol_target_unfavorable (6%).
    Baseline : vol_target = vol_target_base (8%).

    scale = min(1.0, vol_target / portfolio_vol)
    portfolio_vol = rolling 21d std of H021 daily returns * sqrt(252).
    """

    name = "risk_budget_expansion"

    def __init__(
        self,
        vol_lookback: int = 60,
        vol_target_favorable: float = 0.10,
        vol_target_base: float = 0.08,
        vol_target_unfavorable: float = 0.06,
        vix_pct_window: int = 252,
        vix_pct_threshold: float = 0.60,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vol_target_favorable = vol_target_favorable
        self.vol_target_base = vol_target_base
        self.vol_target_unfavorable = vol_target_unfavorable
        self.vix_pct_window = vix_pct_window
        self.vix_pct_threshold = vix_pct_threshold

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # H021 portfolio daily returns
        port_ret = w_spy.shift(1) * spy_r + w_tlt.shift(1) * tlt_r
        port_vol = port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
        port_vol = port_vol.shift(1).replace(0, np.nan)

        # VRP signal
        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_pos = vrp > 0

        # SPY 252d momentum
        spy_ann = prices["adj_close"].pct_change(252).shift(1)
        spy_up = spy_ann > 0

        # VIX 60th percentile (rolling)
        vix_shifted = prices["vix_close"].shift(1)
        vix_pct60 = vix_shifted.rolling(self.vix_pct_window, min_periods=63).quantile(
            self.vix_pct_threshold
        )
        vix_elevated = vix_shifted > vix_pct60

        favorable = vrp_pos & spy_up
        unfavorable = (~vrp_pos) | vix_elevated

        vol_target = pd.Series(
            np.select(
                [favorable, unfavorable],
                [self.vol_target_favorable, self.vol_target_unfavorable],
                default=self.vol_target_base,
            ),
            index=prices.index,
        )

        scale = (vol_target / port_vol).clip(upper=1.0).fillna(1.0)

        return pd.DataFrame(
            {"SPY": w_spy * scale, "TLT": w_tlt * scale},
            index=prices.index,
        )


class DefensiveMacroStateMachine(MultiAssetStrategy):
    """H320: 3-state Risk-On / Neutral / Risk-Off machine with VIX hysteresis.

    Risk-On  : VRP > 0 AND SPY 252d > 0 AND SPY 63d > TLT 63d -> H021 x risk_on_scale
    Risk-Off : VIX > vix_threshold OR SPY 252d < 0 (hysteresis: calm_days consecutive
               non-triggering days required to exit)
    Neutral  : all else -> H021 x 1.0

    In Risk-Off: SPY weight = H021_SPY x risk_off_spy_scale; freed weight -> TLT.
    """

    name = "defensive_macro_state_machine"

    def __init__(
        self,
        vol_lookback: int = 60,
        vix_threshold: float = 28.0,
        calm_days: int = 5,
        risk_on_scale: float = 1.15,
        risk_off_spy_scale: float = 0.65,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vix_threshold = vix_threshold
        self.calm_days = calm_days
        self.risk_on_scale = risk_on_scale
        self.risk_off_spy_scale = risk_off_spy_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vix = prices["vix_close"].shift(1)
        spy_ann = prices["adj_close"].pct_change(252).shift(1)
        spy_63 = prices["adj_close"].pct_change(63).shift(1)
        tlt_63 = prices["tlt_close"].pct_change(63).shift(1)

        vrp = _vrp_series(spy_r, prices["vix_close"])

        risk_off_raw = (vix > self.vix_threshold) | (spy_ann < 0)
        risk_on_raw = (vrp > 0) & (spy_ann > 0) & (spy_63 > tlt_63)

        # Apply hysteresis to Risk-Off: stay in Risk-Off until calm_days consecutive
        # non-triggering days
        n = len(prices)
        state = np.zeros(n, dtype=np.int8)  # 0=Neutral, 1=Risk-On, -1=Risk-Off
        calm_count = 0
        for i in range(n):
            ro = bool(risk_off_raw.iloc[i]) if not pd.isna(risk_off_raw.iloc[i]) else False
            ron = bool(risk_on_raw.iloc[i]) if not pd.isna(risk_on_raw.iloc[i]) else False
            prev = state[i - 1] if i > 0 else 0

            if prev == -1:
                # Already in Risk-Off -- need calm_days to exit
                if ro:
                    calm_count = 0
                    state[i] = -1
                else:
                    calm_count += 1
                    if calm_count >= self.calm_days:
                        state[i] = 0
                        calm_count = 0
                    else:
                        state[i] = -1
            elif ro:
                state[i] = -1
                calm_count = 0
            elif ron:
                state[i] = 1
                calm_count = 0
            else:
                state[i] = 0
                calm_count = 0

        state_s = pd.Series(state, index=prices.index)

        # Build weights per state
        # Risk-Off: SPY reduced, freed weight to TLT
        freed = w_spy * (1.0 - self.risk_off_spy_scale)
        spy_out = pd.Series(
            np.select(
                [state_s == -1, state_s == 1],
                [w_spy * self.risk_off_spy_scale, (w_spy * self.risk_on_scale).clip(upper=1.0)],
                default=w_spy,
            ),
            index=prices.index,
        )
        tlt_out = pd.Series(
            np.select(
                [state_s == -1, state_s == 1],
                [(w_tlt + freed).clip(upper=1.0), (w_tlt * self.risk_on_scale).clip(upper=1.0)],
                default=w_tlt,
            ),
            index=prices.index,
        )

        return pd.DataFrame({"SPY": spy_out, "TLT": tlt_out}, index=prices.index)


# -- Set 4 Batch 6: Vol Regime Refinements (H326-H330) --------------------------


class DualFrequencyVolTarget(MultiAssetStrategy):
    """H326: Vol-target using max(21d, 63d) portfolio vol — more conservative than H086."""

    name = "dual_frequency_vol_target"

    def __init__(
        self,
        vol_target: float = 0.08,
        short_window: int = 21,
        long_window: int = 63,
        vol_lookback: int = 60,
    ) -> None:
        self.vol_target = vol_target
        self.short_window = short_window
        self.long_window = long_window
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        port_r = w_spy * spy_r + w_tlt * tlt_r
        vol_short = port_r.rolling(self.short_window, min_periods=10).std().shift(1) * np.sqrt(252)
        vol_long = port_r.rolling(self.long_window, min_periods=20).std().shift(1) * np.sqrt(252)
        vol_conservative = vol_short.combine(vol_long, np.maximum).replace(0, np.nan)
        scale = (self.vol_target / vol_conservative).clip(upper=1.0).fillna(1.0)

        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class VolAccelerationDetector(MultiAssetStrategy):
    """H327: H086 vol-target with penalty when vol accelerates, bonus when decelerating."""

    name = "vol_acceleration_detector"

    def __init__(
        self,
        vol_target: float = 0.08,
        vol_lookback: int = 60,
        accel_threshold: float = 1.5,
        decel_threshold: float = 0.75,
        accel_penalty: float = 0.75,
        decel_bonus: float = 1.15,
    ) -> None:
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.accel_threshold = accel_threshold
        self.decel_threshold = decel_threshold
        self.accel_penalty = accel_penalty
        self.decel_bonus = decel_bonus

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        port_r = w_spy * spy_r + w_tlt * tlt_r
        vol_21 = port_r.rolling(21, min_periods=10).std().shift(1) * np.sqrt(252)
        vol_63 = port_r.rolling(63, min_periods=20).std().shift(1) * np.sqrt(252)

        # H086 base scale
        vol_scale = (self.vol_target / vol_21.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)

        # Acceleration ratio; guard against zero denominator
        vol_ratio = (vol_21 / vol_63.replace(0, np.nan)).fillna(1.0)
        penalty = pd.Series(
            np.select(
                [vol_ratio > self.accel_threshold, vol_ratio < self.decel_threshold],
                [self.accel_penalty, self.decel_bonus],
                default=1.0,
            ),
            index=prices.index,
        )

        scale = (vol_scale * penalty).clip(upper=1.0)
        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class VolPercentileExposure(MultiAssetStrategy):
    """H328: 3-tier exposure based on 252d rolling percentile of 21d portfolio vol."""

    name = "vol_percentile_exposure"

    def __init__(
        self,
        vol_target: float = 0.08,
        vol_window: int = 21,
        pctile_window: int = 252,
        low_threshold: float = 0.33,
        high_threshold: float = 0.67,
        low_scale: float = 1.20,
        vol_lookback: int = 60,
    ) -> None:
        self.vol_target = vol_target
        self.vol_window = vol_window
        self.pctile_window = pctile_window
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.low_scale = low_scale
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        port_r = w_spy * spy_r + w_tlt * tlt_r
        vol_21 = port_r.rolling(self.vol_window, min_periods=10).std().shift(1) * np.sqrt(252)

        # Rolling percentile rank of current vol
        def _rolling_pctile(s: pd.Series, window: int) -> pd.Series:
            return s.rolling(window, min_periods=window // 2).apply(
                lambda x: float(np.sum(x[:-1] <= x[-1])) / max(len(x) - 1, 1),
                raw=True,
            )

        vol_pctile = _rolling_pctile(vol_21, self.pctile_window)

        vol_target_high = (
            (self.vol_target / vol_21.replace(0, np.nan)).clip(upper=0.60).fillna(0.60)
        )
        scale = pd.Series(
            np.select(
                [vol_pctile < self.low_threshold, vol_pctile > self.high_threshold],
                [self.low_scale, vol_target_high],
                default=1.0,
            ),
            index=prices.index,
        )

        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class VolRegimeRiskParity(MultiAssetStrategy):
    """H329: 3-state vol regime — low vol allows 1.20x; high vol applies tighter 7% target."""

    name = "vol_regime_risk_parity"

    def __init__(
        self,
        vol_window: int = 21,
        regime_window: int = 252,
        low_mult: float = 0.70,
        high_mult: float = 1.30,
        vol_target_high: float = 0.07,
        low_scale: float = 1.20,
        vol_lookback: int = 60,
    ) -> None:
        self.vol_window = vol_window
        self.regime_window = regime_window
        self.low_mult = low_mult
        self.high_mult = high_mult
        self.vol_target_high = vol_target_high
        self.low_scale = low_scale
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        port_r = w_spy * spy_r + w_tlt * tlt_r
        vol_21 = port_r.rolling(self.vol_window, min_periods=10).std().shift(1) * np.sqrt(252)
        vol_mean = vol_21.rolling(self.regime_window, min_periods=self.regime_window // 2).mean()

        is_low = vol_21 < self.low_mult * vol_mean
        is_high = vol_21 > self.high_mult * vol_mean

        high_scale = (self.vol_target_high / vol_21.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
        scale = pd.Series(
            np.select(
                [is_low, is_high],
                [self.low_scale, high_scale],
                default=1.0,
            ),
            index=prices.index,
        )

        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class RatchetVolTarget(MultiAssetStrategy):
    """H330: min(H086 vol-scale, H087 ratchet drawdown-scale) — most conservative wins."""

    name = "ratchet_vol_target"

    def __init__(
        self,
        vol_target: float = 0.08,
        drawdown_sensitivity: float = 3.0,
        vol_lookback: int = 60,
    ) -> None:
        self.vol_target = vol_target
        self.drawdown_sensitivity = drawdown_sensitivity
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        port_r = w_spy * spy_r + w_tlt * tlt_r

        # Vol scale (H086 mechanism)
        port_vol = port_r.rolling(21, min_periods=10).std().shift(1) * np.sqrt(252)
        vol_scale = (self.vol_target / port_vol.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)

        # Ratchet drawdown scale (H087 mechanism)
        cum = (1 + port_r).cumprod()
        peak = cum.cummax()
        dd_depth = ((peak - cum) / peak.replace(0, np.nan)).fillna(0.0)
        dd_scale = (1.0 - dd_depth * self.drawdown_sensitivity).clip(lower=0.10)

        # Take the more conservative of the two
        final_scale = np.minimum(vol_scale, dd_scale)

        return pd.DataFrame(
            {"SPY": w_spy * final_scale, "TLT": w_tlt * final_scale}, index=prices.index
        )


# ---------------------------------------------------------------------------
# Set 4 Batch 7: Calendar + VRP Composites (H331-H335)
# ---------------------------------------------------------------------------


class MacroConfirmedSeasonal(MultiAssetStrategy):
    """H331: SeasonalRiskParity (H078) + macro gate in Nov-Apr window.

    May-Oct: 100% TLT (same as H078).
    Nov-Apr:
      if VRP > 0 OR TLT 252d return > SHY 252d return → H021 weights
      else → 60% SPY / 40% TLT (reduced but not fully defensive)

    The macro gate prevents committing to full risk-parity exposure during
    winters where both VRP is negative AND the yield curve is inverted —
    i.e., the market is not compensating for variance risk AND bond carry
    is not signaling risk-on. In those conditions, a conservative 60/40
    blend is used instead of the dynamic H021 weights.
    """

    name = "macro_confirmed_seasonal"

    def __init__(
        self,
        vol_lookback: int = 60,
        term_lookback: int = 252,
        favorable_months: tuple = (11, 12, 1, 2, 3, 4),
        fallback_spy: float = 0.60,
        fallback_tlt: float = 0.40,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.term_lookback = term_lookback
        self.favorable_months = frozenset(favorable_months)
        self.fallback_spy = fallback_spy
        self.fallback_tlt = fallback_tlt

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy_rp, w_tlt_rp = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # VRP signal
        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_positive = (vrp > 0).fillna(False)

        # Term spread signal: TLT 252d return vs SHY 252d return
        tlt_ret = prices["tlt_close"].pct_change(self.term_lookback).shift(1)
        shy_ret = prices["shy_close"].pct_change(self.term_lookback).shift(1)
        term_positive = (tlt_ret > shy_ret) & tlt_ret.notna() & shy_ret.notna()

        macro_ok = vrp_positive | term_positive

        in_season = pd.Series(prices.index.month.isin(self.favorable_months), index=prices.index)
        tlt_avail = prices["tlt_close"].notna()

        # Start with H021
        w_spy = w_spy_rp.copy()
        w_tlt = w_tlt_rp.copy()

        # May-Oct: 100% TLT
        summer = ~in_season
        w_spy = w_spy.where(~summer, 0.0)
        w_tlt = w_tlt.where(~summer, tlt_avail.astype(float))

        # Nov-Apr, macro gate fails: fallback 60/40
        winter_no_macro = in_season & ~macro_ok
        w_spy = w_spy.where(~winter_no_macro, self.fallback_spy)
        w_tlt = w_tlt.where(~winter_no_macro, self.fallback_tlt)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class VRPEnhancedHalloween(MultiAssetStrategy):
    """H332: HalloweenMomentumConfirm (H046) + VRP scaler within Oct-Apr.

    May-Sep: 100% TLT (defensive, same as H046 logic).
    Oct-Apr: H046 base SPY weight × VRP_modifier, clipped to [0, 1].
      H046 base weight: 1.0 if SPY 6m return > 0, else 0.5.
      VRP_modifier: 1.10 if VRP > 0, 0.90 if VRP ≤ 0.
    Remainder in TLT.

    VRP is the variance risk premium — when positive, the market over-pays for
    options, meaning realized vol typically comes in below implied. Using VRP
    as a scaler within the seasonal window adds a vol-regime dimension to the
    pure calendar signal.
    """

    name = "vrp_enhanced_halloween"

    def __init__(
        self,
        vol_lookback: int = 60,
        momentum_days: int = 126,
        favorable_months: tuple = (10, 11, 12, 1, 2, 3, 4),
        vrp_up: float = 1.10,
        vrp_down: float = 0.90,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.momentum_days = momentum_days
        self.favorable_months = frozenset(favorable_months)
        self.vrp_up = vrp_up
        self.vrp_down = vrp_down

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()

        # H046 base SPY weight: 1.0 if momentum > 0 else 0.5 (within active period)
        mom = (prices["adj_close"] / prices["adj_close"].shift(self.momentum_days) - 1).shift(1)
        mom_positive = (mom > 0) & mom.notna()
        base_spy = mom_positive.map({True: 1.0, False: 0.5})

        # VRP modifier
        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_positive = (vrp > 0).fillna(False)
        vrp_mod = vrp_positive.map({True: self.vrp_up, False: self.vrp_down})

        in_season = pd.Series(prices.index.month.isin(self.favorable_months), index=prices.index)
        tlt_avail = prices["tlt_close"].notna()

        # In season: apply scaled SPY weight, rest to TLT
        spy_wt = (base_spy * vrp_mod).clip(0.0, 1.0)
        tlt_wt = (1.0 - spy_wt).clip(0.0, 1.0)

        # Out of season (May-Sep): 100% TLT
        w_spy = spy_wt.where(in_season, 0.0)
        w_tlt = tlt_wt.where(in_season, tlt_avail.astype(float))

        # Pre-TLT era: SPY only
        w_spy = w_spy.where(tlt_avail, 1.0)
        w_tlt = w_tlt.where(tlt_avail, 0.0)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class TripleOverlayCalendar(MultiAssetStrategy):
    """H334: H078 framework + VRP scaler + funding stress in Nov-Apr.

    May-Oct: 100% TLT.
    Nov-Apr:
      Base: H021 weights.
      VRP multiplier: vrp_z scaled to [0.80, 1.20] (same as H301 z-score approach).
      Stress reduction: if VIX > 20th pctile AND HYG 63d < LQD 63d → multiply by stress_mult.
      Final SPY weight = H021_spy × VRP_mult × stress_mult, clipped [0, 1].

    Stacks three independent orthogonal signals on the Nov-Apr exposure:
    the vol-risk-premium (compensation for variance), macro stress (credit + VIX),
    and the seasonal structure itself.
    """

    name = "triple_overlay_calendar"

    def __init__(
        self,
        vol_lookback: int = 60,
        vrp_window: int = 252,
        vrp_sensitivity: float = 0.25,
        stress_mult: float = 0.80,
        vix_pctile_window: int = 252,
        vix_pctile_threshold: float = 0.20,
        favorable_months: tuple = (11, 12, 1, 2, 3, 4),
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vrp_window = vrp_window
        self.vrp_sensitivity = vrp_sensitivity
        self.stress_mult = stress_mult
        self.vix_pctile_window = vix_pctile_window
        self.vix_pctile_threshold = vix_pctile_threshold
        self.favorable_months = frozenset(favorable_months)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy_rp, w_tlt_rp = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # VRP z-score scaler [0.80, 1.20]
        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_mean = vrp.rolling(self.vrp_window, min_periods=63).mean()
        vrp_std = vrp.rolling(self.vrp_window, min_periods=63).std().replace(0, np.nan)
        vrp_z = ((vrp - vrp_mean) / vrp_std).clip(-2.0, 2.0).fillna(0.0)
        vrp_mult = (1.0 + self.vrp_sensitivity * vrp_z).clip(0.80, 1.20)

        # Stress: VIX above low pctile AND credit stress
        vix = prices["vix_close"].shift(1).fillna(0.0)
        vix_20th = vix.rolling(self.vix_pctile_window, min_periods=63).quantile(
            self.vix_pctile_threshold
        )
        vix_elevated = vix > vix_20th

        credit_stress = pd.Series(False, index=prices.index)
        if "hyg_close" in prices.columns and "lqd_close" in prices.columns:
            hyg_ret = prices["hyg_close"].pct_change(63).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(63).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            credit_stress = ((hyg_ret < lqd_ret) & hyg_avail & hyg_ret.notna()).fillna(False)

        stress_flag = vix_elevated & credit_stress
        stress_mult_s = stress_flag.map({True: self.stress_mult, False: 1.0})

        in_season = pd.Series(prices.index.month.isin(self.favorable_months), index=prices.index)
        tlt_avail = prices["tlt_close"].notna()

        # H021 base with VRP and stress multipliers applied to SPY weight
        combined_mult = vrp_mult * stress_mult_s
        w_spy_active = (w_spy_rp * combined_mult).clip(0.0, 1.0)
        w_tlt_active = w_tlt_rp  # TLT weight unchanged

        # Season switch
        summer = ~in_season
        w_spy = w_spy_active.where(~summer, 0.0)
        w_tlt = w_tlt_active.where(~summer, tlt_avail.astype(float))

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class MacroStateH089(MultiAssetStrategy):
    """H335: H089 (H043+H021 50/50) with macro scaler on the H021 half only.

    H089 baseline: 50% H043 (SellInMayBondConfirm) weights + 50% H021 weights.
    Macro signal: 3-component stress composite (same as H306):
      vix_pctile_252d, credit_stress, vol_accel.
    Macro scaler applied to H021 half:
      H021_half × (0.7 + 0.6 × (1 - stress_composite)) ∈ [0.70×, 1.30×].
    H043 half stays fixed (seasonal, binary, not vol-driven).

    H043 brings its May-Oct TLT anchor + winter bond-confirm gate.
    H021 brings dynamic inverse-vol allocation.
    The macro scaler amplifies/dampens the H021 half based on stress conditions.
    """

    name = "macro_state_h089"

    def __init__(
        self,
        vol_lookback: int = 60,
        tlt_ma_period: int = 200,
        stress_sensitivity: float = 0.60,
        vix_pctile_window: int = 252,
        favorable_months: tuple = (11, 12, 1, 2, 3, 4),
    ) -> None:
        self.vol_lookback = vol_lookback
        self.tlt_ma_period = tlt_ma_period
        self.stress_sensitivity = stress_sensitivity
        self.vix_pctile_window = vix_pctile_window
        self.favorable_months = frozenset(favorable_months)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        # --- H021 half ---
        w_spy_rp, w_tlt_rp = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # --- H043 half (SellInMayBondConfirm) ---
        # In season AND TLT NOT in uptrend → long SPY; else flat
        tlt_ma = (
            prices["tlt_close"]
            .rolling(self.tlt_ma_period, min_periods=self.tlt_ma_period)
            .mean()
            .shift(1)
        )
        in_season = pd.Series(prices.index.month.isin(self.favorable_months), index=prices.index)
        bond_riskoff = (prices["tlt_close"].shift(1) > tlt_ma) & tlt_ma.notna()
        h043_spy = (in_season & ~bond_riskoff).astype(float)
        h043_tlt = pd.Series(0.0, index=prices.index)  # H043 goes flat in May-Oct / crisis

        # --- Macro stress composite (H306 approach) ---
        vix = prices["vix_close"].shift(1)
        vix_hi = vix.rolling(self.vix_pctile_window, min_periods=63).quantile(0.75)
        vix_lo = vix.rolling(self.vix_pctile_window, min_periods=63).quantile(0.25)
        vix_range = (vix_hi - vix_lo).replace(0, np.nan)
        vix_pctile = ((vix - vix_lo) / vix_range).clip(0.0, 1.0).fillna(0.5)

        spy_vol_21 = spy_r.rolling(21, min_periods=10).std().shift(1)
        spy_vol_63 = spy_r.rolling(63, min_periods=30).std().shift(1)
        vol_accel = (spy_vol_21 > spy_vol_63 * 1.5).fillna(False).astype(float)

        n_signals = 2.0
        credit_stress = pd.Series(0.0, index=prices.index)
        if "hyg_close" in prices.columns and "lqd_close" in prices.columns:
            hyg_ret = prices["hyg_close"].pct_change(63).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(63).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            credit_ok = hyg_avail & hyg_ret.notna() & lqd_ret.notna()
            credit_stress = (hyg_ret < lqd_ret).where(credit_ok, False).astype(float)
            n_signals = 3.0

        stress = (vix_pctile + credit_stress + vol_accel) / n_signals
        # Macro scaler: 0.70 + 0.60 × (1 - stress) → [0.70, 1.30]
        macro_scaler = (0.70 + self.stress_sensitivity * (1.0 - stress)).clip(0.70, 1.30)

        # H021 half with macro scaler applied to SPY weight
        w_spy_h021_scaled = (w_spy_rp * macro_scaler).clip(0.0, 1.0)
        w_tlt_h021_scaled = w_tlt_rp  # TLT weight of H021 half unchanged

        # Blend: 50% H043 + 50% H021 (macro-scaled H021 half)
        w_spy = 0.5 * h043_spy + 0.5 * w_spy_h021_scaled
        w_tlt = 0.5 * h043_tlt + 0.5 * w_tlt_h021_scaled

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


# ---------------------------------------------------------------------------
# Set 4 Batch 8: Multi-Signal Macro + Momentum Overlays (H336-H340)
# ---------------------------------------------------------------------------


class VRPRefinedH060(MultiAssetStrategy):
    """H336: CarryMomentumSPYTLT (H060) + VRP overlay.

    H060 base: 100% SPY if curve steep (TLT>SHY 63d) AND SPY above 200d MA;
               100% TLT if curve inverted; flat otherwise.
    VRP modifier: if H060 would be 100% SPY and VRP < 0 →
                  spy_weight × vrp_cut (0.70 SPY + 0.30 TLT).
    When VRP ≥ 0 or H060 is already in TLT/flat: no change.

    VRP < 0 means the market under-pays for variance insurance — realized vol
    is running above implied. This is a mild stress signal that justifies
    trimming the pure-equity bet H060 would otherwise make.
    """

    name = "vrp_refined_h060"

    def __init__(
        self,
        mom_days: int = 63,
        spy_ma: int = 200,
        vrp_cut: float = 0.70,
    ) -> None:
        self.mom_days = mom_days
        self.spy_ma = spy_ma
        self.vrp_cut = vrp_cut

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()

        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt = prices.get("tlt_close", pd.Series(np.nan, index=prices.index))

        shy_r = shy.shift(1) / shy.shift(1 + self.mom_days) - 1
        tlt_r = tlt.shift(1) / tlt.shift(1 + self.mom_days) - 1
        spy_ma = prices["adj_close"].shift(1).rolling(self.spy_ma).mean()

        valid = shy_r.notna() & tlt_r.notna()
        inverted = valid & (shy_r > tlt_r)
        spy_trend_up = prices["adj_close"].shift(1) > spy_ma

        # H060 base weights
        w_spy = pd.Series(0.0, index=prices.index)
        w_tlt = pd.Series(0.0, index=prices.index)
        w_spy[valid & ~inverted & spy_trend_up] = 1.0
        w_tlt[inverted] = 1.0

        # VRP overlay: reduce SPY when VRP < 0 and H060 is in 100% SPY
        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_negative = (vrp < 0).fillna(False)
        fully_spy = w_spy >= 1.0
        cut_condition = vrp_negative & fully_spy

        w_spy = w_spy.where(~cut_condition, self.vrp_cut)
        w_tlt = w_tlt.where(~cut_condition, 1.0 - self.vrp_cut)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class FundingStressH077(MultiAssetStrategy):
    """H337: RiskParityMomentumFilter (H077) + funding stress override.

    H077 base: H021 weights if SPY 12m momentum > 0; else 100% TLT.
    Stress override: if VIX > vix_threshold AND HYG 63d < LQD 63d →
                     force to crisis_tlt% TLT / (1-crisis_tlt)% SPY,
                     regardless of the momentum gate.

    Pre-HYG era: stress override requires only VIX > vix_threshold.

    The momentum gate and stress override address different regimes: momentum
    filters out secular bear markets; funding stress catches sudden liquidity
    crises where momentum may still be positive at the time of the shock.
    """

    name = "funding_stress_h077"

    def __init__(
        self,
        vol_lookback: int = 60,
        mom_days: int = 252,
        vix_threshold: float = 30.0,
        crisis_tlt: float = 0.80,
        credit_lookback: int = 63,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.mom_days = mom_days
        self.vix_threshold = vix_threshold
        self.crisis_tlt = crisis_tlt
        self.credit_lookback = credit_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy_rp, w_tlt_rp = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # H077 momentum gate: SPY 12m > 0 → H021; else 100% TLT
        spy_mom = prices["adj_close"].pct_change(self.mom_days).shift(1)
        mom_ok = (spy_mom > 0) & spy_mom.notna()
        tlt_avail = prices["tlt_close"].notna()

        w_spy = w_spy_rp.where(mom_ok, 0.0)
        w_tlt = w_tlt_rp.where(mom_ok, tlt_avail.astype(float))

        # Funding stress override
        vix = prices["vix_close"].shift(1).fillna(0.0)
        vix_stress = vix > self.vix_threshold

        if "hyg_close" in prices.columns and "lqd_close" in prices.columns:
            hyg_ret = prices["hyg_close"].pct_change(self.credit_lookback).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(self.credit_lookback).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            credit_off = (hyg_ret < lqd_ret) & hyg_avail & hyg_ret.notna()
            stress_flag = (vix_stress & credit_off).where(hyg_avail, vix_stress)
        else:
            stress_flag = vix_stress

        # Override to crisis allocation
        w_spy = w_spy.where(~stress_flag, 1.0 - self.crisis_tlt)
        w_tlt = w_tlt.where(~stress_flag, self.crisis_tlt)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class DoubleCheckH080(MultiAssetStrategy):
    """H338: RiskParityYieldCurveOverlay (H080) + credit spread double-confirm.

    H080 base: if SHY 63d return > TLT 63d return (curve inverted) →
               reduce SPY weight by 50%; else H021.
    Double-confirm logic:
      Both conditions (curve inverted AND HYG 63d < LQD 63d) → SPY × 0.60 (strong cut).
      Only H080 curve signal → SPY × 0.85 (partial cut, cautious).
      Only credit stress → H021 unchanged (credit stress alone not enough to cut SPY).
      Neither → H021 full.

    Pre-HYG era: falls back to H080 base (curve signal only, partial cut).
    """

    name = "double_check_h080"

    def __init__(
        self,
        vol_lookback: int = 60,
        curve_days: int = 63,
        credit_lookback: int = 63,
        both_scale: float = 0.60,
        curve_only_scale: float = 0.85,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.curve_days = curve_days
        self.credit_lookback = credit_lookback
        self.both_scale = both_scale
        self.curve_only_scale = curve_only_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # H080 curve signal
        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt_px = prices["tlt_close"]
        shy_ret = shy.shift(1) / shy.shift(1 + self.curve_days) - 1
        tlt_ret_curve = tlt_px.shift(1) / tlt_px.shift(1 + self.curve_days) - 1
        curve_inverted = (shy_ret > tlt_ret_curve) & shy_ret.notna() & tlt_ret_curve.notna()

        # Credit signal
        has_credit = "hyg_close" in prices.columns and "lqd_close" in prices.columns
        credit_stress = pd.Series(False, index=prices.index)
        if has_credit:
            hyg_ret = prices["hyg_close"].pct_change(self.credit_lookback).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(self.credit_lookback).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            credit_stress = ((hyg_ret < lqd_ret) & hyg_avail & hyg_ret.notna()).fillna(False)

        both = curve_inverted & credit_stress
        curve_only = curve_inverted & ~credit_stress

        # Apply tiered SPY cut
        freed_both = w_spy * (1.0 - self.both_scale)
        freed_curve = w_spy * (1.0 - self.curve_only_scale)

        new_spy = pd.Series(
            np.select(
                [both, curve_only],
                [w_spy * self.both_scale, w_spy * self.curve_only_scale],
                default=w_spy,
            ),
            index=prices.index,
        )
        new_tlt = pd.Series(
            np.select(
                [both, curve_only],
                [(w_tlt + freed_both).clip(upper=1.0), (w_tlt + freed_curve).clip(upper=1.0)],
                default=w_tlt,
            ),
            index=prices.index,
        )

        return pd.DataFrame({"SPY": new_spy, "TLT": new_tlt}, index=prices.index)


class VRPCorrelationRegime(MultiAssetStrategy):
    """H339: CorrelationRiskParityHybrid (H097) + VRP boost in normal regime.

    H097 base: if 60d rolling SPY-TLT correlation > 0 → crisis (80% TLT);
               else H021 weights.
    VRP boost: in normal regime (corr < 0) AND VRP > 252d median VRP →
               SPY weight × vrp_boost (1.10), clipped to 1.0.

    When correlation is negative (SPY and TLT moving independently as expected),
    a positive VRP confirms the market is in a compensating regime, making the
    risk-parity position even more attractive. The boost captures this regime.
    """

    name = "vrp_correlation_regime"

    def __init__(
        self,
        vol_lookback: int = 60,
        corr_lookback: int = 60,
        crisis_tlt: float = 0.90,
        vrp_window: int = 252,
        vrp_boost: float = 1.10,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.corr_lookback = corr_lookback
        self.crisis_tlt = crisis_tlt
        self.vrp_window = vrp_window
        self.vrp_boost = vrp_boost

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy_h021, w_tlt_h021 = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # H097 correlation signal
        corr = spy_r.rolling(self.corr_lookback).corr(tlt_r).shift(1)
        crisis = (corr > 0) & prices["tlt_close"].notna()

        tlt_avail = prices["tlt_close"].notna()
        w_spy = w_spy_h021.where(~crisis, 1.0 - self.crisis_tlt)
        w_tlt = w_tlt_h021.where(~crisis, self.crisis_tlt)
        w_spy = w_spy.where(tlt_avail, 0.0)
        w_tlt = w_tlt.where(tlt_avail, 0.0)

        # VRP boost in normal regime
        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_median = vrp.rolling(self.vrp_window, min_periods=63).median()
        vrp_above_median = (vrp > vrp_median) & vrp_median.notna()

        normal_regime = ~crisis
        boost_condition = normal_regime & vrp_above_median
        w_spy_boosted = (w_spy * self.vrp_boost).clip(upper=1.0)
        w_spy = w_spy.where(~boost_condition, w_spy_boosted)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class GrandCrisisProtocol(MultiAssetStrategy):
    """H340: Union of four crisis signals with tiered SPY/TLT response.

    Signal 1: H097 correlation crisis (60d SPY-TLT corr > 0).
    Signal 2: H079 crisis (portfolio drawdown > dd_threshold AND VIX > vix_threshold).
    Signal 3: H306 high stress (stress_score > stress_high).
    Signal 4: VRP < vrp_severe (severe negative VRP).

    Tiered response:
      2+ signals active → 20% SPY / 80% TLT
      1  signal active  → 40% SPY / 60% TLT
      0  signals active → H021 inverse-vol weights

    Diversifies across four independent crisis detectors: correlation contagion,
    drawdown depth, macro stress, and VRP. Any single detector can fire without
    triggering maximum defense; the most defensive posture requires corroboration.
    """

    name = "grand_crisis_protocol"

    def __init__(
        self,
        vol_lookback: int = 60,
        corr_lookback: int = 60,
        dd_threshold: float = 0.15,
        vix_threshold: float = 30.0,
        stress_high: float = 0.67,
        vrp_severe: float = -0.005,
        spy_one: float = 0.40,
        tlt_one: float = 0.60,
        spy_two: float = 0.20,
        tlt_two: float = 0.80,
        vix_pctile_window: int = 252,
        credit_lookback: int = 63,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.corr_lookback = corr_lookback
        self.dd_threshold = dd_threshold
        self.vix_threshold = vix_threshold
        self.stress_high = stress_high
        self.vrp_severe = vrp_severe
        self.spy_one = spy_one
        self.tlt_one = tlt_one
        self.spy_two = spy_two
        self.tlt_two = tlt_two
        self.vix_pctile_window = vix_pctile_window
        self.credit_lookback = credit_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0914
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy_h021, w_tlt_h021 = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # Signal 1: H097 correlation crisis
        corr = spy_r.rolling(self.corr_lookback).corr(tlt_r).shift(1)
        s1 = ((corr > 0) & prices["tlt_close"].notna()).fillna(False).astype(int)

        # Signal 2: H079 portfolio drawdown + VIX
        port_r = w_spy_h021.shift(1) * spy_r + w_tlt_h021.shift(1) * tlt_r
        cum = (1 + port_r).cumprod()
        peak = cum.cummax()
        dd_depth = ((peak - cum) / peak.replace(0, np.nan)).fillna(0.0)
        vix = prices["vix_close"].shift(1).fillna(0.0)
        s2 = ((dd_depth > self.dd_threshold) & (vix > self.vix_threshold)).astype(int)

        # Signal 3: H306-style stress composite
        vix_s = prices["vix_close"].shift(1)
        vix_hi = vix_s.rolling(self.vix_pctile_window, min_periods=63).quantile(0.75)
        vix_lo = vix_s.rolling(self.vix_pctile_window, min_periods=63).quantile(0.25)
        vix_range = (vix_hi - vix_lo).replace(0, np.nan)
        vix_pctile = ((vix_s - vix_lo) / vix_range).clip(0.0, 1.0).fillna(0.5)

        spy_vol_21 = spy_r.rolling(21, min_periods=10).std().shift(1)
        spy_vol_63 = spy_r.rolling(63, min_periods=30).std().shift(1)
        vol_accel = (spy_vol_21 > spy_vol_63 * 1.5).fillna(False).astype(float)

        n_sig = 2.0
        credit_c = pd.Series(0.0, index=prices.index)
        if "hyg_close" in prices.columns and "lqd_close" in prices.columns:
            hyg_ret = prices["hyg_close"].pct_change(self.credit_lookback).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(self.credit_lookback).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            credit_ok = hyg_avail & hyg_ret.notna() & lqd_ret.notna()
            credit_c = (hyg_ret < lqd_ret).where(credit_ok, False).astype(float)
            n_sig = 3.0

        stress_score = (vix_pctile + credit_c + vol_accel) / n_sig
        s3 = (stress_score > self.stress_high).astype(int)

        # Signal 4: Severe negative VRP
        vrp = _vrp_series(spy_r, prices["vix_close"])
        s4 = (vrp < self.vrp_severe).fillna(False).astype(int)

        total_signals = s1 + s2 + s3 + s4

        tlt_avail = prices["tlt_close"].notna()
        w_spy = pd.Series(
            np.select(
                [total_signals >= 2, total_signals == 1],
                [self.spy_two, self.spy_one],
                default=w_spy_h021,
            ),
            index=prices.index,
        )
        w_tlt = pd.Series(
            np.select(
                [total_signals >= 2, total_signals == 1],
                [self.tlt_two, self.tlt_one],
                default=w_tlt_h021,
            ),
            index=prices.index,
        )

        # Pre-TLT era: SPY only
        w_spy = w_spy.where(tlt_avail, 1.0)
        w_tlt = w_tlt.where(tlt_avail, 0.0)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


# ===========================================================================
# SET 6 — H421–H425: Macro Sector & Policy Exposure Proxies
# ===========================================================================


class RegulatedSectorTiming(MultiAssetStrategy):
    """H421: H021 + TLT trend gate to tilt into regulated defensive sectors.

    When TLT > 200d MA (rates falling = policy tailwind for defensives):
      SPY weight reduced by tilt_pct, split evenly between XLP and XLU.
    Otherwise: pure H021. Cohen, Diether & Malloy (2013): policy events
    benefit regulated utility and consumer staples sectors.
    """

    name = "regulated_sector_timing"

    def __init__(
        self,
        vol_lookback: int = 60,
        tlt_ma: int = 200,
        tilt_pct: float = 0.15,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.tlt_ma = tlt_ma
        self.tilt_pct = tilt_pct

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        xlp_avail = (
            prices["xlp_close"].notna()
            if "xlp_close" in prices.columns
            else pd.Series(False, index=prices.index)
        )
        xlu_avail = (
            prices["xlu_close"].notna()
            if "xlu_close" in prices.columns
            else pd.Series(False, index=prices.index)
        )
        tlt_ma_s = prices["tlt_close"].rolling(self.tlt_ma, min_periods=self.tlt_ma // 2).mean()
        rate_falling = (prices["tlt_close"] > tlt_ma_s) & tlt_avail & xlp_avail & xlu_avail
        half_tilt = self.tilt_pct / 2.0
        w_spy_adj = w_spy.where(~rate_falling, w_spy * (1.0 - self.tilt_pct))
        w_xlp = pd.Series(0.0, index=prices.index).where(~rate_falling, w_spy * half_tilt)
        w_xlu = pd.Series(0.0, index=prices.index).where(~rate_falling, w_spy * half_tilt)
        w_spy_adj = w_spy_adj.where(tlt_avail, 1.0)
        w_tlt_final = w_tlt.where(tlt_avail, 0.0)
        return pd.DataFrame(
            {"SPY": w_spy_adj, "TLT": w_tlt_final, "XLP": w_xlp, "XLU": w_xlu}, index=prices.index
        )


class PolicyRateSectorRotate(MultiAssetStrategy):
    """H422: H021 with yield-curve tilt into rate-sensitive defensive sectors.

    When TLT 12m return > SHY 12m return (rates falling, bonds outperforming):
      Reduce SPY, add XLU + XLP weight.
    Policy rate changes most directly affect utility/consumer staples sectors.
    """

    name = "policy_rate_sector_rotate"

    def __init__(
        self,
        vol_lookback: int = 60,
        curve_lookback: int = 252,
        tilt_pct: float = 0.15,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.curve_lookback = curve_lookback
        self.tilt_pct = tilt_pct

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        has_shy = "shy_close" in prices.columns
        has_xlp = "xlp_close" in prices.columns
        has_xlu = "xlu_close" in prices.columns
        tlt_ret = prices["tlt_close"] / prices["tlt_close"].shift(self.curve_lookback) - 1
        shy_ret = (
            prices["shy_close"] / prices["shy_close"].shift(self.curve_lookback) - 1
            if has_shy
            else pd.Series(0.0, index=prices.index)
        )
        shy_avail = prices["shy_close"].notna() if has_shy else pd.Series(False, index=prices.index)
        xlp_avail = prices["xlp_close"].notna() if has_xlp else pd.Series(False, index=prices.index)
        xlu_avail = prices["xlu_close"].notna() if has_xlu else pd.Series(False, index=prices.index)
        rate_falling = (
            (tlt_ret > shy_ret) & tlt_avail & shy_avail & xlp_avail & xlu_avail & tlt_ret.notna()
        )
        half_tilt = self.tilt_pct / 2.0
        w_spy_adj = w_spy.where(~rate_falling, w_spy * (1.0 - self.tilt_pct))
        w_xlp = pd.Series(0.0, index=prices.index).where(~rate_falling, w_spy * half_tilt)
        w_xlu = pd.Series(0.0, index=prices.index).where(~rate_falling, w_spy * half_tilt)
        w_spy_adj = w_spy_adj.where(tlt_avail, 1.0)
        w_tlt_final = w_tlt.where(tlt_avail, 0.0)
        return pd.DataFrame(
            {"SPY": w_spy_adj, "TLT": w_tlt_final, "XLP": w_xlp, "XLU": w_xlu}, index=prices.index
        )


class CreditCycleSectorFilter(MultiAssetStrategy):
    """H423: Credit cycle gates sector tilt into defensive XLP/XLU.

    When HYG 63d return < LQD 63d return (credit spreading → risk-off):
      Reduce SPY, add XLP + XLU as defensive sectors.
    Credit improving: pure H021. Cohen et al. (2013): credit stress = policy risk.
    """

    name = "credit_cycle_sector_filter"

    def __init__(
        self,
        vol_lookback: int = 60,
        credit_lookback: int = 63,
        tilt_pct: float = 0.20,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.credit_lookback = credit_lookback
        self.tilt_pct = tilt_pct

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        has_hyg = "hyg_close" in prices.columns
        has_lqd = "lqd_close" in prices.columns
        has_xlp = "xlp_close" in prices.columns
        has_xlu = "xlu_close" in prices.columns
        hyg_ret = (
            prices["hyg_close"].pct_change(self.credit_lookback)
            if has_hyg
            else pd.Series(np.nan, index=prices.index)
        )
        lqd_ret = (
            prices["lqd_close"].pct_change(self.credit_lookback)
            if has_lqd
            else pd.Series(np.nan, index=prices.index)
        )
        hyg_avail = prices["hyg_close"].notna() if has_hyg else pd.Series(False, index=prices.index)
        lqd_avail = prices["lqd_close"].notna() if has_lqd else pd.Series(False, index=prices.index)
        xlp_avail = prices["xlp_close"].notna() if has_xlp else pd.Series(False, index=prices.index)
        xlu_avail = prices["xlu_close"].notna() if has_xlu else pd.Series(False, index=prices.index)
        credit_stress = (
            (hyg_ret < lqd_ret) & hyg_avail & lqd_avail & xlp_avail & xlu_avail & hyg_ret.notna()
        )
        half_tilt = self.tilt_pct / 2.0
        w_spy_adj = w_spy.where(~credit_stress, w_spy * (1.0 - self.tilt_pct))
        w_xlp = pd.Series(0.0, index=prices.index).where(~credit_stress, w_spy * half_tilt)
        w_xlu = pd.Series(0.0, index=prices.index).where(~credit_stress, w_spy * half_tilt)
        w_spy_adj = w_spy_adj.where(tlt_avail, 1.0)
        w_tlt_final = w_tlt.where(tlt_avail, 0.0)
        return pd.DataFrame(
            {"SPY": w_spy_adj, "TLT": w_tlt_final, "XLP": w_xlp, "XLU": w_xlu}, index=prices.index
        )


class CommodityExposureTiming(MultiAssetStrategy):
    """H424: DBC momentum signals commodity regime; tilt SPY vs XLP defensively.

    DBC downtrend (DBC < 63d MA): reduce SPY, add XLP (defensive staples).
    DBC uptrend: standard H021 (commodity inflation = broad growth environment).
    """

    name = "commodity_exposure_timing"

    def __init__(
        self,
        vol_lookback: int = 60,
        dbc_ma: int = 63,
        tilt_pct: float = 0.15,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.dbc_ma = dbc_ma
        self.tilt_pct = tilt_pct

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        has_dbc = "dbc_close" in prices.columns
        has_xlp = "xlp_close" in prices.columns
        if has_dbc:
            dbc_ma_s = prices["dbc_close"].rolling(self.dbc_ma, min_periods=self.dbc_ma // 2).mean()
            dbc_avail = prices["dbc_close"].notna()
            xlp_avail = (
                prices["xlp_close"].notna() if has_xlp else pd.Series(False, index=prices.index)
            )
            dbc_downtrend = (
                (prices["dbc_close"] < dbc_ma_s) & dbc_avail & xlp_avail & dbc_ma_s.notna()
            )
        else:
            dbc_downtrend = pd.Series(False, index=prices.index)
        w_spy_adj = w_spy.where(~dbc_downtrend, w_spy * (1.0 - self.tilt_pct))
        w_xlp = pd.Series(0.0, index=prices.index).where(~dbc_downtrend, w_spy * self.tilt_pct)
        w_spy_adj = w_spy_adj.where(tlt_avail, 1.0)
        w_tlt_final = w_tlt.where(tlt_avail, 0.0)
        return pd.DataFrame(
            {"SPY": w_spy_adj, "TLT": w_tlt_final, "XLP": w_xlp}, index=prices.index
        )


class MacroRegimeSectorBlend(MultiAssetStrategy):
    """H425: 3-signal macro regime → sector-blended allocation.

    Signals: (1) VIX < 25, (2) TLT > 63d MA, (3) HYG > HYG.shift(63).
    3 bullish: 80% SPY / 20% TLT.
    2 bullish: 55% SPY / 25% TLT / 10% XLP / 10% XLU.
    0-1 bullish: 30% SPY / 40% TLT / 15% XLP / 15% XLU.
    """

    name = "macro_regime_sector_blend"

    def __init__(
        self,
        vix_threshold: float = 25.0,
        tlt_ma: int = 63,
        credit_lookback: int = 63,
    ) -> None:
        self.vix_threshold = vix_threshold
        self.tlt_ma = tlt_ma
        self.credit_lookback = credit_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        tlt_avail = prices["tlt_close"].notna()
        has_hyg = "hyg_close" in prices.columns
        has_xlp = "xlp_close" in prices.columns
        has_xlu = "xlu_close" in prices.columns
        vix_low = prices["vix_close"] < self.vix_threshold
        tlt_ma_s = prices["tlt_close"].rolling(self.tlt_ma, min_periods=self.tlt_ma // 2).mean()
        tlt_up = (prices["tlt_close"] > tlt_ma_s) & tlt_avail
        if has_hyg:
            hyg_avail = prices["hyg_close"].notna()
            hyg_up = (
                prices["hyg_close"] > prices["hyg_close"].shift(self.credit_lookback)
            ) & hyg_avail
        else:
            hyg_up = pd.Series(False, index=prices.index)
        xlp_avail = prices["xlp_close"].notna() if has_xlp else pd.Series(False, index=prices.index)
        xlu_avail = prices["xlu_close"].notna() if has_xlu else pd.Series(False, index=prices.index)
        can_sector = tlt_avail & xlp_avail & xlu_avail
        n_bull = vix_low.astype(int) + tlt_up.astype(int) + hyg_up.astype(int)
        w_spy = pd.Series(1.0, index=prices.index)
        w_tlt = pd.Series(0.0, index=prices.index)
        w_xlp = pd.Series(0.0, index=prices.index)
        w_xlu = pd.Series(0.0, index=prices.index)
        cond3 = (n_bull == 3) & can_sector
        w_spy = w_spy.where(~cond3, 0.80)
        w_tlt = w_tlt.where(~cond3, 0.20)
        cond2 = (n_bull == 2) & can_sector
        w_spy = w_spy.where(~cond2, 0.55)
        w_tlt = w_tlt.where(~cond2, 0.25)
        w_xlp = w_xlp.where(~cond2, 0.10)
        w_xlu = w_xlu.where(~cond2, 0.10)
        cond01 = (n_bull <= 1) & can_sector
        w_spy = w_spy.where(~cond01, 0.30)
        w_tlt = w_tlt.where(~cond01, 0.40)
        w_xlp = w_xlp.where(~cond01, 0.15)
        w_xlu = w_xlu.where(~cond01, 0.15)
        w_spy = w_spy.where(tlt_avail, 1.0)
        w_tlt = w_tlt.where(tlt_avail, 0.0)
        w_xlp = w_xlp.where(can_sector, 0.0)
        w_xlu = w_xlu.where(can_sector, 0.0)
        return pd.DataFrame(
            {"SPY": w_spy, "TLT": w_tlt, "XLP": w_xlp, "XLU": w_xlu}, index=prices.index
        )


# ===========================================================================
# SET 7 — H481–H490: Hybrid AA+CS Aggregate Fundamental Signals for ETF Timing
# ===========================================================================


def _h021_base_weights(prices: pd.DataFrame, vol_lookback: int = 60) -> tuple[pd.Series, pd.Series]:
    """Compute H021 inverse-vol base weights. Returns (w_spy, w_tlt)."""
    spy_r = prices["adj_close"].pct_change()
    tlt_r = prices["tlt_close"].pct_change()
    spy_vol = spy_r.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1) * np.sqrt(252)
    tlt_vol = tlt_r.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1) * np.sqrt(252)
    inv_spy = 1.0 / spy_vol.replace(0, np.nan)
    inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
    total = inv_spy + inv_tlt
    w_spy = (inv_spy / total).fillna(0.0)
    w_tlt = (inv_tlt / total).fillna(0.0)
    return w_spy, w_tlt


class MarketQualityRegimePort(MultiAssetStrategy):
    """H481: H021 overlaid with market-wide GP/TA quality regime signal.

    Signal: rolling 252d median GP/TA across S&P 500 universe (pre-computed).
    Signal > its own 252d trailing mean → equity-favorable → SPY weight × 1.20.
    Signal ≤ mean → defensive → SPY weight × 0.80.
    Freed/added weight shifts to/from TLT. Capped at 1.0 total exposure.
    """

    name = "market_quality_regime_port"

    def __init__(
        self,
        market_gpta: pd.Series,
        vol_lookback: int = 60,
        signal_window: int = 252,
        bull_scale: float = 1.20,
        bear_scale: float = 0.80,
    ) -> None:
        self.market_gpta = market_gpta
        self.vol_lookback = vol_lookback
        self.signal_window = signal_window
        self.bull_scale = bull_scale
        self.bear_scale = bear_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        sig = self.market_gpta.reindex(prices.index).ffill()
        sig_ma = (
            sig.rolling(self.signal_window, min_periods=self.signal_window // 2).mean().shift(1)
        )
        bull = (sig.shift(1) > sig_ma) & tlt_avail
        scale = pd.Series(1.0, index=prices.index)
        scale = scale.where(~bull, self.bull_scale)
        scale = (
            scale.where(bull, scale)
            .where(bull | ~tlt_avail, self.bear_scale)
            .where(bull, self.bull_scale)
        )
        # Simpler: bull → scale up SPY, else scale down
        spy_scaled = (w_spy * self.bull_scale).where(bull, w_spy * self.bear_scale)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        spy_final = spy_scaled.where(tlt_avail, 1.0)
        tlt_final = tlt_adj.where(tlt_avail, 0.0)
        return pd.DataFrame({"SPY": spy_final, "TLT": tlt_final}, index=prices.index)


class MarketAccrualTimingPort(MultiAssetStrategy):
    """H482: H021 overlaid with market-wide accrual quality signal.

    Signal: rolling 252d mean of (NI-OCF)/TA across universe (pre-computed).
    High accruals (above 252d trailing mean) → earnings quality risk → SPY × 0.80.
    Low accruals → cash-backed earnings → SPY × 1.20.
    """

    name = "market_accrual_timing_port"

    def __init__(
        self,
        market_accruals: pd.Series,
        vol_lookback: int = 60,
        signal_window: int = 252,
        bull_scale: float = 1.20,
        bear_scale: float = 0.80,
    ) -> None:
        self.market_accruals = market_accruals
        self.vol_lookback = vol_lookback
        self.signal_window = signal_window
        self.bull_scale = bull_scale
        self.bear_scale = bear_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        sig = self.market_accruals.reindex(prices.index).ffill()
        sig_ma = (
            sig.rolling(self.signal_window, min_periods=self.signal_window // 2).mean().shift(1)
        )
        # High accruals (sig > ma) = BAD → bear_scale; low accruals = bull_scale
        high_accruals = (sig.shift(1) > sig_ma) & tlt_avail
        spy_scaled = (w_spy * self.bear_scale).where(high_accruals, w_spy * self.bull_scale)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        spy_final = spy_scaled.where(tlt_avail, 1.0)
        tlt_final = tlt_adj.where(tlt_avail, 0.0)
        return pd.DataFrame({"SPY": spy_final, "TLT": tlt_final}, index=prices.index)


class CSReturnDispersionAAPort(MultiAssetStrategy):
    """H483: H021 overlaid with cross-sectional return dispersion regime.

    Signal: rolling 21d mean of daily CS std of 5d returns (pre-computed).
    Low dispersion (below 252d median) → passive/ETF regime → SPY × 0.75.
    High dispersion → active/fundamental regime → normal H021.
    Pure price signal, full 12-fold coverage.
    """

    name = "cs_return_dispersion_aa_port"

    def __init__(
        self,
        cs_dispersion: pd.Series,
        vol_lookback: int = 60,
        signal_window: int = 252,
        low_disp_scale: float = 0.75,
    ) -> None:
        self.cs_dispersion = cs_dispersion
        self.vol_lookback = vol_lookback
        self.signal_window = signal_window
        self.low_disp_scale = low_disp_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        sig = self.cs_dispersion.reindex(prices.index).ffill()
        sig_median = (
            sig.rolling(self.signal_window, min_periods=self.signal_window // 2).median().shift(1)
        )
        low_disp = (sig.shift(1) < sig_median) & tlt_avail
        spy_scaled = (w_spy * self.low_disp_scale).where(low_disp, w_spy)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        spy_final = spy_scaled.where(tlt_avail, 1.0)
        tlt_final = tlt_adj.where(tlt_avail, 0.0)
        return pd.DataFrame({"SPY": spy_final, "TLT": tlt_final}, index=prices.index)


class QualityBreadthAAPort(MultiAssetStrategy):
    """H484: H021 overlaid with cross-sectional quality improvement breadth.

    Signal: fraction of S&P 500 stocks with GP/TA above own 252d GP/TA median (pre-computed).
    Breadth > 0.60 → broad quality improvement → SPY × 1.20.
    Breadth < 0.40 → quality deterioration → SPY × 0.80.
    """

    name = "quality_breadth_aa_port"

    def __init__(
        self,
        quality_breadth: pd.Series,
        vol_lookback: int = 60,
        bull_thresh: float = 0.60,
        bear_thresh: float = 0.40,
        bull_scale: float = 1.20,
        bear_scale: float = 0.80,
    ) -> None:
        self.quality_breadth = quality_breadth
        self.vol_lookback = vol_lookback
        self.bull_thresh = bull_thresh
        self.bear_thresh = bear_thresh
        self.bull_scale = bull_scale
        self.bear_scale = bear_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        breadth = self.quality_breadth.reindex(prices.index).ffill().shift(1)
        bull = (breadth > self.bull_thresh) & tlt_avail
        bear = (breadth < self.bear_thresh) & tlt_avail
        spy_scaled = w_spy.copy()
        spy_scaled = (w_spy * self.bull_scale).where(bull, spy_scaled)
        spy_scaled = (w_spy * self.bear_scale).where(bear, spy_scaled)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        spy_final = spy_scaled.where(tlt_avail, 1.0)
        tlt_final = tlt_adj.where(tlt_avail, 0.0)
        return pd.DataFrame({"SPY": spy_final, "TLT": tlt_final}, index=prices.index)


class MarketVolBreadthAAPort(MultiAssetStrategy):
    """H485: H021 overlaid with cross-sectional vol-decline breadth signal.

    Signal: fraction of S&P 500 stocks where 21d realized vol < 63d realized vol (pre-computed).
    Breadth > 0.60 → systemic vol normalization → SPY × 1.15.
    Breadth < 0.40 → systemic vol buildup → SPY × 0.85.
    Pure price signal, full 12-fold coverage.
    """

    name = "market_vol_breadth_aa_port"

    def __init__(
        self,
        vol_breadth: pd.Series,
        vol_lookback: int = 60,
        bull_thresh: float = 0.60,
        bear_thresh: float = 0.40,
        bull_scale: float = 1.15,
        bear_scale: float = 0.85,
    ) -> None:
        self.vol_breadth = vol_breadth
        self.vol_lookback = vol_lookback
        self.bull_thresh = bull_thresh
        self.bear_thresh = bear_thresh
        self.bull_scale = bull_scale
        self.bear_scale = bear_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        breadth = self.vol_breadth.reindex(prices.index).ffill().shift(1)
        bull = (breadth > self.bull_thresh) & tlt_avail
        bear = (breadth < self.bear_thresh) & tlt_avail
        spy_scaled = w_spy.copy()
        spy_scaled = (w_spy * self.bull_scale).where(bull, spy_scaled)
        spy_scaled = (w_spy * self.bear_scale).where(bear, spy_scaled)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        spy_final = spy_scaled.where(tlt_avail, 1.0)
        tlt_final = tlt_adj.where(tlt_avail, 0.0)
        return pd.DataFrame({"SPY": spy_final, "TLT": tlt_final}, index=prices.index)


class SectorFundamentalRotation(MultiAssetStrategy):
    """H486: H021 overlaid with cross-sector GP/TA quality dispersion signal.

    Signal: rolling 63d cross-sector std of median GP/TA (pre-computed).
    High dispersion (technology/quality sectors dominating) → SPY × 1.10.
    Low dispersion (quality flattening across sectors) → SPY × 0.90.
    """

    name = "sector_fundamental_rotation"

    def __init__(
        self,
        sector_quality_dispersion: pd.Series,
        vol_lookback: int = 60,
        signal_window: int = 252,
        bull_scale: float = 1.10,
        bear_scale: float = 0.90,
    ) -> None:
        self.sector_quality_dispersion = sector_quality_dispersion
        self.vol_lookback = vol_lookback
        self.signal_window = signal_window
        self.bull_scale = bull_scale
        self.bear_scale = bear_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        sig = self.sector_quality_dispersion.reindex(prices.index).ffill()
        sig_ma = (
            sig.rolling(self.signal_window, min_periods=self.signal_window // 2).mean().shift(1)
        )
        high_disp = (sig.shift(1) > sig_ma) & tlt_avail
        spy_scaled = (w_spy * self.bull_scale).where(high_disp, w_spy * self.bear_scale)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        spy_final = spy_scaled.where(tlt_avail, 1.0)
        tlt_final = tlt_adj.where(tlt_avail, 0.0)
        return pd.DataFrame({"SPY": spy_final, "TLT": tlt_final}, index=prices.index)


class MomentumBreadthAAPort(MultiAssetStrategy):
    """H487: H021 overlaid with percentage of S&P 500 stocks above their 200d MA.

    Classic market breadth indicator applied as rule-based AA overlay.
    Breadth > 0.60 → broad market trend → SPY × 1.20.
    Breadth < 0.40 → trend deterioration → SPY × 0.80.
    Pure price signal, full 12-fold coverage.
    """

    name = "momentum_breadth_aa_port"

    def __init__(
        self,
        breadth_signal: pd.Series,
        vol_lookback: int = 60,
        bull_thresh: float = 0.60,
        bear_thresh: float = 0.40,
        bull_scale: float = 1.20,
        bear_scale: float = 0.80,
    ) -> None:
        self.breadth_signal = breadth_signal
        self.vol_lookback = vol_lookback
        self.bull_thresh = bull_thresh
        self.bear_thresh = bear_thresh
        self.bull_scale = bull_scale
        self.bear_scale = bear_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        breadth = self.breadth_signal.reindex(prices.index).ffill().shift(1)
        bull = (breadth > self.bull_thresh) & tlt_avail
        bear = (breadth < self.bear_thresh) & tlt_avail
        spy_scaled = w_spy.copy()
        spy_scaled = (w_spy * self.bull_scale).where(bull, spy_scaled)
        spy_scaled = (w_spy * self.bear_scale).where(bear, spy_scaled)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        spy_final = spy_scaled.where(tlt_avail, 1.0)
        tlt_final = tlt_adj.where(tlt_avail, 0.0)
        return pd.DataFrame({"SPY": spy_final, "TLT": tlt_final}, index=prices.index)


class TLTMomentumOverlay(MultiAssetStrategy):
    """H488: H021 with TLT replaced by SHY during TLT downtrends.

    TLT 6m return > 0 → maintain standard risk-parity TLT allocation.
    TLT 6m return ≤ 0 → replace TLT with SHY (reduce duration, preserve low-risk allocation).
    Preserves H021 diversification without bond duration drag in rising rate environments.
    """

    name = "tlt_momentum_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
        mom_window: int = 126,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.mom_window = mom_window

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        has_shy = "shy_close" in prices.columns
        tlt_mom = prices["tlt_close"].shift(1) / prices["tlt_close"].shift(1 + self.mom_window) - 1
        tlt_trending = (tlt_mom >= 0) | ~tlt_avail
        if has_shy:
            shy_avail = prices["shy_close"].notna()
            use_shy = ~tlt_trending & tlt_avail & shy_avail
            w_tlt_out = w_tlt.where(~use_shy, 0.0)
            w_shy_out = w_tlt.where(use_shy, 0.0)
            return pd.DataFrame(
                {"SPY": w_spy.where(tlt_avail, 1.0), "TLT": w_tlt_out, "SHY": w_shy_out},
                index=prices.index,
            )
        return pd.DataFrame(
            {"SPY": w_spy.where(tlt_avail, 1.0), "TLT": w_tlt.where(tlt_trending, w_tlt, 0.0)},
            index=prices.index,
        )


class ThreeAssetMomentumBridge(MultiAssetStrategy):
    """H489: SPY/TLT/GLD risk-parity base with 3-month momentum overlay.

    Base: H021-style inverse-vol weights across 3 assets.
    Momentum overlay: asset with highest 3m return gets weight × 1.5;
    lowest gets × 0.5; middle unchanged. Weights renormalized to sum ≤ 1.
    Bridges trend-following (H007) with multi-asset risk parity (H076).
    """

    name = "three_asset_momentum_bridge"

    def __init__(
        self,
        vol_lookback: int = 60,
        mom_window: int = 63,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.mom_window = mom_window

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        gld_r = prices["gld_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        gld_vol = gld_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        inv_gld = 1.0 / gld_vol.replace(0, np.nan)

        tlt_avail = prices["tlt_close"].notna()
        gld_avail = prices["gld_close"].notna()
        all_avail = tlt_avail & gld_avail

        total = (inv_spy + inv_tlt.where(tlt_avail, 0.0) + inv_gld.where(gld_avail, 0.0)).replace(
            0, np.nan
        )
        w_spy = (inv_spy / total).fillna(1.0)
        w_tlt = (inv_tlt.where(tlt_avail, 0.0) / total).fillna(0.0)
        w_gld = (inv_gld.where(gld_avail, 0.0) / total).fillna(0.0)

        # 3-month momentum
        spy_mom = prices["adj_close"].shift(1) / prices["adj_close"].shift(1 + self.mom_window) - 1
        tlt_mom = prices["tlt_close"].shift(1) / prices["tlt_close"].shift(1 + self.mom_window) - 1
        gld_mom = prices["gld_close"].shift(1) / prices["gld_close"].shift(1 + self.mom_window) - 1

        moms = pd.DataFrame({"SPY": spy_mom, "TLT": tlt_mom, "GLD": gld_mom})
        ranks = moms.rank(axis=1, method="first")  # 1=lowest, 3=highest
        spy_mult = ranks["SPY"].map({1: 0.5, 2: 1.0, 3: 1.5}).fillna(1.0)
        tlt_mult = ranks["TLT"].map({1: 0.5, 2: 1.0, 3: 1.5}).fillna(1.0)
        gld_mult = ranks["GLD"].map({1: 0.5, 2: 1.0, 3: 1.5}).fillna(1.0)

        w_spy_m = (w_spy * spy_mult).where(all_avail, w_spy)
        w_tlt_m = (w_tlt * tlt_mult).where(all_avail, w_tlt)
        w_gld_m = (w_gld * gld_mult).where(all_avail, w_gld)

        total_m = (w_spy_m + w_tlt_m + w_gld_m).replace(0, np.nan)
        w_spy_f = (w_spy_m / total_m).fillna(w_spy)
        w_tlt_f = (w_tlt_m / total_m).fillna(w_tlt)
        w_gld_f = (w_gld_m / total_m).fillna(w_gld)

        return pd.DataFrame({"SPY": w_spy_f, "TLT": w_tlt_f, "GLD": w_gld_f}, index=prices.index)


# ===========================================================================
# SET 7 — BATCH 5 (H491–H496): Continuous Macro Overlay AA Strategies
# ===========================================================================


class ContinuousYieldCurveScaler(MultiAssetStrategy):
    """H491: H021 with linear SPY-weight scaling from TLT-SHY return spread z-score.

    Signal: trailing 252d TLT return minus SHY return (yield-curve steepness proxy).
    Z-scored over rolling 252d window → SPY scale in [scale_lo, scale_hi].
    High z (steep curve) → equity tilt; low z (flat/inverted) → defensive.
    Continuous analog of H311's binary yield-curve signal.
    """

    name = "continuous_yield_curve_scaler"

    def __init__(
        self,
        vol_lookback: int = 60,
        spread_window: int = 252,
        scale_lo: float = 0.60,
        scale_hi: float = 1.40,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.spread_window = spread_window
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()

        tlt_ret = prices["tlt_close"].pct_change(self.spread_window)
        shy_ret = prices["shy_close"].pct_change(self.spread_window)
        spread = (tlt_ret - shy_ret).shift(1)

        spread_mean = spread.rolling(self.spread_window, min_periods=self.spread_window // 2).mean()
        spread_std = spread.rolling(self.spread_window, min_periods=self.spread_window // 2).std()
        z = (spread - spread_mean) / spread_std.replace(0, np.nan)
        z = z.clip(-2.0, 2.0).fillna(0.0)

        # z in [-2, 2] → scale in [scale_lo, scale_hi] via linear mapping
        mid = (self.scale_lo + self.scale_hi) / 2.0
        half_range = (self.scale_hi - self.scale_lo) / 2.0
        scale = (mid + z * half_range / 2.0).clip(self.scale_lo, self.scale_hi)

        spy_scaled = (w_spy * scale).clip(0.0, 1.0)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        return pd.DataFrame(
            {"SPY": spy_scaled.where(tlt_avail, 1.0), "TLT": tlt_adj.where(tlt_avail, 0.0)},
            index=prices.index,
        )


class VIXPercentileScalerPort(MultiAssetStrategy):
    """H492: H021 with linear SPY-weight scaling from VIX trailing percentile rank.

    Signal: VIX 252d trailing percentile rank (0=lowest VIX in past year, 1=highest).
    scale = 1.0 + (0.5 - vix_pct) * (scale_hi - scale_lo)
    → VIX at min (pct=0): scale=scale_hi (equity tilt).
    → VIX at median (pct=0.5): scale=1.0 (neutral).
    → VIX at max (pct=1): scale=scale_lo (defensive).
    Continuous analog of H066/H070's binary VIX threshold.
    """

    name = "vix_percentile_scaler_port"

    def __init__(
        self,
        vol_lookback: int = 60,
        pct_window: int = 252,
        scale_lo: float = 0.60,
        scale_hi: float = 1.40,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.pct_window = pct_window
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()

        vix = prices["vix_close"].shift(1)
        vix_pct = vix.rolling(self.pct_window, min_periods=self.pct_window // 2).rank(pct=True)

        scale = (
            (1.0 + (0.5 - vix_pct) * (self.scale_hi - self.scale_lo))
            .clip(self.scale_lo, self.scale_hi)
            .fillna(1.0)
        )

        spy_scaled = (w_spy * scale).clip(0.0, 1.0)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        return pd.DataFrame(
            {"SPY": spy_scaled.where(tlt_avail, 1.0), "TLT": tlt_adj.where(tlt_avail, 0.0)},
            index=prices.index,
        )


class CreditSpreadContinuousPort(MultiAssetStrategy):
    """H493: H021 with linear SPY-weight scaling from HYG-LQD return spread z-score.

    Signal: trailing 252d HYG return minus LQD return (credit spread proxy).
    Positive z → HYG outperforms → tight credit spreads → equity tilt.
    Negative z → HYG underperforms → wide spreads → defensive.
    Continuous analog of H308/H312 binary credit-spread signals.
    """

    name = "credit_spread_continuous_port"

    def __init__(
        self,
        vol_lookback: int = 60,
        spread_window: int = 252,
        scale_lo: float = 0.60,
        scale_hi: float = 1.40,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.spread_window = spread_window
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()

        hyg_ret = prices["hyg_close"].pct_change(self.spread_window)
        lqd_ret = prices["lqd_close"].pct_change(self.spread_window)
        spread = (hyg_ret - lqd_ret).shift(1)

        spread_mean = spread.rolling(self.spread_window, min_periods=self.spread_window // 2).mean()
        spread_std = spread.rolling(self.spread_window, min_periods=self.spread_window // 2).std()
        z = (spread - spread_mean) / spread_std.replace(0, np.nan)
        z = z.clip(-2.0, 2.0).fillna(0.0)

        mid = (self.scale_lo + self.scale_hi) / 2.0
        half_range = (self.scale_hi - self.scale_lo) / 2.0
        scale = (mid + z * half_range / 2.0).clip(self.scale_lo, self.scale_hi)

        spy_scaled = (w_spy * scale).clip(0.0, 1.0)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        return pd.DataFrame(
            {"SPY": spy_scaled.where(tlt_avail, 1.0), "TLT": tlt_adj.where(tlt_avail, 0.0)},
            index=prices.index,
        )


class MultiplexMacroComposite(MultiAssetStrategy):
    """H494: H021 with SPY-weight scaling from 3-signal continuous macro composite.

    Composite: average of (1) VIX percentile scaler, (2) yield-curve z-score scaler,
    (3) credit-spread z-score scaler. Each maps to [scale_lo, scale_hi]; average applied.
    Avoids binary-threshold whipsaw by keeping all signals continuous.
    """

    name = "multiplex_macro_composite"

    def __init__(
        self,
        vol_lookback: int = 60,
        signal_window: int = 252,
        scale_lo: float = 0.60,
        scale_hi: float = 1.40,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.signal_window = signal_window
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi

    def _compute_scale(self, z: pd.Series) -> pd.Series:
        z = z.clip(-2.0, 2.0).fillna(0.0)
        mid = (self.scale_lo + self.scale_hi) / 2.0
        half_range = (self.scale_hi - self.scale_lo) / 2.0
        return (mid + z * half_range / 2.0).clip(self.scale_lo, self.scale_hi)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        w = self.signal_window
        min_p = w // 2

        # (1) VIX percentile scaler
        vix = prices["vix_close"].shift(1)
        vix_pct = vix.rolling(w, min_periods=min_p).rank(pct=True)
        scale_vix = (
            (1.0 + (0.5 - vix_pct) * (self.scale_hi - self.scale_lo))
            .clip(self.scale_lo, self.scale_hi)
            .fillna(1.0)
        )

        # (2) Yield curve z-score scaler
        tlt_ret = prices["tlt_close"].pct_change(w)
        shy_ret = prices["shy_close"].pct_change(w)
        yc_spread = (tlt_ret - shy_ret).shift(1)
        yc_mean = yc_spread.rolling(w, min_periods=min_p).mean()
        yc_std = yc_spread.rolling(w, min_periods=min_p).std()
        yc_z = (yc_spread - yc_mean) / yc_std.replace(0, np.nan)
        scale_yc = self._compute_scale(yc_z)

        # (3) Credit spread z-score scaler
        hyg_ret = prices["hyg_close"].pct_change(w)
        lqd_ret = prices["lqd_close"].pct_change(w)
        cs_spread = (hyg_ret - lqd_ret).shift(1)
        cs_mean = cs_spread.rolling(w, min_periods=min_p).mean()
        cs_std = cs_spread.rolling(w, min_periods=min_p).std()
        cs_z = (cs_spread - cs_mean) / cs_std.replace(0, np.nan)
        scale_cs = self._compute_scale(cs_z)

        # Average: weight VIX and YC equally; CS falls back to neutral when HYG unavailable
        cs_avail = prices["hyg_close"].notna() & prices["lqd_close"].notna()
        scale_cs_safe = scale_cs.where(cs_avail, 1.0)
        scale = (scale_vix + scale_yc + scale_cs_safe) / 3.0

        spy_scaled = (w_spy * scale).clip(0.0, 1.0)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        return pd.DataFrame(
            {"SPY": spy_scaled.where(tlt_avail, 1.0), "TLT": tlt_adj.where(tlt_avail, 0.0)},
            index=prices.index,
        )


class GLDMomentumOverlay(MultiAssetStrategy):
    """H495: SPY/TLT/GLD 3-asset parity with GLD 6m momentum tilt.

    Base: H076-style inverse-vol weights across SPY, TLT, GLD.
    GLD 6m return > 0 → GLD weight × mom_up_mult (default 1.2); renormalize.
    GLD 6m return ≤ 0 → GLD weight × mom_dn_mult (default 0.5), floor at gld_floor.
    Tests whether GLD trend improves H076 multi-asset parity.
    """

    name = "gld_momentum_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
        mom_window: int = 126,
        mom_up_mult: float = 1.2,
        mom_dn_mult: float = 0.5,
        gld_floor: float = 0.10,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.mom_window = mom_window
        self.mom_up_mult = mom_up_mult
        self.mom_dn_mult = mom_dn_mult
        self.gld_floor = gld_floor

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        gld_r = prices["gld_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        gld_vol = gld_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        tlt_avail = prices["tlt_close"].notna()
        gld_avail = prices["gld_close"].notna()

        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        inv_gld = 1.0 / gld_vol.replace(0, np.nan)

        total_base = (
            inv_spy + inv_tlt.where(tlt_avail, 0.0) + inv_gld.where(gld_avail, 0.0)
        ).replace(0, np.nan)
        w_spy = (inv_spy / total_base).fillna(1.0)
        w_tlt = (inv_tlt.where(tlt_avail, 0.0) / total_base).fillna(0.0)
        w_gld = (inv_gld.where(gld_avail, 0.0) / total_base).fillna(0.0)

        gld_mom = prices["gld_close"].shift(1) / prices["gld_close"].shift(1 + self.mom_window) - 1
        gld_up = (gld_mom >= 0) & gld_avail

        gld_mult = pd.Series(self.mom_up_mult, index=prices.index).where(gld_up, self.mom_dn_mult)
        w_gld_adj = (w_gld * gld_mult).where(gld_avail, 0.0)
        # Apply floor when GLD is down-trending
        w_gld_floor = pd.Series(self.gld_floor, index=prices.index).where(gld_avail & ~gld_up, 0.0)
        w_gld_adj = w_gld_adj.clip(lower=w_gld_floor)

        total_adj = (w_spy + w_tlt + w_gld_adj).replace(0, np.nan)
        w_spy_f = (w_spy / total_adj).fillna(w_spy)
        w_tlt_f = (w_tlt / total_adj).fillna(w_tlt)
        w_gld_f = (w_gld_adj / total_adj).fillna(0.0)

        return pd.DataFrame({"SPY": w_spy_f, "TLT": w_tlt_f, "GLD": w_gld_f}, index=prices.index)


class VIXMeanReversionPort(MultiAssetStrategy):
    """H496: Contrarian VIX overlay on H021 — buy when VIX is historically extreme.

    Signal: VIX / VIX_252d_MA ratio.
    ratio ≥ high_thresh (default 1.5): extreme fear → SPY weight × scale_hi (contrarian buy).
    ratio < low_thresh (default 0.7): extreme calm → SPY normal (scale 1.0).
    Between thresholds: linear interpolation.
    Tests if VIX mean reversion adds value vs H066/H492 (which penalize high VIX).
    """

    name = "vix_mean_reversion_port"

    def __init__(
        self,
        vol_lookback: int = 60,
        vix_ma_window: int = 252,
        low_thresh: float = 0.70,
        high_thresh: float = 1.50,
        scale_lo: float = 1.00,
        scale_hi: float = 1.20,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vix_ma_window = vix_ma_window
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()

        vix = prices["vix_close"].shift(1)
        vix_ma = vix.rolling(self.vix_ma_window, min_periods=self.vix_ma_window // 2).mean()
        ratio = (vix / vix_ma.replace(0, np.nan)).clip(0.0, 3.0).fillna(1.0)

        # Linear ramp: ratio ≤ low_thresh → scale_lo; ratio ≥ high_thresh → scale_hi
        t_lo, t_hi = self.low_thresh, self.high_thresh
        scale = np.where(
            ratio >= t_hi,
            self.scale_hi,
            np.where(
                ratio < t_lo,
                self.scale_lo,
                self.scale_lo + (ratio - t_lo) / (t_hi - t_lo) * (self.scale_hi - self.scale_lo),
            ),
        )
        scale = pd.Series(scale, index=prices.index)

        spy_scaled = (w_spy * scale).clip(0.0, 1.0)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        return pd.DataFrame(
            {"SPY": spy_scaled.where(tlt_avail, 1.0), "TLT": tlt_adj.where(tlt_avail, 0.0)},
            index=prices.index,
        )


# ---------------------------------------------------------------------------
# H571: QQQTLTRiskParity
# ---------------------------------------------------------------------------


class QQQTLTRiskParity(MultiAssetStrategy):
    """H571: QQQ/TLT inverse-vol risk parity.

    Identical mechanism to H021 (RiskParity) but uses QQQ (Nasdaq-100) instead
    of SPY for the equity leg. QQQ has higher historical CAGR and deeper crashes
    than SPY, giving TLT a more valuable hedge role. QQQ available from 1999-03;
    folds 0-1 mostly cash (no QQQ+TLT overlap until fold 2).
    """

    name = "qqq_tlt_risk_parity"

    def __init__(self, vol_lookback: int = 60) -> None:
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        qqq_r = prices["qqq_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        qqq_vol = qqq_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        inv_qqq = 1.0 / qqq_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_qqq + inv_tlt

        w_qqq = (inv_qqq / total).fillna(0.0)
        w_tlt = (inv_tlt / total).fillna(0.0)

        return pd.DataFrame({"QQQ": w_qqq, "TLT": w_tlt}, index=prices.index)


# ---------------------------------------------------------------------------
# H572: IWMTLTRiskParity
# ---------------------------------------------------------------------------


class IWMTLTRiskParity(MultiAssetStrategy):
    """H572: IWM/TLT inverse-vol risk parity.

    H021-style risk parity but using IWM (Russell 2000 small-cap) instead of
    SPY. Small-cap has lower correlation with TLT than large-cap in normal
    regimes; deeper crashes in recessions. IWM available from 2000-05;
    folds 0-1 mostly cash.
    """

    name = "iwm_tlt_risk_parity"

    def __init__(self, vol_lookback: int = 60) -> None:
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        iwm_r = prices["iwm_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        iwm_vol = iwm_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        inv_iwm = 1.0 / iwm_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_iwm + inv_tlt

        w_iwm = (inv_iwm / total).fillna(0.0)
        w_tlt = (inv_tlt / total).fillna(0.0)

        return pd.DataFrame({"IWM": w_iwm, "TLT": w_tlt}, index=prices.index)


# ---------------------------------------------------------------------------
# H573: QQQIWMTLTRiskParity
# ---------------------------------------------------------------------------


class QQQIWMTLTRiskParity(MultiAssetStrategy):
    """H573: QQQ/IWM/TLT three-asset inverse-vol risk parity.

    Growth (QQQ) + value/small-cap (IWM) + bonds (TLT). Gracefully degrades
    to fewer assets when data is unavailable. Both QQQ (1999-03) and IWM
    (2000-05) start after TLT (2002-07); folds 0-2 mostly cash.
    """

    name = "qqq_iwm_tlt_risk_parity"

    def __init__(self, vol_lookback: int = 60) -> None:
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {"QQQ": "qqq_close", "IWM": "iwm_close", "TLT": "tlt_close"}
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        vols: dict[str, pd.Series] = {}
        for asset, col in available.items():
            r = prices[col].pct_change()
            vols[asset] = r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
                1
            ) * np.sqrt(252)

        inv_vols = {a: 1.0 / v.replace(0, np.nan) for a, v in vols.items()}
        total = sum(inv_vols.values())
        weights = pd.DataFrame(index=prices.index, columns=list(available.keys()), dtype=float)
        for asset in available:
            weights[asset] = (inv_vols[asset] / total).fillna(0.0)

        return weights


# ---------------------------------------------------------------------------
# H578: VNQTLTRiskParity
# ---------------------------------------------------------------------------


class VNQTLTRiskParity(MultiAssetStrategy):
    """H578: VNQ/TLT inverse-vol risk parity.

    REITs (VNQ) paired with long-duration Treasuries (TLT). REITs benefit from
    falling rates like TLT but also from rising real estate values (anti-TLT).
    The ambiguous VNQ-TLT correlation may produce a distinctive risk profile.
    VNQ available from 2004-09; folds 0-3 mostly cash.
    """

    name = "vnq_tlt_risk_parity"

    def __init__(self, vol_lookback: int = 60) -> None:
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        vnq_r = prices["vnq_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        vnq_vol = vnq_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        inv_vnq = 1.0 / vnq_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_vnq + inv_tlt

        w_vnq = (inv_vnq / total).fillna(0.0)
        w_tlt = (inv_tlt / total).fillna(0.0)

        return pd.DataFrame({"VNQ": w_vnq, "TLT": w_tlt}, index=prices.index)


# ---------------------------------------------------------------------------
# H577: SectorMomentumVolTarget
# ---------------------------------------------------------------------------


class SectorMomentumVolTarget(MultiAssetStrategy):
    """H577: Sector momentum with vol-targeting overlay.

    Ranks equity ETFs (XLP, XLU, IVE, IVW, QQQ, IWM, VNQ) by 6-month
    momentum; holds top-N risk-parity weighted. Applies vol-targeting:
    scales total exposure by min(1, target_vol/realized_vol). Pure momentum
    signal without regime conditioning (contrast with H063/H065).
    """

    name = "sector_momentum_vol_target"

    _ETFS = ["XLP", "XLU", "IVE", "IVW", "QQQ", "IWM", "VNQ"]

    def __init__(
        self,
        mom_window: int = 126,
        vol_lookback: int = 60,
        top_n: int = 3,
        target_vol: float = 0.12,
    ) -> None:
        self.mom_window = mom_window
        self.vol_lookback = vol_lookback
        self.top_n = top_n
        self.target_vol = target_vol

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        col_map = {a: f"{a.lower()}_close" for a in self._ETFS}
        available = {a: col for a, col in col_map.items() if col in prices.columns}

        if not available:
            return pd.DataFrame({"SPY": pd.Series(1.0, index=prices.index)})

        mom_df = pd.DataFrame(
            {a: prices[col].pct_change(self.mom_window) for a, col in available.items()}
        )
        ret_df = pd.DataFrame({a: prices[col].pct_change() for a, col in available.items()})
        vol_df = ret_df.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std()
        inv_vol_df = vol_df.replace(0, np.nan).rdiv(1).fillna(1.0)

        mom_rank = mom_df.rank(axis=1, ascending=False, na_option="keep")
        top_mask = (mom_rank <= self.top_n) & mom_rank.notna()

        raw = inv_vol_df.where(top_mask, 0.0)
        total = raw.sum(axis=1).replace(0, np.nan)
        etf_w = raw.div(total, axis=0).fillna(0.0)

        # Vol-targeting: scale total ETF exposure
        port_r = (etf_w.shift(1) * ret_df).sum(axis=1)
        port_vol = port_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        scale = (self.target_vol / port_vol.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)

        etf_w_scaled = etf_w.multiply(scale, axis=0)

        no_etf = top_mask.sum(axis=1) == 0
        etf_w_scaled["SPY"] = no_etf.astype(float)

        return etf_w_scaled.shift(1).fillna(0.0)
