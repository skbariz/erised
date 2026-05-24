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
