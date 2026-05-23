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
