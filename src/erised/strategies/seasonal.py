"""Calendar and seasonality strategies (Sell-in-May, turn of month, Halloween, holidays)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import _calendar_month_ranks, _calendar_quarter_rank_end, _h021_weights, _vrp_series
from .base import MultiAssetStrategy, Strategy


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
