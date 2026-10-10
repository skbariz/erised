"""Earnings and fundamental-growth signals: post-earnings drift, surprises, and revenue growth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import (
    _align_fund,
    _cs_zscore,
    _filing_window,
    _monthly_rebalance,
    _periodic_rebalance,
    _sector_neutral_weights,
)
from .base import CrossSectionalStrategy


class EarningsTrendLong(CrossSectionalStrategy):
    """H180: Long top quintile by YoY net income growth rate, sector-neutral.

    SUE proxy = (NI_t - NI_{t-252bd}) / |NI_{t-252bd}|. Uses EDGAR filing-date
    forward-filled data (point-in-time). Captures post-filing earnings drift.
    """

    name = "earnings_trend_long"

    def __init__(
        self,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        yoy_window: int = 252,
        monthly_rebalance: bool = True,
    ):
        self.net_income = net_income
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.yoy_window = yoy_window
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ni_lag = ni.shift(self.yoy_window)
        sue = (ni - ni_lag) / ni_lag.abs().replace(0, np.nan)
        daily_w = _sector_neutral_weights(
            sue, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class RevenueEarningsDoubleGrowth(CrossSectionalStrategy):
    """H181: Long stocks where BOTH net income AND revenue grew YoY.

    Jegadeesh-Livnat double-confirmation proxy using EDGAR data.
    Signal = (z_ni_growth + z_rev_growth)/2, applied only where both > 0.
    """

    name = "revenue_earnings_double_growth"

    def __init__(
        self,
        net_income: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        yoy_window: int = 252,
        monthly_rebalance: bool = True,
    ):
        self.net_income = net_income
        self.revenue = revenue
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.yoy_window = yoy_window
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)

        ni_lag = ni.shift(self.yoy_window)
        rev_lag = rev.shift(self.yoy_window)
        ni_growth = (ni - ni_lag) / ni_lag.abs().replace(0, np.nan)
        rev_growth = (rev - rev_lag) / rev_lag.abs().replace(0, np.nan)

        # Only score stocks where both components grew
        both_pos = (ni_growth > 0) & (rev_growth > 0)
        composite = ((_cs_zscore(ni_growth) + _cs_zscore(rev_growth)) / 2.0).where(both_pos)

        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class FilingDateGapContinuation(CrossSectionalStrategy):
    """H182: Long stocks with a positive price reaction on a recent EDGAR filing date.

    Detects new filings as the day net_income first changes in the PIT panel.
    If the price returned > gap_threshold on that day, holds the stock for
    hold_days trading days (post-filing drift).
    """

    name = "filing_date_gap_continuation"

    def __init__(
        self,
        net_income: pd.DataFrame,
        gap_threshold: float = 0.02,
        hold_days: int = 20,
        within_sector_quantile: float = 0.20,
    ):
        self.net_income = net_income
        self.gap_threshold = gap_threshold
        self.hold_days = hold_days
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        # Filing days: value changed from prior day or first appearance
        is_new_filing = (ni.diff().abs() > 0) | (ni.notna() & ni.shift(1).isna())
        daily_rets = prices_panel.pct_change()
        # Price return on filing day; NaN on non-filing days
        filing_rets = daily_rets.where(is_new_filing)
        # Positive filing reaction in last hold_days days
        has_pos = (filing_rets > self.gap_threshold).rolling(self.hold_days, min_periods=1).max()
        # Equal weight all stocks with recent positive filing reaction
        mem = membership.reindex_like(filing_rets).fillna(False)
        selected = (has_pos > 0) & mem
        count = selected.sum(axis=1).replace(0, np.nan)
        return selected.astype(float).div(count, axis=0).fillna(0.0)


class EarningsAcceleration(CrossSectionalStrategy):
    """H198: Earnings acceleration -- change in YoY NI growth rate.

    Signal = sue_t - sue_{t - accel_window} where sue = YoY NI growth rate.
    Stocks with accelerating earnings growth (increasing SUE) should outperform.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "earnings_acceleration"

    def __init__(
        self,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        yoy_window: int = 252,
        accel_window: int = 63,
        within_sector_quantile: float = 0.20,
    ):
        self.net_income = net_income
        self.sectors = sectors
        self.yoy_window = yoy_window
        self.accel_window = accel_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ni_lag = ni.shift(self.yoy_window)
        sue = (ni - ni_lag) / ni_lag.abs().replace(0, np.nan)
        acceleration = sue - sue.shift(self.accel_window)
        daily_w = _sector_neutral_weights(
            acceleration,
            membership,
            self.sectors,
            self.within_sector_quantile,
            ascending=False,
        )
        return _monthly_rebalance(daily_w, prices_panel)


class RevenueGrowth(CrossSectionalStrategy):
    """H228: YoY revenue growth, sector-neutral top 20%."""

    name = "revenue_growth"

    def __init__(
        self,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        growth_window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.revenue = revenue
        self.sectors = sectors
        self.growth_window = growth_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        rev_lag = rev.shift(self.growth_window)
        rev_growth = (rev - rev_lag) / rev_lag.where(rev_lag > 0)
        rev_growth = rev_growth.clip(-1.0, 3.0)
        daily_w = _sector_neutral_weights(
            rev_growth, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class GrossProfitImprovementFactor(CrossSectionalStrategy):
    """H281: Long top-quintile by YoY gross profit dollar growth, sector-neutral.

    Distinct from GP/revenue level (H266), GP margin change (H267), GP/TA (H268),
    and NI change (H180). Tests whether GP dollar expansion predicts returns.
    """

    name = "gross_profit_improvement_factor"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        yoy_window: int = 252,
    ):
        self.gross_profit = gross_profit
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.yoy_window = yoy_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        gp_lag = gp.shift(self.yoy_window)
        gp_change = (gp - gp_lag) / gp_lag.abs().replace(0, np.nan)
        gp_change = gp_change.clip(-5, 5)
        daily_w = _sector_neutral_weights(
            gp_change, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class RevenueGrowthFactor(CrossSectionalStrategy):
    """H282: Long top-quintile by YoY revenue dollar growth, sector-neutral.

    Pure top-line growth signal. Distinct from H181 (requires NI growth too),
    H267 (GP margin change), and H281 (GP dollar growth, not total revenue).
    """

    name = "revenue_growth_factor"

    def __init__(
        self,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        yoy_window: int = 252,
    ):
        self.revenue = revenue
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.yoy_window = yoy_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        rev_lag = rev.shift(self.yoy_window)
        rev_change = (rev - rev_lag) / rev_lag.abs().replace(0, np.nan)
        rev_change = rev_change.clip(-5, 5)
        daily_w = _sector_neutral_weights(
            rev_change, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class OCFGrowthFactor(CrossSectionalStrategy):
    """H283: Long top-quintile by YoY operating cash flow growth, sector-neutral.

    Cash flow growth as standalone signal. Distinct from H183 (OCF/NI ratio as
    filter on NI change) and H281/H282 (GP and revenue growth). Tests whether
    OCF expansion itself — not cash backing of earnings — predicts returns.
    """

    name = "ocf_growth_factor"

    def __init__(
        self,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        yoy_window: int = 252,
    ):
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.yoy_window = yoy_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ocf = _align_fund(self.operating_cf, prices_panel)
        ocf_lag = ocf.shift(self.yoy_window)
        ocf_change = (ocf - ocf_lag) / ocf_lag.abs().replace(0, np.nan)
        ocf_change = ocf_change.clip(-5, 5)
        daily_w = _sector_neutral_weights(
            ocf_change, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class FundamentalMomentumComposite(CrossSectionalStrategy):
    """H285: z(GP_change) + z(NI_change) + z(6m_momentum) — earnings attention composite.

    Combines GP improvement (H281), NI improvement (H180 analog), and 6m price
    confirmation. Distinct from H184 (NI+63d_mom only). Tests whether adding GP
    growth and using medium-term momentum amplifies earnings drift.
    """

    name = "fundamental_momentum_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        yoy_window: int = 252,
        mom_window: int = 126,
    ):
        self.gross_profit = gross_profit
        self.net_income = net_income
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.yoy_window = yoy_window
        self.mom_window = mom_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)

        gp_lag = gp.shift(self.yoy_window)
        ni_lag = ni.shift(self.yoy_window)
        gp_change = (gp - gp_lag) / gp_lag.abs().replace(0, np.nan)
        ni_change = (ni - ni_lag) / ni_lag.abs().replace(0, np.nan)
        gp_change = gp_change.clip(-5, 5)
        ni_change = ni_change.clip(-5, 5)

        mom_6m = prices_panel / prices_panel.shift(self.mom_window) - 1

        score = _cs_zscore(gp_change) + _cs_zscore(ni_change) + _cs_zscore(mom_6m)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class EarningsWindowQualityBoost(CrossSectionalStrategy):
    """H376: GP/TA quality rank with 1.5x weight boost in post-filing window.

    Within 5 trading days of an EDGAR filing (detected from NI panel change),
    boost position weight by filing_boost factor. Outside window: base_scale.
    Engelberg-McLean-Pontiff 2018: anomaly payoffs concentrated around news.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "earnings_window_quality_boost"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        filing_window_days: int = 5,
        filing_boost: float = 1.5,
        base_scale: float = 0.8,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.sectors = sectors
        self.filing_window_days = filing_window_days
        self.filing_boost = filing_boost
        self.base_scale = base_scale
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)

        in_event = _filing_window(ni, self.filing_window_days)
        scale = in_event.astype(float) * (self.filing_boost - self.base_scale) + self.base_scale

        base_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        scaled_w = (base_w * scale).clip(upper=1.0)
        row_sum = scaled_w.sum(axis=1).replace(0, np.nan)
        scaled_w = scaled_w.div(row_sum, axis=0).fillna(0.0)
        return _periodic_rebalance(scaled_w, self.rebalance_period)


class EarningsSurpriseDrift(CrossSectionalStrategy):
    """H377: Post-earnings announcement drift scaled by earnings surprise magnitude.

    Surprise = |delta_NI / total_assets| at most recent filing date.
    Position sized by surprise rank: larger surprises get proportionally larger
    weight when NI change is positive. Engelberg-McLean-Pontiff 2018.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "earnings_surprise_drift"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ta_safe = ta.where(ta > 0)

        delta_ni = ni.diff()
        surprise_raw = (delta_ni / ta_safe).clip(-0.5, 0.5)
        positive_surprise = surprise_raw.where(surprise_raw > 0, 0.0)

        daily_w = _sector_neutral_weights(
            positive_surprise,
            membership,
            self.sectors,
            self.within_sector_quantile,
            ascending=False,
        )
        return _monthly_rebalance(daily_w, prices_panel)


class QuietPeriodMomentum(CrossSectionalStrategy):
    """H378: 12-1 momentum reduced to zero in post-filing window.

    During the 5 trading days after an EDGAR filing (from NI panel changes),
    set position weight to zero for that stock and redistribute to remaining
    members. Reduces gap risk around earnings. Engelberg-McLean-Pontiff 2018.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "quiet_period_momentum"

    def __init__(
        self,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        filing_window_days: int = 5,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.net_income = net_income
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_window = skip_window
        self.filing_window_days = filing_window_days
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        mom = prices_panel.pct_change(self.mom_window).shift(self.skip_window)
        in_event = _filing_window(ni, self.filing_window_days)
        quiet_mom = mom.where(~in_event, np.nan)
        daily_w = _sector_neutral_weights(
            quiet_mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class EarningsConfirmedQuality(CrossSectionalStrategy):
    """H379: GP/TA quality rank filtered by non-negative NI change at last filing.

    Only hold GP-ranked stocks where the most recent EDGAR filing showed
    NI change >= 0 (positive earnings confirmation). Exit when NI disappoints.
    Tetlock et al. 2008: positive news tone confirms fundamental quality.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "earnings_confirmed_quality"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        ni_confirmed = ni.diff() >= 0
        gated_gp = gp_ta.where(ni_confirmed, np.nan)
        daily_w = _sector_neutral_weights(
            gated_gp, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class EventAwareComposite(CrossSectionalStrategy):
    """H380: Equal-weight composite of batch-6 earnings-event-aware signals.

    Blends: GP quality (baseline), earnings-confirmed quality, quiet-period
    momentum, positive-surprise drift. Each signal clipped before combining.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "event_aware_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        clip_val: float = 2.0,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.sectors = sectors
        self.clip_val = clip_val
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        ni_confirmed = ni.diff() >= 0
        delta_ni = ni.diff()
        positive_surprise = (delta_ni / ta_safe).clip(0.0, 0.5)
        mom = prices_panel.pct_change(252).shift(21)
        in_event = _filing_window(ni, 5)
        quiet_mom = mom.where(~in_event, np.nan)
        confirmed_gp = gp_ta.where(ni_confirmed, np.nan)

        z1 = _cs_zscore(gp_ta).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z2 = _cs_zscore(confirmed_gp).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z3 = _cs_zscore(positive_surprise).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z4 = _cs_zscore(quiet_mom).clip(-self.clip_val, self.clip_val).fillna(0.0)

        score = z1 + z2 + z3 + z4
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CapitalIntensityAvoidPort(CrossSectionalStrategy):
    """H522: (TA - CA) / Revenue = fixed asset intensity. Long lowest (asset-light). Quarterly.

    Long firms with lowest fixed-capital input per revenue dollar.
    Asset-light firms (software, services) generate more revenue per unit of structural capacity.
    Input-side complement to H452 (GP/(TA-CA) = output efficiency per fixed asset).
    """

    name = "capital_intensity_avoid_port"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
        self.current_assets = current_assets
        self.revenue = revenue
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel).clip(lower=0.0)
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        fixed_assets = (ta - ca).clip(lower=0.0)
        intensity = (fixed_assets / rev_safe).clip(0.0, 20.0)
        # ascending=True: lower intensity (asset-light) is better
        daily_w = _sector_neutral_weights(
            intensity, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class OCFRevenueGrowthCombo(CrossSectionalStrategy):
    """H528: z(OCF/Revenue level) + z(Revenue YoY growth). 2-factor z-score. Quarterly.

    Cash margin (OCF/Revenue captures full cash conversion stack) + revenue growth.
    Extends H233 (z(rev_growth)+z(GP/TA): 0.803) by replacing quality with OCF margin.
    OCF/Revenue is post-working-capital unlike GP/Revenue -- more conservative cash view.
    """

    name = "ocf_revenue_growth_combo"

    def __init__(
        self,
        operating_cf: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.operating_cf = operating_cf
        self.revenue = revenue
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ocf = _align_fund(self.operating_cf, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        ocf_margin = (ocf / rev_safe).clip(-2.0, 5.0)
        rev_growth = (rev / rev.shift(252) - 1).clip(-1.0, 5.0)
        composite = _cs_zscore(ocf_margin).fillna(0.0) + _cs_zscore(rev_growth).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)
