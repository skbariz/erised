"""Value factors: book-to-market, earnings yield, and mispricing scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import (
    _align_fund,
    _buy_hold_band,
    _cs_rank_norm,
    _cs_zscore,
    _equal_weight_top_quantile,
    _monthly_rebalance,
    _periodic_rebalance,
    _sector_neutral_weights,
)
from .base import CrossSectionalStrategy


class LargeCapValue(CrossSectionalStrategy):
    """H160: Long top quintile by composite value: 0.5×rank(B/M) + 0.5×rank(E/P).

    B/M = book_equity / (price × shares_outstanding).
    E/P = net_income / (price × shares_outstanding). NaN for negative earnings.
    Fundamental panels are point-in-time (filed_date forward-fill).
    """

    name = "large_cap_value"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        net_income: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        top_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.book_equity = book_equity
        self.net_income = net_income
        self.shares_outstanding = shares_outstanding
        self.top_quantile = top_quantile
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        be = _align_fund(self.book_equity, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)

        mktcap = prices_panel * sh
        bm = (be / mktcap).where(be > 0)  # exclude negative book equity
        ep = (ni / mktcap).where(ni > 0)  # exclude negative earnings

        composite = 0.5 * _cs_rank_norm(bm) + 0.5 * _cs_rank_norm(ep)
        daily_w = _equal_weight_top_quantile(
            composite, membership, self.top_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class BookToMarketMomentum(CrossSectionalStrategy):
    """H226: z(book_equity/price) + z(12-1m momentum) -- value-momentum composite."""

    name = "book_to_market_momentum"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_days: int = 21,
        within_sector_quantile: float = 0.20,
    ):
        self.book_equity = book_equity
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_days = skip_days
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        be = _align_fund(self.book_equity, prices_panel)
        # B/M proxy: book_equity / price (price-normalized across stocks)
        bm = be / prices_panel.where(prices_panel > 0)
        price_ret = prices_panel.pct_change(self.mom_window - self.skip_days).shift(self.skip_days)
        score = _cs_zscore(bm) + _cs_zscore(price_ret)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class FCFYield(CrossSectionalStrategy):
    """H241: Free cash flow yield = OCF / market_cap (price × shares_outstanding)."""

    name = "fcf_yield"

    def __init__(
        self,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.operating_cf = operating_cf
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ocf = _align_fund(self.operating_cf, prices_panel)
        shares = _align_fund(self.shares_outstanding, prices_panel)
        market_cap = prices_panel * shares
        fcf_yield = ocf / market_cap.where(market_cap > 0)
        fcf_yield = fcf_yield.clip(-0.5, 0.5)
        daily_w = _sector_neutral_weights(
            fcf_yield, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class LowTurnoverValueFactor(CrossSectionalStrategy):
    """H291: Earnings yield with buy/hold band (buy top-20%, hold until top-40% exit).

    Novy-Marx & Velikov (2016): value factors have heavy churn. Band reduces
    round-trip costs by ~62% while maintaining similar gross exposure.
    """

    name = "low_turnover_value_factor"

    def __init__(
        self,
        net_income: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        buy_quantile: float = 0.20,
        hold_quantile: float = 0.40,
        rebalance_period: int = 21,
    ):
        self.net_income = net_income
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.buy_quantile = buy_quantile
        self.hold_quantile = hold_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        market_cap = prices_panel * sh
        ey = ni / market_cap.replace(0, np.nan)

        strict_w = _sector_neutral_weights(
            ey, membership, self.sectors, self.buy_quantile, ascending=False
        )
        extended_w = _sector_neutral_weights(
            ey, membership, self.sectors, self.hold_quantile, ascending=False
        )
        return _buy_hold_band(strict_w, extended_w, membership, self.rebalance_period)


class ManagementMispricing(CrossSectionalStrategy):
    """H356: Management-related mispricing composite (Stambaugh-Yuan 2017).

    z(-accruals) + z(-asset_growth) + z(-dilution).
    Proxies management actions that reduce shareholder value.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "management_mispricing"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(self.lookback)) / 2).where(ta > 0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(self.lookback)) / ta_safe.shift(self.lookback)).clip(
            -1.0, 5.0
        )
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        score = (
            _cs_zscore(-accruals).fillna(0.0)
            + _cs_zscore(-asset_growth).fillna(0.0)
            + _cs_zscore(-dilution).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class PerformanceMispricing(CrossSectionalStrategy):
    """H357: Performance-related mispricing composite (Stambaugh-Yuan 2017).

    z(ROA) + z(delta_ROA) + z(OCF/TA).
    Proxies improving operating performance being under-priced by the market.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "performance_mispricing"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        roa = (ni / ta_safe).clip(-1.0, 2.0)
        delta_roa = (roa - roa.shift(self.lookback)).clip(-1.0, 1.0)
        ocf_ta = (ocf / ta_safe).clip(-1.0, 2.0)
        score = (
            _cs_zscore(roa).fillna(0.0)
            + _cs_zscore(delta_roa).fillna(0.0)
            + _cs_zscore(ocf_ta).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MispricingCombo(CrossSectionalStrategy):
    """H358: 50/50 blend of management and performance mispricing composites.

    Combines Stambaugh-Yuan's two mispricing clusters. Tests whether the two
    clusters together are more robust than either alone.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "mispricing_combo"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(self.lookback)) / 2).where(ta > 0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(self.lookback)) / ta_safe.shift(self.lookback)).clip(
            -1.0, 5.0
        )
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        mgmt = (
            _cs_zscore(-accruals).fillna(0.0)
            + _cs_zscore(-asset_growth).fillna(0.0)
            + _cs_zscore(-dilution).fillna(0.0)
        )
        roa = (ni / ta_safe).clip(-1.0, 2.0)
        delta_roa = (roa - roa.shift(self.lookback)).clip(-1.0, 1.0)
        ocf_ta = (ocf / ta_safe).clip(-1.0, 2.0)
        perf = (
            _cs_zscore(roa).fillna(0.0)
            + _cs_zscore(delta_roa).fillna(0.0)
            + _cs_zscore(ocf_ta).fillna(0.0)
        )
        score = 0.5 * _cs_zscore(mgmt).fillna(0.0) + 0.5 * _cs_zscore(perf).fillna(0.0)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MispricingAvoidFilter(CrossSectionalStrategy):
    """H359: Equal-weight S&P 500 excluding worst-decile combined mispricing.

    Long-only avoid strategy: exclude the bottom 10% by combined mispricing score
    (management + performance), hold remaining equal-weight.
    Tests whether mispricing avoidance alone adds value. Quarterly rebalance.
    """

    name = "mispricing_avoid_filter"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        exclude_pct: float = 0.10,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.lookback = lookback
        self.exclude_pct = exclude_pct
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(self.lookback)) / 2).where(ta > 0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(self.lookback)) / ta_safe.shift(self.lookback)).clip(
            -1.0, 5.0
        )
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        roa = (ni / ta_safe).clip(-1.0, 2.0)
        delta_roa = (roa - roa.shift(self.lookback)).clip(-1.0, 1.0)
        ocf_ta = (ocf / ta_safe).clip(-1.0, 2.0)
        combined = (
            _cs_zscore(-accruals).fillna(0.0)
            + _cs_zscore(-asset_growth).fillna(0.0)
            + _cs_zscore(-dilution).fillna(0.0)
            + _cs_zscore(roa).fillna(0.0)
            + _cs_zscore(delta_roa).fillna(0.0)
            + _cs_zscore(ocf_ta).fillna(0.0)
        )
        mem = membership.reindex_like(combined).fillna(False)
        n_members = mem.sum(axis=1).replace(0, np.nan)
        excl_n = (n_members * self.exclude_pct).clip(lower=1)
        ranks_asc = combined.where(mem).rank(axis=1, ascending=True, na_option="keep")
        excluded = ranks_asc.le(excl_n, axis=0) & ranks_asc.notna()
        included = mem & ~excluded
        n_inc = included.sum(axis=1).replace(0, np.nan)
        daily_w = included.astype(float).div(n_inc, axis=0).fillna(0.0)
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MispricingQualityComposite(CrossSectionalStrategy):
    """H360: Mispricing combo (H358) filtered by GP/TA quality gate.

    Requires GP/TA > median within membership before applying mispricing rank.
    Tests whether quality x mispricing interaction outperforms either alone.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "mispricing_quality_composite"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        gross_profit: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.shares_outstanding = shares_outstanding
        self.gross_profit = gross_profit
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(self.lookback)) / 2).where(ta > 0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(self.lookback)) / ta_safe.shift(self.lookback)).clip(
            -1.0, 5.0
        )
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        roa = (ni / ta_safe).clip(-1.0, 2.0)
        delta_roa = (roa - roa.shift(self.lookback)).clip(-1.0, 1.0)
        ocf_ta = (ocf / ta_safe).clip(-1.0, 2.0)
        mispricing = (
            _cs_zscore(-accruals).fillna(0.0)
            + _cs_zscore(-asset_growth).fillna(0.0)
            + _cs_zscore(-dilution).fillna(0.0)
            + _cs_zscore(roa).fillna(0.0)
            + _cs_zscore(delta_roa).fillna(0.0)
            + _cs_zscore(ocf_ta).fillna(0.0)
        )
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        mem = membership.reindex_like(gp_ta).fillna(False)
        gp_median = gp_ta.where(mem).median(axis=1)
        quality_gate = gp_ta.ge(gp_median, axis=0) & mem
        gated = mispricing.where(quality_gate, np.nan)
        daily_w = _sector_neutral_weights(
            gated, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)
