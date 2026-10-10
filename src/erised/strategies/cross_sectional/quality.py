"""Quality and profitability factors: gross profitability, accruals, margins, investment, and capital discipline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import (
    _align_fund,
    _buy_hold_band,
    _cs_zscore,
    _equal_weight_top_quantile,
    _monthly_rebalance,
    _periodic_rebalance,
    _sector_neutral_weights,
)
from .base import CrossSectionalStrategy


class PiotroskiValueScreen(CrossSectionalStrategy):
    """H164: Among top-quintile B/M stocks, buy only those with F-score ≥ min_fscore.

    Step 1: rank by book-to-market, identify top quintile (cheap stocks).
    Step 2: within that set, keep only stocks with Piotroski F-score >= min_fscore.
    Step 3: equal-weight the survivors.

    fscore_panel: pre-computed daily F-score panel (rows=dates, cols=tickers).
    """

    name = "piotroski_value_screen"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        fscore_panel: pd.DataFrame,
        bm_quantile: float = 0.20,
        min_fscore: float = 6.0,
        monthly_rebalance: bool = True,
    ):
        self.book_equity = book_equity
        self.shares_outstanding = shares_outstanding
        self.fscore_panel = fscore_panel
        self.bm_quantile = bm_quantile
        self.min_fscore = min_fscore
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        be = _align_fund(self.book_equity, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        fs = _align_fund(self.fscore_panel, prices_panel)

        mktcap = prices_panel * sh
        bm = (be / mktcap).where(be > 0)

        # Identify top-quintile by B/M (highest B/M = cheapest)
        mem_f = membership.reindex_like(bm).fillna(False)
        masked_bm = bm.where(mem_f)
        ranks = masked_bm.rank(axis=1, ascending=False, na_option="keep")
        n_members = mem_f.sum(axis=1)
        threshold = (n_members * self.bm_quantile).clip(lower=1)
        in_value_quintile = ranks.le(threshold, axis=0) & ranks.notna()

        # Apply F-score gate
        passes_fscore = fs >= self.min_fscore
        selected = in_value_quintile & passes_fscore & mem_f

        count = selected.sum(axis=1).replace(0, np.nan)
        daily_w = selected.astype(float).div(count, axis=0).fillna(0.0)
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class GrossProfitabilityRank(CrossSectionalStrategy):
    """H165: Long top quintile by gross profitability = gross_profit / total_assets.

    Sector-neutral: select top `within_sector_quantile` within each GICS sector.
    High GP/A = durable pricing power = better expected return (Novy-Marx 2013).
    """

    name = "gross_profitability_rank"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)

        gpa = (gp / ta).where(ta > 0)  # exclude non-positive total assets
        daily_w = _sector_neutral_weights(
            gpa, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class LowAccrualsQuality(CrossSectionalStrategy):
    """H166: Long bottom quintile by accruals = (net_income - operating_cf) / total_assets.

    Low accruals (cash-driven earnings) signal higher earnings quality.
    ascending=True so the lowest accruals rank first.
    """

    name = "low_accruals_quality"

    def __init__(
        self,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        total_assets: pd.DataFrame,
        top_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.total_assets = total_assets
        self.top_quantile = top_quantile
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)

        # accruals = (net_income - operating_cf) / |total_assets|
        accruals = ((ni - ocf) / ta.abs()).where(ta.abs() > 0)
        # ascending=True: low accruals = high quality = selected
        daily_w = _equal_weight_top_quantile(
            accruals, membership, self.top_quantile, ascending=True
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class CompositeQualityScore(CrossSectionalStrategy):
    """H168: Multi-dimensional quality composite (QMJ-inspired).

    Three equally-weighted quality dimensions (all sector-neutral z-scored):
      1. Profitability: avg z(GP/A, ROE, cash ROE)
      2. Growth: z(1-yr change in GP/A)
      3. Safety: avg z(-beta_252d, -LTD/TA, current_ratio)

    Long top `within_sector_quantile` within each GICS sector.
    spy_rets required for rolling beta computation.
    """

    name = "composite_quality_score"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        book_equity: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        sectors: pd.Series,
        spy_rets: pd.Series,
        within_sector_quantile: float = 0.20,
        beta_window: int = 252,
        monthly_rebalance: bool = True,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.book_equity = book_equity
        self.long_term_debt = long_term_debt
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.sectors = sectors
        self.spy_rets = spy_rets
        self.within_sector_quantile = within_sector_quantile
        self.beta_window = beta_window
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel).fillna(0.0)
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)

        # Dimension 1: Profitability
        gpa = (gp / ta).where(ta > 0)
        roe = (ni / be).where(be > 0)
        cash_roe = (ocf / be).where(be > 0)
        profitability = (
            _cs_zscore(gpa).fillna(0.0)
            + _cs_zscore(roe).fillna(0.0)
            + _cs_zscore(cash_roe).fillna(0.0)
        ) / 3.0

        # Dimension 2: Growth (1-yr change in GP/A)
        gpa_lag = gpa.shift(252)
        gpa_growth = (gpa - gpa_lag).where(gpa.notna() & gpa_lag.notna())
        growth = _cs_zscore(gpa_growth)

        # Dimension 3: Safety
        daily_rets = prices_panel.pct_change()
        spy = self.spy_rets.reindex(prices_panel.index).fillna(0.0)
        min_p = self.beta_window // 2
        cov_sm = daily_rets.rolling(self.beta_window, min_periods=min_p).cov(spy)
        var_m = spy.rolling(self.beta_window, min_periods=min_p).var()
        beta = cov_sm.div(var_m, axis=0)
        leverage = (ltd / ta).where(ta > 0).fillna(0.0)
        curr_ratio = (ca / cl.replace(0, np.nan)).where(ca.notna() & cl.notna())
        safety = (
            _cs_zscore(-beta).fillna(0.0)
            + _cs_zscore(-leverage).fillna(0.0)
            + _cs_zscore(curr_ratio).fillna(0.0)
        ) / 3.0

        # Composite: equal-weight of three dimensions
        composite = (
            _cs_zscore(profitability).fillna(0.0)
            + _cs_zscore(growth).fillna(0.0)
            + _cs_zscore(safety).fillna(0.0)
        ) / 3.0

        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class QualityValueComposite(CrossSectionalStrategy):
    """H169: (1/3)*z(B/M) + (1/3)*z(GP/A) + (1/3)*z(-accruals), sector-neutral.

    Combines value (book-to-market), quality (gross profitability),
    and earnings quality (low accruals) into one composite.
    Sector-neutral selection of top `within_sector_quantile`.
    """

    name = "quality_value_composite"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.book_equity = book_equity
        self.shares_outstanding = shares_outstanding
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        be = _align_fund(self.book_equity, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)

        mktcap = prices_panel * sh
        bm = (be / mktcap).where(be > 0)
        gpa = (gp / ta).where(ta > 0)
        accruals = ((ni - ocf) / ta.abs()).where(ta.abs() > 0)

        composite = (
            _cs_zscore(bm).fillna(0.0)
            + _cs_zscore(gpa).fillna(0.0)
            + _cs_zscore(-accruals).fillna(0.0)  # negate: low accruals = high quality
        ) / 3.0

        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class FF5InvestmentFactor(CrossSectionalStrategy):
    """H170: Long bottom quintile by 1-year total-asset growth rate.

    Low asset growth = conservative investment = higher expected return.
    Cooper et al. (2008); Fama-French (2015) CMA factor.
    Sector-neutral selection.
    """

    name = "ff5_investment_factor"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        growth_window: int = 252,
        monthly_rebalance: bool = True,
    ):
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.growth_window = growth_window
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ta_lag = ta.shift(self.growth_window)
        asset_growth = (ta - ta_lag) / ta_lag.where(ta_lag > 0)

        # ascending=True: low asset growth selected (conservative investing)
        daily_w = _sector_neutral_weights(
            asset_growth, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class OperatingProfitabilityFF5(CrossSectionalStrategy):
    """H171: Long top quintile by gross profit / book equity (FF2015 RMW variant).

    GP/BE is the RMW (robust-minus-weak) profitability signal.
    Distinct from H165 (GP/TA): uses book equity, capturing return on equity capital.
    Sector-neutral selection.
    """

    name = "operating_profitability_ff5"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.gross_profit = gross_profit
        self.book_equity = book_equity
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)

        op = (gp / be).where(be > 0)
        daily_w = _sector_neutral_weights(
            op, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class ReturnOnAssets(CrossSectionalStrategy):
    """H173: Long top quintile by return on assets = net_income / total_assets.

    Distinct from H165 (gross profitability GP/TA): net income is post-tax and post-SGA,
    so it captures overall earnings efficiency rather than just pricing power.
    Sector-neutral selection.
    """

    name = "return_on_assets"

    def __init__(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)

        roa = (ni / ta).where(ta > 0)
        daily_w = _sector_neutral_weights(
            roa, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class CashOperatingProfitability(CrossSectionalStrategy):
    """H174: Long top quintile by cash operating profitability = operating_cf / book_equity.

    Ball et al. (2016): cash-based OP filters out accruals and is more predictive
    than standard earnings-based OP. Sector-neutral selection.
    """

    name = "cash_operating_profitability"

    def __init__(
        self,
        operating_cf: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.operating_cf = operating_cf
        self.book_equity = book_equity
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ocf = _align_fund(self.operating_cf, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)

        cop = (ocf / be).where(be > 0)
        daily_w = _sector_neutral_weights(
            cop, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class CashEarningsDrift(CrossSectionalStrategy):
    """H183: NI YoY growth, filtered to stocks with cash-backed earnings.

    Quality gate: operating_cf / net_income > cash_ratio_min.
    Stocks with accrual-heavy earnings are excluded (Sloan 1996 inspired).
    """

    name = "cash_earnings_drift"

    def __init__(
        self,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        yoy_window: int = 252,
        cash_ratio_min: float = 0.5,
        monthly_rebalance: bool = True,
    ):
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.yoy_window = yoy_window
        self.cash_ratio_min = cash_ratio_min
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)

        ni_lag = ni.shift(self.yoy_window)
        sue = (ni - ni_lag) / ni_lag.abs().replace(0, np.nan)

        # Cash quality: OCF / |NI| > threshold (exclude accrual-heavy earnings)
        cash_ratio = ocf / ni.abs().replace(0, np.nan)
        cash_backed = cash_ratio > self.cash_ratio_min

        composite = sue.where(cash_backed & (sue > 0))
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class LowAssetGrowth(CrossSectionalStrategy):
    """H206: Long bottom quintile by 2-year total-asset growth rate.

    Extended Cooper et al. (2008): firms with sustained 2-year asset growth
    underperform more persistently than 1-year growth (H170/FF5InvestmentFactor).
    Low sustained asset growth = conservative capital allocation.
    Sector-neutral bottom 20%, monthly rebalance.
    """

    name = "low_asset_growth"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        growth_window: int = 504,
    ):
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.growth_window = growth_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ta_lag = ta.shift(self.growth_window)
        asset_growth = (ta - ta_lag) / ta_lag.where(ta_lag > 0)
        daily_w = _sector_neutral_weights(
            asset_growth, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class ShareholderYield(CrossSectionalStrategy):
    """H207: Long bottom quintile by 1-year share count change rate.

    Net issuance = (shares_t - shares_{t-252}) / shares_{t-252}.
    Negative = buyback (share count reduction).
    Stocks reducing share count signal management confidence in valuation.
    Sector-neutral bottom 20% (most negative = most buybacks), monthly rebalance.
    """

    name = "shareholder_yield"

    def __init__(
        self,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        issuance_window: int = 252,
    ):
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.issuance_window = issuance_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        sh = _align_fund(self.shares_outstanding, prices_panel)
        sh_lag = sh.shift(self.issuance_window)
        net_issuance = (sh - sh_lag) / sh_lag.where(sh_lag > 0)
        # ascending=True: most negative (most buyback) selected in bottom quintile
        daily_w = _sector_neutral_weights(
            net_issuance, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class ConservativeQuality(CrossSectionalStrategy):
    """H208: z(-asset_growth) + z(-net_issuance) + z(OCF/TA) composite.

    Three capital discipline dimensions combined via cross-sectional z-scores:
    - Low asset growth (H206 signal)
    - Low share issuance / high buyback (H207 signal)
    - High operating cash flow relative to assets (cash quality signal)
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "conservative_quality"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        window: int = 252,
    ):
        self.total_assets = total_assets
        self.shares_outstanding = shares_outstanding
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.window = window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)

        ta_lag = ta.shift(self.window)
        asset_growth = (ta - ta_lag) / ta_lag.where(ta_lag > 0)
        sh_lag = sh.shift(self.window)
        net_issuance = (sh - sh_lag) / sh_lag.where(sh_lag > 0)
        ocf_ta = ocf / ta.where(ta > 0)

        z_ag = _cs_zscore(-asset_growth)
        z_ni = _cs_zscore(-net_issuance)
        z_ocf = _cs_zscore(ocf_ta)

        composite = z_ag.add(z_ni, fill_value=0.0).add(z_ocf.fillna(0.0), fill_value=0.0)
        n_signals = (
            z_ag.notna().astype(float) + z_ni.notna().astype(float) + z_ocf.notna().astype(float)
        )
        composite = composite / n_signals.replace(0, np.nan)

        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class BuybackMomentum(CrossSectionalStrategy):
    """H209: z(-net_issuance) + z(12-1m price momentum) composite.

    Management confidence (share buyback) + market confirmation (price momentum).
    Stocks both reducing share count and showing intermediate-term price strength.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "buyback_momentum"

    def __init__(
        self,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        issuance_window: int = 252,
        mom_window: int = 252,
        skip_window: int = 21,
    ):
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.issuance_window = issuance_window
        self.mom_window = mom_window
        self.skip_window = skip_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        sh = _align_fund(self.shares_outstanding, prices_panel)
        sh_lag = sh.shift(self.issuance_window)
        net_issuance = (sh - sh_lag) / sh_lag.where(sh_lag > 0)

        z_buyback = _cs_zscore(-net_issuance)
        z_mom = _cs_zscore(prices_panel.pct_change(self.mom_window).shift(self.skip_window))

        composite = z_buyback.add(z_mom, fill_value=0.0)
        n = z_buyback.notna().astype(float) + z_mom.notna().astype(float)
        composite = composite / n.replace(0, np.nan)

        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class CapitalDiscipline(CrossSectionalStrategy):
    """H210: z(-asset_growth) + z(-net_issuance) + z(-leverage_change) composite.

    Three capital allocation dimensions:
    - Low asset growth: avoid empire-building
    - Low share issuance: prefer buybacks
    - Low leverage increase: avoid debt-financed growth
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "capital_discipline"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        window: int = 252,
    ):
        self.total_assets = total_assets
        self.shares_outstanding = shares_outstanding
        self.long_term_debt = long_term_debt
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.window = window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)

        ta_lag = ta.shift(self.window)
        asset_growth = (ta - ta_lag) / ta_lag.where(ta_lag > 0)
        sh_lag = sh.shift(self.window)
        net_issuance = (sh - sh_lag) / sh_lag.where(sh_lag > 0)
        leverage = ltd / ta.where(ta > 0)
        leverage_change = leverage - leverage.shift(self.window)

        z_ag = _cs_zscore(-asset_growth)
        z_ni = _cs_zscore(-net_issuance)
        z_lev = _cs_zscore(-leverage_change)

        composite = z_ag.add(z_ni, fill_value=0.0).add(z_lev.fillna(0.0), fill_value=0.0)
        n = z_ag.notna().astype(float) + z_ni.notna().astype(float) + z_lev.notna().astype(float)
        composite = composite / n.replace(0, np.nan)

        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class BalanceSheetAccruals(CrossSectionalStrategy):
    """H211: Long bottom quintile by balance-sheet accruals.

    Balance-sheet accruals = Δ(NWC) / avg_TA where NWC = current_assets - current_liabilities.
    Sloan (1996) original balance-sheet approach: NWC growth signals low earnings quality.
    Distinct from H166 (LowAccrualsQuality) which uses the cash-flow approach (NI-OCF)/TA.
    Sector-neutral bottom 20%, monthly rebalance.
    """

    name = "balance_sheet_accruals"

    def __init__(
        self,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        accrual_window: int = 252,
    ):
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.accrual_window = accrual_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)

        nwc = ca - cl
        nwc_lag = nwc.shift(self.accrual_window)
        avg_ta = (ta + ta.shift(self.accrual_window)) / 2
        accruals = (nwc - nwc_lag) / avg_ta.where(avg_ta > 0)

        daily_w = _sector_neutral_weights(
            accruals, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class DualMomentumEarnings(CrossSectionalStrategy):
    """H212: Long stocks in top quantile of BOTH price momentum AND earnings momentum.

    AND condition (not composite): stock must rank in top quantile on both signals.
    Fewer but higher-conviction positions. Monthly rebalance.
    """

    name = "dual_momentum_earnings"

    def __init__(
        self,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        yoy_window: int = 252,
        top_quantile: float = 0.30,
        within_sector_quantile: float = 0.20,
    ):
        self.net_income = net_income
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_window = skip_window
        self.yoy_window = yoy_window
        self.top_quantile = top_quantile
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        price_mom = prices_panel.pct_change(self.mom_window).shift(self.skip_window)

        ni = _align_fund(self.net_income, prices_panel)
        ni_lag = ni.shift(self.yoy_window)
        sue = (ni - ni_lag) / ni_lag.abs().replace(0, np.nan)

        # Rank within cross-section (pct=True gives 0..1 where 1=highest)
        price_rank = price_mom.rank(axis=1, ascending=True, na_option="keep", pct=True)
        sue_rank = sue.rank(axis=1, ascending=True, na_option="keep", pct=True)

        # Both must be in the top fraction
        both_top = (price_rank >= (1 - self.top_quantile)) & (sue_rank >= (1 - self.top_quantile))

        composite = price_mom.where(both_top, np.nan)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class EarningsQualityMomentum(CrossSectionalStrategy):
    """H213: z(-balance_sheet_accruals) + z(12-1m price momentum) composite.

    Low accruals (high earnings quality) + strong price momentum.
    Combines Sloan (1996) balance-sheet accruals with Jegadeesh-Titman momentum.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "earnings_quality_momentum"

    def __init__(
        self,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        accrual_window: int = 252,
        mom_window: int = 252,
        skip_window: int = 21,
    ):
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.accrual_window = accrual_window
        self.mom_window = mom_window
        self.skip_window = skip_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)

        nwc = ca - cl
        nwc_lag = nwc.shift(self.accrual_window)
        avg_ta = (ta + ta.shift(self.accrual_window)) / 2
        accruals = (nwc - nwc_lag) / avg_ta.where(avg_ta > 0)

        z_accruals = _cs_zscore(-accruals)
        z_mom = _cs_zscore(prices_panel.pct_change(self.mom_window).shift(self.skip_window))

        composite = z_accruals.add(z_mom, fill_value=0.0)
        n = z_accruals.notna().astype(float) + z_mom.notna().astype(float)
        composite = composite / n.replace(0, np.nan)

        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class GrossMarginMomentum(CrossSectionalStrategy):
    """H214: 12-1m momentum gated by improving gross margin.

    Filter: gross_margin_t > gross_margin_{t-252} (YoY improvement in GP/revenue).
    Only buy momentum stocks with expanding gross margins.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "gross_margin_momentum"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        margin_window: int = 252,
        mom_window: int = 252,
        skip_window: int = 21,
    ):
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.margin_window = margin_window
        self.mom_window = mom_window
        self.skip_window = skip_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)

        gross_margin = gp / rev.where(rev > 0)
        margin_improving = gross_margin > gross_margin.shift(self.margin_window)

        price_mom = prices_panel.pct_change(self.mom_window).shift(self.skip_window)
        gated_mom = price_mom.where(margin_improving, np.nan)

        daily_w = _sector_neutral_weights(
            gated_mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class FundamentalReversal(CrossSectionalStrategy):
    """H215: Monthly price reversal gated by fundamental stability.

    Buy oversold stocks (bottom quintile by 21d return) where NI > 0 AND OCF > 0.
    Avoids fundamental deterioration traps in short-term reversal.
    Sector-neutral bottom 20% by 21d return within qualifying stocks, monthly rebalance.
    """

    name = "fundamental_reversal"

    def __init__(
        self,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        reversal_window: int = 21,
    ):
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.reversal_window = reversal_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)

        quality_ok = (ni > 0) & (ocf > 0)
        reversal = prices_panel.pct_change(self.reversal_window)
        gated_reversal = reversal.where(quality_ok, np.nan)

        # ascending=True: buy the biggest losers among quality-stable stocks
        daily_w = _sector_neutral_weights(
            gated_reversal, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class ROEMomentum(CrossSectionalStrategy):
    """H221: YoY improvement in Return on Equity (NI / book_equity)."""

    name = "roe_momentum"

    def __init__(
        self,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        roe_window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.net_income = net_income
        self.book_equity = book_equity
        self.sectors = sectors
        self.roe_window = roe_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        roe = ni / be.where(be > 0)
        roe_change = roe - roe.shift(self.roe_window)
        daily_w = _sector_neutral_weights(
            roe_change, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class CashConversionQuality(CrossSectionalStrategy):
    """H222: OCF / gross_profit -- cash conversion efficiency (quality factor)."""

    name = "cash_conversion_quality"

    def __init__(
        self,
        operating_cf: pd.DataFrame,
        gross_profit: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.operating_cf = operating_cf
        self.gross_profit = gross_profit
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ocf = _align_fund(self.operating_cf, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        cash_conv = ocf / gp.where(gp > 0)
        daily_w = _sector_neutral_weights(
            cash_conv, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class LeverageReduction(CrossSectionalStrategy):
    """H223: Long stocks with largest YoY decrease in LTD/TA (financial de-leveraging)."""

    name = "leverage_reduction"

    def __init__(
        self,
        long_term_debt: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        lev_window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.long_term_debt = long_term_debt
        self.total_assets = total_assets
        self.sectors = sectors
        self.lev_window = lev_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        lev = ltd / ta.where(ta > 0)
        lev_change = lev - lev.shift(self.lev_window)  # negative = de-leveraging
        daily_w = _sector_neutral_weights(
            lev_change, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class PiotroskiFScore(CrossSectionalStrategy):
    """H224: Piotroski (2000) F-Score -- 9-component binary quality score."""

    name = "piotroski_f_score"

    def __init__(
        self,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        score_window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.total_assets = total_assets
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.long_term_debt = long_term_debt
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.score_window = score_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)

        w = self.score_window
        ta_lag = ta.shift(w)
        avg_ta = (ta + ta_lag) / 2

        roa = ni / avg_ta.where(avg_ta > 0)
        roa_lag = ni.shift(w) / ta_lag.where(ta_lag > 0)

        lev = ltd / avg_ta.where(avg_ta > 0)
        lev_lag = ltd.shift(w) / ta_lag.where(ta_lag > 0)

        curr_ratio = ca / cl.where(cl > 0)
        curr_ratio_lag = ca.shift(w) / cl.shift(w).where(cl.shift(w) > 0)

        gm = gp / rev.where(rev > 0)
        gm_lag = gp.shift(w) / rev.shift(w).where(rev.shift(w) > 0)

        at = rev / avg_ta.where(avg_ta > 0)
        at_lag = rev.shift(w) / ta_lag.where(ta_lag > 0)

        accruals_quality = ocf / avg_ta.where(avg_ta > 0) - roa

        f = (
            (roa > 0).astype(float)  # F1: ROA positive
            + (ocf > 0).astype(float)  # F2: OCF positive
            + (roa > roa_lag).astype(float)  # F3: ROA improving
            + (accruals_quality > 0).astype(float)  # F4: OCF > NI (quality)
            + (lev < lev_lag).astype(float)  # F5: leverage decreasing
            + (curr_ratio > curr_ratio_lag).astype(float)  # F6: current ratio improving
            + (sh <= sh.shift(w)).astype(float)  # F7: no share dilution
            + (gm > gm_lag).astype(float)  # F8: gross margin improving
            + (at > at_lag).astype(float)  # F9: asset turnover improving
        )
        # Zero out rows with no fundamental coverage
        f = f.where(roa.notna())

        daily_w = _sector_neutral_weights(
            f, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class EarningsConsistency(CrossSectionalStrategy):
    """H225: z(1yr NI growth) + z(2yr NI growth) -- consistent multi-period earners."""

    name = "earnings_consistency"

    def __init__(
        self,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        window_1yr: int = 252,
        window_2yr: int = 504,
        within_sector_quantile: float = 0.20,
    ):
        self.net_income = net_income
        self.sectors = sectors
        self.window_1yr = window_1yr
        self.window_2yr = window_2yr
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ni_lag1 = ni.shift(self.window_1yr)
        ni_lag2 = ni.shift(self.window_2yr)
        growth_1yr = (ni - ni_lag1) / ni_lag1.abs().where(ni_lag1 != 0)
        growth_2yr = (ni - ni_lag2) / ni_lag2.abs().where(ni_lag2 != 0)
        growth_1yr = growth_1yr.clip(-1.0, 3.0)
        growth_2yr = growth_2yr.clip(-1.0, 3.0)
        score = _cs_zscore(growth_1yr) + _cs_zscore(growth_2yr)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class GrossProfitability(CrossSectionalStrategy):
    """H227: Novy-Marx (2013) GP/TA gross profitability factor."""

    name = "gross_profitability"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        gp_ta = gp / ta.where(ta > 0)
        daily_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class OperatingLeverage(CrossSectionalStrategy):
    """H229: GP growth minus revenue growth -- improving operating margin leverage."""

    name = "operating_leverage"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        gp_lag = gp.shift(self.window)
        rev_lag = rev.shift(self.window)
        gp_growth = (gp - gp_lag) / gp_lag.abs().where(gp_lag != 0)
        rev_growth = (rev - rev_lag) / rev_lag.abs().where(rev_lag != 0)
        # GP grows faster than revenue = improving gross margin = positive operating leverage
        op_lev = gp_growth - rev_growth
        op_lev = op_lev.clip(-2.0, 2.0)
        daily_w = _sector_neutral_weights(
            op_lev, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class AssetTurnover(CrossSectionalStrategy):
    """H230: YoY improvement in asset turnover (revenue / total_assets)."""

    name = "asset_turnover"

    def __init__(
        self,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.revenue = revenue
        self.total_assets = total_assets
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        at = rev / ta.where(ta > 0)
        at_change = at - at.shift(self.window)
        daily_w = _sector_neutral_weights(
            at_change, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class ValueQuality(CrossSectionalStrategy):
    """H232: z(BE/price) + z(GP/TA) -- value-quality composite (QMJ-style)."""

    name = "value_quality"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.book_equity = book_equity
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        be = _align_fund(self.book_equity, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        bm = be / prices_panel.where(prices_panel > 0)
        gp_ta = gp / ta.where(ta > 0)
        score = _cs_zscore(bm) + _cs_zscore(gp_ta)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class GrowthQuality(CrossSectionalStrategy):
    """H233: z(rev_growth) + z(GP/TA) -- growth + profitability composite."""

    name = "growth_quality"

    def __init__(
        self,
        revenue: pd.DataFrame,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        growth_window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.revenue = revenue
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.growth_window = growth_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        rev_lag = rev.shift(self.growth_window)
        rev_growth = (rev - rev_lag) / rev_lag.where(rev_lag > 0)
        rev_growth = rev_growth.clip(-1.0, 3.0)
        gp_ta = gp / ta.where(ta > 0)
        score = _cs_zscore(rev_growth) + _cs_zscore(gp_ta)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class EfficientGrowth(CrossSectionalStrategy):
    """H235: z(rev_growth) + z(OCF/TA) -- revenue growth backed by operating cash flows."""

    name = "efficient_growth"

    def __init__(
        self,
        revenue: pd.DataFrame,
        operating_cf: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        growth_window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.revenue = revenue
        self.operating_cf = operating_cf
        self.total_assets = total_assets
        self.sectors = sectors
        self.growth_window = growth_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        rev_lag = rev.shift(self.growth_window)
        rev_growth = (rev - rev_lag) / rev_lag.where(rev_lag > 0)
        rev_growth = rev_growth.clip(-1.0, 3.0)
        ocf_ta = ocf / ta.where(ta > 0)
        score = _cs_zscore(rev_growth) + _cs_zscore(ocf_ta)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class LiquidityMomentum(CrossSectionalStrategy):
    """H236: YoY improvement in current ratio (CA / CL)."""

    name = "liquidity_momentum"

    def __init__(
        self,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        curr_ratio = ca / cl.where(cl > 0)
        cr_change = curr_ratio - curr_ratio.shift(self.window)
        daily_w = _sector_neutral_weights(
            cr_change, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class DebtCoverage(CrossSectionalStrategy):
    """H237: OCF / long_term_debt solvency ratio -- higher = better debt coverage."""

    name = "debt_coverage"

    def __init__(
        self,
        operating_cf: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.operating_cf = operating_cf
        self.long_term_debt = long_term_debt
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ocf = _align_fund(self.operating_cf, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        # Clip LTD to a small positive floor to handle zero-debt companies gracefully
        debt_coverage = ocf / ltd.where(ltd > 0)
        debt_coverage = debt_coverage.clip(-2.0, 10.0)
        daily_w = _sector_neutral_weights(
            debt_coverage, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class NetProfitMargin(CrossSectionalStrategy):
    """H238: YoY improvement in net profit margin (NI / revenue)."""

    name = "net_profit_margin"

    def __init__(
        self,
        net_income: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.net_income = net_income
        self.revenue = revenue
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        margin = ni / rev.where(rev > 0)
        margin_change = margin - margin.shift(self.window)
        daily_w = _sector_neutral_weights(
            margin_change, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class GrossMarginLevel(CrossSectionalStrategy):
    """H239: Level of gross margin (GP / revenue) -- persistent margin quality."""

    name = "gross_margin_level"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        gross_margin = gp / rev.where(rev > 0)
        daily_w = _sector_neutral_weights(
            gross_margin, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class ROAMomentum(CrossSectionalStrategy):
    """H240: YoY improvement in Return on Assets (NI / total_assets)."""

    name = "roa_momentum"

    def __init__(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        avg_ta = (ta + ta.shift(self.window)) / 2
        roa = ni / avg_ta.where(avg_ta > 0)
        roa_lag = ni.shift(self.window) / ta.shift(self.window).where(ta.shift(self.window) > 0)
        roa_change = roa - roa_lag
        daily_w = _sector_neutral_weights(
            roa_change, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class CashReturnOnEquity(CrossSectionalStrategy):
    """H242: Cash return on equity = OCF / book_equity."""

    name = "cash_return_on_equity"

    def __init__(
        self,
        operating_cf: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.operating_cf = operating_cf
        self.book_equity = book_equity
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ocf = _align_fund(self.operating_cf, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        croe = ocf / be.where(be > 0)
        croe = croe.clip(-1.0, 5.0)
        daily_w = _sector_neutral_weights(
            croe, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class GrossProfitGrowth(CrossSectionalStrategy):
    """H243: YoY absolute gross profit growth scaled by total assets."""

    name = "gross_profit_growth"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        gp_growth = (gp - gp.shift(self.window)) / ta.shift(self.window).where(
            ta.shift(self.window) > 0
        )
        gp_growth = gp_growth.clip(-1.0, 2.0)
        daily_w = _sector_neutral_weights(
            gp_growth, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class OCFMomentum(CrossSectionalStrategy):
    """H245: YoY improvement in OCF/total_assets (cash return on assets)."""

    name = "ocf_momentum"

    def __init__(
        self,
        operating_cf: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.operating_cf = operating_cf
        self.total_assets = total_assets
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ocf_ta = ocf / ta.where(ta > 0)
        ocf_ta_change = ocf_ta - ocf_ta.shift(self.window)
        daily_w = _sector_neutral_weights(
            ocf_ta_change, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class FinancialHealthComposite(CrossSectionalStrategy):
    """H249: Composite financial health score: z(OCF/TA) + z(-LTD/TA) + z(curr_ratio) + z(NI/TA)."""

    name = "financial_health_composite"

    def __init__(
        self,
        operating_cf: pd.DataFrame,
        total_assets: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.operating_cf = operating_cf
        self.total_assets = total_assets
        self.long_term_debt = long_term_debt
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.net_income = net_income
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)

        ocf_ta = ocf / ta.where(ta > 0)
        ltd_ta = ltd / ta.where(ta > 0)
        curr_ratio = ca / cl.where(cl > 0)
        ni_ta = ni / ta.where(ta > 0)

        score = (
            _cs_zscore(ocf_ta) + _cs_zscore(-ltd_ta) + _cs_zscore(curr_ratio) + _cs_zscore(ni_ta)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class LowNetOperatingAssets(CrossSectionalStrategy):
    """H261: NOA proxy = (total_assets - current_liabilities) / total_assets; long bottom-quintile."""

    name = "low_net_operating_assets"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.total_assets = total_assets
        self.current_liabilities = current_liabilities
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)

        noa = (ta - cl) / ta.where(ta > 0)
        daily_w = _sector_neutral_weights(
            noa, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class AccrualReliabilityFactor(CrossSectionalStrategy):
    """H262: Total accruals = (net_income - operating_cf) / avg_total_assets; long low-accrual quintile."""

    name = "accrual_reliability_factor"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)

        avg_ta = (ta + ta.shift(self.window)) / 2
        accruals = (ni - ocf) / avg_ta.where(avg_ta > 0)

        daily_w = _sector_neutral_weights(
            accruals, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class ConservativeCapexInvestment(CrossSectionalStrategy):
    """H263: YoY change in long-term assets / total_assets as capex proxy; long bottom-quintile."""

    name = "conservative_capex_investment"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.total_assets = total_assets
        self.current_assets = current_assets
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)

        lt_assets = (ta - ca).clip(lower=0)
        delta_lt = (lt_assets - lt_assets.shift(self.window)) / ta.where(ta > 0)

        daily_w = _sector_neutral_weights(
            delta_lt, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class SustainableEarningsComposite(CrossSectionalStrategy):
    """H264: 3-factor composite: z(-NOA) + z(-accruals) + z(CFO/NI); long top-quintile."""

    name = "sustainable_earnings_composite"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.total_assets = total_assets
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)

        noa = (ta - cl) / ta.where(ta > 0)
        avg_ta = (ta + ta.shift(self.window)) / 2
        accruals = (ni - ocf) / avg_ta.where(avg_ta > 0)
        # Cash purity: OCF / NI; winsorize to [-5, 5] to handle near-zero NI
        cfo_ni = (ocf / ni.where(ni.abs() > 1e-6)).clip(-5.0, 5.0)

        score = _cs_zscore(-noa) + _cs_zscore(-accruals) + _cs_zscore(cfo_ni)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class SustainabilityFilteredProfitability(CrossSectionalStrategy):
    """H265: GP/TA filtered by sustainability composite; top-half sustainability required."""

    name = "sustainability_filtered_profitability"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        sustainability_pct: float = 0.50,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.sustainability_pct = sustainability_pct
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)

        gp_ta = gp / ta.where(ta > 0)

        noa = (ta - cl) / ta.where(ta > 0)
        avg_ta = (ta + ta.shift(self.window)) / 2
        accruals = (ni - ocf) / avg_ta.where(avg_ta > 0)
        cfo_ni = (ocf / ni.where(ni.abs() > 1e-6)).clip(-5.0, 5.0)
        sust_score = _cs_zscore(-noa) + _cs_zscore(-accruals) + _cs_zscore(cfo_ni)

        mem = membership.reindex_like(gp_ta).fillna(False)
        sust_rank = sust_score.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        filtered_mem = membership & (sust_rank >= (1.0 - self.sustainability_pct)).fillna(False)

        daily_w = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class GrossMarginIntensityFactor(CrossSectionalStrategy):
    """H266: Gross margin (GP/revenue) as intangible intensity proxy; long top-quintile."""

    name = "gross_margin_intensity_factor"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)

        gm = gp / rev.where(rev > 0)
        daily_w = _sector_neutral_weights(
            gm, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class GrossMarginChangeFactor(CrossSectionalStrategy):
    """H267: YoY change in gross margin; long top-quintile (improving margins)."""

    name = "gross_margin_change_factor"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)

        gm = gp / rev.where(rev > 0)
        delta_gm = gm - gm.shift(self.window)

        daily_w = _sector_neutral_weights(
            delta_gm, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class KnowledgeReturnProxy(CrossSectionalStrategy):
    """H268: GP/total_assets as intangible ROA proxy; top-decile concentration."""

    name = "knowledge_return_proxy"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.10,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)

        gp_ta = gp / ta.where(ta > 0)
        daily_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class IntangibleValueScreen(CrossSectionalStrategy):
    """H269: High gross margin + cheap B/M + low distress three-way screen."""

    name = "intangible_value_screen"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        book_equity: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        gm_pct: float = 0.33,
        btm_pct: float = 0.50,
        distress_pct: float = 0.50,
        vol_window: int = 252,
        mom_window: int = 63,
    ):
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.total_assets = total_assets
        self.book_equity = book_equity
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.gm_pct = gm_pct
        self.btm_pct = btm_pct
        self.distress_pct = distress_pct
        self.vol_window = vol_window
        self.mom_window = mom_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        shs = _align_fund(self.shares_outstanding, prices_panel)

        gm = gp / rev.where(rev > 0)
        mcap = (prices_panel * shs).where(shs > 0)
        btm = be / mcap.where(mcap > 0)

        liab = (ta - be).clip(lower=0)
        leverage = liab / ta.where(ta > 0)
        roa = ni / ta.where(ta > 0)
        cash_ratio = ocf / ta.where(ta > 0)
        log_size = np.log(mcap.where(mcap > 0))
        daily_rets = prices_panel.pct_change()
        mom3 = prices_panel / prices_panel.shift(self.mom_window) - 1
        vol12m = daily_rets.rolling(self.vol_window, min_periods=self.vol_window // 4).std()

        distress = (
            _cs_zscore(leverage)
            + _cs_zscore(-roa)
            + _cs_zscore(-mom3)
            + _cs_zscore(vol12m)
            + _cs_zscore(-cash_ratio)
            + _cs_zscore(-log_size)
        )

        mem = membership.reindex_like(gm).fillna(False)
        gm_rank = gm.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        btm_rank = btm.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        dist_rank = distress.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")

        filtered_mem = (
            membership
            & (gm_rank >= (1.0 - self.gm_pct)).fillna(False)
            & (btm_rank >= (1.0 - self.btm_pct)).fillna(False)
            & (dist_rank <= self.distress_pct).fillna(False)
        )

        n_valid = filtered_mem.sum(axis=1).replace(0, np.nan)
        daily_w = filtered_mem.div(n_valid, axis=0).fillna(0.0)
        return _monthly_rebalance(daily_w, prices_panel)


class IntangibleQualityComposite(CrossSectionalStrategy):
    """H270: Composite of gross margin level + YoY change + CFO/NI cash quality."""

    name = "intangible_quality_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)

        gm = gp / rev.where(rev > 0)
        delta_gm = gm - gm.shift(self.window)
        cfo_ni = (ocf / ni.where(ni.abs() > 1e-6)).clip(-5.0, 5.0)

        score = _cs_zscore(gm) + _cs_zscore(delta_gm) + _cs_zscore(cfo_ni)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class LowCapexGrowthFactor(CrossSectionalStrategy):
    """H286: Long bottom-quintile by LT-asset-change acceleration (capex proxy 2nd derivative).

    H263 uses the level of LT asset change (1st derivative). H286 tests the
    acceleration (2nd derivative): how quickly is capex growing? Empire-building
    firms accelerate investment before the peak — the acceleration is the warning signal.
    """

    name = "low_capex_growth_factor"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        window: int = 252,
    ):
        self.total_assets = total_assets
        self.current_assets = current_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.window = window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        lt_assets = ta - ca
        # 1st derivative: annual LT-asset change normalized by TA
        delta_lt = (lt_assets - lt_assets.shift(self.window)) / ta.replace(0, np.nan)
        # 2nd derivative: how much faster is capex growing
        capex_growth = (delta_lt - delta_lt.shift(self.window)) / delta_lt.shift(
            self.window
        ).abs().replace(0, np.nan)
        capex_growth = capex_growth.clip(-10, 10)
        # ascending=True: lowest acceleration = most disciplined = best
        daily_w = _sector_neutral_weights(
            capex_growth, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class InvestmentEfficiencyRatio(CrossSectionalStrategy):
    """H287: Long top-quintile by revenue growth per unit of capital invested.

    Productive investment generates revenue. efficiency = rev_growth / (capex_intensity + 0.01).
    High ratio = revenue growth from modest capital = productive investment.
    Low ratio = capex without revenue payoff = empire-building.
    """

    name = "investment_efficiency_ratio"

    def __init__(
        self,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        window: int = 252,
        capex_floor: float = 0.01,
    ):
        self.revenue = revenue
        self.total_assets = total_assets
        self.current_assets = current_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.window = window
        self.capex_floor = capex_floor

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        lt_assets = ta - ca

        rev_growth = (rev - rev.shift(self.window)) / rev.shift(self.window).abs().replace(
            0, np.nan
        )
        rev_growth = rev_growth.clip(-5, 5)
        # capex intensity: annual LT asset change / TA (annualized fraction)
        capex_intensity = (lt_assets - lt_assets.shift(self.window)) / ta.replace(0, np.nan)
        # Use only positive capex (investment, not divestiture)
        capex_intensity = capex_intensity.clip(lower=0)
        efficiency = rev_growth / (capex_intensity + self.capex_floor)
        efficiency = efficiency.clip(-100, 100)

        daily_w = _sector_neutral_weights(
            efficiency, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class ROICCapexDiscipline(CrossSectionalStrategy):
    """H288: z(ROIC) + z(-capex_growth) — earnings quality + investment restraint.

    ROIC = NI / (TA - CL): how well assets earn.
    capex_growth = acceleration of LT-asset change (from H286).
    Combines high-earnings-quality with low investment acceleration.
    """

    name = "roic_capex_discipline"

    def __init__(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        window: int = 252,
    ):
        self.net_income = net_income
        self.total_assets = total_assets
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.window = window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        lt_assets = ta - ca

        invested_capital = (ta - cl).where((ta - cl) > 0)
        roic = ni / invested_capital.replace(0, np.nan)

        delta_lt = (lt_assets - lt_assets.shift(self.window)) / ta.replace(0, np.nan)
        capex_growth = (delta_lt - delta_lt.shift(self.window)) / delta_lt.shift(
            self.window
        ).abs().replace(0, np.nan)
        capex_growth = capex_growth.clip(-10, 10)

        score = _cs_zscore(roic) + _cs_zscore(-capex_growth)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class AssetGrowthPenalty(CrossSectionalStrategy):
    """H289: z(total_asset_growth) + z(lt_asset_change) — comprehensive investment activity.

    Combines total-asset growth (H206 signal) with LT-asset change (H263 signal).
    Rank ascending: lowest combined investment = most disciplined = best.
    Distinct from H206 (total-asset only) and H263 (LT-asset only).
    """

    name = "asset_growth_penalty"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        window: int = 252,
    ):
        self.total_assets = total_assets
        self.current_assets = current_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.window = window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        lt_assets = ta - ca

        asset_growth = (ta - ta.shift(self.window)) / ta.shift(self.window).replace(0, np.nan)
        asset_growth = asset_growth.clip(-5, 5)
        lt_change = (lt_assets - lt_assets.shift(self.window)) / ta.replace(0, np.nan)
        lt_change = lt_change.clip(-5, 5)

        comprehensive = _cs_zscore(asset_growth) + _cs_zscore(lt_change)
        # ascending=True: lowest combined investment activity = best
        daily_w = _sector_neutral_weights(
            comprehensive, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class CapitalEfficiencyQualityComposite(CrossSectionalStrategy):
    """H290: z(ROIC) + z(efficiency_ratio) + z(-NOA) — 3-factor capital quality.

    ROIC = NI/(TA-CL), efficiency = rev_growth/(capex_intensity+0.01), NOA = (TA-CL)/TA.
    3-factor composite testing the dilution boundary (program finding: composites above 2-3 factors dilute).
    """

    name = "capital_efficiency_quality_composite"

    def __init__(
        self,
        net_income: pd.DataFrame,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        window: int = 252,
        capex_floor: float = 0.01,
    ):
        self.net_income = net_income
        self.revenue = revenue
        self.total_assets = total_assets
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.window = window
        self.capex_floor = capex_floor

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        lt_assets = ta - ca

        invested_capital = (ta - cl).where((ta - cl) > 0)
        roic = ni / invested_capital.replace(0, np.nan)

        noa = (ta - cl) / ta.replace(0, np.nan)

        rev_growth = (rev - rev.shift(self.window)) / rev.shift(self.window).abs().replace(
            0, np.nan
        )
        rev_growth = rev_growth.clip(-5, 5)
        capex_intensity = (lt_assets - lt_assets.shift(self.window)) / ta.replace(0, np.nan)
        capex_intensity = capex_intensity.clip(lower=0)
        efficiency = (rev_growth / (capex_intensity + self.capex_floor)).clip(-100, 100)

        score = _cs_zscore(roic) + _cs_zscore(efficiency) + _cs_zscore(-noa)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class LowTurnoverQualityFactor(CrossSectionalStrategy):
    """H292: GP/TA with quarterly rebalance (every 63 trading days).

    Tests whether the stable GP/TA signal retains alpha at quarterly frequency.
    Identical signal to H227; only rebalance cadence differs.
    """

    name = "low_turnover_quality_factor"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
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
        gp_ta = gp / ta.replace(0, np.nan)
        daily_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class TurnoverCappedProfitabilityFactor(CrossSectionalStrategy):
    """H294: GP/TA with 20% monthly replacement cap.

    At each rebalance, limit portfolio turnover to 20% of current holdings.
    Existing stocks in top-40% are held; new additions capped at 20% of portfolio.
    Reduces market-impact during periods of high factor-rank volatility.
    """

    name = "turnover_capped_profitability_factor"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        buy_quantile: float = 0.20,
        hold_quantile: float = 0.40,
        max_replace_fraction: float = 0.20,
        rebalance_period: int = 21,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.buy_quantile = buy_quantile
        self.hold_quantile = hold_quantile
        self.max_replace_fraction = max_replace_fraction
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        gp_ta = gp / ta.replace(0, np.nan)

        # Score panel: used to rank new additions by quality
        score_arr = gp_ta.values

        strict_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.buy_quantile, ascending=False
        )
        extended_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.hold_quantile, ascending=False
        )

        dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        n_dates = len(dates)
        n_stocks = len(tickers)

        strict_arr = strict_w.values > 0
        extended_arr = extended_w.values > 0
        mem_arr = membership.reindex_like(prices_panel).fillna(False).values

        held = np.zeros(n_stocks, dtype=bool)
        out = np.zeros((n_dates, n_stocks))

        for i in range(n_dates):
            is_rebal = i % self.rebalance_period == 0
            mem_today = mem_arr[i]

            if is_rebal:
                # Forced exits (fell out of extended zone) and new buys
                keep = held & extended_arr[i] & mem_today
                potential_new = strict_arr[i] & ~held & mem_today

                n_held = int(held.sum())
                max_new = (
                    max(1, int(self.max_replace_fraction * n_held))
                    if n_held > 0
                    else int(strict_arr[i].sum())
                )

                # Rank potential new additions by score (best first)
                if potential_new.sum() > 0 and max_new > 0:
                    scores_today = score_arr[i]
                    new_scores = np.where(potential_new, scores_today, -np.inf)
                    top_new_idx = np.argsort(-np.nan_to_num(new_scores, nan=-np.inf))[:max_new]
                    add_mask = np.zeros(n_stocks, dtype=bool)
                    for idx in top_new_idx:
                        if potential_new[idx]:
                            add_mask[idx] = True
                    held = keep | add_mask
                else:
                    held = keep
            else:
                held = held & mem_today

            n_active = held.sum()
            if n_active > 0:
                out[i] = held.astype(float) / n_active

        return pd.DataFrame(out, index=dates, columns=tickers)


class ExecutionEfficientComposite(CrossSectionalStrategy):
    """H295: z(gp_ta) + z(efficiency_ratio), bi-monthly rebalance + buy/hold band.

    Combines best fundamental signal (GP/TA) with best capital-efficiency signal
    (investment efficiency ratio). Buy when rank >= 0.70, hold until < 0.50.
    Bi-monthly rebalance (42 days) to reduce momentum-level turnover costs.
    """

    name = "execution_efficient_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        buy_quantile: float = 0.20,
        hold_quantile: float = 0.50,
        rebalance_period: int = 42,
        window: int = 252,
        capex_floor: float = 0.01,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.current_assets = current_assets
        self.revenue = revenue
        self.sectors = sectors
        self.buy_quantile = buy_quantile
        self.hold_quantile = hold_quantile
        self.rebalance_period = rebalance_period
        self.window = window
        self.capex_floor = capex_floor

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        lt_assets = ta - ca

        gp_ta = gp / ta.replace(0, np.nan)
        rev_growth = (rev - rev.shift(self.window)) / rev.shift(self.window).abs().replace(
            0, np.nan
        )
        rev_growth = rev_growth.clip(-5, 5)
        capex_intensity = (lt_assets - lt_assets.shift(self.window)) / ta.replace(0, np.nan)
        capex_intensity = capex_intensity.clip(lower=0)
        efficiency = (rev_growth / (capex_intensity + self.capex_floor)).clip(-100, 100)

        score = _cs_zscore(gp_ta) + _cs_zscore(efficiency)

        strict_w = _sector_neutral_weights(
            score, membership, self.sectors, self.buy_quantile, ascending=False
        )
        extended_w = _sector_neutral_weights(
            score, membership, self.sectors, self.hold_quantile, ascending=False
        )
        return _buy_hold_band(strict_w, extended_w, membership, self.rebalance_period)


class AccountingSustainabilityQualityComposite(CrossSectionalStrategy):
    """H298: z(GP/TA) + z(-accruals) + z(-NOA) accounting sustainability composite.

    GP/TA (productive), low accruals (cash-backed earnings), low NOA (lean balance sheet).
    accruals = (NI - OCF) / total_assets; noa = (TA - CL) / TA.
    Tests 3-factor composite at the dilution boundary vs H265 2-factor filtered quality.
    """

    name = "accounting_sustainability_quality_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.current_liabilities = current_liabilities
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)

        ta_safe = ta.replace(0, np.nan)

        gp_ta = gp / ta_safe
        accruals = ((ni - ocf) / ta_safe).clip(-1, 1)
        noa = ((ta - cl) / ta_safe).clip(0, 5)

        score = (
            _cs_zscore(gp_ta).fillna(0.0)
            + _cs_zscore(-accruals).fillna(0.0)
            + _cs_zscore(-noa).fillna(0.0)
        )

        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class QInvestmentRank(CrossSectionalStrategy):
    """H351: Long lowest-quartile of asset-growth rate (q-factor investment signal).

    investment_rate = (total_assets_t - total_assets_{t-252bd}) / total_assets_{t-252bd}
    Low investment signals capital discipline and predicts higher future returns.
    Sector-neutral top 20% by lowest investment rate. Quarterly rebalance.
    """

    name = "q_investment_rank"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
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
        ta_safe = ta.where(ta > 0)
        inv_rate = (ta - ta.shift(self.lookback)) / ta_safe.shift(self.lookback)
        inv_rate = inv_rate.clip(-1.0, 5.0)
        daily_w = _sector_neutral_weights(
            inv_rate, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class QROELevel(CrossSectionalStrategy):
    """H352: Long top-quartile by ROE = net_income / book_equity (q-factor profitability).

    Persistent high ROE predicts higher future returns (Hou-Xue-Zhang 2015).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "q_roe_level"

    def __init__(
        self,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.book_equity = book_equity
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        be_safe = be.where(be > 0)
        roe = (ni / be_safe).clip(-2.0, 5.0)
        daily_w = _sector_neutral_weights(
            roe, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class QInvestProfCombo(CrossSectionalStrategy):
    """H353: z(-investment_rate) + z(ROE) composite -- dual q-factor signal.

    Combines capital discipline (low asset growth) and profitability (high ROE).
    Tests whether the q-factor framework adds incremental value vs single signals.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "q_invest_prof_combo"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
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
        be = _align_fund(self.book_equity, prices_panel)
        ta_safe = ta.where(ta > 0)
        inv_rate = ((ta - ta.shift(self.lookback)) / ta_safe.shift(self.lookback)).clip(-1.0, 5.0)
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        score = _cs_zscore(-inv_rate).fillna(0.0) + _cs_zscore(roe).fillna(0.0)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class QResidualMomentum(CrossSectionalStrategy):
    """H354: 12-1 momentum only for stocks where q-factor composite > median.

    Quality gate (low investment + high ROE) ensures momentum is backed by
    fundamentals, not deteriorating firms. Sector-neutral top 20%, monthly rebalance.
    """

    name = "q_residual_momentum"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        ta_lookback: int = 252,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_window = skip_window
        self.ta_lookback = ta_lookback
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ta_safe = ta.where(ta > 0)
        inv_rate = ((ta - ta.shift(self.ta_lookback)) / ta_safe.shift(self.ta_lookback)).clip(
            -1.0, 5.0
        )
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        q_score = _cs_zscore(-inv_rate).fillna(0.0) + _cs_zscore(roe).fillna(0.0)
        q_median = q_score.median(axis=1)
        q_gate = q_score.ge(q_median, axis=0)
        mom = prices_panel.pct_change(self.mom_window).shift(self.skip_window)
        gated_mom = mom.where(q_gate, np.nan)
        daily_w = _sector_neutral_weights(
            gated_mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class QFactorComposite(CrossSectionalStrategy):
    """H355: z(-investment) + z(ROE) + z(GP/TA) three-signal q-factor composite.

    Combines q-factor investment and profitability pillars with gross profitability.
    Tests whether q-factors add over stand-alone GP (H239 baseline).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "q_factor_composite"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        gross_profit: pd.DataFrame,
        sectors: pd.Series,
        ta_lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.gross_profit = gross_profit
        self.sectors = sectors
        self.ta_lookback = ta_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        ta_safe = ta.where(ta > 0)
        inv_rate = ((ta - ta.shift(self.ta_lookback)) / ta_safe.shift(self.ta_lookback)).clip(
            -1.0, 5.0
        )
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        score = (
            _cs_zscore(-inv_rate).fillna(0.0)
            + _cs_zscore(roe).fillna(0.0)
            + _cs_zscore(gp_ta).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class LowTurnoverGP(CrossSectionalStrategy):
    """H396: GP/TA rank with threshold rebalancing (only when rank moves >2 deciles).

    Rebalances a stock's weight only when its GP/TA rank changes by more than
    threshold_deciles × (1/n_deciles) of the cross-section. Average rebalance
    frequency halved vs quarterly. Hendershott et al. 2011: turnover costs
    compound; threshold rebalancing captures most signal at lower cost.
    Sector-neutral, threshold rebalance.
    """

    name = "low_turnover_gp"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        n_deciles: int = 10,
        threshold_deciles: int = 2,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.n_deciles = n_deciles
        self.threshold_deciles = threshold_deciles
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)

        mem = membership.reindex_like(gp_ta).fillna(False)
        pct_rank = (
            gp_ta.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep").fillna(0.0)
        )

        # Compute base sector-neutral weights at monthly frequency
        base_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        base_w_monthly = _periodic_rebalance(base_w, self.rebalance_period)

        # Apply threshold: only update if rank changed by > threshold_deciles
        threshold = self.threshold_deciles / self.n_deciles
        prev_rank = pct_rank.shift(self.rebalance_period)
        rank_change = (pct_rank - prev_rank).abs()
        rebalance_needed = rank_change.gt(threshold) | rank_change.isna()

        # Build thresholded weights: carry forward when no rebalance needed
        final_w = base_w_monthly.copy()
        for i in range(1, len(final_w)):
            for t in final_w.columns:
                if not rebalance_needed.iloc[i][t]:
                    final_w.iloc[i, final_w.columns.get_loc(t)] = final_w.iloc[
                        i - 1, final_w.columns.get_loc(t)
                    ]

        # Renormalize rows
        row_sum = final_w.sum(axis=1).replace(0, np.nan)
        return final_w.div(row_sum, axis=0).fillna(0.0)


class AccrualQualityPort(CrossSectionalStrategy):
    """H401: Long low-accrual stocks — cash earnings dominance as PEAD proxy.

    Accruals = (NI − OCF) / avg_TA. Low accruals = high-quality cash earnings.
    Market overvalues accrual-heavy earnings; corrects when cash reality hits
    (Sloan 1996; proxy for Bernard & Thomas delayed price response).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "accrual_quality_port"

    def __init__(
        self,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        avg_ta = ((ta + ta.shift(252)) / 2).where(ta > 0)
        # Accruals: lower is better (ascending=True in selection)
        accruals = ((ni - ocf) / avg_ta).clip(-1.0, 1.0)
        # ascending=True → smallest accruals rank first (best)
        daily_w = _sector_neutral_weights(
            accruals, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class EarningsMomentumQuality(CrossSectionalStrategy):
    """H402: Long top-quintile stocks by NI growth confirmed by OCF growth.

    Both NI/TA YoY growth and OCF/TA YoY growth positive and growing together
    signals genuine earnings quality (not accrual-inflated). Proxy for PEAD:
    stocks with persistent positive earnings surprises continue to outperform.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "earnings_momentum_quality"

    def __init__(
        self,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.total_assets = total_assets
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        ocf_ta = (ocf / ta_safe).clip(-2.0, 5.0)
        ni_growth = ni_ta - ni_ta.shift(self.lookback)
        ocf_growth = ocf_ta - ocf_ta.shift(self.lookback)
        composite = _cs_zscore(ni_growth).fillna(0.0) + _cs_zscore(ocf_growth).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class TorpedoRiskAvoid(CrossSectionalStrategy):
    """H403: Long high book/market + positive NI growth (torpedo-safe zone).

    Skinner & Sloan (2002): expensive growth stocks (low B/M) punished severely
    on negative earnings surprise. We avoid the torpedo quadrant (low B/M + NI
    declining) and concentrate on cheap/quality names. Market cap = price × shares.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "torpedo_risk_avoid"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.book_equity = book_equity
        self.net_income = net_income
        self.total_assets = total_assets
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
        be = _align_fund(self.book_equity, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        shares = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        # Market cap proxy; shares in millions, price in dollars
        mktcap = (prices_panel * shares.clip(lower=0.0)).replace(0, np.nan)
        bm_ratio = (be / mktcap).clip(0.0, 10.0)
        # NI growth (YoY change in NI/TA)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        ni_growth = ni_ta - ni_ta.shift(self.lookback)
        ni_growth_positive = (ni_growth > 0).astype(float)
        # Torpedo-safe score: high B/M × positive NI growth
        torpedo_safe = _cs_zscore(bm_ratio).fillna(0.0) + _cs_zscore(ni_growth_positive).fillna(0.0)
        daily_w = _sector_neutral_weights(
            torpedo_safe, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class RevenueEarningsAcceleration(CrossSectionalStrategy):
    """H404: Long top-quintile by simultaneous revenue AND NI growth acceleration.

    2nd derivative signal: Δ(revenue growth YoY) + Δ(NI growth YoY) > 0.
    Acceleration captures regime shift from poor to improving earnings — the
    setup for a positive surprise that hasn't been priced yet (PEAD proxy).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "revenue_earnings_acceleration"

    def __init__(
        self,
        revenue: pd.DataFrame,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        annual_lag: int = 252,
        accel_lag: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.revenue = revenue
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.annual_lag = annual_lag
        self.accel_lag = accel_lag
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        # YoY growth rates
        rev_growth = (rev - rev.shift(self.annual_lag)) / rev.shift(self.annual_lag).abs().clip(
            lower=1.0
        )
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        ni_growth = ni_ta - ni_ta.shift(self.annual_lag)
        # 2nd derivative: acceleration vs one quarter ago
        rev_accel = rev_growth - rev_growth.shift(self.accel_lag)
        ni_accel = ni_growth - ni_growth.shift(self.accel_lag)
        composite = _cs_zscore(rev_accel.clip(-2.0, 2.0)).fillna(0.0) + _cs_zscore(
            ni_accel.clip(-2.0, 2.0)
        ).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CashEarningsPersistence(CrossSectionalStrategy):
    """H405: Long stocks with persistently cash-dominant earnings.

    Rolling fraction of periods where OCF > 0 AND NI > 0 AND OCF/NI > 0.8.
    High persistence = analyst error risk is low → less torpedo risk → better returns.
    Proxy for So (2013): predictable analyst forecast errors from historical earnings patterns.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "cash_earnings_persistence"

    def __init__(
        self,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        window: int = 504,
        ocf_ni_threshold: float = 0.8,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.window = window
        self.ocf_ni_threshold = ocf_ni_threshold
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        # Indicator: OCF > 0, NI > 0, and OCF/NI >= threshold
        ni_pos = ni > 0
        ocf_pos = ocf > 0
        ocf_ni_ratio = (ocf / ni.replace(0, np.nan)).clip(0.0, 10.0)
        cash_dominant = (ni_pos & ocf_pos & (ocf_ni_ratio >= self.ocf_ni_threshold)).astype(float)
        # Rolling fraction of qualifying periods
        persistence = cash_dominant.rolling(self.window, min_periods=self.window // 2).mean()
        daily_w = _sector_neutral_weights(
            persistence, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class NetProfitMarginLevelPort(CrossSectionalStrategy):
    """H451: NI/Revenue (net profit margin) LEVEL.

    Persistent high net margin = structural pricing power + cost efficiency.
    H238 tested YoY NI/Revenue CHANGE; this tests the LEVEL.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "net_profit_margin_level_port"

    def __init__(
        self,
        net_income: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.revenue = revenue
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        signal = (ni / rev_safe).clip(-2.0, 2.0)
        daily_w = _sector_neutral_weights(
            signal, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class AssetTurnoverLevelPort(CrossSectionalStrategy):
    """H452: Revenue/TA (asset turnover) LEVEL.

    High asset turnover = efficient revenue generation per unit of capital.
    H230 tested YoY Revenue/TA CHANGE; this tests the LEVEL.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "asset_turnover_level_port"

    def __init__(
        self,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.revenue = revenue
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        signal = (rev / ta_safe).clip(0.0, 10.0)
        daily_w = _sector_neutral_weights(
            signal, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class ReturnOnEquityLevelPort(CrossSectionalStrategy):
    """H453: NI/BE (return on equity) LEVEL.

    Persistent high ROE = compounding structural advantage + efficient equity deployment.
    H221 tested YoY ROE CHANGE; this tests the LEVEL.
    Excludes negative book equity (BE <= 0 → NaN → neutral rank).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "return_on_equity_level_port"

    def __init__(
        self,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.book_equity = book_equity
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        be_safe = be.where(be > 0)
        signal = (ni / be_safe).clip(-2.0, 5.0)
        daily_w = _sector_neutral_weights(
            signal, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CashProfitMarginPort(CrossSectionalStrategy):
    """H454: OCF/Revenue (operating cash flow margin) LEVEL.

    High OCF margin = cash conversion efficiency; revenue converts to real cash.
    Not previously tested in any form.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "cash_profit_margin_port"

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
        signal = (ocf / rev_safe).clip(-2.0, 2.0)
        daily_w = _sector_neutral_weights(
            signal, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class GrossProfitFixedAssetsPort(CrossSectionalStrategy):
    """H455: GP/(TA-CA) = gross profit per unit of long-term/fixed assets.

    Rewards capital-light businesses generating high gross profit per fixed investment.
    TA-CA = long-term (structural) assets; excludes current assets from denominator.
    Not previously tested in any form.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "gross_profit_fixed_assets_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.current_assets = current_assets
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
        ca = _align_fund(self.current_assets, prices_panel)
        fixed_assets = (ta - ca).where(lambda x: x > 0)
        signal = (gp / fixed_assets).clip(0.0, 20.0)
        daily_w = _sector_neutral_weights(
            signal, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class GrossProfitDebtCoveragePort(CrossSectionalStrategy):
    """H456: GP/LTD (gross profit to long-term debt) LEVEL.

    Financial fortress: high GP relative to debt load.
    H237 tested OCF/LTD; this uses gross profit (more structural, pre-SGA).
    Firms with LTD < 1% of TA are excluded (NaN → neutral rank, not top rank).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "gross_profit_debt_coverage_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        min_debt_frac: float = 0.01,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.long_term_debt = long_term_debt
        self.total_assets = total_assets
        self.sectors = sectors
        self.min_debt_frac = min_debt_frac
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        # Only compute ratio for firms with meaningful debt (avoid near-zero LTD firms dominating)
        ltd_meaningful = ltd.where(ltd >= ta_safe * self.min_debt_frac)
        signal = (gp / ltd_meaningful).clip(0.0, 50.0)
        daily_w = _sector_neutral_weights(
            signal, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CashConversionEfficiencyPort(CrossSectionalStrategy):
    """H457: OCF/(CA-CL) = operating cash flow per unit of net working capital.

    Measures how efficiently the firm's working capital cycle generates cash.
    Firms with CA <= CL (negative working capital) are excluded (NaN → neutral).
    Not previously tested in any form.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "cash_conversion_efficiency_port"

    def __init__(
        self,
        operating_cf: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.operating_cf = operating_cf
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ocf = _align_fund(self.operating_cf, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        nwc = (ca - cl).where(lambda x: x > 0)
        signal = (ocf / nwc).clip(-5.0, 20.0)
        daily_w = _sector_neutral_weights(
            signal, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class NetMarginMomentumPort(CrossSectionalStrategy):
    """H458: z(NI/Revenue level) + z(6m price momentum). All-fold template.

    Net profit margin (full cost stack) + price trend confirmation.
    Monthly rebalance for all-fold coverage via momentum component.
    """

    name = "net_margin_momentum_port"

    def __init__(
        self,
        net_income: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.net_income = net_income
        self.revenue = revenue
        self.sectors = sectors
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        margin = (ni / rev_safe).clip(-2.0, 2.0)
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        composite = _cs_zscore(margin).fillna(0.0) + _cs_zscore(momentum).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class AssetTurnoverMomentumPort(CrossSectionalStrategy):
    """H459: z(Revenue/TA level) + z(6m price momentum). All-fold template.

    Asset productivity (operational efficiency) + price trend confirmation.
    Monthly rebalance for all-fold coverage via momentum component.
    """

    name = "asset_turnover_momentum_port"

    def __init__(
        self,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.revenue = revenue
        self.total_assets = total_assets
        self.sectors = sectors
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        turnover = (rev / ta_safe).clip(0.0, 10.0)
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        composite = _cs_zscore(turnover).fillna(0.0) + _cs_zscore(momentum).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class ROELevelMomentumPort(CrossSectionalStrategy):
    """H460: z(NI/BE level) + z(6m price momentum). All-fold template.

    ROE includes financial leverage (unlike GP/TA) + price trend confirmation.
    Excludes negative book equity. Monthly rebalance for all-fold coverage.
    """

    name = "roe_level_momentum_port"

    def __init__(
        self,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.net_income = net_income
        self.book_equity = book_equity
        self.sectors = sectors
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        be_safe = be.where(be > 0)
        roe = (ni / be_safe).clip(-2.0, 5.0)
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        composite = _cs_zscore(roe).fillna(0.0) + _cs_zscore(momentum).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class AdaptiveFrequencyQualityPort(CrossSectionalStrategy):
    """H500: GP/TA quality with VIX-adaptive rebalance frequency.

    High VIX (≥ vix_threshold, default 25): rebalance monthly (21 days).
    Low VIX (< vix_threshold): rebalance quarterly (63 days).
    H292 established quarterly beats monthly for GP/TA (0.884 vs 0.852).
    H500 tests whether high-vol regimes require faster signal refresh.
    """

    name = "adaptive_frequency_quality_port"

    def __init__(
        self,
        sectors: pd.Series,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        vix_series: pd.Series,
        vix_threshold: float = 25.0,
        period_high_vix: int = 21,
        period_low_vix: int = 63,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.sectors = sectors
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.vix_series = vix_series
        self.vix_threshold = vix_threshold
        self.period_high_vix = period_high_vix
        self.period_low_vix = period_low_vix
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        tickers = prices_panel.columns.tolist()
        idx = prices_panel.index

        gp = self.gross_profit.reindex(columns=tickers, index=idx)
        ta = self.total_assets.reindex(columns=tickers, index=idx)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)

        daily_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )

        vix = self.vix_series.reindex(idx).ffill().shift(1)
        high_vix = vix >= self.vix_threshold

        result = daily_w.copy()
        last_w = None
        days_since = 0
        for i in range(len(idx)):
            period = self.period_high_vix if bool(high_vix.iloc[i]) else self.period_low_vix
            if last_w is None or days_since >= period:
                last_w = daily_w.iloc[i].copy()
                days_since = 0
            result.iloc[i] = last_w
            days_since += 1

        return result


class AnnualRebalanceQualityPort(CrossSectionalStrategy):
    """H501: GP/TA quality, sector-neutral top 20%, annual rebalance (252 days).

    Identical signal to H227 (monthly) and H292 (quarterly). Tests whether
    annual rebalance further improves Sharpe by eliminating residual turnover costs.
    GP/TA is an annual filing signal — annual rebalance aligns cadence with information.
    """

    name = "annual_rebalance_quality_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 252,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
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
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        daily_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class AnnualRebalanceNetMarginPort(CrossSectionalStrategy):
    """H502: NI/Revenue (net profit margin) LEVEL, sector-neutral top 20%, annual rebalance.

    NI/Revenue level captures persistent full-cost-stack efficiency (including SGA,
    interest, taxes). Annual rebalance tests whether annual filing cadence optimal.
    Companion to H451 (monthly) to test frequency effect on net margin signal.
    """

    name = "annual_rebalance_net_margin_port"

    def __init__(
        self,
        net_income: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 252,
    ) -> None:
        self.net_income = net_income
        self.revenue = revenue
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        ni_margin = (ni / rev_safe).clip(-2.0, 5.0)
        daily_w = _sector_neutral_weights(
            ni_margin, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class SemiAnnualAssetTurnoverPort(CrossSectionalStrategy):
    """H503: Revenue/TA (asset turnover) LEVEL, sector-neutral top 20%, semi-annual rebalance (126 days).

    Revenue/TA captures structural business model efficiency. Semi-annual cadence
    sits between quarterly (H452) and annual — tests optimal frequency for this signal.
    """

    name = "semi_annual_asset_turnover_port"

    def __init__(
        self,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 126,
    ) -> None:
        self.revenue = revenue
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        asset_turn = (rev / ta_safe).clip(0.0, 10.0)
        daily_w = _sector_neutral_weights(
            asset_turn, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class LowTurnoverNetMarginPort(CrossSectionalStrategy):
    """H504: NI/Revenue LEVEL with buy-hold band (enter top 20%, exit below 25th percentile).

    Quarterly rebalance with asymmetric entry/exit thresholds to reduce turnover.
    Buy: top 20% NI/Revenue (sector-neutral). Hold: until ranked below 25th percentile.
    Tests whether reducing exit sensitivity improves net margin signal net performance.
    """

    name = "low_turnover_net_margin_port"

    def __init__(
        self,
        net_income: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        buy_quantile: float = 0.20,
        hold_quantile: float = 0.75,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.revenue = revenue
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
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        ni_margin = (ni / rev_safe).clip(-2.0, 5.0)
        strict_w = _sector_neutral_weights(
            ni_margin, membership, self.sectors, self.buy_quantile, ascending=False
        )
        extended_w = _sector_neutral_weights(
            ni_margin, membership, self.sectors, self.hold_quantile, ascending=False
        )
        return _buy_hold_band(strict_w, extended_w, membership, self.rebalance_period)


class LargeCapQualityFilter(CrossSectionalStrategy):
    """H507: GP/TA quality restricted to stocks above sector-median market cap.

    Market cap = price × shares_outstanding (point-in-time, 63d lag).
    Filter to top 50% by market cap within each sector before applying quality ranking.
    Large-cap stocks have better liquidity → lower realized slippage than 1.5 bps assumed.
    Sector-neutral top 20% within large-cap filtered universe. Quarterly rebalance.
    """

    name = "large_cap_quality_filter"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        cap_include_pct: float = 0.50,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.cap_include_pct = cap_include_pct
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        shares = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        mktcap = (prices_panel * shares.clip(lower=0.0)).replace(0, np.nan)
        mem_f = membership.reindex_like(mktcap).fillna(False)

        # Identify large-cap stocks (top cap_include_pct by market cap within sector)
        large_cap = pd.DataFrame(False, index=prices_panel.index, columns=prices_panel.columns)
        for sector in self.sectors.dropna().unique():
            tickers = [
                t for t in self.sectors[self.sectors == sector].index if t in prices_panel.columns
            ]
            if not tickers:
                continue
            sec_cap = mktcap[tickers].where(mem_f[tickers])
            n_mem = mem_f[tickers].sum(axis=1)
            # Threshold: rank > (1 - cap_include_pct) × n_members
            thresh = (n_mem * (1.0 - self.cap_include_pct)).clip(lower=0)
            rank = sec_cap.rank(axis=1, ascending=True, na_option="keep")
            large_cap[tickers] = rank.gt(thresh, axis=0)

        large_cap_mem = membership & large_cap
        daily_w = _sector_neutral_weights(
            gp_ta, large_cap_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class LowTurnoverROEPort(CrossSectionalStrategy):
    """H509: NI/BE (ROE) LEVEL with buy-hold band (enter top 20%, exit below 25th percentile).

    Quarterly rebalance. ROE level captures sustained equity capital efficiency.
    Asymmetric entry/exit: buy top 20%, hold until below 25th percentile.
    Negative book equity stocks excluded (assigned NaN signal → bottom rank).
    """

    name = "low_turnover_roe_port"

    def __init__(
        self,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        buy_quantile: float = 0.20,
        hold_quantile: float = 0.75,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.book_equity = book_equity
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
        be = _align_fund(self.book_equity, prices_panel)
        # Exclude negative book equity (leveraged distress)
        be_positive = be.where(be > 0)
        roe = (ni / be_positive).clip(-1.0, 5.0)
        strict_w = _sector_neutral_weights(
            roe, membership, self.sectors, self.buy_quantile, ascending=False
        )
        extended_w = _sector_neutral_weights(
            roe, membership, self.sectors, self.hold_quantile, ascending=False
        )
        return _buy_hold_band(strict_w, extended_w, membership, self.rebalance_period)


class GrossCrossEfficiencyPort(CrossSectionalStrategy):
    """H523: (GP/TA) x (Revenue/TA) = GP*Revenue/TA^2. Long top 20%. Quarterly.

    Rewards firms excelling on BOTH gross profitability AND asset efficiency simultaneously.
    GP/TA (H227: 0.879) and Revenue/TA are both independently alpha-generating.
    Their product selects firms that are strong on both dimensions, not just one.
    """

    name = "gross_cross_efficiency_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        rev_ta = (rev / ta_safe).clip(0.0, 10.0)
        cross_eff = (gp_ta * rev_ta).clip(0.0, 50.0)
        daily_w = _sector_neutral_weights(
            cross_eff, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CashCycleEfficiencyPort(CrossSectionalStrategy):
    """H524: OCF / (CA - CL) = cash flow per unit net working capital. Long top 20%. Quarterly.

    Measures how efficiently the operating cycle converts working capital to cash.
    Firms with CA < CL (negative NWC) are excluded (assigned NaN signal).
    High ratio = lean operations; low ratio = capital trapped in operating cycle.
    """

    name = "cash_cycle_efficiency_port"

    def __init__(
        self,
        operating_cf: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.operating_cf = operating_cf
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ocf = _align_fund(self.operating_cf, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel).clip(lower=0.0)
        cl = _align_fund(self.current_liabilities, prices_panel).clip(lower=0.0)
        nwc = ca - cl
        # Only include firms with positive NWC (ca > cl)
        nwc_positive = nwc.where(nwc > 0)
        cash_cycle = (ocf / nwc_positive).clip(-5.0, 20.0)
        daily_w = _sector_neutral_weights(
            cash_cycle, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class SectorNeutralizedROEPort(CrossSectionalStrategy):
    """H525: NI/BE demeaned by sector median NI/BE. Long highest sector-adjusted ROE. Quarterly.

    Removes structural ROE level differences across sectors (financials naturally have high
    ROE from leverage; sector demeaning removes this bias). Distinct from H453 which ranks
    ROE within sector: H525 uses the explicit demeaned value, allowing cross-sector comparison.
    """

    name = "sector_neutralized_roe_port"

    def __init__(
        self,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.book_equity = book_equity
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        be_positive = be.where(be > 0)
        roe = (ni / be_positive).clip(-1.0, 5.0)
        # Sector-demean: subtract sector median per date
        roe_demeaned = roe.copy()
        for sector in self.sectors.dropna().unique():
            cols = [t for t in self.sectors[self.sectors == sector].index if t in roe.columns]
            if not cols:
                continue
            sector_med = roe[cols].median(axis=1)
            roe_demeaned[cols] = roe[cols].values - sector_med.values[:, None]
        daily_w = _sector_neutral_weights(
            roe_demeaned, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class ProfitabilityConsistencyPort(CrossSectionalStrategy):
    """H526: Fraction of trailing 8 quarters where NI/BE > sector median NI/BE. Quarterly.

    Persistent above-sector ROE = structural competitive advantage (durable moat).
    H225 tested earnings growth persistence; H526 tests ROE LEVEL persistence above benchmark.
    Rolling 504-day window approximates 8 quarterly periods.
    """

    name = "profitability_consistency_port"

    def __init__(
        self,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.book_equity = book_equity
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        be_positive = be.where(be > 0)
        roe = (ni / be_positive).clip(-1.0, 5.0)
        # Compute sector median ROE per date
        sector_med_roe = pd.DataFrame(0.0, index=roe.index, columns=roe.columns)
        for sector in self.sectors.dropna().unique():
            cols = [t for t in self.sectors[self.sectors == sector].index if t in roe.columns]
            if not cols:
                continue
            med = roe[cols].median(axis=1)
            sector_med_roe[cols] = np.tile(med.values[:, None], (1, len(cols)))
        # Fraction of days in lookback where ROE > sector median
        above_med = (roe > sector_med_roe).astype(float)
        consistency = above_med.rolling(self.lookback, min_periods=self.lookback // 2).mean()
        daily_w = _sector_neutral_weights(
            consistency, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class RevenueMarginConsistencyPort(CrossSectionalStrategy):
    """H527: Fraction of 8 quarters where rev_growth>0 AND GP/Revenue change > -1%. Quarterly.

    Joint persistence condition: sustained revenue growth AND margin stability simultaneously.
    H411-H415 tested these signals separately. H527 requires BOTH conditions for 8 consecutive
    quarters -- a joint quality-growth persistence signal.
    """

    name = "revenue_margin_consistency_port"

    def __init__(
        self,
        revenue: pd.DataFrame,
        gross_profit: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 504,
        quarterly_step: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.revenue = revenue
        self.gross_profit = gross_profit
        self.sectors = sectors
        self.lookback = lookback
        self.quarterly_step = quarterly_step
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        rev_safe = rev.where(rev > 0)
        # Quarterly revenue growth > 0
        rev_growth = rev.pct_change(self.quarterly_step)
        rev_growth_pos = (rev_growth > 0).astype(float)
        # Quarterly GP margin change > -1%
        gp_margin = (gp / rev_safe).clip(0.0, 2.0)
        margin_change = gp_margin.pct_change(self.quarterly_step)
        margin_stable = (margin_change > -0.01).astype(float)
        # Both conditions simultaneously
        both = rev_growth_pos * margin_stable
        consistency = both.rolling(self.lookback, min_periods=self.lookback // 2).mean()
        daily_w = _sector_neutral_weights(
            consistency, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class NetMarginAssetTurnoverCombo(CrossSectionalStrategy):
    """H529: z(NI/Revenue level) + z(Revenue/TA level). 2-factor DuPont z-score. Quarterly.

    DuPont decomposition: NI/TA = NI/Revenue x Revenue/TA (ROA = margin x turnover).
    Ranking on BOTH z-scores separately rewards firms strong on both margins AND efficiency.
    Different from H173 (ROA = product): z-score sum selects firms excelling on each dimension.
    """

    name = "net_margin_asset_turnover_combo"

    def __init__(
        self,
        net_income: pd.DataFrame,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.revenue = revenue
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        rev_safe = rev.where(rev > 0)
        ta_safe = ta.where(ta > 0)
        ni_margin = (ni / rev_safe).clip(-2.0, 5.0)
        asset_turn = (rev / ta_safe).clip(0.0, 10.0)
        composite = _cs_zscore(ni_margin).fillna(0.0) + _cs_zscore(asset_turn).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class SectorDeltaQualityPort(CrossSectionalStrategy):
    """H531: GP/TA - sector_median(GP/TA). Long largest positive deviation. Quarterly.

    Sector-relative quality premium: firms MUCH better than sector peers get highest score.
    H227 ranks GP/TA within sector (top 20% per sector). H531 uses absolute deviation --
    selects stocks with largest surplus over sector median, can concentrate in best sectors.
    """

    name = "sector_delta_quality_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
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
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        gp_ta_delta = gp_ta.copy()
        for sector in self.sectors.dropna().unique():
            cols = [t for t in self.sectors[self.sectors == sector].index if t in gp_ta.columns]
            if not cols:
                continue
            med = gp_ta[cols].median(axis=1)
            gp_ta_delta[cols] = gp_ta[cols].values - med.values[:, None]
        daily_w = _sector_neutral_weights(
            gp_ta_delta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class TwoStageQualityFilter(CrossSectionalStrategy):
    """H532: Stage 1: filter top 40% by GP/TA. Stage 2: rank NI/Revenue, take top 20%. Quarterly.

    Sequential double-quality filter: GP/TA gates into high-quality universe, then NI/Revenue
    selects the most profitable among them. H265 (filter+rank: 0.952) used accruals gate;
    H532 uses NI/Revenue (full cost-stack profitability) as second quality dimension.
    """

    name = "two_stage_quality_filter"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        stage1_quantile: float = 0.40,
        stage2_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.revenue = revenue
        self.sectors = sectors
        self.stage1_quantile = stage1_quantile
        self.stage2_quantile = stage2_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        ta_safe = ta.where(ta > 0)
        rev_safe = rev.where(rev > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        ni_margin = (ni / rev_safe).clip(-2.0, 5.0)
        top40_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.stage1_quantile, ascending=False
        )
        filtered_mem = membership & (top40_w > 0)
        daily_w = _sector_neutral_weights(
            ni_margin, filtered_mem, self.sectors, self.stage2_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class GPTAAccelerationPort(CrossSectionalStrategy):
    """H534: Second derivative of GP/TA (acceleration of quality improvement). Quarterly.

    d1 = GP/TA(t) - GP/TA(t-1Q)  [quarterly improvement rate]
    d2 = d1(t) - d1(t-1Q)         [acceleration: rate of change of improvement]
    Long top 20% with fastest-accelerating GP/TA. Inflection-point signal.
    """

    name = "gpta_acceleration_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        quarterly_step: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.quarterly_step = quarterly_step
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        d1 = gp_ta - gp_ta.shift(self.quarterly_step)
        d2 = (d1 - d1.shift(self.quarterly_step)).clip(-2.0, 2.0)
        daily_w = _sector_neutral_weights(
            d2, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)
