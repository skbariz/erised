"""Multi-factor composites and portfolio-construction variants combining several signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import (
    _align_fund,
    _cs_rank_norm,
    _cs_zscore,
    _equal_weight_top_quantile,
    _monthly_rebalance,
    _periodic_rebalance,
    _sector_neutral_weights,
)
from .base import CrossSectionalStrategy


class CompositeValueMomentum(CrossSectionalStrategy):
    """H162: 50/50 cross-sectional z-score blend of momentum and long-term reversal.

    score = 0.5 * z(12-1m return) + 0.5 * z(−36m return)
    Negative sign on 36m makes losers score positively (value component).
    """

    name = "composite_value_momentum"

    def __init__(
        self,
        long_days: int = 252,
        skip_days: int = 21,
        reversal_days: int = 756,
        top_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.long_days = long_days
        self.skip_days = skip_days
        self.reversal_days = reversal_days
        self.top_quantile = top_quantile
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        mom = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1
        rev = prices_panel.shift(self.skip_days) / prices_panel.shift(self.reversal_days) - 1

        def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
            mu = df.mean(axis=1)
            sd = df.std(axis=1).replace(0, np.nan)
            return df.sub(mu, axis=0).div(sd, axis=0)

        mom_z = _cs_zscore(mom)
        rev_z = _cs_zscore(-rev)  # negate: loser → high z-score
        score = 0.5 * mom_z + 0.5 * rev_z

        daily_w = _equal_weight_top_quantile(score, membership, self.top_quantile, ascending=False)
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class ValueMomentumComposite(CrossSectionalStrategy):
    """H162: 50/50 z-score blend of value (B/M + E/P) and 12-1m momentum.

    composite = 0.5 × z(value_rank) + 0.5 × z(momentum)
    Value component updates quarterly (on filing dates).
    Momentum component updates daily. Portfolio rebalances monthly.
    """

    name = "value_momentum_composite"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        net_income: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        long_days: int = 252,
        skip_days: int = 21,
        top_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.book_equity = book_equity
        self.net_income = net_income
        self.shares_outstanding = shares_outstanding
        self.long_days = long_days
        self.skip_days = skip_days
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
        bm = (be / mktcap).where(be > 0)
        ep = (ni / mktcap).where(ni > 0)
        value_composite = 0.5 * _cs_rank_norm(bm) + 0.5 * _cs_rank_norm(ep)

        mom = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1

        score = 0.5 * _cs_zscore(value_composite) + 0.5 * _cs_zscore(mom)
        daily_w = _equal_weight_top_quantile(score, membership, self.top_quantile, ascending=False)
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class SectorNeutralValueMomentum(CrossSectionalStrategy):
    """H163: Value+Momentum composite (H162) ranked within GICS sectors.

    For each of the 11 GICS sectors: rank stocks by the composite z-score
    within the sector, select top 30% within sector, scale each sector to
    equal weight in the portfolio.

    sectors: pd.Series, index=ticker, values=GICS sector string.
    """

    name = "sector_neutral_value_momentum"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        net_income: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        long_days: int = 252,
        skip_days: int = 21,
        within_sector_quantile: float = 0.30,
        monthly_rebalance: bool = True,
    ):
        self.book_equity = book_equity
        self.net_income = net_income
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.long_days = long_days
        self.skip_days = skip_days
        self.within_sector_quantile = within_sector_quantile
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
        bm = (be / mktcap).where(be > 0)
        ep = (ni / mktcap).where(ni > 0)
        value_composite = 0.5 * _cs_rank_norm(bm) + 0.5 * _cs_rank_norm(ep)

        mom = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1
        score = 0.5 * _cs_zscore(value_composite) + 0.5 * _cs_zscore(mom)

        mem_f = membership.reindex_like(score).fillna(False)
        unique_sectors = self.sectors.dropna().unique()
        sector_weights = pd.DataFrame(0.0, index=prices_panel.index, columns=prices_panel.columns)

        for sector in unique_sectors:
            sec_tickers = self.sectors[self.sectors == sector].index.tolist()
            sec_tickers = [t for t in sec_tickers if t in score.columns]
            if not sec_tickers:
                continue
            sec_score = score[sec_tickers]
            sec_mem = mem_f[sec_tickers]
            sec_w = _equal_weight_top_quantile(
                sec_score, sec_mem, self.within_sector_quantile, ascending=False
            )
            sector_weights[sec_tickers] = sec_w

        # Scale so all sectors contribute equally: divide each sector's weights
        # by the number of active sectors on each date
        sec_totals: dict[str, pd.Series] = {}
        for sector in unique_sectors:
            sec_tickers = [
                t
                for t in self.sectors[self.sectors == sector].index.tolist()
                if t in sector_weights.columns
            ]
            sec_totals[sector] = sector_weights[sec_tickers].sum(axis=1)

        n_active = sum((st > 0).astype(int) for st in sec_totals.values())
        n_active = n_active.replace(0, np.nan)

        for sector in unique_sectors:
            sec_tickers = [
                t
                for t in self.sectors[self.sectors == sector].index.tolist()
                if t in sector_weights.columns
            ]
            st = sec_totals[sector].replace(0, np.nan)
            scale = (1.0 / n_active) / st
            sector_weights[sec_tickers] = sector_weights[sec_tickers].mul(scale, axis=0)

        daily_w = sector_weights.fillna(0.0)
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class FF5Composite(CrossSectionalStrategy):
    """H172: (1/3)*z(B/M) + (1/3)*z(GP/BE) + (1/3)*z(-delta_TA/TA), sector-neutral.

    Combines the three non-market FF5 signals: value (HML), profitability (RMW),
    and investment (CMA, negated so low growth ranks high).
    """

    name = "ff5_composite"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        growth_window: int = 252,
        monthly_rebalance: bool = True,
    ):
        self.book_equity = book_equity
        self.shares_outstanding = shares_outstanding
        self.gross_profit = gross_profit
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
        be = _align_fund(self.book_equity, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)

        mktcap = prices_panel * sh
        bm = (be / mktcap).where(be > 0)
        op = (gp / be).where(be > 0)

        ta_lag = ta.shift(self.growth_window)
        asset_growth = (ta - ta_lag) / ta_lag.where(ta_lag > 0)

        composite = (
            _cs_zscore(bm).fillna(0.0)
            + _cs_zscore(op).fillna(0.0)
            + _cs_zscore(-asset_growth).fillna(0.0)  # negate: low growth ranks high
        ) / 3.0

        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class MultiSignalComposite(CrossSectionalStrategy):
    """H199: Equal-weight z-score composite of momentum, reversal, and earnings.

    Blends three orthogonal signals:
      1. 12-1 month price momentum (H150 signal)
      2. 5-day short-term reversal negated (H185 signal, aligned: losers = positive)
      3. YoY NI growth SUE proxy (H180 signal)

    Each z-scored cross-sectionally before blending. NaN signals drop gracefully.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "multi_signal_composite"

    def __init__(
        self,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        reversal_window: int = 5,
        yoy_window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.net_income = net_income
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_window = skip_window
        self.reversal_window = reversal_window
        self.yoy_window = yoy_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        z_mom = _cs_zscore(prices_panel.pct_change(self.mom_window).shift(self.skip_window))
        z_rev = _cs_zscore(-prices_panel.pct_change(self.reversal_window))
        ni = _align_fund(self.net_income, prices_panel)
        ni_lag = ni.shift(self.yoy_window)
        sue = (ni - ni_lag) / ni_lag.abs().replace(0, np.nan)
        z_sue = _cs_zscore(sue)

        composite = z_mom.add(z_rev, fill_value=0.0).add(z_sue.fillna(0.0), fill_value=0.0)
        n_signals = (
            z_mom.notna().astype(float) + z_rev.notna().astype(float) + z_sue.notna().astype(float)
        )
        composite = composite / n_signals.replace(0, np.nan)

        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class MultiFactorComposite(CrossSectionalStrategy):
    """H234: z(mom) + z(GP/TA) + z(-BS_accruals) + z(-asset_growth) -- 4-factor composite."""

    name = "multi_factor_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_days: int = 21,
        fundamental_window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_days = skip_days
        self.fundamental_window = fundamental_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)

        w = self.fundamental_window
        # Price momentum
        price_ret = prices_panel.pct_change(self.mom_window - self.skip_days).shift(self.skip_days)
        # Gross profitability (quality)
        gp_ta = gp / ta.where(ta > 0)
        # Balance-sheet accruals (lower = better quality)
        nwc = ca - cl
        avg_ta = (ta + ta.shift(w)) / 2
        accruals = (nwc - nwc.shift(w)) / avg_ta.where(avg_ta > 0)
        # Asset growth (lower = better capital discipline)
        asset_growth = (ta - ta.shift(w)) / ta.shift(w).where(ta.shift(w) > 0)

        score = (
            _cs_zscore(price_ret)
            + _cs_zscore(gp_ta)
            + _cs_zscore(-accruals)
            + _cs_zscore(-asset_growth)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class GrandFundamentalComposite(CrossSectionalStrategy):
    """H250: Terminal composite: z(GP/TA) + z(-accruals) + z(-asset_growth) + z(OCF/TA) + z(-LTD/TA).

    Combines the best cross-batch signals: Novy-Marx profitability, accruals quality,
    conservative investment, cash generation, and low leverage.
    """

    name = "grand_fundamental_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        operating_cf: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.operating_cf = operating_cf
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.long_term_debt = long_term_debt
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
        ocf = _align_fund(self.operating_cf, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)

        # Novy-Marx profitability
        gp_ta = gp / ta.where(ta > 0)

        # Balance-sheet accruals (Sloan): delta(NWC) / avg_TA
        nwc = ca - cl
        avg_ta = (ta + ta.shift(self.window)) / 2
        accruals = (nwc - nwc.shift(self.window)) / avg_ta.where(avg_ta > 0)

        # Conservative investment: low asset growth
        asset_growth = (ta - ta.shift(self.window)) / ta.shift(self.window).where(
            ta.shift(self.window) > 0
        )

        # Cash generation
        ocf_ta = ocf / ta.where(ta > 0)

        # Low leverage
        ltd_ta = ltd / ta.where(ta > 0)

        score = (
            _cs_zscore(gp_ta)
            + _cs_zscore(-accruals)
            + _cs_zscore(-asset_growth)
            + _cs_zscore(ocf_ta)
            + _cs_zscore(-ltd_ta)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class SpreadAwarePositionSizer(CrossSectionalStrategy):
    """H397: GP/TA quality with position sizes inversely proportional to realized vol.

    Realized vol proxies for effective spread (higher vol = wider spread for
    large caps). Weights scaled by inv_vol within the selected top-quality basket.
    Hendershott et al. 2011: effective spread varies even within S&P 500.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "spread_aware_position_sizer"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        vol_window: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.vol_window = vol_window
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
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0)
        sust = _cs_zscore(-accruals).fillna(0.0) + _cs_zscore(-asset_growth).fillna(0.0)
        mem = membership.reindex_like(gp_ta).fillna(False)
        sust_rank = sust.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        filtered_mem = membership & (sust_rank >= 0.50).fillna(False)
        base_w = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        realized_vol = (
            prices_panel.pct_change().rolling(self.vol_window, min_periods=20).std().shift(1)
        )
        inv_vol = (1.0 / realized_vol.replace(0, np.nan)).clip(0.0, 100.0)
        # Scale within selected basket
        selected = base_w > 0
        scaled = base_w * inv_vol.where(selected, 0.0)
        row_sum = scaled.sum(axis=1).replace(0, np.nan)
        rescaled = scaled.div(row_sum, axis=0).fillna(0.0)
        return _periodic_rebalance(rescaled, self.rebalance_period)


class LiquidityRegimeCapPort(CrossSectionalStrategy):
    """H398: GP/TA quality with liquidity-regime participation cap.

    When a stock's realized vol drops below its 10th historical percentile
    (unusual quiet = potential liquidity event ahead), cap position at 50%.
    When realized vol is above 90th percentile (illiquid stress), cap at 30%.
    Normal regime: full weight. Hendershott et al. 2011: liquidity regime matters.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "liquidity_regime_cap_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        vol_window: int = 21,
        lookback: int = 252,
        low_pct: float = 0.10,
        high_pct: float = 0.90,
        low_cap: float = 0.50,
        high_cap: float = 0.30,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.vol_window = vol_window
        self.lookback = lookback
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.low_cap = low_cap
        self.high_cap = high_cap
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
        base_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        realized_vol = (
            prices_panel.pct_change().rolling(self.vol_window, min_periods=5).std().shift(1)
        )
        vol_low = realized_vol.rolling(self.lookback, min_periods=63).quantile(self.low_pct)
        vol_high = realized_vol.rolling(self.lookback, min_periods=63).quantile(self.high_pct)
        in_low_regime = realized_vol.lt(vol_low)
        in_high_regime = realized_vol.gt(vol_high)
        cap = pd.DataFrame(1.0, index=prices_panel.index, columns=prices_panel.columns)
        cap = cap.where(~in_low_regime, self.low_cap)
        cap = cap.where(~in_high_regime, self.high_cap)
        scaled = (base_w * cap).clip(lower=0.0)
        row_sum = scaled.sum(axis=1).replace(0, np.nan)
        rescaled = scaled.div(row_sum, axis=0).fillna(0.0)
        return _periodic_rebalance(rescaled, self.rebalance_period)


class ExecutionOptimizedComposite(CrossSectionalStrategy):
    """H399: Normalized blend of batch-10 execution-refined signals.

    Combines: spread-aware GP (vol-scaled), liquidity-regime capped GP, and
    plain GP/TA anchor. Tests whether execution cost adjustments collectively
    improve net-of-cost risk-adjusted returns.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "execution_optimized_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
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
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0)
        sust = _cs_zscore(-accruals).fillna(0.0) + _cs_zscore(-asset_growth).fillna(0.0)
        mem = membership.reindex_like(gp_ta).fillna(False)
        sust_rank = sust.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        filtered_mem = membership & (sust_rank >= 0.50).fillna(False)
        # Component 1: spread-aware (inv-vol scaled) within quality filter
        realized_vol = prices_panel.pct_change().rolling(63, min_periods=20).std().shift(1)
        inv_vol = (1.0 / realized_vol.replace(0, np.nan)).clip(0.0, 100.0)
        base_filt = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        spread_w = base_filt * inv_vol.where(base_filt > 0, 0.0)
        rs1 = spread_w.sum(axis=1).replace(0, np.nan)
        w1 = spread_w.div(rs1, axis=0).fillna(0.0)
        # Component 2: liquidity-regime capped plain GP
        vol_low = realized_vol.rolling(252, min_periods=63).quantile(0.10)
        vol_high = realized_vol.rolling(252, min_periods=63).quantile(0.90)
        in_low = realized_vol.lt(vol_low)
        in_high = realized_vol.gt(vol_high)
        cap = pd.DataFrame(1.0, index=prices_panel.index, columns=prices_panel.columns)
        cap = cap.where(~in_low, 0.50).where(~in_high, 0.30)
        base_plain = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        regime_w = (base_plain * cap).clip(lower=0.0)
        rs2 = regime_w.sum(axis=1).replace(0, np.nan)
        w2 = regime_w.div(rs2, axis=0).fillna(0.0)
        # Component 3: plain GP anchor
        w3 = base_plain.copy()
        combined = w1 + w2 + w3
        row_sum = combined.sum(axis=1).replace(0, np.nan)
        daily_w = combined.div(row_sum, axis=0).fillna(0.0)
        return _periodic_rebalance(daily_w, self.rebalance_period)


class Set6FundamentalQualityBlend(CrossSectionalStrategy):
    """H449: Balance sheet strength + Altman Z proxy + shareholder alignment composite.

    z(BS improvement) + z(Altman Z proxy) + z(low dilution + OCF).
    Multi-dimensional fundamental quality across all three Set 6 quality themes.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "set6_fundamental_quality_blend"

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
        revenue: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.book_equity = book_equity
        self.long_term_debt = long_term_debt
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.revenue = revenue
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
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        shares = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        ltd_safe = ltd.where(ltd > 0).fillna(1.0)
        # Balance sheet improvement (like H409)
        debt_improve = -(ltd / ta_safe - (ltd / ta_safe).shift(self.lookback)).clip(-2.0, 2.0)
        ocf_improve = (ocf / ta_safe - (ocf / ta_safe).shift(self.lookback)).clip(-2.0, 2.0)
        be_improve = (be / ta_safe - (be / ta_safe).shift(self.lookback)).clip(-2.0, 2.0)
        bs_score = (
            _cs_zscore(debt_improve).fillna(0.0)
            + _cs_zscore(ocf_improve).fillna(0.0)
            + _cs_zscore(be_improve).fillna(0.0)
        )
        # Altman Z proxy (like H434)
        ni_mean = (ni / ta_safe).rolling(self.lookback, min_periods=63).mean().clip(-2.0, 2.0)
        z_proxy = (
            1.2 * ((ca - cl) / ta_safe).clip(-2.0, 2.0)
            + 1.4 * ni_mean
            + 3.3 * (gp / ta_safe).clip(0.0, 5.0)
            + 0.6 * (be / ltd_safe).clip(0.0, 20.0)
            + 1.0 * (rev / ta_safe).clip(0.0, 10.0)
        )
        # Shareholder alignment (like H441 simplified)
        no_dilution = (shares <= shares.shift(self.lookback)).fillna(False).astype(float)
        ocf_pos = (ocf > 0).fillna(False).astype(float)
        alignment_score = _cs_zscore(no_dilution + ocf_pos).fillna(0.0)
        composite = (
            _cs_zscore(bs_score).fillna(0.0) + _cs_zscore(z_proxy).fillna(0.0) + alignment_score
        )
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class ReturnConsistencyPort(CrossSectionalStrategy):
    """H463: Long stocks with highest fraction of positive return days.

    Fraction of positive-return days over trailing 252 days. High = consistent winner.
    Pure price signal → full 12-fold coverage. Distinct from H155 SmoothMomentum
    (multiplicative path modifier); this ranks independently on consistency alone.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "return_consistency_port"

    def __init__(
        self,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        pos_frac = (daily_rets > 0).rolling(self.window, min_periods=self.window // 4).mean()
        # ascending=False: highest fraction (most consistent) selected
        daily_w = _sector_neutral_weights(
            pos_frac, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class UpsideSharpePort(CrossSectionalStrategy):
    """H465: Long stocks with highest upside-Sharpe ratio.

    Signal: mean(positive-day returns) / total daily return std, over 252 days.
    Rewards stocks efficient on the upside relative to total risk.
    Not equivalent to negated skewness — combines upside magnitude and frequency.
    Sector-neutral top 20%, monthly rebalance. Full 12-fold.
    """

    name = "upside_sharpe_port"

    def __init__(
        self,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        min_p = self.window // 4
        up_rets = daily_rets.where(daily_rets > 0)
        mean_up = up_rets.rolling(self.window, min_periods=min_p).mean()
        total_vol = daily_rets.rolling(self.window, min_periods=min_p).std().replace(0, np.nan)
        upside_sharpe = mean_up.div(total_vol, axis=1)
        # ascending=False: highest upside Sharpe selected
        daily_w = _sector_neutral_weights(
            upside_sharpe, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class RegimeAdaptiveCSSignal(CrossSectionalStrategy):
    """H497: VIX-regime-conditional CS signal selection.

    VIX < vix_threshold (default 20): use 6m price momentum (top 20%).
    VIX ≥ vix_threshold: use GP/TA quality (top 20%).
    Sector-neutral, monthly rebalance. Tests momentum-calm / quality-stress hypothesis.
    """

    name = "regime_adaptive_cs_signal"

    def __init__(
        self,
        sectors: pd.Series,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        vix_series: pd.Series,
        vix_threshold: float = 20.0,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.vix_series = vix_series
        self.vix_threshold = vix_threshold
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

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

        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1

        vix = self.vix_series.reindex(idx).ffill().shift(1)
        stressed = vix >= self.vix_threshold

        selected = pd.DataFrame(
            np.where(stressed.values[:, None], gp_ta.values, momentum.values),
            index=idx,
            columns=tickers,
        )
        daily_w = _sector_neutral_weights(
            selected, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class VolAccelerationCSSwitch(CrossSectionalStrategy):
    """H499: SPY vol acceleration-conditional CS signal selection.

    SPY 21d realized vol > 63d realized vol × 1.2 (vol accelerating):
        use GP/TA quality signal (top 20%, sector-neutral).
    Otherwise (vol normal or decelerating):
        use 6m price momentum (top 20%, sector-neutral).
    Monthly rebalance. Applies H327's vol-acceleration concept to CS signal selection.
    """

    name = "vol_acceleration_cs_switch"

    def __init__(
        self,
        sectors: pd.Series,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        spy_returns: pd.Series,
        accel_ratio: float = 1.20,
        vol_short: int = 21,
        vol_long: int = 63,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.spy_returns = spy_returns
        self.accel_ratio = accel_ratio
        self.vol_short = vol_short
        self.vol_long = vol_long
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

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

        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1

        spy_r = self.spy_returns.reindex(idx).fillna(0.0)
        vol_21 = spy_r.rolling(self.vol_short, min_periods=self.vol_short // 2).std()
        vol_63 = spy_r.rolling(self.vol_long, min_periods=self.vol_long // 2).std()
        accel = (vol_21 > vol_63 * self.accel_ratio).shift(1)

        selected = pd.DataFrame(
            np.where(accel.values[:, None], gp_ta.values, momentum.values),
            index=idx,
            columns=tickers,
        )
        daily_w = _sector_neutral_weights(
            selected, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)
