"""Risk anomalies: low beta, low volatility, tail risk, skewness, and distress avoidance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import (
    _align_fund,
    _cs_zscore,
    _monthly_rebalance,
    _periodic_rebalance,
    _sector_neutral_weights,
)
from .base import CrossSectionalStrategy


class BettingAgainstBeta(CrossSectionalStrategy):
    """H175: Long bottom quintile by rolling 252d beta vs SPY (long-only BAB).

    Frazzini-Pedersen (2014): low-beta stocks earn higher risk-adjusted returns.
    Low beta is empirically associated with outperformance on a risk-adjusted basis.
    Sector-neutral to avoid sector concentration.
    """

    name = "betting_against_beta"

    def __init__(
        self,
        spy_rets: pd.Series,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        beta_window: int = 252,
        monthly_rebalance: bool = True,
    ):
        self.spy_rets = spy_rets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.beta_window = beta_window
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        spy = self.spy_rets.reindex(prices_panel.index).fillna(0.0)
        min_p = self.beta_window // 2

        cov_sm = daily_rets.rolling(self.beta_window, min_periods=min_p).cov(spy)
        var_m = spy.rolling(self.beta_window, min_periods=min_p).var()
        beta = cov_sm.div(var_m, axis=0)

        # ascending=True: low beta selected (BAB long leg)
        daily_w = _sector_neutral_weights(
            beta, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class LowVolatilityAnomaly(CrossSectionalStrategy):
    """H176: Long bottom quintile by 252d realized volatility.

    Baker, Bradley, Wurgler (2011): low-volatility stocks outperform high-volatility
    stocks on a risk-adjusted basis. Sector-neutral selection.
    """

    name = "low_volatility_anomaly"

    def __init__(
        self,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        vol_window: int = 252,
        monthly_rebalance: bool = True,
    ):
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.vol_window = vol_window
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        min_p = self.vol_window // 2
        realized_vol = daily_rets.rolling(self.vol_window, min_periods=min_p).std()

        # ascending=True: low vol selected
        daily_w = _sector_neutral_weights(
            realized_vol, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class LowBetaQuality(CrossSectionalStrategy):
    """H179: Intersection of low beta and high quality (GP/A), sector-neutral.

    Composite = (1/2)*z(-beta) + (1/2)*z(GP/A).
    Combines Frazzini-Pedersen (2014) BAB signal with Novy-Marx (2013) profitability.
    Low beta + high quality = defensive quality = should outperform in risk-off regimes.
    """

    name = "low_beta_quality"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        spy_rets: pd.Series,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        beta_window: int = 252,
        monthly_rebalance: bool = True,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.spy_rets = spy_rets
        self.sectors = sectors
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
        gpa = (gp / ta).where(ta > 0)

        daily_rets = prices_panel.pct_change()
        spy = self.spy_rets.reindex(prices_panel.index).fillna(0.0)
        min_p = self.beta_window // 2
        cov_sm = daily_rets.rolling(self.beta_window, min_periods=min_p).cov(spy)
        var_m = spy.rolling(self.beta_window, min_periods=min_p).var()
        beta = cov_sm.div(var_m, axis=0)

        composite = (
            _cs_zscore(-beta).fillna(0.0)  # low beta = good
            + _cs_zscore(gpa).fillna(0.0)  # high GP/A = good
        ) / 2.0

        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class LowSkewnessAnomaly(CrossSectionalStrategy):
    """H197: Low historical return skewness predicts outperformance.

    Bali et al. (2011): investors overpay for lottery-like (positively skewed)
    stocks. Low-skewness stocks are underpriced and outperform.
    Signal = -rolling_skewness. Sector-neutral top 20%, monthly rebalance.
    """

    name = "low_skewness_anomaly"

    def __init__(
        self,
        sectors: pd.Series,
        skew_window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ):
        self.sectors = sectors
        self.skew_window = skew_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        skew = daily_rets.rolling(self.skew_window, min_periods=self.skew_window // 2).skew()
        score = -skew  # negate: low skewness = high score
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class AltmanZProxy(CrossSectionalStrategy):
    """H244: Simplified Altman Z-score proxy from available fundamentals.

    Composite of 4 components: NWC/TA + NI/TA + BE/TL + revenue/TA.
    """

    name = "altman_z_proxy"

    def __init__(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        book_equity: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.net_income = net_income
        self.total_assets = total_assets
        self.book_equity = book_equity
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.revenue = revenue
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)

        nwc_ta = (ca - cl) / ta.where(ta > 0)
        ni_ta = ni / ta.where(ta > 0)
        total_liab = (ta - be).where((ta - be) > 0)
        be_tl = be / total_liab.where(total_liab > 0)
        rev_ta = rev / ta.where(ta > 0)

        score = _cs_zscore(nwc_ta) + _cs_zscore(ni_ta) + _cs_zscore(be_tl) + _cs_zscore(rev_ta)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class EquityMultiplier(CrossSectionalStrategy):
    """H246: Low equity multiplier (TA/BE) = low financial leverage. DuPont component."""

    name = "equity_multiplier"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.total_assets = total_assets
        self.book_equity = book_equity
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        em = ta / be.where(be > 0)
        em = em.clip(1.0, 20.0)
        # Low equity multiplier = low leverage = better
        daily_w = _sector_neutral_weights(
            em, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class IdiosyncraticVolFactor(CrossSectionalStrategy):
    """H251: Long bottom quintile by 63-day idiosyncratic (residual) volatility.

    Ang, Hodrick, Xing, Zhang (2006): high idio vol predicts low returns.
    Idio vol = std of residuals from rolling OLS of stock return on SPY return.
    Distinct from H176 (total vol): residual vol strips out market beta.
    """

    name = "idiosyncratic_vol_factor"

    def __init__(
        self,
        sectors: pd.Series,
        spy_rets: pd.Series,
        idio_vol_window: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ):
        self.sectors = sectors
        self.spy_rets = spy_rets
        self.idio_vol_window = idio_vol_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        spy = self.spy_rets.reindex(prices_panel.index).fillna(0.0)

        w = self.idio_vol_window
        min_p = w // 2

        # Rolling beta = cov(stock, SPY) / var(SPY)
        cov_sm = daily_rets.rolling(w, min_periods=min_p).cov(spy)
        var_m = spy.rolling(w, min_periods=min_p).var().replace(0, np.nan)
        beta = cov_sm.div(var_m, axis=0)

        # Residual return = stock_ret - beta * spy_ret
        spy_broadcast = pd.DataFrame(
            np.outer(spy.values, np.ones(prices_panel.shape[1])),
            index=prices_panel.index,
            columns=prices_panel.columns,
        )
        residuals = daily_rets - beta.shift(1) * spy_broadcast

        idio_vol = residuals.rolling(w, min_periods=min_p).std()

        # ascending=True: low idio vol selected
        daily_w = _sector_neutral_weights(
            idio_vol, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MaxReturnLotteryAvoid(CrossSectionalStrategy):
    """H252: Long bottom quintile by 21-day maximum daily return (MAX21).

    Bali, Cakici, Whitelaw (2011): investors overpay for lottery-like payoffs.
    Stocks with extreme recent single-day returns earn lower subsequent returns.
    """

    name = "max_return_lottery_avoid"

    def __init__(
        self,
        sectors: pd.Series,
        max_window: int = 21,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ):
        self.sectors = sectors
        self.max_window = max_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        max21 = daily_rets.rolling(self.max_window, min_periods=self.max_window // 2).max()

        # ascending=True: low MAX selected
        daily_w = _sector_neutral_weights(
            max21, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class DownsideBetaFactor(CrossSectionalStrategy):
    """H253: Long bottom quintile by 252-day downside beta.

    Harvey & Siddique (2000): crash sensitivity (beta on SPY down-days only)
    is priced. Low downside-beta stocks are preferred by crash-averse investors.
    Downside beta = sum(r_i * r_m | r_m < 0) / sum(r_m^2 | r_m < 0) over window.
    """

    name = "downside_beta_factor"

    def __init__(
        self,
        sectors: pd.Series,
        spy_rets: pd.Series,
        downside_window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ):
        self.sectors = sectors
        self.spy_rets = spy_rets
        self.downside_window = downside_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        spy = self.spy_rets.reindex(prices_panel.index).fillna(0.0)

        w = self.downside_window
        min_p = w // 4  # fewer observations on down days; relax min_periods

        # Mask: zero out positive-SPY days for regression-through-origin
        down_mask = (spy < 0).astype(float)
        spy_down = spy * down_mask  # spy on down days, 0 elsewhere

        spy_down_broadcast = pd.DataFrame(
            np.outer(spy_down.values, np.ones(prices_panel.shape[1])),
            index=prices_panel.index,
            columns=prices_panel.columns,
        )
        stock_down = daily_rets.multiply(down_mask, axis=0)

        # Downside beta = sum(r_i * r_m on down days) / sum(r_m^2 on down days)
        numerator = (stock_down * spy_down_broadcast).rolling(w, min_periods=min_p).sum()
        denominator = (spy_down**2).rolling(w, min_periods=min_p).sum().replace(0, np.nan)
        downside_beta = numerator.div(denominator, axis=0)

        # ascending=True: low downside beta selected
        daily_w = _sector_neutral_weights(
            downside_beta, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class TailRiskComposite(CrossSectionalStrategy):
    """H254: Equal z-score composite of four tail-risk signals.

    Score = z(-idio_vol_63d) + z(-MAX21) + z(-downside_beta_252d) + z(skewness_63d).
    All four oriented so high score = low tail risk. Sector-neutral top 20%.
    """

    name = "tail_risk_composite"

    def __init__(
        self,
        sectors: pd.Series,
        spy_rets: pd.Series,
        idio_vol_window: int = 63,
        max_window: int = 21,
        downside_window: int = 252,
        skew_window: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ):
        self.sectors = sectors
        self.spy_rets = spy_rets
        self.idio_vol_window = idio_vol_window
        self.max_window = max_window
        self.downside_window = downside_window
        self.skew_window = skew_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        spy = self.spy_rets.reindex(prices_panel.index).fillna(0.0)

        # --- Idio vol ---
        w_iv = self.idio_vol_window
        cov_sm = daily_rets.rolling(w_iv, min_periods=w_iv // 2).cov(spy)
        var_m = spy.rolling(w_iv, min_periods=w_iv // 2).var().replace(0, np.nan)
        beta = cov_sm.div(var_m, axis=0)
        spy_bc = pd.DataFrame(
            np.outer(spy.values, np.ones(prices_panel.shape[1])),
            index=prices_panel.index,
            columns=prices_panel.columns,
        )
        residuals = daily_rets - beta.shift(1) * spy_bc
        idio_vol = residuals.rolling(w_iv, min_periods=w_iv // 2).std()

        # --- MAX21 ---
        max21 = daily_rets.rolling(self.max_window, min_periods=self.max_window // 2).max()

        # --- Downside beta ---
        w_db = self.downside_window
        down_mask = (spy < 0).astype(float)
        spy_down = spy * down_mask
        spy_down_bc = pd.DataFrame(
            np.outer(spy_down.values, np.ones(prices_panel.shape[1])),
            index=prices_panel.index,
            columns=prices_panel.columns,
        )
        stock_down = daily_rets.multiply(down_mask, axis=0)
        num = (stock_down * spy_down_bc).rolling(w_db, min_periods=w_db // 4).sum()
        den = (spy_down**2).rolling(w_db, min_periods=w_db // 4).sum().replace(0, np.nan)
        downside_beta = num.div(den, axis=0)

        # --- Skewness ---
        skew = daily_rets.rolling(self.skew_window, min_periods=self.skew_window // 2).skew()

        # Composite: all terms oriented so high = low tail risk
        score = (
            _cs_zscore(-idio_vol).fillna(0.0)
            + _cs_zscore(-max21).fillna(0.0)
            + _cs_zscore(-downside_beta).fillna(0.0)
            + _cs_zscore(skew).fillna(0.0)
        )

        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class TailFilteredMomentum(CrossSectionalStrategy):
    """H255: 12-1 month momentum with lottery exclusion filter.

    Before ranking by momentum, exclude stocks in the top decile of idio vol
    (63d) OR top decile of MAX21 (21d). Tests whether removing lottery noise
    from the momentum universe improves quality vs H157 (Sharpe 1.079).
    """

    name = "tail_filtered_momentum"

    def __init__(
        self,
        sectors: pd.Series,
        spy_rets: pd.Series,
        momentum_window: int = 252,
        momentum_skip: int = 21,
        idio_vol_window: int = 63,
        max_window: int = 21,
        exclude_decile: float = 0.10,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ):
        self.sectors = sectors
        self.spy_rets = spy_rets
        self.momentum_window = momentum_window
        self.momentum_skip = momentum_skip
        self.idio_vol_window = idio_vol_window
        self.max_window = max_window
        self.exclude_decile = exclude_decile
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        spy = self.spy_rets.reindex(prices_panel.index).fillna(0.0)

        # --- 12-1 month momentum ---
        mom = prices_panel.shift(self.momentum_skip) / prices_panel.shift(self.momentum_window) - 1

        # --- Idio vol ---
        w_iv = self.idio_vol_window
        cov_sm = daily_rets.rolling(w_iv, min_periods=w_iv // 2).cov(spy)
        var_m = spy.rolling(w_iv, min_periods=w_iv // 2).var().replace(0, np.nan)
        beta = cov_sm.div(var_m, axis=0)
        spy_bc = pd.DataFrame(
            np.outer(spy.values, np.ones(prices_panel.shape[1])),
            index=prices_panel.index,
            columns=prices_panel.columns,
        )
        residuals = daily_rets - beta.shift(1) * spy_bc
        idio_vol = residuals.rolling(w_iv, min_periods=w_iv // 2).std()

        # --- MAX21 ---
        max21 = daily_rets.rolling(self.max_window, min_periods=self.max_window // 2).max()

        # Identify lottery stocks: top decile of idio vol OR top decile of MAX21
        mem = membership.reindex_like(idio_vol).fillna(False)
        idio_rank = idio_vol.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        max_rank = max21.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        is_lottery = (idio_rank > (1.0 - self.exclude_decile)) | (
            max_rank > (1.0 - self.exclude_decile)
        )

        # Exclude lottery stocks from the selection universe
        filtered_membership = membership & ~is_lottery.fillna(False)

        daily_w = _sector_neutral_weights(
            mom, filtered_membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class OhlsonOScoreDistress(CrossSectionalStrategy):
    """H256: Proxy Ohlson O-score from quarterly fundamentals; long bottom-quintile (low distress)."""

    name = "ohlson_oscore_distress"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        book_equity: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.total_assets = total_assets
        self.book_equity = book_equity
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)

        liab = (ta - be).clip(lower=0)
        leverage = liab / ta.where(ta > 0)
        wc_ratio = (ca - cl) / ta.where(ta > 0)
        curr_ratio = ca / cl.where(cl > 0)
        roa = ni / ta.where(ta > 0)
        cfo_lev = ocf / liab.where(liab > 0)

        # High score = high distress; rank ascending (low distress = best)
        score = (
            _cs_zscore(leverage)
            + _cs_zscore(-wc_ratio)
            + _cs_zscore(-curr_ratio)
            + _cs_zscore(-roa)
            + _cs_zscore(-cfo_lev)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class CHSMarketDistress(CrossSectionalStrategy):
    """H257: CHS-style distress composite (market + accounting); long bottom-quintile."""

    name = "chs_market_distress"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        book_equity: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        vol_window: int = 252,
        mom_window: int = 63,
        within_sector_quantile: float = 0.20,
    ):
        self.total_assets = total_assets
        self.book_equity = book_equity
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.vol_window = vol_window
        self.mom_window = mom_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        shs = _align_fund(self.shares_outstanding, prices_panel)

        liab = (ta - be).clip(lower=0)
        leverage = liab / ta.where(ta > 0)
        roa = ni / ta.where(ta > 0)
        cash_ratio = ocf / ta.where(ta > 0)

        mcap = (prices_panel * shs).where(shs > 0)
        log_size = np.log(mcap.where(mcap > 0))

        daily_rets = prices_panel.pct_change()
        momentum = prices_panel / prices_panel.shift(self.mom_window) - 1
        vol12m = daily_rets.rolling(self.vol_window, min_periods=self.vol_window // 4).std()

        # High score = high distress; rank ascending
        score = (
            _cs_zscore(leverage)
            + _cs_zscore(-roa)
            + _cs_zscore(-momentum)
            + _cs_zscore(vol12m)
            + _cs_zscore(-cash_ratio)
            + _cs_zscore(-log_size)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class DistressTrendFactor(CrossSectionalStrategy):
    """H258: QoQ change in O-score proxy; long improving (falling) distress."""

    name = "distress_trend_factor"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        book_equity: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        lag: int = 63,
        within_sector_quantile: float = 0.20,
    ):
        self.total_assets = total_assets
        self.book_equity = book_equity
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.lag = lag
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)

        liab = (ta - be).clip(lower=0)
        score = (
            _cs_zscore(liab / ta.where(ta > 0))
            + _cs_zscore(-(ca - cl) / ta.where(ta > 0))
            + _cs_zscore(-ca / cl.where(cl > 0))
            + _cs_zscore(-ni / ta.where(ta > 0))
            + _cs_zscore(-ocf / liab.where(liab > 0))
        )

        # Falling score = improving distress = best; rank ascending on delta
        delta_score = score - score.shift(self.lag)
        daily_w = _sector_neutral_weights(
            delta_score, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class DistressFilteredValue(CrossSectionalStrategy):
    """H259: Book-to-market value intersected with low CHS distress filter."""

    name = "distress_filtered_value"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        book_equity: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        value_quantile: float = 0.20,
        distress_pct: float = 0.50,
        vol_window: int = 252,
        mom_window: int = 63,
    ):
        self.total_assets = total_assets
        self.book_equity = book_equity
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.value_quantile = value_quantile
        self.distress_pct = distress_pct
        self.vol_window = vol_window
        self.mom_window = mom_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        shs = _align_fund(self.shares_outstanding, prices_panel)

        # Book-to-market (high = cheap)
        mcap = (prices_panel * shs).where(shs > 0)
        btm = be / mcap.where(mcap > 0)

        # CHS distress score
        liab = (ta - be).clip(lower=0)
        leverage = liab / ta.where(ta > 0)
        roa = ni / ta.where(ta > 0)
        cash_ratio = ocf / ta.where(ta > 0)
        log_size = np.log(mcap.where(mcap > 0))
        daily_rets = prices_panel.pct_change()
        momentum = prices_panel / prices_panel.shift(self.mom_window) - 1
        vol12m = daily_rets.rolling(self.vol_window, min_periods=self.vol_window // 4).std()

        distress = (
            _cs_zscore(leverage)
            + _cs_zscore(-roa)
            + _cs_zscore(-momentum)
            + _cs_zscore(vol12m)
            + _cs_zscore(-cash_ratio)
            + _cs_zscore(-log_size)
        )

        # Filter: top-quintile BTM (cheap) AND bottom-half distress
        mem = membership.reindex_like(btm).fillna(False)
        btm_rank = btm.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        distress_rank = distress.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")

        filtered_mem = (
            membership
            & (btm_rank >= (1.0 - self.value_quantile)).fillna(False)
            & (distress_rank <= self.distress_pct).fillna(False)
        )

        # Equal-weight surviving names
        n_valid = filtered_mem.sum(axis=1).replace(0, np.nan)
        daily_w = filtered_mem.div(n_valid, axis=0).fillna(0.0)
        return _monthly_rebalance(daily_w, prices_panel)


class LowDistressQualityMomentum(CrossSectionalStrategy):
    """H260: Three-way screen: top-third GP/TA + top-half momentum + bottom-half CHS distress."""

    name = "low_distress_quality_momentum"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        book_equity: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        quality_pct: float = 0.33,
        momentum_pct: float = 0.50,
        distress_pct: float = 0.50,
        momentum_window: int = 252,
        momentum_skip: int = 21,
        vol_window: int = 252,
        mom3_window: int = 63,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.book_equity = book_equity
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.quality_pct = quality_pct
        self.momentum_pct = momentum_pct
        self.distress_pct = distress_pct
        self.momentum_window = momentum_window
        self.momentum_skip = momentum_skip
        self.vol_window = vol_window
        self.mom3_window = mom3_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        shs = _align_fund(self.shares_outstanding, prices_panel)

        # Quality: GP/TA
        gp_ta = gp / ta.where(ta > 0)

        # Momentum: 12-1 month
        mom = prices_panel.shift(self.momentum_skip) / prices_panel.shift(self.momentum_window) - 1

        # CHS distress
        liab = (ta - be).clip(lower=0)
        mcap = (prices_panel * shs).where(shs > 0)
        leverage = liab / ta.where(ta > 0)
        roa = ni / ta.where(ta > 0)
        cash_ratio = ocf / ta.where(ta > 0)
        log_size = np.log(mcap.where(mcap > 0))
        daily_rets = prices_panel.pct_change()
        mom3 = prices_panel / prices_panel.shift(self.mom3_window) - 1
        vol12m = daily_rets.rolling(self.vol_window, min_periods=self.vol_window // 4).std()

        distress = (
            _cs_zscore(leverage)
            + _cs_zscore(-roa)
            + _cs_zscore(-mom3)
            + _cs_zscore(vol12m)
            + _cs_zscore(-cash_ratio)
            + _cs_zscore(-log_size)
        )

        mem = membership.reindex_like(gp_ta).fillna(False)
        q_rank = gp_ta.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        m_rank = mom.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        d_rank = distress.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")

        filtered_mem = (
            membership
            & (q_rank >= (1.0 - self.quality_pct)).fillna(False)
            & (m_rank >= (1.0 - self.momentum_pct)).fillna(False)
            & (d_rank <= self.distress_pct).fillna(False)
        )

        n_valid = filtered_mem.sum(axis=1).replace(0, np.nan)
        daily_w = filtered_mem.div(n_valid, axis=0).fillna(0.0)
        return _monthly_rebalance(daily_w, prices_panel)


class NegativeSkewnessAvoidance(CrossSectionalStrategy):
    """H276: Long most-positive-skew stocks (Harvey & Siddique 2000)."""

    name = "negative_skewness_avoidance"

    def __init__(
        self,
        sectors: pd.Series,
        skew_window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.sectors = sectors
        self.skew_window = skew_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rets = prices_panel.pct_change()
        skewness = rets.rolling(self.skew_window, min_periods=self.skew_window // 4).skew()

        # ascending=True: most positive skew = best (lowest rank value in "ascending" sense is worst)
        # We want HIGH skewness → ascending=False actually makes sense:
        # ascending=False means higher score is better → higher skewness is better
        daily_w = _sector_neutral_weights(
            skewness, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class CoskewnessPortfolioPenalty(CrossSectionalStrategy):
    """H277: Long low-coskewness stocks — diversification value (Harvey & Siddique 2000)."""

    name = "coskewness_portfolio_penalty"

    def __init__(
        self,
        spy_prices: pd.Series,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.spy_prices = spy_prices
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rets = prices_panel.pct_change()
        spy_rets = self.spy_prices.pct_change().reindex(prices_panel.index)

        # Rolling demeaned returns
        ri_mean = rets.rolling(self.window, min_periods=self.window // 4).mean()
        rm_mean = spy_rets.rolling(self.window, min_periods=self.window // 4).mean()

        ri_dm = rets - ri_mean
        rm_dm = (spy_rets - rm_mean).fillna(0.0)
        rm_dm2 = rm_dm**2

        # Coskewness numerator: rolling mean of (ri - μi)(rm - μm)²
        cross = ri_dm.multiply(rm_dm2, axis=0)
        cosk_num = cross.rolling(self.window, min_periods=self.window // 4).mean()

        sigma_i = rets.rolling(self.window, min_periods=self.window // 4).std()
        sigma_m = spy_rets.rolling(self.window, min_periods=self.window // 4).std()
        denom = sigma_i.multiply((sigma_m**2), axis=0)

        coskewness = cosk_num / denom.where(denom.abs() > 1e-12)

        # ascending=True: most negative coskewness = best (lowest score → best rank)
        daily_w = _sector_neutral_weights(
            coskewness, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class LeftTailLossFactor(CrossSectionalStrategy):
    """H278: Long lowest-CVaR stocks — conditional value at risk signal."""

    name = "left_tail_loss_factor"

    def __init__(
        self,
        sectors: pd.Series,
        window: int = 252,
        cvar_pct: float = 0.05,
        within_sector_quantile: float = 0.20,
    ):
        self.sectors = sectors
        self.window = window
        self.cvar_pct = cvar_pct
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rets = prices_panel.pct_change()
        n_tail = max(1, int(self.window * self.cvar_pct))

        # CVaR: mean of worst n_tail returns in each rolling window
        # Use expanding then rolling: nsmallest approach via apply
        cvar = rets.rolling(self.window, min_periods=self.window // 4).apply(
            lambda x: np.sort(x)[:n_tail].mean(), raw=True
        )

        # ascending=False: highest CVaR value = least negative = lowest tail loss = best
        daily_w = _sector_neutral_weights(
            cvar, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class CrashSensitivityRegime(CrossSectionalStrategy):
    """H279: Low coskewness + regime tightening when SPY vol > 30%."""

    name = "crash_sensitivity_regime"

    def __init__(
        self,
        spy_prices: pd.Series,
        sectors: pd.Series,
        window: int = 252,
        cvar_pct: float = 0.05,
        base_quantile: float = 0.20,
        stress_quantile: float = 0.10,
        spy_stress_window: int = 21,
        spy_stress_vol_threshold: float = 0.30,
    ):
        self.spy_prices = spy_prices
        self.sectors = sectors
        self.window = window
        self.cvar_pct = cvar_pct
        self.base_quantile = base_quantile
        self.stress_quantile = stress_quantile
        self.spy_stress_window = spy_stress_window
        self.spy_stress_vol_threshold = spy_stress_vol_threshold

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rets = prices_panel.pct_change()
        spy_rets = self.spy_prices.pct_change().reindex(prices_panel.index)

        # Coskewness
        ri_mean = rets.rolling(self.window, min_periods=self.window // 4).mean()
        rm_mean = spy_rets.rolling(self.window, min_periods=self.window // 4).mean()
        ri_dm = rets - ri_mean
        rm_dm = (spy_rets - rm_mean).fillna(0.0)
        rm_dm2 = rm_dm**2
        cross = ri_dm.multiply(rm_dm2, axis=0)
        cosk_num = cross.rolling(self.window, min_periods=self.window // 4).mean()
        sigma_i = rets.rolling(self.window, min_periods=self.window // 4).std()
        sigma_m = spy_rets.rolling(self.window, min_periods=self.window // 4).std()
        denom = sigma_i.multiply((sigma_m**2), axis=0)
        coskewness = cosk_num / denom.where(denom.abs() > 1e-12)

        # CVaR for stress regime additional filter
        n_tail = max(1, int(self.window * self.cvar_pct))
        cvar = rets.rolling(self.window, min_periods=self.window // 4).apply(
            lambda x: np.sort(x)[:n_tail].mean(), raw=True
        )

        # SPY stress regime flag (21d annualized vol > threshold)
        spy_vol_21d = spy_rets.rolling(
            self.spy_stress_window, min_periods=self.spy_stress_window // 2
        ).std() * np.sqrt(252)
        stress = spy_vol_21d > self.spy_stress_vol_threshold

        # Base weights: top-quintile low coskewness
        base_w = _sector_neutral_weights(
            coskewness, membership, self.sectors, self.base_quantile, ascending=True
        )

        # Stress weights: top-decile low coskewness AND better-half CVaR (less negative = lower tail loss)
        mem_f = membership.reindex_like(cvar).fillna(False)
        cvar_rank = cvar.where(mem_f).rank(axis=1, pct=True, na_option="keep")
        stress_mem = membership & (cvar_rank >= 0.50).fillna(False)
        stress_w = _sector_neutral_weights(
            coskewness, stress_mem, self.sectors, self.stress_quantile, ascending=True
        )

        # Switch between base and stress by date
        stress_mask = stress.reindex(prices_panel.index).fillna(False)
        daily_w = base_w.copy()
        daily_w[stress_mask] = stress_w[stress_mask]

        return _monthly_rebalance(daily_w, prices_panel)


class SkewAdjustedMomentum(CrossSectionalStrategy):
    """H280: Momentum minus coskewness and CVaR penalties (percentile composite)."""

    name = "skew_adjusted_momentum"

    def __init__(
        self,
        spy_prices: pd.Series,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        cosk_window: int = 252,
        cvar_window: int = 252,
        cvar_pct: float = 0.05,
        cosk_weight: float = 0.4,
        cvar_weight: float = 0.3,
        within_sector_quantile: float = 0.20,
    ):
        self.spy_prices = spy_prices
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_window = skip_window
        self.cosk_window = cosk_window
        self.cvar_window = cvar_window
        self.cvar_pct = cvar_pct
        self.cosk_weight = cosk_weight
        self.cvar_weight = cvar_weight
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rets = prices_panel.pct_change()
        spy_rets = self.spy_prices.pct_change().reindex(prices_panel.index)

        # Momentum 12-1
        mom = prices_panel.shift(self.skip_window) / prices_panel.shift(self.mom_window) - 1

        # Coskewness
        ri_mean = rets.rolling(self.cosk_window, min_periods=self.cosk_window // 4).mean()
        rm_mean = spy_rets.rolling(self.cosk_window, min_periods=self.cosk_window // 4).mean()
        ri_dm = rets - ri_mean
        rm_dm = (spy_rets - rm_mean).fillna(0.0)
        rm_dm2 = rm_dm**2
        cross = ri_dm.multiply(rm_dm2, axis=0)
        cosk_num = cross.rolling(self.cosk_window, min_periods=self.cosk_window // 4).mean()
        sigma_i = rets.rolling(self.cosk_window, min_periods=self.cosk_window // 4).std()
        sigma_m = spy_rets.rolling(self.cosk_window, min_periods=self.cosk_window // 4).std()
        denom = sigma_i.multiply((sigma_m**2), axis=0)
        coskewness = cosk_num / denom.where(denom.abs() > 1e-12)

        # CVaR
        n_tail = max(1, int(self.cvar_window * self.cvar_pct))
        cvar = rets.rolling(self.cvar_window, min_periods=self.cvar_window // 4).apply(
            lambda x: np.sort(x)[:n_tail].mean(), raw=True
        )

        mem = membership.reindex_like(prices_panel).fillna(False)
        mom_pct = mom.where(mem).rank(axis=1, pct=True, na_option="keep")
        cosk_pct = coskewness.where(mem).rank(axis=1, pct=True, na_option="keep")
        # negate CVaR so rank 1.0 = most negative CVaR (worst tail risk) = maximal penalty
        cvar_pct_rank = (-cvar).where(mem).rank(axis=1, pct=True, na_option="keep")

        score = mom_pct - self.cosk_weight * cosk_pct - self.cvar_weight * cvar_pct_rank
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class TailRiskFilteredCrossProgram(CrossSectionalStrategy):
    """H296: GP/TA quality factor with tail-risk stock exclusion.

    Extends the program's best cross-sectional signal (GP/TA, H227/H268) with a
    pre-filter that excludes high-tail-risk stocks. Tail risk composite (same as H254):
    z(-idio_vol_63d) + z(-MAX21) + z(-downside_beta_252d).
    Bottom-decile tail-risk stocks excluded from the selection universe.
    Tests whether removing lottery/crash names from the quality universe improves net alpha.
    """

    name = "tail_risk_filtered_cross_program"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        spy_rets: pd.Series,
        exclude_quantile: float = 0.10,
        idio_vol_window: int = 63,
        max_window: int = 21,
        downside_window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.spy_rets = spy_rets
        self.exclude_quantile = exclude_quantile
        self.idio_vol_window = idio_vol_window
        self.max_window = max_window
        self.downside_window = downside_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        spy = self.spy_rets.reindex(prices_panel.index).fillna(0.0)

        # Idio vol (rolling beta residual)
        w_iv = self.idio_vol_window
        cov_sm = daily_rets.rolling(w_iv, min_periods=w_iv // 2).cov(spy)
        var_m = spy.rolling(w_iv, min_periods=w_iv // 2).var().replace(0, np.nan)
        beta_iv = cov_sm.div(var_m, axis=0)
        spy_bc = pd.DataFrame(
            np.outer(spy.values, np.ones(prices_panel.shape[1])),
            index=prices_panel.index,
            columns=prices_panel.columns,
        )
        residuals = daily_rets - beta_iv.shift(1) * spy_bc
        idio_vol = residuals.rolling(w_iv, min_periods=w_iv // 2).std()

        # MAX21
        max21 = daily_rets.rolling(self.max_window, min_periods=self.max_window // 2).max()

        # Downside beta
        w_db = self.downside_window
        down_mask = (spy < 0).astype(float)
        spy_down = spy * down_mask
        spy_down_bc = pd.DataFrame(
            np.outer(spy_down.values, np.ones(prices_panel.shape[1])),
            index=prices_panel.index,
            columns=prices_panel.columns,
        )
        stock_down = daily_rets.multiply(down_mask, axis=0)
        num = (stock_down * spy_down_bc).rolling(w_db, min_periods=w_db // 4).sum()
        den = (spy_down**2).rolling(w_db, min_periods=w_db // 4).sum().replace(0, np.nan)
        downside_beta = num.div(den, axis=0)

        # Tail risk composite: high = low tail risk
        tail_score = (
            _cs_zscore(-idio_vol).fillna(0.0)
            + _cs_zscore(-max21).fillna(0.0)
            + _cs_zscore(-downside_beta).fillna(0.0)
        )

        # Build filtered membership: exclude bottom-decile tail-risk stocks
        tail_rank = tail_score.rank(axis=1, pct=True, na_option="keep")
        high_tail_risk = tail_rank <= self.exclude_quantile
        filtered_mem = membership.reindex_like(prices_panel).fillna(False) & ~high_tail_risk.fillna(
            False
        )

        # GP/TA signal on filtered universe
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        gp_ta = gp / ta.replace(0, np.nan)
        daily_w = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class DistressScreenedValueQuality(CrossSectionalStrategy):
    """H297: GP/TA + earnings yield, filtered by low CHS-inspired distress score.

    CHS distress (simplified): z(leverage) + z(-ROA) + z(-12m_momentum) + z(vol12m)
    + z(-net_working_capital_ratio). Filter: keep bottom-half distress names.
    Signal: z(gp_ta) + z(earnings_yield) on the filtered universe.
    """

    name = "distress_screened_value_quality"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        distress_pct: float = 0.50,
        mom_window: int = 252,
        mom_skip: int = 21,
        vol_window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.long_term_debt = long_term_debt
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.distress_pct = distress_pct
        self.mom_window = mom_window
        self.mom_skip = mom_skip
        self.vol_window = vol_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)

        ta_safe = ta.replace(0, np.nan)

        # CHS-inspired distress components (high score = more distressed)
        leverage = (ltd / ta_safe).clip(-5, 5)
        neg_roa = -(ni / ta_safe).clip(-5, 5)
        daily_rets = prices_panel.pct_change()
        neg_mom = -(
            prices_panel.shift(self.mom_skip) / prices_panel.shift(self.mom_window) - 1
        ).clip(-5, 5)
        vol12m = daily_rets.rolling(
            self.vol_window, min_periods=self.vol_window // 4
        ).std() * np.sqrt(252)
        nwc = (ca - cl) / ta_safe
        neg_nwc = (-nwc).clip(-5, 5)

        distress_score = (
            _cs_zscore(leverage).fillna(0.0)
            + _cs_zscore(neg_roa).fillna(0.0)
            + _cs_zscore(neg_mom).fillna(0.0)
            + _cs_zscore(vol12m).fillna(0.0)
            + _cs_zscore(neg_nwc).fillna(0.0)
        )

        # Filter: keep bottom-half distress (low distress firms)
        distress_rank = distress_score.rank(axis=1, pct=True, na_option="keep")
        low_distress = distress_rank <= self.distress_pct
        filtered_mem = membership.reindex_like(prices_panel).fillna(False) & low_distress.fillna(
            False
        )

        # Value-quality score on filtered universe
        gp_ta = gp / ta_safe
        market_cap = prices_panel * sh.replace(0, np.nan)
        earnings_yield = (ni / market_cap.replace(0, np.nan)).clip(-5, 5)
        score = _cs_zscore(gp_ta).fillna(0.0) + _cs_zscore(earnings_yield).fillna(0.0)

        daily_w = _sector_neutral_weights(
            score, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class MertonD2DPort(CrossSectionalStrategy):
    """H431: Simplified distance-to-default proxy from available fundamentals.

    D2D proxy = (BE / (BE + LTD)) / realized_vol_21d.
    High equity fraction + low volatility = high D2D = far from default.
    Merton (1974): equity is call option; D2D = strikes distance.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "merton_d2d_port"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        sectors: pd.Series,
        vol_window: int = 21,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.book_equity = book_equity
        self.long_term_debt = long_term_debt
        self.sectors = sectors
        self.vol_window = vol_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        be = _align_fund(self.book_equity, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        total_cap = (be + ltd.clip(lower=0.0)).replace(0, np.nan)
        equity_fraction = (be / total_cap).clip(0.0, 1.0)
        realized_vol = (
            prices_panel.pct_change()
            .rolling(self.vol_window, min_periods=5)
            .std()
            .shift(1)
            .replace(0, np.nan)
        )
        d2d = (equity_fraction / realized_vol).clip(0.0, 100.0)
        daily_w = _sector_neutral_weights(
            d2d, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class LeverageAdjustedMomentum(CrossSectionalStrategy):
    """H432: 6m price momentum filtered to low-leverage names (LTD/TA < median).

    Avramov et al. (2007): momentum profits stronger in high-quality firms.
    Conditioning on leverage removes distress tail risk from momentum signal.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "leverage_adjusted_momentum"

    def __init__(
        self,
        long_term_debt: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.long_term_debt = long_term_debt
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
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        leverage = (ltd / ta_safe).clip(0.0, 5.0)
        mem_f = membership.reindex_like(leverage).fillna(False)
        leverage_median = leverage.where(mem_f).median(axis=1)
        low_leverage = (leverage.le(leverage_median, axis=0)).fillna(False)
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        quality_mem = membership & low_leverage
        daily_w = _sector_neutral_weights(
            momentum, quality_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CreditHealthQuality(CrossSectionalStrategy):
    """H433: Credit-healthy quality composite: GP/TA + low leverage + OCF/TA.

    z(GP/TA) + z(-LTD/TA) + z(OCF/TA). Credit-healthy quality portfolio.
    Vassalou & Xing (2004): default risk must be handled with controls.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "credit_health_quality"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.long_term_debt = long_term_debt
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
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        ltd_ta = (ltd / ta_safe).clip(0.0, 5.0)
        ocf_ta = (ocf / ta_safe).clip(-2.0, 5.0)
        composite = (
            _cs_zscore(gp_ta).fillna(0.0)
            + _cs_zscore(-ltd_ta).fillna(0.0)
            + _cs_zscore(ocf_ta).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class FinancialFragilityAvoid(CrossSectionalStrategy):
    """H434: Altman Z-score proxy from available Compustat fundamentals.

    Z_proxy = 1.2*(CA-CL)/TA + 1.4*(NI/TA rolling mean) + 3.3*(GP/TA)
              + 0.6*(BE/max(LTD,1)) + 1.0*(Rev/TA)
    Long top 20% most financially sound (highest Z proxy).
    Kisgen (2006): firms near rating thresholds behave differently.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "financial_fragility_avoid"

    def __init__(
        self,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        gross_profit: pd.DataFrame,
        book_equity: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        ni_window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.total_assets = total_assets
        self.net_income = net_income
        self.gross_profit = gross_profit
        self.book_equity = book_equity
        self.long_term_debt = long_term_debt
        self.revenue = revenue
        self.sectors = sectors
        self.ni_window = ni_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        ta_safe = ta.where(ta > 0)
        ltd_safe = ltd.where(ltd > 0).fillna(1.0)
        # Altman Z components (simplified with available data)
        z1 = ((ca - cl) / ta_safe).clip(-2.0, 2.0)
        z2 = (ni / ta_safe).rolling(self.ni_window, min_periods=63).mean().clip(-2.0, 2.0)
        z3 = (gp / ta_safe).clip(0.0, 5.0)
        z4 = (be / ltd_safe).clip(0.0, 20.0)
        z5 = (rev / ta_safe).clip(0.0, 10.0)
        z_proxy = 1.2 * z1 + 1.4 * z2 + 3.3 * z3 + 0.6 * z4 + 1.0 * z5
        daily_w = _sector_neutral_weights(
            z_proxy, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CreditConditionedQualityPort(CrossSectionalStrategy):
    """H435: GP/TA quality restricted to credit-healthy subsample.

    Membership filter: LTD/TA < sector median AND OCF > 0.
    Applies GP/TA quality signal only to financially healthy names.
    Vassalou & Xing (2004): quality signals must control for default risk.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "credit_conditioned_quality_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.long_term_debt = long_term_debt
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
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        leverage = (ltd / ta_safe).clip(0.0, 5.0)
        mem_f = membership.reindex_like(leverage).fillna(False)
        lev_median = leverage.where(mem_f).median(axis=1)
        low_lev = leverage.le(lev_median, axis=0).fillna(False)
        ocf_pos = (ocf > 0).fillna(False)
        credit_healthy = membership & low_lev & ocf_pos
        daily_w = _sector_neutral_weights(
            gp_ta, credit_healthy, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MinReturnCrashRiskPort(CrossSectionalStrategy):
    """H461: Long stocks with smallest absolute worst-day return (low crash risk).

    MIN1 = minimum 1-day return in trailing 21 days. Low |MIN1| = low crash risk.
    H252 tested MAX1 avoidance (best day); this tests worst-day avoidance.
    Sector-neutral top 20%, monthly rebalance. Full 12-fold (pure price).
    """

    name = "min_return_crash_risk_port"

    def __init__(
        self,
        sectors: pd.Series,
        min_window: int = 21,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.min_window = min_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        # MIN1 is negative; higher (less negative) = safer stock
        min1 = daily_rets.rolling(self.min_window, min_periods=self.min_window // 2).min()
        # ascending=False: highest min1 (closest to 0) = least crash risk = long
        daily_w = _sector_neutral_weights(
            min1, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class ReturnAsymmetryPort(CrossSectionalStrategy):
    """H462: Long stocks with symmetric or downside-safe return distributions.

    Signal: MAX1 - |MIN1|. High positive = lottery-like upside asymmetry (bad).
    Low/negative = symmetric or downside-safe (good). Sector-neutral, monthly.
    Combines MAX1 and MIN1 into a net asymmetry spread — distinct from either alone.
    """

    name = "return_asymmetry_port"

    def __init__(
        self,
        sectors: pd.Series,
        window: int = 21,
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
        max1 = daily_rets.rolling(self.window, min_periods=self.window // 2).max()
        min1 = daily_rets.rolling(self.window, min_periods=self.window // 2).min()
        asymmetry = max1 - min1.abs()
        # ascending=True: low asymmetry (safe/symmetric) selected
        daily_w = _sector_neutral_weights(
            asymmetry, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class DownsideCapturePort(CrossSectionalStrategy):
    """H464: Long stocks with lowest realized downside capture vs SPY.

    Average daily return on SPY-negative days normalized by own total vol.
    Captures non-linear downside sensitivity missed by OLS beta regression.
    H253 used regression beta on down days; this uses realized average return.
    Sector-neutral top 20%, monthly rebalance. Full 12-fold.
    """

    name = "downside_capture_port"

    def __init__(
        self,
        sectors: pd.Series,
        spy_rets: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.spy_rets = spy_rets
        self.window = window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        spy = self.spy_rets.reindex(prices_panel.index).fillna(0.0)
        min_p = self.window // 4
        down_mask = (spy < 0).astype(float)
        # Average return on down days
        stock_down = daily_rets.multiply(down_mask, axis=0)
        down_count = down_mask.rolling(self.window, min_periods=min_p).sum()
        down_sum = stock_down.rolling(self.window, min_periods=min_p).sum()
        avg_down_ret = down_sum.div(down_count.replace(0, np.nan), axis=0)
        # Normalize by own total vol
        total_vol = daily_rets.rolling(self.window, min_periods=min_p).std().replace(0, np.nan)
        downside_capture = avg_down_ret.div(total_vol, axis=1)
        # ascending=True: lowest downside capture (safest) selected
        daily_w = _sector_neutral_weights(
            downside_capture, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class VolOfVolStabilityPort(CrossSectionalStrategy):
    """H466: Long stocks with most stable volatility process (low vol-of-vol).

    Rolling std of twelve 21-day realized vol estimates over 252 days.
    Low vol-of-vol = predictable risk profile = institutional preference.
    H251 tested vol LEVEL; this tests vol STABILITY. Distinct signal.
    Sector-neutral top 20%, monthly rebalance. Full 12-fold.
    """

    name = "vol_of_vol_stability_port"

    def __init__(
        self,
        sectors: pd.Series,
        vol_window: int = 21,
        n_windows: int = 12,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.vol_window = vol_window
        self.n_windows = n_windows
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        total_window = self.vol_window * self.n_windows
        min_p = total_window // 4
        # Compute rolling 21-day realized vol at each point; then take std over 252d window
        realized_vol = daily_rets.rolling(self.vol_window, min_periods=self.vol_window // 2).std()
        vol_of_vol = realized_vol.rolling(total_window, min_periods=min_p).std()
        # ascending=True: lowest vol-of-vol (most stable) selected
        daily_w = _sector_neutral_weights(
            vol_of_vol, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class BetaStabilityPort(CrossSectionalStrategy):
    """H467: Long stocks with most stable systematic risk (low beta variance).

    Std of four non-overlapping 63-day rolling beta estimates over 252 days.
    Low variance = predictable beta = lower institutional uncertainty premium.
    H175 tested beta LEVEL (BAB); H253 tested downside beta. This tests beta STABILITY.
    Sector-neutral top 20%, monthly rebalance. Full 12-fold.
    """

    name = "beta_stability_port"

    def __init__(
        self,
        sectors: pd.Series,
        spy_rets: pd.Series,
        beta_window: int = 63,
        n_windows: int = 4,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.spy_rets = spy_rets
        self.beta_window = beta_window
        self.n_windows = n_windows
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        spy = self.spy_rets.reindex(prices_panel.index).fillna(0.0)
        w = self.beta_window
        min_p = w // 2
        total_window = w * self.n_windows
        # Rolling beta over 63-day window
        cov_sm = daily_rets.rolling(w, min_periods=min_p).cov(spy)
        var_m = spy.rolling(w, min_periods=min_p).var().replace(0, np.nan)
        beta = cov_sm.div(var_m, axis=0)
        # Std of beta series over total window = beta instability
        beta_instability = beta.rolling(total_window, min_periods=total_window // 4).std()
        # ascending=True: lowest beta variance (most stable) selected
        daily_w = _sector_neutral_weights(
            beta_instability, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class NetDebtQualityPort(CrossSectionalStrategy):
    """H521: Net leverage = (LTD - CA) / TA. Long lowest (most net cash). Sector-neutral, quarterly.

    Net debt is the TRUE economic leverage after offsetting liquid assets.
    Negative values = net cash position (CA > LTD) → top ranked.
    H246 tested gross leverage (TA/BE); H237 tested cash coverage (OCF/LTD).
    Net debt ratio not previously tested.
    """

    name = "net_debt_quality_port"

    def __init__(
        self,
        long_term_debt: pd.DataFrame,
        current_assets: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.long_term_debt = long_term_debt
        self.current_assets = current_assets
        self.total_assets = total_assets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ltd = _align_fund(self.long_term_debt, prices_panel).clip(lower=0.0)
        ca = _align_fund(self.current_assets, prices_panel).clip(lower=0.0)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        # Net leverage: negative = net cash (good). Clip at [-2, 2] to limit outlier influence.
        net_leverage = ((ltd - ca) / ta_safe).clip(-2.0, 2.0)
        # ascending=True: lower (more negative = more net cash) is better
        daily_w = _sector_neutral_weights(
            net_leverage, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)
