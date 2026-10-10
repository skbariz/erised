"""Cross-sectional reversal strategies: long-term losers and short-term (weekly/monthly) reversal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import (
    _align_fund,
    _cs_zscore,
    _equal_weight_top_quantile,
    _monthly_rebalance,
    _periodic_rebalance,
    _sector_neutral_weights,
)
from .base import CrossSectionalStrategy


class LongTermReversal(CrossSectionalStrategy):
    """H160: Buy 3-year losers — long bottom quintile by 36-month past return.

    Skips the last month (skip_days=21) to avoid momentum contamination.
    ascending=True so the most negative return ranks first.
    """

    name = "long_term_reversal"

    def __init__(
        self,
        long_days: int = 756,
        skip_days: int = 21,
        top_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.long_days = long_days
        self.skip_days = skip_days
        self.top_quantile = top_quantile
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        reversal = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1
        daily_w = _equal_weight_top_quantile(
            reversal, membership, self.top_quantile, ascending=True
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class FiveYearReversal(CrossSectionalStrategy):
    """H161: Buy 5-year losers — same mechanism as H160 but 60-month lookback."""

    name = "five_year_reversal"

    def __init__(
        self,
        long_days: int = 1260,
        skip_days: int = 21,
        top_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.long_days = long_days
        self.skip_days = skip_days
        self.top_quantile = top_quantile
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        reversal = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1
        daily_w = _equal_weight_top_quantile(
            reversal, membership, self.top_quantile, ascending=True
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class NearFiftyTwoWeekLow(CrossSectionalStrategy):
    """H164: Buy stocks nearest to their 52-week low (contrarian reversal).

    Signal: current_price / rolling_min_price over `window` days.
    Ratio close to 1.0 = trading near 52-week low = most beaten-down.
    ascending=True so the lowest ratio (nearest low) ranks first.
    """

    name = "near_52wk_low"

    def __init__(
        self,
        window: int = 252,
        top_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.window = window
        self.top_quantile = top_quantile
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        roll_min = prices_panel.rolling(self.window, min_periods=self.window // 2).min()
        # Proximity to 52-week low: lower ratio = closer to 52-week low
        score = prices_panel / roll_min
        # ascending=True → lowest ratio (closest to 52-week low) ranks first
        daily_w = _equal_weight_top_quantile(score, membership, self.top_quantile, ascending=True)
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class QualityFilteredReversal(CrossSectionalStrategy):
    """H161: Buy 5-year losers that pass a quality filter.

    Base signal: negative 60-month price return (buy biggest losers).
    Quality gate (all must be True to enter portfolio):
      - 6-month price return > 0 (recovery trend has started)
      - operating_cf > 0 (cash-generative, not financially distressed)
      - current_ratio > 1.0 (solvent — current_assets > current_liabilities)
    """

    name = "quality_filtered_reversal"

    def __init__(
        self,
        operating_cf: pd.DataFrame,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        long_days: int = 1260,
        skip_days: int = 21,
        recovery_days: int = 126,
        top_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.operating_cf = operating_cf
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.long_days = long_days
        self.skip_days = skip_days
        self.recovery_days = recovery_days
        self.top_quantile = top_quantile
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        # 5-year reversal signal (ascending=True → buy lowest returns)
        reversal = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1

        # Quality gates (all must be True)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)

        recovery = prices_panel / prices_panel.shift(self.recovery_days) - 1
        gate = (
            (recovery > 0)  # positive 6-month price return (recovery started)
            & (ocf > 0)  # positive operating cash flow
            & (ca > cl)  # current ratio > 1
        )
        # Apply quality gate: set reversal score to NaN for stocks that fail
        gated_reversal = reversal.where(gate)

        daily_w = _equal_weight_top_quantile(
            gated_reversal, membership, self.top_quantile, ascending=True
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


class WeeklyReversalSectorNeutral(CrossSectionalStrategy):
    """H185: Long prior 5-day losers within each sector; rebalance weekly.

    Liquidity-provision reversal: stocks that fall on stock-specific selling
    pressure tend to bounce back in the following week.
    """

    name = "weekly_reversal_sector_neutral"

    def __init__(
        self,
        sectors: pd.Series,
        reversal_window: int = 5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ):
        self.sectors = sectors
        self.reversal_window = reversal_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ret_5d = prices_panel.pct_change(self.reversal_window)
        score = -ret_5d  # negate: want losers (bottom quintile by return)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class ResidualWeeklyReversal(CrossSectionalStrategy):
    """H186: Reverse the idiosyncratic (market-stripped) 5-day return.

    Residual = 5d_return - beta * SPY_5d_return. Trades the stock-specific
    move only, removing market-wide effects.
    """

    name = "residual_weekly_reversal"

    def __init__(
        self,
        spy_rets: pd.Series,
        sectors: pd.Series,
        reversal_window: int = 5,
        beta_window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ):
        self.spy_rets = spy_rets
        self.sectors = sectors
        self.reversal_window = reversal_window
        self.beta_window = beta_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        spy = self.spy_rets.reindex(prices_panel.index).fillna(0.0)
        min_p = self.beta_window // 2
        cov = daily_rets.rolling(self.beta_window, min_periods=min_p).cov(spy)
        var = spy.rolling(self.beta_window, min_periods=min_p).var()
        beta = cov.div(var, axis=0)
        # 5-day residual return
        spy_5d = spy.rolling(self.reversal_window).sum()
        rets_5d = prices_panel.pct_change(self.reversal_window)
        residual_5d = rets_5d.sub(beta.mul(spy_5d, axis=0))
        # Reverse residual: long biggest negative idiosyncratic movers
        score = -residual_5d
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class LiquidityFilteredReversal(CrossSectionalStrategy):
    """H187: 5-day reversal applied only to stocks with high realized volatility.

    High realized vol proxies for illiquidity within the S&P 500. The reversal
    premium is larger in less liquid names; the vol filter concentrates the signal.
    """

    name = "liquidity_filtered_reversal"

    def __init__(
        self,
        sectors: pd.Series,
        reversal_window: int = 5,
        vol_window: int = 21,
        within_sector_quantile: float = 0.20,
        illiq_quantile: float = 0.50,
        rebalance_period: int = 5,
    ):
        self.sectors = sectors
        self.reversal_window = reversal_window
        self.vol_window = vol_window
        self.within_sector_quantile = within_sector_quantile
        self.illiq_quantile = illiq_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        illiq = daily_rets.rolling(self.vol_window, min_periods=self.vol_window // 2).std()
        mem = membership.reindex_like(illiq).fillna(False)
        n_mem = mem.sum(axis=1)
        thresh = (n_mem * self.illiq_quantile).clip(lower=1)
        illiq_rank = illiq.where(mem).rank(axis=1, ascending=False, na_option="keep")
        high_illiq = illiq_rank.le(thresh, axis=0) & illiq_rank.notna()

        ret_5d = prices_panel.pct_change(self.reversal_window)
        score = (-ret_5d).where(high_illiq)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MonthlyReversalLong(CrossSectionalStrategy):
    """H188: Long prior-month losers within sector; monthly rebalance.

    Lower cost than weekly reversal (~10x fewer rebalances). Tests if the
    reversal signal survives at monthly holding frequency.
    """

    name = "monthly_reversal_long"

    def __init__(
        self,
        sectors: pd.Series,
        reversal_window: int = 21,
        within_sector_quantile: float = 0.20,
    ):
        self.sectors = sectors
        self.reversal_window = reversal_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ret_21d = prices_panel.pct_change(self.reversal_window)
        score = -ret_21d  # negate: losers get high score
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class IntangibleReturnReversal(CrossSectionalStrategy):
    """H218: Fade stocks with high intangible return = price > fundamental return.

    intangible_return = 12m price return - fundamental_return_proxy.
    fundamental_return_proxy = YoY NI growth (clipped to [-1, 2]).
    High intangible return stocks are hype-driven; fade (short-term reversal in disguise).
    Sector-neutral bottom quintile by intangible return, monthly rebalance.
    """

    name = "intangible_return_reversal"

    def __init__(
        self,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        price_window: int = 252,
        skip_window: int = 21,
        yoy_window: int = 252,
    ):
        self.net_income = net_income
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.price_window = price_window
        self.skip_window = skip_window
        self.yoy_window = yoy_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        # Price return (12m, skip 1m)
        price_return = prices_panel.pct_change(self.price_window).shift(self.skip_window)

        # Fundamental return proxy: YoY NI growth clipped to [-1, 2]
        ni = _align_fund(self.net_income, prices_panel)
        ni_lag = ni.shift(self.yoy_window)
        ni_growth = (ni - ni_lag) / ni_lag.abs().replace(0, np.nan)
        ni_growth = ni_growth.clip(-1.0, 2.0)

        # Intangible return = price driven by non-fundamental factors
        intangible = price_return - ni_growth.fillna(0.0)

        # Low intangible return = price grounded in fundamentals = buy
        daily_w = _sector_neutral_weights(
            intangible, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _monthly_rebalance(daily_w, prices_panel)


class AttentionReversalScreen(CrossSectionalStrategy):
    """H274: Avoid top-decile MAX21 + top-decile vol + weak GP/TA. Long non-froth."""

    name = "attention_reversal_screen"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        max_window: int = 21,
        vol_window: int = 21,
        max_quantile: float = 0.10,
        vol_quantile: float = 0.10,
        quality_pct: float = 0.50,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.max_window = max_window
        self.vol_window = vol_window
        self.max_quantile = max_quantile
        self.vol_quantile = vol_quantile
        self.quality_pct = quality_pct

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)

        rets = prices_panel.pct_change()
        max21 = rets.rolling(self.max_window, min_periods=self.max_window // 2).max()
        vol21 = rets.rolling(self.vol_window, min_periods=self.vol_window // 2).std()
        gp_ta = gp / ta.where(ta > 0)

        mem = membership.reindex_like(prices_panel).fillna(False)
        max_rank = max21.where(mem).rank(axis=1, pct=True, na_option="keep")
        vol_rank = vol21.where(mem).rank(axis=1, pct=True, na_option="keep")
        quality_rank = gp_ta.where(mem).rank(axis=1, pct=True, na_option="keep")

        # Flag as froth: top-decile MAX + top-decile vol + bottom-half quality
        froth = (
            (max_rank >= (1.0 - self.max_quantile)).fillna(False)
            & (vol_rank >= (1.0 - self.vol_quantile)).fillna(False)
            & (quality_rank < self.quality_pct).fillna(False)
        )

        filtered_mem = membership & ~froth
        n_valid = filtered_mem.sum(axis=1).replace(0, np.nan)
        daily_w = filtered_mem.div(n_valid, axis=0).fillna(0.0)
        return _monthly_rebalance(daily_w, prices_panel)


class ReversionQualityCombo(CrossSectionalStrategy):
    """H538: z(5-day within-sector reversal) + z(GP/TA). Weekly rebalance.

    Quality stocks temporarily dislocated by short-term price movements.
    5-day reversal: negative recent return = potential reversal buy.
    GP/TA quality filter acts as persistent signal (quarterly, forward-filled).
    """

    name = "reversion_quality_combo"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        reversal_window: int = 5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.reversal_window = reversal_window
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
        ret_5d = prices_panel.pct_change(self.reversal_window)
        reversal = -ret_5d
        composite = _cs_zscore(reversal).fillna(0.0) + _cs_zscore(gp_ta).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class FundamentalReversalWeekly(CrossSectionalStrategy):
    """H540: z(5-day idiosyncratic reversal) + z(-accruals). Weekly rebalance.

    Idiosyncratic = within-sector residual (5d return minus sector mean).
    Accruals = (NI - OCF) / TA; low accruals = cash-backed earnings (quality).
    Selects quality stocks with short-term dislocations. Weekly reversal + quarterly quality.
    """

    name = "fundamental_reversal_weekly"

    def __init__(
        self,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        reversal_window: int = 5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ) -> None:
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.total_assets = total_assets
        self.sectors = sectors
        self.reversal_window = reversal_window
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
        accruals = ((ni - ocf) / ta_safe).clip(-1.0, 1.0)
        ret_5d = prices_panel.pct_change(self.reversal_window)
        idio_ret = ret_5d.copy()
        for sector in self.sectors.dropna().unique():
            cols = [t for t in self.sectors[self.sectors == sector].index if t in ret_5d.columns]
            if not cols:
                continue
            sec_mean = ret_5d[cols].mean(axis=1)
            idio_ret[cols] = ret_5d[cols].values - sec_mean.values[:, None]
        composite = _cs_zscore(-idio_ret).fillna(0.0) + _cs_zscore(-accruals).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class ThreeFactorDailyReversal(CrossSectionalStrategy):
    """H575: Daily reversal using 3-factor (market, size, value) residuals.

    Computes each stock's residual daily return after removing exposure to
    market (SPY), size (IWM-SPY), and value (IVE-IVW) factors via OLS.
    Long bottom decile (most negative residual, oversold), short top decile
    (most positive residual, overbought). Dollar-neutral, daily rebalance.
    More economically grounded than PCA-based H191/H192.
    """

    name = "three_factor_daily_reversal"

    def __init__(
        self,
        spy_rets: pd.Series,
        size_rets: pd.Series,
        value_rets: pd.Series,
        sectors: pd.Series,
        quantile: float = 0.10,
    ) -> None:
        self.spy_rets = spy_rets
        self.size_rets = size_rets
        self.value_rets = value_rets
        self.sectors = sectors
        self.quantile = quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        stock_rets = prices_panel.pct_change()
        mkt = self.spy_rets.reindex(stock_rets.index).fillna(0.0)
        size = self.size_rets.reindex(stock_rets.index).fillna(0.0)
        val = self.value_rets.reindex(stock_rets.index).fillna(0.0)

        # 3-factor betas via 60d rolling covariance
        window = 60
        cov_mf = stock_rets.rolling(window, min_periods=window // 2).cov(mkt)
        cov_sf = stock_rets.rolling(window, min_periods=window // 2).cov(size)
        cov_vf = stock_rets.rolling(window, min_periods=window // 2).cov(val)
        var_mkt = mkt.rolling(window, min_periods=window // 2).var()
        var_size = size.rolling(window, min_periods=window // 2).var()
        var_val = val.rolling(window, min_periods=window // 2).var()

        beta_m = cov_mf.div(var_mkt.replace(0, np.nan), axis=0).fillna(1.0)
        beta_s = cov_sf.div(var_size.replace(0, np.nan), axis=0).fillna(0.0)
        beta_v = cov_vf.div(var_val.replace(0, np.nan), axis=0).fillna(0.0)

        predicted = (
            beta_m.multiply(mkt, axis=0)
            + beta_s.multiply(size, axis=0)
            + beta_v.multiply(val, axis=0)
        )
        residual = stock_rets - predicted

        # Negate residual: most negative residual (oversold) gets highest score
        signal = -residual

        # Vectorized long/short selection via percentile rank
        active = membership.reindex(prices_panel.index, method="ffill").fillna(False)
        sig_active = signal.where(active, np.nan)

        rank_pct = sig_active.rank(axis=1, pct=True, na_option="keep", ascending=False)
        long_mask = rank_pct <= self.quantile
        short_mask = rank_pct >= (1.0 - self.quantile)

        long_n = long_mask.sum(axis=1).replace(0, np.nan)
        short_n = short_mask.sum(axis=1).replace(0, np.nan)
        long_w = long_mask.div(long_n, axis=0).fillna(0.0)
        short_w = -short_mask.div(short_n, axis=0).fillna(0.0)

        return (long_w + short_w).shift(1).fillna(0.0)
