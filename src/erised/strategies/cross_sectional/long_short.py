"""Market-neutral long/short versions of momentum, reversal, and factor signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import (
    _align_fund,
    _cs_zscore,
    _fit_pca_components,
    _long_short_normalize,
    _periodic_rebalance,
    _sector_neutral_weights,
)
from .base import CrossSectionalStrategy


class LongShortMomentum(CrossSectionalStrategy):
    """H193: Market-neutral long-short momentum.

    Long top-quintile momentum stocks, short bottom-quintile, sector-neutral.
    Tests whether the short leg adds alpha over long-only MomentumLongOnly.
    Monthly rebalance; gross exposure = 1.0 (50% long, 50% short).
    """

    name = "long_short_momentum"

    def __init__(
        self,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ):
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_window = skip_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        mom = prices_panel.pct_change(self.mom_window).shift(self.skip_window)
        long_w = _sector_neutral_weights(
            mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        short_w = _sector_neutral_weights(
            -mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        combined = _long_short_normalize(long_w, short_w)
        return _periodic_rebalance(combined, self.rebalance_period)


class LongShortReversal(CrossSectionalStrategy):
    """H194: Market-neutral long-short weekly reversal.

    Long prior-5-day sector losers, short prior-5-day sector winners.
    Tests whether the short leg adds alpha over long-only WeeklyReversalSectorNeutral.
    Weekly rebalance; gross exposure = 1.0 (50% long, 50% short).
    """

    name = "long_short_reversal"

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
        long_w = _sector_neutral_weights(
            -ret_5d, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        short_w = _sector_neutral_weights(
            ret_5d, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        combined = _long_short_normalize(long_w, short_w)
        return _periodic_rebalance(combined, self.rebalance_period)


class LongShortSectorReversal(CrossSectionalStrategy):
    """H471: Dollar-neutral L/S of within-sector 5-day reversal (H185 in L/S form).

    Long prior-5d within-sector losers; short prior-5d within-sector winners.
    Sector-neutral within each leg separately. Weekly rebalance. Full 12-fold.
    H194 is generic reversal; this is explicitly sector-restricted reversal.
    """

    name = "long_short_sector_reversal"

    def __init__(
        self,
        sectors: pd.Series,
        reversal_window: int = 5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ) -> None:
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
        long_w = _sector_neutral_weights(
            -ret_5d, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        short_w = _sector_neutral_weights(
            ret_5d, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        combined = _long_short_normalize(long_w, short_w)
        return _periodic_rebalance(combined, self.rebalance_period)


class LongShortResidualReversal(CrossSectionalStrategy):
    """H472: Dollar-neutral L/S of idiosyncratic 5-day reversal (H186 in L/S form).

    Long lowest-5d-idiosyncratic-return stocks; short highest.
    Stripping SPY beta before L/S amplifies the pure idiosyncratic reversal.
    Weekly rebalance. Full 12-fold.
    """

    name = "long_short_residual_reversal"

    def __init__(
        self,
        spy_rets: pd.Series,
        sectors: pd.Series,
        reversal_window: int = 5,
        beta_window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ) -> None:
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
        var = spy.rolling(self.beta_window, min_periods=min_p).var().replace(0, np.nan)
        beta = cov.div(var, axis=0)
        spy_5d = spy.rolling(self.reversal_window).sum()
        rets_5d = prices_panel.pct_change(self.reversal_window)
        residual_5d = rets_5d.sub(beta.mul(spy_5d, axis=0))
        long_w = _sector_neutral_weights(
            -residual_5d, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        short_w = _sector_neutral_weights(
            residual_5d, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        combined = _long_short_normalize(long_w, short_w)
        return _periodic_rebalance(combined, self.rebalance_period)


class LongShortPCAReversal(CrossSectionalStrategy):
    """H473: Dollar-neutral L/S of PCA residual 5-day reversal (H191 in L/S form).

    Long lowest-5d-PCA-residual stocks; short highest. K=10 PCA factors.
    PCA stripping leaves the purest idiosyncratic signal for L/S amplification.
    Weekly rebalance. Full 12-fold. Primary MDR ceiling crack candidate.
    """

    name = "long_short_pca_reversal"

    def __init__(
        self,
        train_prices: pd.DataFrame,
        sectors: pd.Series,
        n_components: int = 10,
        reversal_window: int = 5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ) -> None:
        self.sectors = sectors
        self.n_components = n_components
        self.reversal_window = reversal_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period
        self.components_df = _fit_pca_components(train_prices, n_components)

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        common = prices_panel.columns.intersection(self.components_df.columns)
        daily_rets = prices_panel[common].pct_change().fillna(0.0)
        comps = self.components_df[common].values
        mat = daily_rets.values
        factor_rets = mat @ comps.T
        reconstruction = factor_rets @ comps
        residuals = pd.DataFrame(mat - reconstruction, index=daily_rets.index, columns=common)
        resid_5d = residuals.rolling(
            self.reversal_window, min_periods=self.reversal_window // 2 + 1
        ).sum()
        mem_cs = membership.reindex(columns=prices_panel.columns)
        long_w = _sector_neutral_weights(
            -resid_5d, mem_cs, self.sectors, self.within_sector_quantile, ascending=False
        )
        short_w = _sector_neutral_weights(
            resid_5d, mem_cs, self.sectors, self.within_sector_quantile, ascending=False
        )
        combined = _long_short_normalize(long_w, short_w)
        return _periodic_rebalance(combined, self.rebalance_period)


class LongShortOUReversal(CrossSectionalStrategy):
    """H474: Dollar-neutral L/S of OU mean reversion (H189 in L/S form).

    Long negative-z (oversold) stocks; short positive-z (overbought) stocks.
    OU z-score threshold ensures positions only taken on statistically significant deviations.
    Weekly rebalance. Full 12-fold.
    """

    name = "long_short_ou_reversal"

    def __init__(
        self,
        spy_rets: pd.Series,
        sectors: pd.Series,
        beta_window: int = 252,
        ou_window: int = 252,
        entry_z: float = 1.5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ) -> None:
        self.spy_rets = spy_rets
        self.sectors = sectors
        self.beta_window = beta_window
        self.ou_window = ou_window
        self.entry_z = entry_z
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
        var = spy.rolling(self.beta_window, min_periods=min_p).var().replace(0, np.nan)
        beta = cov.div(var, axis=0)
        residual = daily_rets.sub(beta.mul(spy, axis=0))
        resid_5d = residual.rolling(5, min_periods=3).sum()
        mu = resid_5d.rolling(self.ou_window, min_periods=self.ou_window // 2).mean()
        sd = (
            resid_5d.rolling(self.ou_window, min_periods=self.ou_window // 2)
            .std()
            .replace(0, np.nan)
        )
        z_score = (resid_5d - mu) / sd
        # Long: z < -entry_z (oversold); short: z > +entry_z (overbought)
        long_signal = (-z_score).where(z_score < -self.entry_z)
        short_signal = z_score.where(z_score > self.entry_z)
        long_w = _sector_neutral_weights(
            long_signal, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        short_w = _sector_neutral_weights(
            short_signal, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        combined = _long_short_normalize(long_w, short_w)
        return _periodic_rebalance(combined, self.rebalance_period)


class LongShortPCAOUReversal(CrossSectionalStrategy):
    """H475: Dollar-neutral L/S of PCA OU mean reversion (H192 in L/S form).

    H192 long-only PCA OU = Sharpe 1.175 -- the best long-only short-term signal.
    L/S amplifies by adding short leg on z > +entry_z stocks (overbought).
    K=10 PCA factors + OU threshold. Primary MDR ceiling crack attempt.
    Weekly rebalance. Full 12-fold.
    """

    name = "long_short_pca_ou_reversal"

    def __init__(
        self,
        train_prices: pd.DataFrame,
        sectors: pd.Series,
        n_components: int = 10,
        ou_window: int = 252,
        entry_z: float = 1.5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ) -> None:
        self.sectors = sectors
        self.n_components = n_components
        self.ou_window = ou_window
        self.entry_z = entry_z
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period
        self.components_df = _fit_pca_components(train_prices, n_components)

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        common = prices_panel.columns.intersection(self.components_df.columns)
        daily_rets = prices_panel[common].pct_change().fillna(0.0)
        comps = self.components_df[common].values
        mat = daily_rets.values
        factor_rets = mat @ comps.T
        reconstruction = factor_rets @ comps
        residuals = pd.DataFrame(mat - reconstruction, index=daily_rets.index, columns=common)
        resid_5d = residuals.rolling(5, min_periods=3).sum()
        mu = resid_5d.rolling(self.ou_window, min_periods=self.ou_window // 2).mean()
        sd = (
            resid_5d.rolling(self.ou_window, min_periods=self.ou_window // 2)
            .std()
            .replace(0, np.nan)
        )
        z_score = (resid_5d - mu) / sd
        mem_cs = membership.reindex(columns=prices_panel.columns)
        long_signal = (-z_score).where(z_score < -self.entry_z)
        short_signal = z_score.where(z_score > self.entry_z)
        long_w = _sector_neutral_weights(
            long_signal, mem_cs, self.sectors, self.within_sector_quantile, ascending=False
        )
        short_w = _sector_neutral_weights(
            short_signal, mem_cs, self.sectors, self.within_sector_quantile, ascending=False
        )
        combined = _long_short_normalize(long_w, short_w)
        return _periodic_rebalance(combined, self.rebalance_period)


class LongShortBABWeekly(CrossSectionalStrategy):
    """H476: Dollar-neutral L/S BAB at weekly rebalance (H175 in L/S form).

    Long low-beta stocks; short high-beta stocks. Equal dollar exposure.
    Frazzini-Pedersen (2014): pure beta-arbitrage alpha from L/S.
    Weekly beta refresh. Full 12-fold.
    """

    name = "long_short_bab_weekly"

    def __init__(
        self,
        spy_rets: pd.Series,
        sectors: pd.Series,
        beta_window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ) -> None:
        self.spy_rets = spy_rets
        self.sectors = sectors
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
        var = spy.rolling(self.beta_window, min_periods=min_p).var().replace(0, np.nan)
        beta = cov.div(var, axis=0)
        # Long low beta; short high beta
        long_w = _sector_neutral_weights(
            -beta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        short_w = _sector_neutral_weights(
            beta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        combined = _long_short_normalize(long_w, short_w)
        return _periodic_rebalance(combined, self.rebalance_period)


class LongShortNegativeSkewness(CrossSectionalStrategy):
    """H477: Dollar-neutral L/S negative skewness avoidance (H276 in L/S form).

    Long most-positive-skew stocks; short most-negative-skew (lottery) stocks.
    Monthly rebalance (12 turns × 3 bps = manageable if gross L/S alpha > 1%).
    Full 12-fold.
    """

    name = "long_short_negative_skewness"

    def __init__(
        self,
        sectors: pd.Series,
        skew_window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
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
        skewness = daily_rets.rolling(self.skew_window, min_periods=self.skew_window // 4).skew()
        # Long high skew (safe); short low/negative skew (lottery)
        long_w = _sector_neutral_weights(
            skewness, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        short_w = _sector_neutral_weights(
            -skewness, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        combined = _long_short_normalize(long_w, short_w)
        return _periodic_rebalance(combined, self.rebalance_period)


class LongShortQualityWeekly(CrossSectionalStrategy):
    """H478: Dollar-neutral L/S GP/TA quality signal at weekly rebalance.

    Long top-20% GP/TA; short bottom-20%. Weekly rebalance tests whether
    the stable cross-sectional GP/TA dispersion generates enough weekly L/S alpha.
    Likely killed — fundamental signal doesn't vary fast enough for weekly L/S.
    Full 12-fold (price + fundamentals, but GP/TA is forward-filled).
    """

    name = "long_short_quality_weekly"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
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
        long_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        short_w = _sector_neutral_weights(
            -gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        combined = _long_short_normalize(long_w, short_w)
        return _periodic_rebalance(combined, self.rebalance_period)


class LongShortShortMomentum(CrossSectionalStrategy):
    """H479: Dollar-neutral L/S of 3-month (63-day) price momentum. Monthly.

    Shorter horizon than H193 (12-1m killed). Higher per-period IC at 3m.
    Monthly: 12 turns × 3 bps = manageable. Falsification test for L/S momentum.
    Full 12-fold.
    """

    name = "long_short_short_momentum"

    def __init__(
        self,
        sectors: pd.Series,
        mom_window: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.mom_window = mom_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        mom_3m = prices_panel.pct_change(self.mom_window)
        long_w = _sector_neutral_weights(
            mom_3m, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        short_w = _sector_neutral_weights(
            -mom_3m, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        combined = _long_short_normalize(long_w, short_w)
        return _periodic_rebalance(combined, self.rebalance_period)


class LongShortCompositeReversal(CrossSectionalStrategy):
    """H480: Dollar-neutral L/S composite: z(5d sector reversal) + z(-MAX1_5d). Weekly.

    Short: 5-day winners that also had an extreme positive day (lottery overshoot).
    Long: 5-day losers with no extreme day (no overshooting, mean-revert cleanly).
    Combines H471 reversal signal with the lottery premium in one composite.
    Weekly rebalance. Full 12-fold.
    """

    name = "long_short_composite_reversal"

    def __init__(
        self,
        sectors: pd.Series,
        reversal_window: int = 5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ) -> None:
        self.sectors = sectors
        self.reversal_window = reversal_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        ret_5d = prices_panel.pct_change(self.reversal_window)
        max1_5d = daily_rets.rolling(
            self.reversal_window, min_periods=self.reversal_window // 2
        ).max()
        # Long signal: prior losers + no extreme day (safe reversal)
        long_score = _cs_zscore(-ret_5d).fillna(0.0) + _cs_zscore(-max1_5d).fillna(0.0)
        # Short signal: prior winners + had extreme positive day (lottery overshoot)
        short_score = _cs_zscore(ret_5d).fillna(0.0) + _cs_zscore(max1_5d).fillna(0.0)
        long_w = _sector_neutral_weights(
            long_score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        short_w = _sector_neutral_weights(
            short_score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        combined = _long_short_normalize(long_w, short_w)
        return _periodic_rebalance(combined, self.rebalance_period)
