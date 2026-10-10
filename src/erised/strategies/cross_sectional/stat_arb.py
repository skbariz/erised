"""Statistical arbitrage: PCA residuals, Ornstein-Uhlenbeck mean reversion, and pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import (
    _fit_pca_components,
    _form_min_distance_pairs,
    _periodic_rebalance,
    _sector_neutral_weights,
)
from .base import CrossSectionalStrategy


class ResidualOUMeanReversion(CrossSectionalStrategy):
    """H189: Long stocks with deeply negative idiosyncratic 5-day return z-score.

    Avellaneda-Lee (2010): idiosyncratic return follows an OU process.
    Stocks with z-score < entry_z (relative to 252-day history) are in extreme
    negative deviation and should mean-revert.
    """

    name = "residual_ou_mean_reversion"

    def __init__(
        self,
        spy_rets: pd.Series,
        sectors: pd.Series,
        beta_window: int = 252,
        ou_window: int = 252,
        entry_z: float = -1.5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ):
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
        var = spy.rolling(self.beta_window, min_periods=min_p).var()
        beta = cov.div(var, axis=0)
        residual = daily_rets.sub(beta.mul(spy, axis=0))
        # 5-day cumulative idiosyncratic return
        resid_5d = residual.rolling(5, min_periods=3).sum()
        # Z-score vs. trailing ou_window distribution
        mu = resid_5d.rolling(self.ou_window, min_periods=self.ou_window // 2).mean()
        sd = (
            resid_5d.rolling(self.ou_window, min_periods=self.ou_window // 2)
            .std()
            .replace(0, np.nan)
        )
        z_score = (resid_5d - mu) / sd
        # Long stocks with deeply negative z-score (mean-reversion entry)
        # Negate so most negative = highest rank in sector-neutral selection
        score = -z_score.where(z_score < self.entry_z)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class SectorPairsTrading(CrossSectionalStrategy):
    """H190: Long-short pairs trading within GICS sectors.

    Training period: identify minimum-SSD pairs of normalized price series
    within each sector (Gatev et al. 2006).
    Test period: monitor log-spread z-score; enter when |z| > entry_z.
    Self-financing: long underperformer, short overperformer.
    """

    name = "sector_pairs_trading"

    def __init__(
        self,
        train_prices: pd.DataFrame,
        sectors: pd.Series,
        n_pairs_per_sector: int = 5,
        zscore_window: int = 126,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ):
        self.sectors = sectors
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.pairs = _form_min_distance_pairs(train_prices, sectors, n_pairs_per_sector)

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices_panel.index, columns=prices_panel.columns)
        log_p = np.log(prices_panel.replace(0.0, np.nan))

        for a, b in self.pairs:
            if a not in prices_panel.columns or b not in prices_panel.columns:
                continue
            spread = log_p[a] - log_p[b]
            mu = spread.rolling(self.zscore_window, min_periods=self.zscore_window // 2).mean()
            sd = (
                spread.rolling(self.zscore_window, min_periods=self.zscore_window // 2)
                .std()
                .replace(0, np.nan)
            )
            z = (spread - mu) / sd
            long_a = (z < -self.entry_z).astype(float)
            long_b = (z > self.entry_z).astype(float)
            weights[a] += long_a - long_b
            weights[b] += long_b - long_a

        abs_sum = weights.abs().sum(axis=1).replace(0, np.nan)
        return weights.div(abs_sum, axis=0).fillna(0.0)


class PCAResidualReversal(CrossSectionalStrategy):
    """H191: Short-term reversal on PCA factor residuals.

    Extends H186 (single-factor beta-stripped reversal) to K PCA factors.
    Training: fit PCA on daily returns to extract dominant risk factors.
    Test: compute multi-factor residuals, apply 5-day reversal signal.
    """

    name = "pca_residual_reversal"

    def __init__(
        self,
        train_prices: pd.DataFrame,
        sectors: pd.Series,
        n_components: int = 10,
        reversal_window: int = 5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ):
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
        comps = self.components_df[common].values  # (k, n_common)

        mat = daily_rets.values  # (n_days, n_common)
        factor_rets = mat @ comps.T  # (n_days, k)
        reconstruction = factor_rets @ comps  # (n_days, n_common)
        residuals = pd.DataFrame(mat - reconstruction, index=daily_rets.index, columns=common)
        resid_5d = residuals.rolling(
            self.reversal_window, min_periods=self.reversal_window // 2 + 1
        ).sum()
        score = -resid_5d

        mem_cs = membership.reindex(columns=prices_panel.columns)
        daily_w = _sector_neutral_weights(
            score, mem_cs, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class PCAOUMeanReversion(CrossSectionalStrategy):
    """H192: OU z-score entry on PCA factor residuals.

    Extends H189 (single-factor OU) to K PCA factors. Uses training period
    to fit PCA components, then monitors residual OU z-score during test.
    Entry: z-score < entry_z (abnormally low idiosyncratic position).
    """

    name = "pca_ou_mean_reversion"

    def __init__(
        self,
        train_prices: pd.DataFrame,
        sectors: pd.Series,
        n_components: int = 10,
        ou_window: int = 252,
        entry_z: float = -1.5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ):
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
        score = -z_score.where(z_score < self.entry_z)

        mem_cs = membership.reindex(columns=prices_panel.columns)
        daily_w = _sector_neutral_weights(
            score, mem_cs, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)
