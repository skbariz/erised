"""Cross-sectional (stock-selection) strategies for H150-H154.

Interface contract:
  generate_weights(prices_panel, membership) -> pd.DataFrame
    prices_panel : adj_close panel, rows=dates, cols=tickers.
    membership   : bool panel, same shape; True = stock in S&P 500 on that date.
    returns      : weights DataFrame, same shape. Long-only rows sum ≤ 1.0.

All signals are computed with a 1-day lag enforced by the panel engine's
shift(1) in run_panel_backtest. Strategies must only use data up to date t
when computing the weight for date t.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class CrossSectionalStrategy(ABC):
    name: str = "cs_unnamed"

    @abstractmethod
    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute target weights for every date in prices_panel.

        Weights must use only data up to (and including) each date t.
        The backtester applies a 1-bar shift, so today's weight earns
        tomorrow's return — no additional shift is required here.
        """
        ...


def _equal_weight_top_quantile(
    scores: pd.DataFrame,
    membership: pd.DataFrame,
    quantile: float,
    ascending: bool = False,
) -> pd.DataFrame:
    """Utility: for each date, equal-weight the top `quantile` fraction of
    in-membership stocks ranked by `scores`.

    ascending=False  -> high score is best (momentum, quality).
    ascending=True   -> low score is best (low accruals, low beta).
    """
    # Mask scores outside membership
    masked = scores.where(membership.reindex_like(scores).fillna(False))

    # Rank within membership (1 = best)
    if ascending:
        ranks = masked.rank(axis=1, ascending=True, na_option="keep")
    else:
        ranks = masked.rank(axis=1, ascending=False, na_option="keep")

    n_members = membership.reindex_like(scores).fillna(False).sum(axis=1)
    threshold = (n_members * quantile).clip(lower=1)

    # Select stocks within top quantile
    selected = ranks.le(threshold, axis=0) & ranks.notna()

    count = selected.sum(axis=1).replace(0, np.nan)
    return selected.astype(float).div(count, axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# H150: Cross-Sectional 12-1 Momentum, Long-Only Equal Weight
# ---------------------------------------------------------------------------


class MomentumLongOnly(CrossSectionalStrategy):
    """H150: Long top quintile of S&P 500 by 12-1 month return, equal-weighted.

    Signal: price(t-21) / price(t-252) - 1 (12-month return skipping last month).
    With monthly_rebalance=True (default), weights are only updated on the first
    trading day of each month — intra-month positions are held constant, reducing
    turnover from ~5000 trades/year to ~1200 trades/year.
    """

    name = "momentum_long_only"

    def __init__(
        self,
        long_days: int = 252,
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
        mom = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1
        daily_w = _equal_weight_top_quantile(mom, membership, self.top_quantile, ascending=False)
        if not self.monthly_rebalance:
            return daily_w
        # Keep weights constant within each month — only update on month-start dates
        is_month_start = prices_panel.index.to_series().dt.month.diff().ne(0)
        out = daily_w.copy()
        prev = pd.Series(dtype=object)
        for date, is_start in is_month_start.items():
            if is_start or prev.empty:
                prev = daily_w.loc[date]
            out.loc[date] = prev
        return out


# ---------------------------------------------------------------------------
# H151: Cross-Sectional Momentum, Market-Neutral Long/Short
# ---------------------------------------------------------------------------


class MomentumMarketNeutral(CrossSectionalStrategy):
    """H151: Long top quintile, short bottom quintile; gross exposure = 1.0.

    Each leg is 0.5 gross; net ≈ 0. Rebalances monthly by default.
    Borrow cost for the short leg is modelled separately in the runner
    (50 bps/year applied as an additional drag to net_returns).
    """

    name = "momentum_market_neutral"

    def __init__(
        self,
        long_days: int = 252,
        skip_days: int = 21,
        top_quantile: float = 0.20,
        bottom_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.long_days = long_days
        self.skip_days = skip_days
        self.top_quantile = top_quantile
        self.bottom_quantile = bottom_quantile
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        mom = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1
        mem = membership.reindex_like(mom).fillna(False)
        masked = mom.where(mem)

        ranks = masked.rank(axis=1, ascending=False, na_option="keep")
        n_members = mem.sum(axis=1)
        top_thresh = (n_members * self.top_quantile).clip(lower=1)
        bot_thresh = (n_members * (1 - self.bottom_quantile)).clip(lower=1)

        long_mask = ranks.le(top_thresh, axis=0) & ranks.notna()
        short_mask = ranks.ge(bot_thresh, axis=0) & ranks.notna()

        long_count = long_mask.sum(axis=1).replace(0, np.nan)
        short_count = short_mask.sum(axis=1).replace(0, np.nan)

        # Each leg gets 0.5 gross exposure → total gross = 1.0
        long_w = long_mask.astype(float).div(long_count, axis=0).fillna(0.0) * 0.5
        short_w = short_mask.astype(float).div(short_count, axis=0).fillna(0.0) * 0.5
        daily_w = long_w - short_w

        if not self.monthly_rebalance:
            return daily_w
        is_month_start = prices_panel.index.to_series().dt.month.diff().ne(0)
        out = daily_w.copy()
        prev = pd.Series(dtype=object)
        for date, is_start in is_month_start.items():
            if is_start or prev.empty:
                prev = daily_w.loc[date]
            out.loc[date] = prev
        return out


# ---------------------------------------------------------------------------
# H152: 52-Week High Momentum (George & Hwang 2004)
# ---------------------------------------------------------------------------


class HighWaterMomentum(CrossSectionalStrategy):
    """H152: Rank stocks by proximity to their 52-week high price.

    Signal: close / rolling_252_day_max. Stocks near their yearly high
    continue outperforming via anchoring-related momentum (George & Hwang 2004).
    """

    name = "high_water_momentum"

    def __init__(self, lookback: int = 252, top_quantile: float = 0.20):
        self.lookback = lookback
        self.top_quantile = top_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rolling_high = prices_panel.rolling(self.lookback, min_periods=self.lookback // 2).max()
        ratio = prices_panel / rolling_high
        # Shift by 1 day so today's price isn't used to compute today's signal
        ratio = ratio.shift(1)
        return _equal_weight_top_quantile(ratio, membership, self.top_quantile, ascending=False)


# ---------------------------------------------------------------------------
# H153: Sector-Neutral Momentum
# ---------------------------------------------------------------------------


class SectorNeutralMomentum(CrossSectionalStrategy):
    """H153: Select top 30% by 12-1 momentum within each GICS sector.

    Requires a sector_map: pd.Series (ticker -> sector string).
    Equal-weights selected stocks within each sector, then normalizes
    across sectors for a flat total portfolio weight.
    """

    name = "sector_neutral_momentum"

    def __init__(
        self,
        sector_map: pd.Series,
        long_days: int = 252,
        skip_days: int = 21,
        top_quantile: float = 0.30,
    ):
        self.sector_map = sector_map  # index=ticker, values=sector string
        self.long_days = long_days
        self.skip_days = skip_days
        self.top_quantile = top_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        mom = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1
        mem = membership.reindex_like(mom).fillna(False)

        # Map tickers to sectors
        tickers = prices_panel.columns.tolist()
        sectors = self.sector_map.reindex(tickers).fillna("Unknown")
        unique_sectors = sectors.unique()

        # For each sector, compute intra-sector momentum rank
        sector_weights = pd.DataFrame(0.0, index=prices_panel.index, columns=tickers)

        for sector in unique_sectors:
            sector_tickers = sectors[sectors == sector].index.tolist()
            if not sector_tickers:
                continue
            sec_mom = mom[sector_tickers]
            sec_mem = mem[sector_tickers]
            sec_w = _equal_weight_top_quantile(sec_mom, sec_mem, self.top_quantile, ascending=False)
            sector_weights[sector_tickers] = sec_w.values

        row_sum = sector_weights.sum(axis=1).replace(0, np.nan)
        return sector_weights.div(row_sum, axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# H154: SPY Trend-Gated Cross-Sectional Momentum
# ---------------------------------------------------------------------------


class TrendGatedMomentum(CrossSectionalStrategy):
    """H154: H150 momentum with a SPY moving-average gate.

    When SPY is above its 200-day MA: full momentum portfolio.
    When SPY is between 100- and 200-day MA: 50% exposure (rest cash).
    When SPY is below 200-day MA: 0% (flat).

    Requires spy_prices: pd.Series of SPY adj_close.
    """

    name = "trend_gated_momentum"

    def __init__(
        self,
        spy_prices: pd.Series,
        long_days: int = 252,
        skip_days: int = 21,
        top_quantile: float = 0.20,
        ma_long: int = 200,
        ma_short: int = 100,
    ):
        self.spy_prices = spy_prices
        self.long_days = long_days
        self.skip_days = skip_days
        self.top_quantile = top_quantile
        self.ma_long = ma_long
        self.ma_short = ma_short

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        # Build base momentum portfolio
        mom = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1
        base_weights = _equal_weight_top_quantile(
            mom, membership, self.top_quantile, ascending=False
        )

        # Compute SPY MA regime (shift(1) so today's signal uses yesterday's price)
        spy = self.spy_prices.reindex(prices_panel.index).ffill()
        ma_l = spy.rolling(self.ma_long, min_periods=self.ma_long).mean().shift(1)
        ma_s = spy.rolling(self.ma_short, min_periods=self.ma_short).mean().shift(1)
        spy_prev = spy.shift(1)

        full = spy_prev >= ma_l
        half = (spy_prev >= ma_s) & (spy_prev < ma_l)
        # below ma_s or ma_l not yet available -> scale = 0

        scale = pd.Series(0.0, index=prices_panel.index)
        scale[full] = 1.0
        scale[half] = 0.5

        return base_weights.mul(scale, axis=0)
