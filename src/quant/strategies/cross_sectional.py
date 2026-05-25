"""Cross-sectional (stock-selection) strategies for H150-H159.

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

    def __init__(
        self,
        lookback: int = 252,
        top_quantile: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.lookback = lookback
        self.top_quantile = top_quantile
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rolling_high = prices_panel.rolling(self.lookback, min_periods=self.lookback // 2).max()
        ratio = prices_panel / rolling_high
        ratio = ratio.shift(1)
        daily_w = _equal_weight_top_quantile(ratio, membership, self.top_quantile, ascending=False)
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
        monthly_rebalance: bool = True,
    ):
        self.sector_map = sector_map  # index=ticker, values=sector string
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
        mem = membership.reindex_like(mom).fillna(False)

        tickers = prices_panel.columns.tolist()
        sectors = self.sector_map.reindex(tickers).fillna("Unknown")
        unique_sectors = sectors.unique()

        # Compute equal-weight intra-sector weights; each sector sums to 1
        sector_weights = pd.DataFrame(0.0, index=prices_panel.index, columns=tickers)
        for sector in unique_sectors:
            sector_tickers = sectors[sectors == sector].index.tolist()
            if not sector_tickers:
                continue
            sec_w = _equal_weight_top_quantile(
                mom[sector_tickers], mem[sector_tickers], self.top_quantile, ascending=False
            )
            sector_weights[sector_tickers] = sec_w.values

        # Equal sector allocation: each active sector gets weight 1/N_sectors
        # Compute sector totals and active-sector count without deprecated groupby(axis=1)
        sec_totals = {}
        for sector in unique_sectors:
            sec_t = sectors[sectors == sector].index.tolist()
            sec_totals[sector] = sector_weights[sec_t].sum(axis=1)

        n_active = sum((st > 0).astype(int) for st in sec_totals.values())
        n_active = n_active.replace(0, np.nan)

        for sector in unique_sectors:
            sector_tickers = sectors[sectors == sector].index.tolist()
            st = sec_totals[sector].replace(0, np.nan)
            scale = (1.0 / n_active) / st
            sector_weights[sector_tickers] = sector_weights[sector_tickers].mul(scale, axis=0)

        daily_w = sector_weights.fillna(0.0)
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
# H154: SPY Trend-Gated Cross-Sectional Momentum
# ---------------------------------------------------------------------------


class TrendGatedMomentum(CrossSectionalStrategy):
    """H154: H150 momentum gated by the S&P 500 equal-weight index trend.

    When the equal-weight index is above its 200-day MA: full exposure.
    When between 100- and 200-day MA: 50% exposure (rest cash).
    When below both MAs: 0% (flat).

    Trend signal is computed internally from prices_panel (equal-weight
    average of all members), so no external SPY data is required.
    spy_prices kept for backward compatibility with tests; if provided it
    overrides the internal equal-weight index.
    """

    name = "trend_gated_momentum"

    def __init__(
        self,
        long_days: int = 252,
        skip_days: int = 21,
        top_quantile: float = 0.20,
        ma_long: int = 200,
        ma_short: int = 100,
        spy_prices: pd.Series | None = None,
    ):
        self.long_days = long_days
        self.skip_days = skip_days
        self.top_quantile = top_quantile
        self.ma_long = ma_long
        self.ma_short = ma_short
        self.spy_prices = spy_prices

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        mom = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1
        base_weights = _equal_weight_top_quantile(
            mom, membership, self.top_quantile, ascending=False
        )

        # Trend index: external spy_prices if provided, else equal-weight S&P 500
        if self.spy_prices is not None:
            index_prices = self.spy_prices.reindex(prices_panel.index).ffill()
        else:
            mem = membership.reindex_like(prices_panel).fillna(False)
            index_prices = prices_panel.where(mem).mean(axis=1).ffill()

        ma_l = index_prices.rolling(self.ma_long, min_periods=self.ma_long).mean().shift(1)
        ma_s = index_prices.rolling(self.ma_short, min_periods=self.ma_short).mean().shift(1)
        idx_prev = index_prices.shift(1)

        scale = pd.Series(0.0, index=prices_panel.index)
        scale[idx_prev >= ma_l] = 1.0
        scale[(idx_prev >= ma_s) & (idx_prev < ma_l)] = 0.5

        return base_weights.mul(scale, axis=0)


# ---------------------------------------------------------------------------
# Batch 2 shared utilities
# ---------------------------------------------------------------------------


def _monthly_rebalance(daily_w: pd.DataFrame, prices_panel: pd.DataFrame) -> pd.DataFrame:
    """Hold weights constant within each calendar month; only update on month-start dates."""
    is_month_start = prices_panel.index.to_series().dt.month.diff().ne(0)
    out = daily_w.copy()
    prev: pd.Series = pd.Series(dtype=float)
    for date, is_start in is_month_start.items():
        if is_start or prev.empty:
            prev = daily_w.loc[date]
        out.loc[date] = prev
    return out


def _portfolio_returns(weights: pd.DataFrame, prices_panel: pd.DataFrame) -> pd.Series:
    """Implied daily portfolio returns from a weights frame (1-day lagged weights)."""
    return (prices_panel.pct_change() * weights.shift(1)).sum(axis=1)


def _vol_target_scale(
    port_rets: pd.Series,
    vol_target: float,
    vol_window: int,
    vol_min: float,
    vol_max: float,
) -> pd.Series:
    """Exposure multiplier = clip(target / realized_vol, vol_min, vol_max).

    Uses trailing `vol_window`-day realized daily vol of `port_rets`.
    Returns 1.0 during warm-up (insufficient history).
    """
    realized = port_rets.rolling(vol_window, min_periods=5).std() * np.sqrt(252)
    return (vol_target / realized).clip(vol_min, vol_max).fillna(1.0)


# ---------------------------------------------------------------------------
# H155: Smooth Momentum (Da, Gurun & Warachka 2014)
# ---------------------------------------------------------------------------


class SmoothMomentum(CrossSectionalStrategy):
    """H155: 12-1 momentum × fraction of positive days in formation window.

    Stocks with smooth upward paths (many small positive days) have more
    durable momentum than stocks whose gain came from a few gap-up events.
    """

    name = "smooth_momentum"

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
        formation = self.long_days - self.skip_days
        daily_rets = prices_panel.pct_change()

        mom = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1
        # Vectorized: count positive days via rolling sum on bool indicator (avoids Python callbacks)
        pos = (daily_rets > 0).astype(float)
        count_pos = pos.rolling(formation, min_periods=formation // 2).sum()
        count_all = daily_rets.rolling(formation, min_periods=formation // 2).count()
        fraction_pos = (count_pos / count_all).shift(self.skip_days)
        smooth_score = mom * fraction_pos
        daily_w = _equal_weight_top_quantile(smooth_score, membership, self.top_quantile)
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H156: Jump-Penalized Momentum
# ---------------------------------------------------------------------------


class JumpPenalizedMomentum(CrossSectionalStrategy):
    """H156: 12-1 momentum computed after removing top/bottom extreme daily returns.

    Trimming the `trim_days` largest and smallest daily returns from the formation
    window removes event-driven jumps and rewards steady trend momentum.
    """

    name = "jump_penalized_momentum"

    def __init__(
        self,
        long_days: int = 252,
        skip_days: int = 21,
        top_quantile: float = 0.20,
        trim_days: int = 5,
        monthly_rebalance: bool = True,
    ):
        self.long_days = long_days
        self.skip_days = skip_days
        self.top_quantile = top_quantile
        self.trim_days = trim_days
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        formation = self.long_days - self.skip_days
        daily_rets = prices_panel.pct_change()
        _n = self.trim_days

        trimmed_mom = (
            daily_rets.rolling(formation, min_periods=formation // 2)
            .apply(
                lambda x: float(np.sort(x)[_n:-_n].sum()) if len(x) > 2 * _n else np.nan,
                raw=True,
            )
            .shift(self.skip_days)
        )
        daily_w = _equal_weight_top_quantile(trimmed_mom, membership, self.top_quantile)
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H157: Volatility-Scaled Momentum (Barroso & Santa-Clara 2015)
# ---------------------------------------------------------------------------


class VolScaledMomentum(CrossSectionalStrategy):
    """H157: H150 portfolio with vol-target exposure scaling.

    Stock selection updates monthly (H150 signal). Total exposure scales daily:
    multiplier = clip(vol_target / realized_21d_vol, vol_min, vol_max).
    Applies the H086 mechanism to cross-sectional momentum.
    """

    name = "vol_scaled_momentum"

    def __init__(
        self,
        long_days: int = 252,
        skip_days: int = 21,
        top_quantile: float = 0.20,
        vol_target: float = 0.12,
        vol_window: int = 21,
        vol_min: float = 0.2,
        vol_max: float = 1.5,
        monthly_rebalance: bool = True,
    ):
        self.long_days = long_days
        self.skip_days = skip_days
        self.top_quantile = top_quantile
        self.vol_target = vol_target
        self.vol_window = vol_window
        self.vol_min = vol_min
        self.vol_max = vol_max
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        mom = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1
        daily_w = _equal_weight_top_quantile(mom, membership, self.top_quantile)
        base_w = _monthly_rebalance(daily_w, prices_panel) if self.monthly_rebalance else daily_w

        port_rets = _portfolio_returns(base_w, prices_panel)
        scale = _vol_target_scale(
            port_rets, self.vol_target, self.vol_window, self.vol_min, self.vol_max
        )
        return base_w.mul(scale, axis=0)


# ---------------------------------------------------------------------------
# H158: Drawdown Circuit Breaker
# ---------------------------------------------------------------------------


class DrawdownCircuitBreaker(CrossSectionalStrategy):
    """H158: H150 portfolio with hysteresis-based drawdown circuit breaker.

    Tracks trailing 252-day drawdown of the unscaled H150 portfolio.
    dd < dd_soft  → reduce to scale_soft (default 50%).
    dd < dd_hard  → reduce to scale_hard (default 25%).
    Resume full exposure only when dd recovers above dd_resume (default -5%).
    """

    name = "drawdown_circuit_breaker"

    def __init__(
        self,
        long_days: int = 252,
        skip_days: int = 21,
        top_quantile: float = 0.20,
        dd_soft: float = -0.10,
        dd_hard: float = -0.15,
        dd_resume: float = -0.05,
        scale_soft: float = 0.5,
        scale_hard: float = 0.25,
        lookback: int = 252,
        monthly_rebalance: bool = True,
    ):
        self.long_days = long_days
        self.skip_days = skip_days
        self.top_quantile = top_quantile
        self.dd_soft = dd_soft
        self.dd_hard = dd_hard
        self.dd_resume = dd_resume
        self.scale_soft = scale_soft
        self.scale_hard = scale_hard
        self.lookback = lookback
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        mom = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1
        daily_w = _equal_weight_top_quantile(mom, membership, self.top_quantile)
        base_w = _monthly_rebalance(daily_w, prices_panel) if self.monthly_rebalance else daily_w

        # Implied equity curve of unscaled portfolio
        port_rets = _portfolio_returns(base_w, prices_panel)
        equity = (1 + port_rets).cumprod()
        rolling_high = equity.rolling(self.lookback, min_periods=1).max()
        dd = (equity / rolling_high - 1).shift(1)  # lag 1: use yesterday's drawdown

        # Hysteresis scale — iterate to maintain state
        scale = pd.Series(1.0, index=prices_panel.index)
        state = 1.0
        for date in prices_panel.index:
            d = dd.loc[date]
            if pd.isna(d):
                pass
            elif state < 1.0:
                if d >= self.dd_resume:
                    state = 1.0
                elif d < self.dd_hard:
                    state = self.scale_hard
                else:
                    state = self.scale_soft
            else:
                if d < self.dd_hard:
                    state = self.scale_hard
                elif d < self.dd_soft:
                    state = self.scale_soft
            scale.loc[date] = state

        return base_w.mul(scale, axis=0)


# ---------------------------------------------------------------------------
# H159: Smooth Momentum + Vol-Target Composite
# ---------------------------------------------------------------------------


class SmoothVolTargetMomentum(CrossSectionalStrategy):
    """H159: H155 smooth signal for stock selection + H157 vol-target for portfolio exposure.

    Addresses two momentum failure modes simultaneously:
    - Bad stock selection: selects smooth-path stocks (not event-driven jumpers)
    - Bad timing: reduces total exposure in high-vol regimes
    """

    name = "smooth_vol_target_momentum"

    def __init__(
        self,
        long_days: int = 252,
        skip_days: int = 21,
        top_quantile: float = 0.20,
        vol_target: float = 0.12,
        vol_window: int = 21,
        vol_min: float = 0.2,
        vol_max: float = 1.5,
        monthly_rebalance: bool = True,
    ):
        self.long_days = long_days
        self.skip_days = skip_days
        self.top_quantile = top_quantile
        self.vol_target = vol_target
        self.vol_window = vol_window
        self.vol_min = vol_min
        self.vol_max = vol_max
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        formation = self.long_days - self.skip_days
        daily_rets = prices_panel.pct_change()

        # H155 smooth momentum signal (vectorized fraction-positive)
        mom = prices_panel.shift(self.skip_days) / prices_panel.shift(self.long_days) - 1
        pos = (daily_rets > 0).astype(float)
        count_pos = pos.rolling(formation, min_periods=formation // 2).sum()
        count_all = daily_rets.rolling(formation, min_periods=formation // 2).count()
        fraction_pos = (count_pos / count_all).shift(self.skip_days)
        smooth_score = mom * fraction_pos
        daily_w = _equal_weight_top_quantile(smooth_score, membership, self.top_quantile)
        base_w = _monthly_rebalance(daily_w, prices_panel) if self.monthly_rebalance else daily_w

        # H157 vol-target scaling on unscaled portfolio returns
        port_rets = _portfolio_returns(base_w, prices_panel)
        scale = _vol_target_scale(
            port_rets, self.vol_target, self.vol_window, self.vol_min, self.vol_max
        )
        return base_w.mul(scale, axis=0)
