"""Cross-sectional (stock-selection) strategies for H150-H164.

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


# ---------------------------------------------------------------------------
# H160: Long-Term Reversal (De Bondt-Thaler 36-month contrarian)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H161: Five-Year Reversal (60-month formation)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H162: Composite Value-Momentum (12-1m momentum + 3yr reversal z-score blend)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H163: Residual Momentum (CAPM-adjusted market-neutral momentum)
# ---------------------------------------------------------------------------


class ResidualMomentum(CrossSectionalStrategy):
    """H163: Rank by cumulative CAPM-residual return over the formation window.

    score_i = sum(r_i,t - beta_i * r_mkt,t) for t in [T-long, T-skip].
    beta_i estimated via rolling cov/var over the same window.
    Market proxy: spy_rets if provided, else equal-weight panel average.
    """

    name = "residual_momentum"

    def __init__(
        self,
        long_days: int = 252,
        skip_days: int = 21,
        top_quantile: float = 0.20,
        monthly_rebalance: bool = True,
        spy_rets: pd.Series | None = None,
    ):
        self.long_days = long_days
        self.skip_days = skip_days
        self.top_quantile = top_quantile
        self.monthly_rebalance = monthly_rebalance
        self.spy_rets = spy_rets

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        window = self.long_days - self.skip_days

        if self.spy_rets is not None:
            mkt_r = self.spy_rets.reindex(prices_panel.index).fillna(0.0)
        else:
            mem_f = membership.reindex_like(daily_rets).fillna(False)
            mkt_r = daily_rets.where(mem_f).mean(axis=1).fillna(0.0)

        # Rolling beta = cov(stock, mkt) / var(mkt)
        cov_sm = daily_rets.rolling(window, min_periods=window // 2).cov(mkt_r)
        var_m = mkt_r.rolling(window, min_periods=window // 2).var()
        beta = cov_sm.div(var_m, axis=0)

        stock_sum = daily_rets.rolling(window, min_periods=window // 2).sum()
        mkt_sum = mkt_r.rolling(window, min_periods=window // 2).sum()

        # Residual momentum shifted by skip_days for lookahead safety
        residual_mom = (stock_sum - beta.mul(mkt_sum, axis=0)).shift(self.skip_days)

        daily_w = _equal_weight_top_quantile(
            residual_mom, membership, self.top_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H164: Near 52-Week Low (contrarian low-proximity reversal)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Shared helpers for fundamental-data strategies (H160-H164)
# ---------------------------------------------------------------------------


def _cs_rank_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank normalised to [0, 1] each row (NaN kept as NaN)."""
    n = df.notna().sum(axis=1).replace(0, np.nan)
    return df.rank(axis=1, ascending=True, na_option="keep").div(n, axis=0)


def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score each row (NaN kept as NaN)."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def _align_fund(fund_panel: pd.DataFrame, prices_panel: pd.DataFrame) -> pd.DataFrame:
    """Reindex fundamental panel to prices_panel index and columns."""
    return fund_panel.reindex(index=prices_panel.index, columns=prices_panel.columns)


# ---------------------------------------------------------------------------
# H160: Large-Cap Value (Book-to-Market + Earnings Yield)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H161: Quality-Filtered Long-Term Reversal (De Bondt-Thaler upgraded)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H162: Value + Momentum Composite (Asness, Moskowitz & Pedersen 2013)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H163: Sector-Neutral Value + Momentum
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H164: Piotroski Value Screen (high B/M filtered by F-score ≥ 6)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Shared helper for Batch 4 sector-neutral strategies
# ---------------------------------------------------------------------------


def _sector_neutral_weights(
    score: pd.DataFrame,
    membership: pd.DataFrame,
    sectors: pd.Series,
    within_sector_quantile: float,
    ascending: bool = False,
) -> pd.DataFrame:
    """Select top `within_sector_quantile` within each GICS sector by score.

    Each active sector contributes equal weight to the portfolio.
    ascending=False means high score is best (profitability, quality).
    ascending=True means low score is best (accruals).
    """
    mem_f = membership.reindex_like(score).fillna(False)
    unique_sectors = sectors.dropna().unique()
    weights = pd.DataFrame(0.0, index=score.index, columns=score.columns)

    for sector in unique_sectors:
        sec_tickers = [t for t in sectors[sectors == sector].index if t in score.columns]
        if not sec_tickers:
            continue
        sec_w = _equal_weight_top_quantile(
            score[sec_tickers], mem_f[sec_tickers], within_sector_quantile, ascending=ascending
        )
        weights[sec_tickers] = sec_w

    # Equal sector allocation
    sec_totals: dict[str, pd.Series] = {}
    for sector in unique_sectors:
        sec_tickers = [t for t in sectors[sectors == sector].index if t in weights.columns]
        sec_totals[sector] = weights[sec_tickers].sum(axis=1)

    n_active = sum((st > 0).astype(int) for st in sec_totals.values())
    n_active = n_active.replace(0, np.nan)

    for sector in unique_sectors:
        sec_tickers = [t for t in sectors[sectors == sector].index if t in weights.columns]
        st = sec_totals[sector].replace(0, np.nan)
        scale = (1.0 / n_active) / st
        weights[sec_tickers] = weights[sec_tickers].mul(scale, axis=0)

    return weights.fillna(0.0)


# ---------------------------------------------------------------------------
# H165: Gross Profitability Rank (Novy-Marx 2013)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H166: Low Accruals Quality (Sloan 1996)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H168: Composite Quality Score (QMJ-Inspired)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H169: Quality-Value Composite (Low Accruals + Gross Profitability + Value)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H170: FF5 Investment Factor (low asset growth = conservative = higher return)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H171: Operating Profitability — FF2015 RMW variant (GP / book equity)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H172: FF5 Composite (B/M + GP/BE + low asset growth)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H173: Return on Assets (NI / TA)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H174: Cash Operating Profitability (OCF / book equity)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H175: Betting Against Beta (long-only low-beta, Frazzini-Pedersen 2014)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H176: Low Volatility Anomaly (long lowest realized vol quintile)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H177: Vol-Targeted Momentum (12-1m momentum with stock-level vol scaling)
# ---------------------------------------------------------------------------


class VolTargetedMomentum(CrossSectionalStrategy):
    """H177: 12-1 month momentum with per-stock vol-targeting overlay.

    Moreira-Muir (2017): scale each stock's weight inversely to its recent variance.
    Selects top `top_quantile` by momentum score, then scales each stock's weight
    by (target_vol / realized_vol) capped at 1.0.
    """

    name = "vol_targeted_momentum"

    def __init__(
        self,
        top_quantile: float = 0.20,
        momentum_skip: int = 21,
        momentum_window: int = 252,
        vol_window: int = 63,
        target_vol: float = 0.20,
        monthly_rebalance: bool = True,
    ):
        self.top_quantile = top_quantile
        self.momentum_skip = momentum_skip
        self.momentum_window = momentum_window
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        # 12-1 month momentum
        mom = prices_panel.pct_change(self.momentum_window - self.momentum_skip).shift(
            self.momentum_skip
        )
        base_w = _equal_weight_top_quantile(mom, membership, self.top_quantile, ascending=False)

        # Per-stock vol scaling (annualized)
        daily_rets = prices_panel.pct_change()
        annual_vol = daily_rets.rolling(
            self.vol_window, min_periods=self.vol_window // 2
        ).std() * np.sqrt(252)
        annual_vol = annual_vol.replace(0, np.nan)

        # Scale: target_vol / realized_vol, capped at 1.0
        scale = (self.target_vol / annual_vol).clip(upper=1.0).fillna(1.0)

        # Apply scale and renormalize to sum to 1
        scaled = base_w * scale
        row_sum = scaled.sum(axis=1).replace(0, np.nan)
        daily_w = scaled.div(row_sum, axis=0).fillna(0.0)

        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H178: Defensive Quality + Momentum Composite
# ---------------------------------------------------------------------------


class DefensiveQualityMomentum(CrossSectionalStrategy):
    """H178: (1/2)*z(quality_composite) + (1/2)*z(12-1m momentum), sector-neutral.

    Quality composite = avg z(GP/A, -beta_252d) — simplified defensive quality.
    Momentum = 12-1 month price return.
    Sector-neutral selection of top `within_sector_quantile`.
    """

    name = "defensive_quality_momentum"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        spy_rets: pd.Series,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        beta_window: int = 252,
        momentum_skip: int = 21,
        momentum_window: int = 252,
        monthly_rebalance: bool = True,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.spy_rets = spy_rets
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.beta_window = beta_window
        self.momentum_skip = momentum_skip
        self.momentum_window = momentum_window
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        # Quality: GP/A z-score + low beta z-score
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        gpa = (gp / ta).where(ta > 0)

        daily_rets = prices_panel.pct_change()
        spy = self.spy_rets.reindex(prices_panel.index).fillna(0.0)
        min_p = self.beta_window // 2
        cov_sm = daily_rets.rolling(self.beta_window, min_periods=min_p).cov(spy)
        var_m = spy.rolling(self.beta_window, min_periods=min_p).var()
        beta = cov_sm.div(var_m, axis=0)

        quality = (
            _cs_zscore(gpa).fillna(0.0)
            + _cs_zscore(-beta).fillna(0.0)  # negate: low beta = high quality
        ) / 2.0

        # Momentum: 12-1 month
        mom = prices_panel.pct_change(self.momentum_window - self.momentum_skip).shift(
            self.momentum_skip
        )

        composite = (_cs_zscore(quality).fillna(0.0) + _cs_zscore(mom).fillna(0.0)) / 2.0

        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H179: Low Beta Quality (BAB + quality intersection)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Shared helper: periodic (e.g. weekly) rebalance
# ---------------------------------------------------------------------------


def _periodic_rebalance(weights: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """Hold weights for `period` trading days; rebalance on days 0, period, 2*period, ..."""
    rebal_mask = pd.Series(False, index=weights.index)
    rebal_mask.iloc[::period] = True
    masked = weights.where(rebal_mask)
    return masked.ffill().fillna(0.0)


# ---------------------------------------------------------------------------
# H180: EarningsTrendLong (YoY net income growth, sector-neutral)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H181: RevenueEarningsDoubleGrowth (both NI and revenue grew YoY)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H182: FilingDateGapContinuation (price reaction on EDGAR filing dates)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H183: CashEarningsDrift (NI growth + cash-backed quality filter)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H184: EarningsMomentumComposite (SUE proxy + 63d price momentum)
# ---------------------------------------------------------------------------


class EarningsMomentumComposite(CrossSectionalStrategy):
    """H184: (z_SUE + z_mom_63d) / 2, sector-neutral.

    Combines earnings trend (fundamental) with price momentum (technical).
    Both signals in the same direction = stronger, more durable drift.
    """

    name = "earnings_momentum_composite"

    def __init__(
        self,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        yoy_window: int = 252,
        mom_window: int = 63,
        monthly_rebalance: bool = True,
    ):
        self.net_income = net_income
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.yoy_window = yoy_window
        self.mom_window = mom_window
        self.monthly_rebalance = monthly_rebalance

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ni_lag = ni.shift(self.yoy_window)
        sue = (ni - ni_lag) / ni_lag.abs().replace(0, np.nan)

        mom = prices_panel.pct_change(self.mom_window)
        composite = (_cs_zscore(sue).fillna(0.0) + _cs_zscore(mom).fillna(0.0)) / 2.0

        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        if not self.monthly_rebalance:
            return daily_w
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H185: WeeklyReversalSectorNeutral (5-day reversal, sector-neutral)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H186: ResidualWeeklyReversal (reversal of idiosyncratic 5-day return)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H187: LiquidityFilteredReversal (reversal concentrated in high-vol stocks)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H188: MonthlyReversalLong (21-day reversal, monthly rebalance, low cost)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H189: ResidualOUMeanReversion (Avellaneda-Lee OU process, z-score entry)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Batch 9 helpers
# ---------------------------------------------------------------------------


def _form_min_distance_pairs(
    train_prices: pd.DataFrame,
    sectors: pd.Series,
    n_pairs_per_sector: int,
) -> list[tuple[str, str]]:
    """Gatev et al.: find top-N pairs within each sector by normalized-price SSD."""
    import itertools

    norm = train_prices.ffill().bfill()
    first_valid = norm.apply(lambda col: col.dropna().iloc[0] if col.notna().any() else 1.0)
    norm = norm.divide(first_valid)

    pairs: list[tuple[str, str]] = []
    for sector in sectors.dropna().unique():
        sec_tickers = [t for t in sectors[sectors == sector].index if t in train_prices.columns]
        if len(sec_tickers) < 2:
            continue
        sub = norm[sec_tickers].fillna(0.0).values  # (n_days, n_tickers)
        n = len(sec_tickers)
        ssd = np.full((n, n), np.inf)
        for i, j in itertools.combinations(range(n), 2):
            diff = sub[:, i] - sub[:, j]
            ssd[i, j] = (diff**2).mean()
            ssd[j, i] = ssd[i, j]
        flat_idx = np.argsort(ssd.ravel())
        seen: set[frozenset[int]] = set()
        for idx in flat_idx:
            if len(seen) >= n_pairs_per_sector:
                break
            i, j = divmod(int(idx), n)
            if i == j:
                continue
            key = frozenset({i, j})
            if key not in seen:
                seen.add(key)
                pairs.append((sec_tickers[i], sec_tickers[j]))
    return pairs


def _fit_pca_components(
    train_prices: pd.DataFrame,
    n_components: int,
) -> pd.DataFrame:
    """PCA on training returns via SVD. Returns (n_components, n_tickers) DataFrame."""
    rets = train_prices.pct_change().dropna(how="all").fillna(0.0)
    mat = rets.values - rets.values.mean(axis=0)
    _, _, vt = np.linalg.svd(mat, full_matrices=False)
    return pd.DataFrame(vt[:n_components], columns=rets.columns)


def _long_short_normalize(
    long_w: pd.DataFrame,
    short_w: pd.DataFrame,
) -> pd.DataFrame:
    """Combine long/short legs into dollar-neutral weights (abs sum = 1.0)."""
    combined = long_w * 0.5 - short_w * 0.5
    abs_sum = combined.abs().sum(axis=1).replace(0, np.nan)
    return combined.div(abs_sum, axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# H190: SectorPairsTrading (Gatev, Goetzmann, Rouwenhorst 2006)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H191: PCAResidualReversal (multi-factor extension of H186)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H192: PCAOUMeanReversion (multi-factor Avellaneda-Lee)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H193: LongShortMomentum (true market-neutral L/S, extends H150)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H194: LongShortReversal (true market-neutral L/S, extends H185)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H195: FiftyTwoWeekHighMomentum (George-Hwang 2004)
# ---------------------------------------------------------------------------


class FiftyTwoWeekHighMomentum(CrossSectionalStrategy):
    """H195: 52-week high proximity as momentum signal.

    George and Hwang (2004): ratio of current price to 52-week rolling maximum
    predicts future returns. Stocks near their 52w high have strong momentum
    (anchor-bias inhibits full price adjustment). Sector-neutral top 20%.
    """

    name = "fifty_two_week_high_momentum"

    def __init__(
        self,
        sectors: pd.Series,
        high_window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ):
        self.sectors = sectors
        self.high_window = high_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        high_52w = prices_panel.rolling(self.high_window, min_periods=self.high_window // 2).max()
        score = prices_panel / high_52w.replace(0, np.nan)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H196: IdiosyncraticMomentum (sector-adjusted 12-1 momentum)
# ---------------------------------------------------------------------------


class IdiosyncraticMomentum(CrossSectionalStrategy):
    """H196: Idiosyncratic (sector-adjusted) momentum.

    Momentum = stock_12m_return - sector_equal_weight_12m_return.
    Removes the systematic sector component to isolate stock-specific momentum.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "idiosyncratic_momentum"

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
        sector_avg = pd.DataFrame(np.nan, index=mom.index, columns=mom.columns)
        for sector in self.sectors.dropna().unique():
            sec_tickers = [
                t for t in self.sectors[self.sectors == sector].index if t in mom.columns
            ]
            if sec_tickers:
                avg = mom[sec_tickers].mean(axis=1)
                sector_avg[sec_tickers] = np.tile(avg.values[:, None], len(sec_tickers))
        idio_mom = mom - sector_avg
        daily_w = _sector_neutral_weights(
            idio_mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H197: LowSkewnessAnomaly (Bali, Cakici, Whitelaw 2011)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H198: EarningsAcceleration (second derivative of NI growth)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H199: MultiSignalComposite (momentum + reversal + earnings z-score blend)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Batch 1 helpers (H202-H204): sector return computation
# ---------------------------------------------------------------------------


def _compute_sector_returns(
    prices_panel: pd.DataFrame,
    membership: pd.DataFrame,
    sectors: pd.Series,
    window: int,
) -> pd.DataFrame:
    """Equal-weight average of member-stock returns over `window` days, per sector.

    Returns DataFrame: index=dates, columns=sector names.
    Only in-membership stocks contribute on each date.
    """
    mem_f = membership.reindex_like(prices_panel).fillna(False)
    ret_window = prices_panel.pct_change(window)
    sec_rets: dict[str, pd.Series] = {}

    for sector in sectors.dropna().unique():
        sec_tickers = [t for t in sectors[sectors == sector].index if t in prices_panel.columns]
        if not sec_tickers:
            continue
        sec_ret = ret_window[sec_tickers].where(mem_f[sec_tickers])
        n_valid = mem_f[sec_tickers].sum(axis=1).replace(0, np.nan)
        sec_rets[sector] = sec_ret.sum(axis=1) / n_valid

    return pd.DataFrame(sec_rets, index=prices_panel.index)


# ---------------------------------------------------------------------------
# H202: IndustryConfirmedMomentum
# ---------------------------------------------------------------------------


class IndustryConfirmedMomentum(CrossSectionalStrategy):
    """H202: Moskowitz-Grinblatt (1999): 12-1m momentum gated by sector 6m trend.

    Signal: standard 12-1m price momentum (H150 signal).
    Gate: stock weight zeroed when its GICS sector 6m equal-weight return < 0.
    Sector return = avg 126d return of in-membership stocks in that sector.
    Rationale: sector momentum explains much of individual momentum; only buy
    winners in winning sectors for higher conviction entries.
    """

    name = "industry_confirmed_momentum"

    def __init__(
        self,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        sector_mom_window: int = 126,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_window = skip_window
        self.sector_mom_window = sector_mom_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        # Standard 12-1m momentum
        mom = prices_panel.pct_change(self.mom_window).shift(self.skip_window)

        # Sector 6m returns (membership-weighted)
        sector_ret_df = _compute_sector_returns(
            prices_panel, membership, self.sectors, self.sector_mom_window
        )
        sector_positive = sector_ret_df > 0  # dates × sectors

        # Map sector gate to each stock
        sec_map = self.sectors.reindex(prices_panel.columns)
        stock_gate = pd.DataFrame(False, index=prices_panel.index, columns=prices_panel.columns)
        for sec in sector_positive.columns:
            tickers = sec_map[sec_map == sec].index.tolist()
            tickers = [t for t in tickers if t in stock_gate.columns]
            if tickers:
                gate_col = sector_positive[sec].reindex(prices_panel.index)
                for t in tickers:
                    stock_gate[t] = gate_col

        # Gate: zero signal for stocks in sectors with negative 6m return
        gated_mom = mom.where(stock_gate, np.nan)

        daily_w = _sector_neutral_weights(
            gated_mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H203: SectorLeaderLongOnly
# ---------------------------------------------------------------------------


class SectorLeaderLongOnly(CrossSectionalStrategy):
    """H203: Allocate entirely to stocks inside the top-N GICS sectors by 6m return.

    Rank GICS sectors by their 6m equal-weight member return. Stocks in the
    top-N sectors are equally weighted; stocks in other sectors get zero weight.
    More aggressive than H202 (gate): concentrates in sector leaders.
    Sector return = avg 126d return of in-membership stocks in that sector.
    """

    name = "sector_leader_long_only"

    def __init__(
        self,
        sectors: pd.Series,
        sector_mom_window: int = 126,
        top_sectors: int = 2,
    ) -> None:
        self.sectors = sectors
        self.sector_mom_window = sector_mom_window
        self.top_sectors = top_sectors

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        mem_f = membership.reindex_like(prices_panel).fillna(False)

        # Sector 6m returns
        sector_ret_df = _compute_sector_returns(
            prices_panel, membership, self.sectors, self.sector_mom_window
        )

        # Rank sectors per day (1=best), select top-N
        sec_rank = sector_ret_df.rank(axis=1, ascending=False, na_option="keep")
        top_sec_mask = (sec_rank <= self.top_sectors) & sec_rank.notna()

        # Map to stocks
        sec_map = self.sectors.reindex(prices_panel.columns)
        in_top_sector = pd.DataFrame(False, index=prices_panel.index, columns=prices_panel.columns)
        for sec in top_sec_mask.columns:
            tickers = sec_map[sec_map == sec].index.tolist()
            tickers = [t for t in tickers if t in in_top_sector.columns]
            if tickers:
                top_col = top_sec_mask[sec].reindex(prices_panel.index)
                for t in tickers:
                    in_top_sector[t] = top_col

        # Equal-weight in-membership stocks inside top sectors
        eligible = in_top_sector & mem_f
        n_eligible = eligible.sum(axis=1).replace(0, np.nan)
        daily_w = eligible.astype(float).div(n_eligible, axis=0).fillna(0.0)

        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H204: IndustryRelativeStrength
# ---------------------------------------------------------------------------


class IndustryRelativeStrength(CrossSectionalStrategy):
    """H204: 21-day return minus sector-average 21d return; sector-neutral top quintile.

    Signal: stock_21d_return - sector_avg_21d_return.
    Captures stocks breaking out above their sector's recent pace — pure
    idiosyncratic short-horizon momentum. Different from H196 (12-1m version):
    H204 uses a 21-day formation with no skip window.
    Sector-neutral so macro sector bets are eliminated.
    """

    name = "industry_relative_strength"

    def __init__(
        self,
        sectors: pd.Series,
        rel_window: int = 21,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.rel_window = rel_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        mem_f = membership.reindex_like(prices_panel).fillna(False)
        ret_21 = prices_panel.pct_change(self.rel_window)

        # Sector average 21d return (membership-weighted, per date)
        sector_ret_df = _compute_sector_returns(
            prices_panel, membership, self.sectors, self.rel_window
        )

        # Map sector avg back to each stock
        sec_map = self.sectors.reindex(prices_panel.columns)
        sector_avg_for_stock = pd.DataFrame(
            np.nan, index=prices_panel.index, columns=prices_panel.columns
        )
        for sec in sector_ret_df.columns:
            tickers = sec_map[sec_map == sec].index.tolist()
            tickers = [t for t in tickers if t in sector_avg_for_stock.columns]
            if tickers:
                avg_col = sector_ret_df[sec].reindex(prices_panel.index)
                for t in tickers:
                    sector_avg_for_stock[t] = avg_col

        # Relative return: stock - sector avg
        rel_return = ret_21 - sector_avg_for_stock

        # Sector-neutral: select top quintile within each sector
        daily_w = _sector_neutral_weights(
            rel_return, mem_f, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# Batch 2: Asset Growth & Net Share Issuance (H206-H210)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Batch 3: Accruals & Earnings Quality (H211-H215)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Batch 4: Residual Momentum Variants (H216-H220)
# ---------------------------------------------------------------------------


class SectorResidualMomentum(CrossSectionalStrategy):
    """H216: Residual momentum stripping both market AND sector beta.

    Extension of H163: after stripping SPY beta, also subtract the equal-weight
    sector-average residual. Pure idiosyncratic momentum: stock return unexplained
    by both market and sector factors. Sector-neutral top 20%, monthly rebalance.
    """

    name = "sector_residual_momentum"

    def __init__(
        self,
        sectors: pd.Series,
        spy_rets: pd.Series,
        long_days: int = 252,
        skip_days: int = 21,
        within_sector_quantile: float = 0.20,
    ):
        self.sectors = sectors
        self.spy_rets = spy_rets
        self.long_days = long_days
        self.skip_days = skip_days
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        window = self.long_days - self.skip_days
        mkt_r = self.spy_rets.reindex(prices_panel.index).fillna(0.0)

        # Step 1: strip market beta (same as H163)
        cov_sm = daily_rets.rolling(window, min_periods=window // 2).cov(mkt_r)
        var_m = mkt_r.rolling(window, min_periods=window // 2).var()
        beta_mkt = cov_sm.div(var_m, axis=0)
        daily_resid1 = daily_rets - beta_mkt.mul(mkt_r, axis=0)

        # Step 2: subtract equal-weight sector-average market-residual
        mem_f = membership.reindex_like(daily_rets).fillna(False)
        sec_map = self.sectors.reindex(prices_panel.columns)
        sector_avg = pd.DataFrame(np.nan, index=prices_panel.index, columns=prices_panel.columns)
        for sec in sec_map.dropna().unique():
            tickers = sec_map[sec_map == sec].index.tolist()
            tickers = [t for t in tickers if t in daily_resid1.columns]
            if not tickers:
                continue
            mask = mem_f[tickers]
            sec_ret = daily_resid1[tickers].where(mask).mean(axis=1)
            for t in tickers:
                sector_avg[t] = sec_ret

        pure_resid = daily_resid1 - sector_avg.fillna(0.0)

        # Cumulate over window, shift by skip_days
        residual_mom = (
            pure_resid.rolling(window, min_periods=window // 2).sum().shift(self.skip_days)
        )

        daily_w = _sector_neutral_weights(
            residual_mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class ResidualMomVolScaled(CrossSectionalStrategy):
    """H217: H163-style residual momentum with idiosyncratic vol-target scaling.

    Each stock's weight is scaled by min(1, target_vol / realized_idiosyncratic_vol).
    Reduces exposure during high-volatility periods, same mechanism as H086 on H021.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "residual_mom_vol_scaled"

    def __init__(
        self,
        sectors: pd.Series,
        spy_rets: pd.Series,
        long_days: int = 252,
        skip_days: int = 21,
        vol_lookback: int = 60,
        target_vol: float = 0.12,
        within_sector_quantile: float = 0.20,
    ):
        self.sectors = sectors
        self.spy_rets = spy_rets
        self.long_days = long_days
        self.skip_days = skip_days
        self.vol_lookback = vol_lookback
        self.target_vol = target_vol
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        window = self.long_days - self.skip_days
        mkt_r = self.spy_rets.reindex(prices_panel.index).fillna(0.0)

        # CAPM residual (same as H163)
        cov_sm = daily_rets.rolling(window, min_periods=window // 2).cov(mkt_r)
        var_m = mkt_r.rolling(window, min_periods=window // 2).var()
        beta_mkt = cov_sm.div(var_m, axis=0)
        daily_resid = daily_rets - beta_mkt.mul(mkt_r, axis=0)

        residual_mom = (
            daily_resid.rolling(window, min_periods=window // 2).sum().shift(self.skip_days)
        )

        # Idiosyncratic realized vol (vol of residuals)
        resid_vol = (
            daily_resid.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2)
            .std()
            .mul(np.sqrt(252))
        )
        # Annual target vol (default 12%)
        vol_scalar = (self.target_vol / resid_vol).clip(upper=1.0)

        # Scale ranking signal before sector-neutral selection
        scaled_score = residual_mom * vol_scalar

        daily_w = _sector_neutral_weights(
            scaled_score, membership, self.sectors, self.within_sector_quantile, ascending=False
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


class FundamentalMomentum(CrossSectionalStrategy):
    """H219: Pure fundamental momentum -- no price signal.

    Composite of YoY growth in NI, gross profit, and OCF.
    z(NI_growth) + z(GP_growth) + z(OCF_growth).
    No price signal: tests whether accounting momentum predicts price returns.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "fundamental_momentum"

    def __init__(
        self,
        net_income: pd.DataFrame,
        gross_profit: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        yoy_window: int = 252,
    ):
        self.net_income = net_income
        self.gross_profit = gross_profit
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.yoy_window = yoy_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        def _yoy_growth(panel: pd.DataFrame) -> pd.DataFrame:
            lag = panel.shift(self.yoy_window)
            return ((panel - lag) / lag.abs().replace(0, np.nan)).clip(-2.0, 3.0)

        ni = _align_fund(self.net_income, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)

        z_ni = _cs_zscore(_yoy_growth(ni))
        z_gp = _cs_zscore(_yoy_growth(gp))
        z_ocf = _cs_zscore(_yoy_growth(ocf))

        composite = z_ni.add(z_gp, fill_value=0.0).add(z_ocf.fillna(0.0), fill_value=0.0)
        n = z_ni.notna().astype(float) + z_gp.notna().astype(float) + z_ocf.notna().astype(float)
        composite = composite / n.replace(0, np.nan)

        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class ResidualMomQuality(CrossSectionalStrategy):
    """H220: H163 residual momentum with high-accruals stocks zeroed out.

    Balance-sheet accruals quality gate: zero weight for stocks in top quintile
    of delta(NWC)/avg_TA (highest accruals = lowest quality = excluded).
    Sector-neutral top 20% of qualifying residual momentum, monthly rebalance.
    """

    name = "residual_mom_quality"

    def __init__(
        self,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        spy_rets: pd.Series,
        long_days: int = 252,
        skip_days: int = 21,
        accrual_window: int = 252,
        accrual_exclude_quantile: float = 0.20,
        within_sector_quantile: float = 0.20,
    ):
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.total_assets = total_assets
        self.sectors = sectors
        self.spy_rets = spy_rets
        self.long_days = long_days
        self.skip_days = skip_days
        self.accrual_window = accrual_window
        self.accrual_exclude_quantile = accrual_exclude_quantile
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        window = self.long_days - self.skip_days
        mkt_r = self.spy_rets.reindex(prices_panel.index).fillna(0.0)

        # CAPM residual momentum (H163 logic)
        cov_sm = daily_rets.rolling(window, min_periods=window // 2).cov(mkt_r)
        var_m = mkt_r.rolling(window, min_periods=window // 2).var()
        beta_mkt = cov_sm.div(var_m, axis=0)
        daily_resid = daily_rets - beta_mkt.mul(mkt_r, axis=0)
        residual_mom = (
            daily_resid.rolling(window, min_periods=window // 2).sum().shift(self.skip_days)
        )

        # Balance-sheet accruals quality gate
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        nwc = ca - cl
        avg_ta = (ta + ta.shift(self.accrual_window)) / 2
        accruals = (nwc - nwc.shift(self.accrual_window)) / avg_ta.where(avg_ta > 0)

        # Exclude top accrual_exclude_quantile (worst quality)
        accrual_rank = accruals.rank(axis=1, ascending=False, na_option="keep", pct=True)
        high_accrual = accrual_rank <= self.accrual_exclude_quantile

        gated_mom = residual_mom.where(~high_accrual, np.nan)

        daily_w = _sector_neutral_weights(
            gated_mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# Batch 5: H221-H225 -- ROE, cash quality, leverage, Piotroski, earnings consistency
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Batch 6: H226-H230 -- Value, gross profitability, revenue growth, operating leverage, asset turnover
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Batch 7: H231-H235 -- composite quality+momentum signals
# ---------------------------------------------------------------------------


class QualityMomentum(CrossSectionalStrategy):
    """H231: z(GP/TA) + z(12-1m momentum) -- Novy-Marx quality + price momentum."""

    name = "quality_momentum"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_days: int = 21,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_days = skip_days
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        gp_ta = gp / ta.where(ta > 0)
        price_ret = prices_panel.pct_change(self.mom_window - self.skip_days).shift(self.skip_days)
        score = _cs_zscore(gp_ta) + _cs_zscore(price_ret)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
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


# ---------------------------------------------------------------------------
# Batch 8: H236-H240 -- solvency, margins, ROA
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Batch 9: Market Value & Cash Flow Factors (H241–H245)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Batch 10: Structural & Terminal Composite Factors (H246–H250)
# ---------------------------------------------------------------------------


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


class RetainedEarningsProxy(CrossSectionalStrategy):
    """H247: YoY book equity growth / total assets = retained earnings proxy."""

    name = "retained_earnings_proxy"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.book_equity = book_equity
        self.total_assets = total_assets
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        be = _align_fund(self.book_equity, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        be_growth = (be - be.shift(self.window)) / ta.shift(self.window).where(
            ta.shift(self.window) > 0
        )
        be_growth = be_growth.clip(-0.5, 1.0)
        daily_w = _sector_neutral_weights(
            be_growth, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class IntangibleQuality(CrossSectionalStrategy):
    """H248: GP/book_equity -- intangible asset intensity proxy.

    High GP relative to book equity signals brand, IP, or other intangibles
    not captured on the balance sheet.
    """

    name = "intangible_quality"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.book_equity = book_equity
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        gp_be = gp / be.where(be > 0)
        gp_be = gp_be.clip(-1.0, 10.0)
        daily_w = _sector_neutral_weights(
            gp_be, membership, self.sectors, self.within_sector_quantile, ascending=False
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


# ---------------------------------------------------------------------------
# H251: IdiosyncraticVolFactor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H252: MaxReturnLotteryAvoid
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H253: DownsideBetaFactor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H254: TailRiskComposite
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H255: TailFilteredMomentum
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H256: OhlsonOScoreDistress
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H257: CHSMarketDistress
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H258: DistressTrendFactor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H259: DistressFilteredValue
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H260: LowDistressQualityMomentum
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H261: LowNetOperatingAssets
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H262: AccrualReliabilityFactor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H263: ConservativeCapexInvestment
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H264: SustainableEarningsComposite
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H265: SustainabilityFilteredProfitability
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H266: GrossMarginIntensityFactor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H267: GrossMarginMomentum
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H268: KnowledgeReturnProxy
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H269: IntangibleValueScreen
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H270: IntangibleQualityComposite
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H271: EarlyStageMomentum
# ---------------------------------------------------------------------------


class EarlyStageMomentum(CrossSectionalStrategy):
    """H271: Momentum minus vol z-score — uncrowded early-stage momentum proxy."""

    name = "early_stage_momentum"

    def __init__(
        self,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        vol_window: int = 21,
        within_sector_quantile: float = 0.20,
    ):
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_window = skip_window
        self.vol_window = vol_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rets = prices_panel.pct_change()
        mom = prices_panel.shift(self.skip_window) / prices_panel.shift(self.mom_window) - 1
        vol = rets.rolling(self.vol_window, min_periods=self.vol_window // 2).std()

        score = _cs_zscore(mom) - _cs_zscore(vol)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H272: VolatilitySurgeReturnPremium
# ---------------------------------------------------------------------------


class VolatilitySurgeReturnPremium(CrossSectionalStrategy):
    """H272: Vol spike AND positive return — attention/visibility proxy."""

    name = "volatility_surge_return_premium"

    def __init__(
        self,
        sectors: pd.Series,
        short_window: int = 21,
        long_window: int = 252,
        vol_ratio_threshold: float = 1.5,
        within_sector_quantile: float = 0.20,
    ):
        self.sectors = sectors
        self.short_window = short_window
        self.long_window = long_window
        self.vol_ratio_threshold = vol_ratio_threshold
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rets = prices_panel.pct_change()
        vol_short = rets.rolling(self.short_window, min_periods=self.short_window // 2).std()
        vol_long = rets.rolling(self.long_window, min_periods=self.long_window // 4).std()
        vol_ratio = vol_short / vol_long.where(vol_long > 1e-10)

        ret_short = prices_panel / prices_panel.shift(self.short_window) - 1
        positive_return = ret_short > 0

        # Only consider flagged stocks (vol spike + positive return); rank by vol_ratio within them
        score = vol_ratio.where(positive_return & (vol_ratio > self.vol_ratio_threshold))
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H273: PriceBreakoutMomentum
# ---------------------------------------------------------------------------


class PriceBreakoutMomentum(CrossSectionalStrategy):
    """H273: 63-day price high breakout — George & Hwang (2004) short-window."""

    name = "price_breakout_momentum"

    def __init__(
        self,
        sectors: pd.Series,
        breakout_window: int = 63,
        within_sector_quantile: float = 0.20,
    ):
        self.sectors = sectors
        self.breakout_window = breakout_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        # Prior day's 63-day max (shift avoids lookahead)
        rolling_max = (
            prices_panel.rolling(self.breakout_window, min_periods=self.breakout_window)
            .max()
            .shift(1)
        )
        breakout = prices_panel > rolling_max

        # Score: 1.0 for breakout stocks, NaN otherwise (non-breakouts excluded)
        score = breakout.astype(float).where(breakout)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H274: AttentionReversalScreen
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H275: MomentumVolatilityScore
# ---------------------------------------------------------------------------


class MomentumVolatilityScore(CrossSectionalStrategy):
    """H275: Momentum minus vol and MAX penalties (percentile composite)."""

    name = "momentum_volatility_score"

    def __init__(
        self,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        vol_window: int = 21,
        max_window: int = 21,
        vol_weight: float = 0.5,
        max_weight: float = 0.3,
        within_sector_quantile: float = 0.20,
    ):
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_window = skip_window
        self.vol_window = vol_window
        self.max_window = max_window
        self.vol_weight = vol_weight
        self.max_weight = max_weight
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rets = prices_panel.pct_change()
        mom = prices_panel.shift(self.skip_window) / prices_panel.shift(self.mom_window) - 1
        vol = rets.rolling(self.vol_window, min_periods=self.vol_window // 2).std()
        max_ret = rets.rolling(self.max_window, min_periods=self.max_window // 2).max()

        mem = membership.reindex_like(prices_panel).fillna(False)
        mom_pct = mom.where(mem).rank(axis=1, pct=True, na_option="keep")
        vol_pct = vol.where(mem).rank(axis=1, pct=True, na_option="keep")
        max_pct = max_ret.where(mem).rank(axis=1, pct=True, na_option="keep")

        score = mom_pct - self.vol_weight * vol_pct - self.max_weight * max_pct
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H276: NegativeSkewnessAvoidance
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H277: CoskewnessPortfolioPenalty
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H278: LeftTailLossFactor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H279: CrashSensitivityRegime
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H280: SkewAdjustedMomentum
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H281: GrossProfitImprovementFactor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H282: RevenueGrowthFactor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H283: OCFGrowthFactor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H284: LowCoverageMomentumProxy
# ---------------------------------------------------------------------------


class LowCoverageMomentumProxy(CrossSectionalStrategy):
    """H284: Momentum × idiosyncratic-vol trajectory as analyst-coverage proxy.

    Hong, Lim & Stein (2000): momentum stronger in low-coverage firms.
    Proxy: idio-vol ratio = rolling_std(idio_ret, 63d) / rolling_std(idio_ret, 252d).
    Low ratio = quiet information phase = low-coverage proxy.
    Score = z(momentum_12_1) - z(idio_vol_ratio).
    Distinct from H271 (total vol level, not idio-vol ratio).
    """

    name = "low_coverage_momentum_proxy"

    def __init__(
        self,
        spy_prices: pd.Series,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        beta_window: int = 63,
        ivol_short: int = 63,
        ivol_long: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.spy_prices = spy_prices
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_window = skip_window
        self.beta_window = beta_window
        self.ivol_short = ivol_short
        self.ivol_long = ivol_long
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rets = prices_panel.pct_change()
        spy_rets = self.spy_prices.pct_change().reindex(prices_panel.index).fillna(0.0)

        # Momentum 12-1
        mom = prices_panel.shift(self.skip_window) / prices_panel.shift(self.mom_window) - 1

        # Rolling beta over beta_window, then idio_ret = stock_ret - beta * spy_ret
        spy_var = spy_rets.rolling(self.beta_window, min_periods=self.beta_window // 4).var()
        # Cross-covariance using axis=0 broadcasting
        spy_mean = spy_rets.rolling(self.beta_window, min_periods=self.beta_window // 4).mean()
        cov_ri_rm = rets.mul(spy_rets, axis=0).rolling(
            self.beta_window, min_periods=self.beta_window // 4
        ).mean() - rets.rolling(self.beta_window, min_periods=self.beta_window // 4).mean().mul(
            spy_mean, axis=0
        )
        beta = cov_ri_rm.div(spy_var.replace(0, np.nan), axis=0)
        idio_ret = rets - beta.mul(spy_rets, axis=0)

        ivol_short = idio_ret.rolling(self.ivol_short, min_periods=self.ivol_short // 4).std()
        ivol_long = idio_ret.rolling(self.ivol_long, min_periods=self.ivol_long // 4).std()
        ivol_ratio = ivol_short / ivol_long.replace(0, np.nan)

        score = _cs_zscore(mom) - _cs_zscore(ivol_ratio)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H285: FundamentalMomentumComposite (EarningsAttentionComposite)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H286: LowCapexGrowthFactor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H287: InvestmentEfficiencyRatio
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H288: ROICCapexDiscipline
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H289: AssetGrowthPenalty
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H290: CapitalEfficiencyQualityComposite
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helper: buy/hold band for stateful strategies
# ---------------------------------------------------------------------------


def _buy_hold_band(
    strict_w: pd.DataFrame,
    extended_w: pd.DataFrame,
    membership: pd.DataFrame,
    rebalance_period: int = 21,
) -> pd.DataFrame:
    """Stateful buy/hold band portfolio.

    Entry: stock enters top-quintile (strict_w > 0) on a rebalance day.
    Hold: stock remains in extended selection (extended_w > 0) at next rebalance.
    Exit: stock falls out of extended selection at any rebalance day.

    Between rebalance days, the previous portfolio is held unchanged (except membership exits).
    Equal-weight among held-and-in-membership stocks.
    """
    dates = strict_w.index
    tickers = strict_w.columns.tolist()
    n_dates = len(dates)
    n_stocks = len(tickers)

    strict_arr = strict_w.values > 0
    extended_arr = extended_w.values > 0
    mem_arr = membership.reindex_like(strict_w).fillna(False).values

    held = np.zeros(n_stocks, dtype=bool)
    out = np.zeros((n_dates, n_stocks))

    for i in range(n_dates):
        is_rebal = i % rebalance_period == 0
        mem_today = mem_arr[i]

        if is_rebal:
            forced_sell = held & ~extended_arr[i]
            new_buy = strict_arr[i] & ~held & mem_today
            held = (held | new_buy) & ~forced_sell & mem_today
        else:
            held = held & mem_today

        n_active = held.sum()
        if n_active > 0:
            out[i] = held.astype(float) / n_active

    return pd.DataFrame(out, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# H291: LowTurnoverValueFactor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H292: LowTurnoverQualityFactor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H293: CostAwareMomentumLongHold
# ---------------------------------------------------------------------------


class CostAwareMomentumLongHold(CrossSectionalStrategy):
    """H293: 12-1 month momentum with bi-monthly rebalance (every 42 trading days).

    Tests whether momentum IC decays within 42 days or remains stable.
    H090 showed risk parity is insensitive to rebalance frequency; momentum may differ.
    """

    name = "cost_aware_momentum_long_hold"

    def __init__(
        self,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 42,
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
        mom = prices_panel.shift(self.skip_window) / prices_panel.shift(self.mom_window) - 1
        daily_w = _sector_neutral_weights(
            mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H294: TurnoverCappedProfitabilityFactor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H295: ExecutionEfficientComposite
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H296: TailRiskFilteredCrossProgram
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H297: DistressScreenedValueQuality
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H298: AccountingSustainabilityQualityComposite
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H352: QROELevel  (q-factor: high profitability / ROE)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H353: QInvestProfCombo  (q-factor: low investment + high ROE composite)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H354: QResidualMomentum  (momentum filtered by q-factor quality gate)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H355: QFactorComposite  (batch-1 ensemble: q-factors + GP)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H356: ManagementMispricing  (Stambaugh-Yuan management cluster)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H357: PerformanceMispricing  (Stambaugh-Yuan performance cluster)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H358: MispricingCombo  (50/50 management + performance mispricing)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H359: MispricingAvoidFilter  (equal-weight ex worst-decile mispricing)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H360: MispricingQualityComposite  (GP filter + mispricing ensemble)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H361: PercentileBinnedGP  (nonlinear quintile bin scoring of GP/TA)
# ---------------------------------------------------------------------------


class PercentileBinnedGP(CrossSectionalStrategy):
    """H361: GP/TA scored by quintile bins instead of continuous z-score.

    Assigns scores 1-5 by quintile. Tests whether the top GP quintile genuinely
    outperforms quintile 4 (nonlinear shape; Freyberger et al. 2020).
    Sector-neutral, quarterly rebalance.
    """

    name = "percentile_binned_gp"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        n_bins: int = 5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.n_bins = n_bins
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
        pct_rank = gp_ta.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        bin_score = (
            (pct_rank * self.n_bins)
            .clip(upper=self.n_bins)
            .fillna(0.0)
            .apply(lambda x: x.astype(int))
            .astype(float)
        )
        daily_w = _sector_neutral_weights(
            bin_score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H362: MonotonicityVotingComposite  (vote across GP, ROE, accruals)
# ---------------------------------------------------------------------------


class MonotonicityVotingComposite(CrossSectionalStrategy):
    """H362: Vote across GP/TA, ROE, accruals -- include only if >=2/3 signals agree.

    Inspired by Green-Hand-Zhang: if signals disagree, the stock is excluded.
    Stocks where 2+ signals place in top quintile are high-conviction longs.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "monotonicity_voting_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        vote_quantile: float = 0.20,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.vote_quantile = vote_quantile
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
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        mem = membership.reindex_like(gp_ta).fillna(False)
        n_members = mem.sum(axis=1).replace(0, np.nan)
        top_n = (n_members * self.vote_quantile).clip(lower=1)
        gp_rank = gp_ta.where(mem).rank(axis=1, ascending=False, na_option="keep")
        roe_rank = roe.where(mem).rank(axis=1, ascending=False, na_option="keep")
        acc_rank = (-accruals).where(mem).rank(axis=1, ascending=False, na_option="keep")
        votes = (
            (gp_rank.le(top_n, axis=0) & gp_rank.notna()).astype(int)
            + (roe_rank.le(top_n, axis=0) & roe_rank.notna()).astype(int)
            + (acc_rank.le(top_n, axis=0) & acc_rank.notna()).astype(int)
        ).astype(float)
        score = votes.where(votes >= 2, np.nan)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H363: ElasticNetReturnForecast  (regularized ML cross-sectional ranker)
# ---------------------------------------------------------------------------


class ElasticNetReturnForecast(CrossSectionalStrategy):
    """H363: Elastic-net regularized cross-sectional expected-return ranker.

    For each quarterly rebalance date, fits sklearn ElasticNet on a rolling
    3-year window of (feature, next-quarter-return) pairs.
    Features: GP/TA, ROE, neg_accruals, neg_asset_growth, momentum, OCF/TA.
    Falls back to GP/TA rank if sklearn unavailable.
    Freyberger et al. (2020) + Kozak et al. (2020).
    """

    name = "elastic_net_return_forecast"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        train_window: int = 756,
        pred_horizon: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.train_window = train_window
        self.pred_horizon = pred_horizon
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            from sklearn.linear_model import ElasticNet
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            gp = _align_fund(self.gross_profit, prices_panel)
            ta = _align_fund(self.total_assets, prices_panel)
            gp_ta = (gp / ta.where(ta > 0)).clip(0.0, 5.0)
            daily_w = _sector_neutral_weights(
                gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
            )
            return _periodic_rebalance(daily_w, self.rebalance_period)

        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        feat_panels = [
            (gp / ta_safe).clip(0.0, 5.0),
            (ni / be.where(be > 0)).clip(-2.0, 5.0),
            -((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0),
            -((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0),
            prices_panel.pct_change(252).shift(21).clip(-1.0, 5.0),
            (ocf / ta_safe).clip(-1.0, 2.0),
        ]

        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        # Initialise with GP/TA fallback so early folds behave like H239
        score_df = feat_panels[0].reindex(index=all_dates, columns=tickers)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos - self.pred_horizon
            train_start = max(0, train_end - self.train_window)
            if train_end - train_start < 40:
                continue
            X_rows, y_rows = [], []
            for tp in range(train_start, train_end - self.rebalance_period, self.rebalance_period):
                t_d = all_dates[tp]
                t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                try:
                    X = np.column_stack([fp.loc[t_d, tickers].values for fp in feat_panels])
                except Exception:
                    continue
                mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                fwd = (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
                valid = mem_t & np.isfinite(fwd) & np.isfinite(X).all(axis=1)
                if valid.sum() < 10:
                    continue
                X_rows.append(X[valid])
                y_rows.append(fwd[valid])
            if not X_rows:
                continue
            X_tr = np.vstack(X_rows)
            y_tr = np.concatenate(y_rows)
            scaler = StandardScaler()
            try:
                Xs = scaler.fit_transform(X_tr)
                model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=2000)
                model.fit(Xs, y_tr)
                X_pred = np.column_stack([fp.loc[rebal_date, tickers].values for fp in feat_panels])
                preds = model.predict(scaler.transform(X_pred))
                score_df.loc[rebal_date, tickers] = preds
            except Exception:
                continue

        score_df = score_df.ffill().fillna(feat_panels[0])
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H364: NonlinearMomentumBins  (extreme-only momentum, top decile)
# ---------------------------------------------------------------------------


class NonlinearMomentumBins(CrossSectionalStrategy):
    """H364: Long only the top decile of 12-1 momentum (J-shaped selection).

    Tests whether momentum alpha is concentrated in the top decile, not quintile.
    Freyberger et al. (2020) nonlinear shape diagnostics.
    Sector-neutral top 10%, monthly rebalance.
    """

    name = "nonlinear_momentum_bins"

    def __init__(
        self,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        top_decile: float = 0.10,
    ) -> None:
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_window = skip_window
        self.top_decile = top_decile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        mom = prices_panel.pct_change(self.mom_window).shift(self.skip_window)
        daily_w = _sector_neutral_weights(
            mom, membership, self.sectors, self.top_decile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H365: ShrunkSignalComposite  (batch-3 ensemble with clipped z-scores)
# ---------------------------------------------------------------------------


class ShrunkSignalComposite(CrossSectionalStrategy):
    """H365: Clipped z-score blend of batch-3 nonlinear and regularized signals.

    Combines four signals with each z-score clipped to [-clip_val, clip_val]:
      clip(z(GP_quintile_bin)) + clip(z(voting_votes)) +
      clip(z(top_decile_momentum)) + clip(z(GP/TA))
    Kozak et al. (2020): clipping acts as implicit regularization.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "shrunk_signal_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        clip_val: float = 2.0,
        n_bins: int = 5,
        vote_quantile: float = 0.20,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.clip_val = clip_val
        self.n_bins = n_bins
        self.vote_quantile = vote_quantile
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
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        mom = prices_panel.pct_change(252).shift(21)
        mem = membership.reindex_like(gp_ta).fillna(False)
        n_members = mem.sum(axis=1).replace(0, np.nan)
        top_n = (n_members * self.vote_quantile).clip(lower=1)

        pct_rank = gp_ta.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        bin_score = (
            (pct_rank * self.n_bins)
            .clip(upper=self.n_bins)
            .fillna(0.0)
            .apply(lambda x: x.astype(int))
            .astype(float)
        )
        z1 = _cs_zscore(bin_score).clip(-self.clip_val, self.clip_val).fillna(0.0)

        gp_rank = gp_ta.where(mem).rank(axis=1, ascending=False, na_option="keep")
        roe_rank = roe.where(mem).rank(axis=1, ascending=False, na_option="keep")
        acc_rank = (-accruals).where(mem).rank(axis=1, ascending=False, na_option="keep")
        votes = (
            (gp_rank.le(top_n, axis=0) & gp_rank.notna()).astype(int)
            + (roe_rank.le(top_n, axis=0) & roe_rank.notna()).astype(int)
            + (acc_rank.le(top_n, axis=0) & acc_rank.notna()).astype(int)
        ).astype(float)
        z2 = _cs_zscore(votes).clip(-self.clip_val, self.clip_val).fillna(0.0)

        mom_rank = mom.where(mem).rank(axis=1, ascending=False, na_option="keep")
        top_dec_n = (n_members * 0.10).clip(lower=1)
        in_top_mom = (mom_rank.le(top_dec_n, axis=0) & mom_rank.notna()).astype(float)
        z3 = _cs_zscore(in_top_mom).clip(-self.clip_val, self.clip_val).fillna(0.0)

        z4 = _cs_zscore(gp_ta).clip(-self.clip_val, self.clip_val).fillna(0.0)

        score = z1 + z2 + z3 + z4
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class RecentEpochOnlyGP(CrossSectionalStrategy):
    """H366: GP/TA rank multiplied by recent-period price confirmation.

    GP/TA z-score x (1 + clip(z(21d_return), -1, 1)). Tilts quality toward
    names with recent positive confirmation, down-weighting quality traps.
    McLean-Pontiff 2016: weighting towards recent evidence reduces stale signal.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "recent_epoch_only_gp"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        recent_window: int = 21,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.recent_window = recent_window
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
        recent_ret = prices_panel.pct_change(self.recent_window)
        tilt = _cs_zscore(recent_ret).clip(-1.0, 1.0).fillna(0.0)
        score = _cs_zscore(gp_ta).fillna(0.0) * (1.0 + 0.3 * tilt)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H367: PostPublicationHaircutComposite  (shrinkage-weighted quality composite)
# ---------------------------------------------------------------------------


class PostPublicationHaircutComposite(CrossSectionalStrategy):
    """H367: Composite of GP/TA, ROE, accruals with inverse-volatility shrinkage.

    Each signal z-score shrunk by its trailing cross-sectional standard deviation:
    signal_weight_i = 1 / (1 + rolling_std(z_i)). Noisier signals get more haircut.
    Approximates post-publication decay (McLean-Pontiff 2016): attenuates weak/noisy
    signals automatically. Sector-neutral top 20%, quarterly rebalance.
    """

    name = "post_publication_haircut_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        shrink_window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.shrink_window = shrink_window
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
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)

        z1 = _cs_zscore(gp_ta)
        z2 = _cs_zscore(roe)
        z3 = _cs_zscore(-accruals)

        def _haircut_weight(z: pd.DataFrame) -> pd.DataFrame:
            row_std = z.std(axis=1).rolling(self.shrink_window, min_periods=63).mean()
            weight = 1.0 / (1.0 + row_std.clip(lower=0.1))
            return z.multiply(weight, axis=0)

        score = (
            _haircut_weight(z1).fillna(0.0)
            + _haircut_weight(z2).fillna(0.0)
            + _haircut_weight(z3).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H368: IncrementalAlphaFilter  (regression-based signal selection)
# ---------------------------------------------------------------------------


class IncrementalAlphaFilter(CrossSectionalStrategy):
    """H368: Keep only cross-sectional signals with t-stat > threshold vs GP+momentum.

    For each quarterly rebalance, regresses 63d forward returns on (candidate signal,
    GP/TA, momentum) using trailing 756d data. Candidate signal kept if |t-stat| > 1.5.
    Passing signals equal-weighted into composite. Feng-Giglio-Xiu 2020.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "incremental_alpha_filter"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        train_window: int = 756,
        pred_horizon: int = 63,
        t_threshold: float = 1.5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.train_window = train_window
        self.pred_horizon = pred_horizon
        self.t_threshold = t_threshold
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
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        mom = prices_panel.pct_change(252).shift(21).clip(-1.0, 5.0)
        asset_growth = ((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0)

        candidates = {
            "roe": roe,
            "neg_accruals": -accruals,
            "neg_asset_growth": -asset_growth,
        }
        controls = [gp_ta, mom]

        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        score_df = gp_ta.reindex(index=all_dates, columns=tickers)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos - self.pred_horizon
            train_start = max(0, train_end - self.train_window)
            if train_end - train_start < 40:
                continue

            passing_signals = []
            for cname, cand_panel in candidates.items():
                X_rows, y_rows = [], []
                for tp in range(
                    train_start, train_end - self.rebalance_period, self.rebalance_period
                ):
                    t_d = all_dates[tp]
                    t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                    try:
                        c_vals = cand_panel.loc[t_d, tickers].values
                        ctrl_vals = np.column_stack([p.loc[t_d, tickers].values for p in controls])
                        X = np.column_stack([c_vals.reshape(-1, 1), ctrl_vals])
                    except Exception:
                        continue
                    mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                    fwd = (
                        (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
                    )
                    valid = mem_t & np.isfinite(fwd) & np.isfinite(X).all(axis=1)
                    if valid.sum() < 20:
                        continue
                    X_rows.append(X[valid])
                    y_rows.append(fwd[valid])

                if not X_rows:
                    continue
                X_tr = np.vstack(X_rows)
                y_tr = np.concatenate(y_rows)
                try:
                    from numpy.linalg import lstsq

                    X_aug = np.column_stack([np.ones(len(X_tr)), X_tr])
                    coef, _, _, _ = lstsq(X_aug, y_tr, rcond=None)
                    residuals = y_tr - X_aug @ coef
                    sigma2 = np.var(residuals)
                    XtX_inv = np.linalg.inv(X_aug.T @ X_aug + 1e-8 * np.eye(X_aug.shape[1]))
                    se = np.sqrt(sigma2 * XtX_inv[1, 1])
                    t_stat = abs(coef[1] / (se + 1e-10))
                    if t_stat > self.t_threshold:
                        passing_signals.append(cand_panel.loc[rebal_date, tickers].values)
                except Exception:
                    continue

            if passing_signals:
                scores = np.mean(np.column_stack(passing_signals), axis=1)
            else:
                scores = gp_ta.loc[rebal_date, tickers].values
            score_df.loc[rebal_date, tickers] = scores

        score_df = score_df.ffill().fillna(gp_ta)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H369: EigenportfolioCharacteristic  (PCA on cross-sectional characteristics)
# ---------------------------------------------------------------------------


class EigenportfolioCharacteristic(CrossSectionalStrategy):
    """H369: PCA on monthly characteristic matrix; PC1 score as ranking signal.

    Characteristics: GP/TA, ROE, neg_accruals, neg_asset_growth, momentum.
    Each quarterly rebalance: fit PCA on past year's cross-sectional data,
    extract PC1 score for current date. Long top 20% by PC1.
    Green-Hand-Zhang 2017: PCA extracts the truly independent dimensions.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "eigenportfolio_characteristic"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        n_components: int = 1,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.n_components = n_components
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
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0)
        mom = prices_panel.pct_change(252).shift(21).clip(-1.0, 5.0)
        feat_panels = [gp_ta, roe, -accruals, -asset_growth, mom]

        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        score_df = gp_ta.reindex(index=all_dates, columns=tickers)

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            daily_w = _sector_neutral_weights(
                gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
            )
            return _periodic_rebalance(daily_w, self.rebalance_period)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos
            train_start = max(0, train_end - 252)
            if train_end - train_start < 30:
                continue
            rows = []
            for tp in range(train_start, train_end, self.rebalance_period):
                t_d = all_dates[tp]
                try:
                    X = np.column_stack([fp.loc[t_d, tickers].values for fp in feat_panels])
                except Exception:
                    continue
                mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                valid = mem_t & np.isfinite(X).all(axis=1)
                if valid.sum() < 20:
                    continue
                rows.append(X)
            if not rows:
                continue
            X_train = np.vstack(rows)
            try:
                sc = StandardScaler()
                Xs = sc.fit_transform(X_train)
                pca = PCA(n_components=self.n_components)
                pca.fit(Xs)
                X_cur = np.column_stack([fp.loc[rebal_date, tickers].values for fp in feat_panels])
                X_cur_scaled = sc.transform(X_cur)
                pc1 = pca.transform(X_cur_scaled)[:, 0]
                score_df.loc[rebal_date, tickers] = pc1
            except Exception:
                continue

        score_df = score_df.ffill().fillna(gp_ta)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H370: DecayAdjustedComposite  (equal-weight blend of batch-4 signals)
# ---------------------------------------------------------------------------


class DecayAdjustedComposite(CrossSectionalStrategy):
    """H370: Equal-weight blend of batch-4 decay/alpha-control signals.

    Combines RecentEpochOnlyGP, PostPublicationHaircut, and GP/TA anchor
    via clipped z-score averaging. Acts as an ensemble with implicit
    regularization from the clipping. Sector-neutral top 20%, quarterly.
    """

    name = "decay_adjusted_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        clip_val: float = 2.0,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
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
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        recent_ret = prices_panel.pct_change(21)
        tilt = _cs_zscore(recent_ret).clip(-1.0, 1.0).fillna(0.0)
        # Signal 1: recent-epoch GP
        z1 = (_cs_zscore(gp_ta).fillna(0.0) * (1.0 + 0.3 * tilt)).clip(
            -self.clip_val, self.clip_val
        )
        # Signal 2: shrunk composite (GP + ROE + accruals)
        row_std_gp = _cs_zscore(gp_ta).std(axis=1).rolling(252, min_periods=63).mean()
        row_std_roe = _cs_zscore(roe).std(axis=1).rolling(252, min_periods=63).mean()
        row_std_acc = _cs_zscore(-accruals).std(axis=1).rolling(252, min_periods=63).mean()
        w_gp = 1.0 / (1.0 + row_std_gp.clip(lower=0.1))
        w_roe = 1.0 / (1.0 + row_std_roe.clip(lower=0.1))
        w_acc = 1.0 / (1.0 + row_std_acc.clip(lower=0.1))
        shrunk = (
            _cs_zscore(gp_ta).fillna(0.0).multiply(w_gp, axis=0)
            + _cs_zscore(roe).fillna(0.0).multiply(w_roe, axis=0)
            + _cs_zscore(-accruals).fillna(0.0).multiply(w_acc, axis=0)
        )
        z2 = _cs_zscore(shrunk).clip(-self.clip_val, self.clip_val).fillna(0.0)
        # Signal 3: plain GP/TA anchor
        z3 = _cs_zscore(gp_ta).clip(-self.clip_val, self.clip_val).fillna(0.0)
        score = z1 + z2 + z3
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H371: LinearCrossSectionalRank  (OLS linear ranker baseline)
# ---------------------------------------------------------------------------


class LinearCrossSectionalRank(CrossSectionalStrategy):
    """H371: OLS cross-sectional regression of forward return on 8 characteristics.

    Features: GP/TA, ROE, neg_accruals, neg_asset_growth, momentum,
    OCF/TA, leverage (D/A), book-to-market. OLS, no regularization.
    Required ML ladder baseline (Chen-Pelger-Zhu 2023).
    Falls back to GP/TA when training data insufficient. Quarterly rebalance.
    """

    name = "linear_cross_sectional_rank"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        sectors: pd.Series,
        train_window: int = 756,
        pred_horizon: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.long_term_debt = long_term_debt
        self.sectors = sectors
        self.train_window = train_window
        self.pred_horizon = pred_horizon
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def _build_feat_panels(self, prices_panel: pd.DataFrame) -> list[pd.DataFrame]:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        return [
            (gp / ta_safe).clip(0.0, 5.0),
            (ni / be.where(be > 0)).clip(-2.0, 5.0),
            -((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0),
            -((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0),
            prices_panel.pct_change(252).shift(21).clip(-1.0, 5.0),
            (ocf / ta_safe).clip(-1.0, 2.0),
            (ltd / ta_safe).clip(0.0, 2.0),
            (be / ta_safe).clip(0.0, 3.0),
        ]

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        feat_panels = self._build_feat_panels(prices_panel)
        gp_ta = feat_panels[0]
        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        score_df = gp_ta.reindex(index=all_dates, columns=tickers)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos - self.pred_horizon
            train_start = max(0, train_end - self.train_window)
            if train_end - train_start < 40:
                continue
            X_rows, y_rows = [], []
            for tp in range(train_start, train_end - self.rebalance_period, self.rebalance_period):
                t_d = all_dates[tp]
                t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                try:
                    X = np.column_stack([fp.loc[t_d, tickers].values for fp in feat_panels])
                except Exception:
                    continue
                mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                fwd = (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
                valid = mem_t & np.isfinite(fwd) & np.isfinite(X).all(axis=1)
                if valid.sum() < 10:
                    continue
                X_rows.append(X[valid])
                y_rows.append(fwd[valid])
            if not X_rows:
                continue
            X_tr = np.vstack(X_rows)
            y_tr = np.concatenate(y_rows)
            try:
                from sklearn.preprocessing import StandardScaler

                sc = StandardScaler()
                Xs = sc.fit_transform(X_tr)
                coef, _, _, _ = np.linalg.lstsq(
                    np.column_stack([np.ones(len(Xs)), Xs]), y_tr, rcond=None
                )
                X_pred = np.column_stack([fp.loc[rebal_date, tickers].values for fp in feat_panels])
                X_pred_s = sc.transform(X_pred)
                preds = np.column_stack([np.ones(len(X_pred_s)), X_pred_s]) @ coef
                score_df.loc[rebal_date, tickers] = preds
            except Exception:
                continue

        score_df = score_df.ffill().fillna(gp_ta)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H372: ElasticNetRankStrong  (elastic net alpha=0.3 -- stronger regularization)
# ---------------------------------------------------------------------------


class ElasticNetRankStrong(CrossSectionalStrategy):
    """H372: Elastic net with stronger regularization (alpha=0.3) on 8 features.

    Same feature set as H371. Higher alpha enforces more shrinkage toward zero,
    reduces overfitting in the 500-stock universe. Falls back to GP/TA.
    Quarterly rebalance. Freyberger et al. (2020) + Kozak et al. (2020).
    """

    name = "elastic_net_rank_strong"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        sectors: pd.Series,
        train_window: int = 756,
        pred_horizon: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
        alpha: float = 0.30,
        l1_ratio: float = 0.5,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.long_term_debt = long_term_debt
        self.sectors = sectors
        self.train_window = train_window
        self.pred_horizon = pred_horizon
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _build_feat_panels(self, prices_panel: pd.DataFrame) -> list[pd.DataFrame]:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        return [
            (gp / ta_safe).clip(0.0, 5.0),
            (ni / be.where(be > 0)).clip(-2.0, 5.0),
            -((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0),
            -((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0),
            prices_panel.pct_change(252).shift(21).clip(-1.0, 5.0),
            (ocf / ta_safe).clip(-1.0, 2.0),
            (ltd / ta_safe).clip(0.0, 2.0),
            (be / ta_safe).clip(0.0, 3.0),
        ]

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            from sklearn.linear_model import ElasticNet
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            gp = _align_fund(self.gross_profit, prices_panel)
            ta = _align_fund(self.total_assets, prices_panel)
            gp_ta = (gp / ta.where(ta > 0)).clip(0.0, 5.0)
            daily_w = _sector_neutral_weights(
                gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
            )
            return _periodic_rebalance(daily_w, self.rebalance_period)

        feat_panels = self._build_feat_panels(prices_panel)
        gp_ta = feat_panels[0]
        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        score_df = gp_ta.reindex(index=all_dates, columns=tickers)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos - self.pred_horizon
            train_start = max(0, train_end - self.train_window)
            if train_end - train_start < 40:
                continue
            X_rows, y_rows = [], []
            for tp in range(train_start, train_end - self.rebalance_period, self.rebalance_period):
                t_d = all_dates[tp]
                t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                try:
                    X = np.column_stack([fp.loc[t_d, tickers].values for fp in feat_panels])
                except Exception:
                    continue
                mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                fwd = (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
                valid = mem_t & np.isfinite(fwd) & np.isfinite(X).all(axis=1)
                if valid.sum() < 10:
                    continue
                X_rows.append(X[valid])
                y_rows.append(fwd[valid])
            if not X_rows:
                continue
            X_tr = np.vstack(X_rows)
            y_tr = np.concatenate(y_rows)
            scaler = StandardScaler()
            try:
                Xs = scaler.fit_transform(X_tr)
                model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=2000)
                model.fit(Xs, y_tr)
                X_pred = np.column_stack([fp.loc[rebal_date, tickers].values for fp in feat_panels])
                preds = model.predict(scaler.transform(X_pred))
                score_df.loc[rebal_date, tickers] = preds
            except Exception:
                continue

        score_df = score_df.ffill().fillna(gp_ta)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H373: GradientBoostingRank  (GB with price + EDGAR features)
# ---------------------------------------------------------------------------


class GradientBoostingRank(CrossSectionalStrategy):
    """H373: Gradient Boosting cross-sectional ranker (8 features: price + EDGAR).

    Uses price features (momentum, reversal, vol -- always available) plus
    EDGAR features (GP/TA, ROE, accruals, OCF/TA, asset_growth) when available.
    Price features enable non-zero predictions in pre-EDGAR folds. Shallow trees
    (max_depth=3) limit overfitting in the 500-stock universe.
    Chen-Pelger-Zhu 2023. Quarterly rebalance.
    """

    name = "gradient_boosting_rank"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        train_window: int = 756,
        pred_horizon: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
        max_depth: int = 3,
        n_estimators: int = 100,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.train_window = train_window
        self.pred_horizon = pred_horizon
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period
        self.max_depth = max_depth
        self.n_estimators = n_estimators

    def _build_feat_panels(self, prices_panel: pd.DataFrame) -> list[pd.DataFrame]:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        return [
            # Price features (always available)
            prices_panel.pct_change(252).shift(21).fillna(0.0).clip(-1.0, 5.0),
            prices_panel.pct_change(21).fillna(0.0).clip(-1.0, 2.0),
            prices_panel.pct_change(63).rolling(63).std().shift(1).fillna(0.0).clip(0.0, 0.5),
            # EDGAR features (NaN in pre-EDGAR folds, model handles missing)
            (gp / ta_safe).clip(0.0, 5.0).fillna(0.0),
            (ni / be.where(be > 0)).clip(-2.0, 5.0).fillna(0.0),
            -((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0).fillna(0.0),
            -((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0).fillna(0.0),
            (ocf / ta_safe).clip(-1.0, 2.0).fillna(0.0),
        ]

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            gp = _align_fund(self.gross_profit, prices_panel)
            ta = _align_fund(self.total_assets, prices_panel)
            gp_ta = (gp / ta.where(ta > 0)).clip(0.0, 5.0)
            daily_w = _sector_neutral_weights(
                gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
            )
            return _periodic_rebalance(daily_w, self.rebalance_period)

        feat_panels = self._build_feat_panels(prices_panel)
        mom_panel = feat_panels[0]  # price fallback
        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        score_df = mom_panel.reindex(index=all_dates, columns=tickers)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos - self.pred_horizon
            train_start = max(0, train_end - self.train_window)
            if train_end - train_start < 40:
                continue
            X_rows, y_rows = [], []
            for tp in range(train_start, train_end - self.rebalance_period, self.rebalance_period):
                t_d = all_dates[tp]
                t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                try:
                    X = np.column_stack([fp.loc[t_d, tickers].values for fp in feat_panels])
                except Exception:
                    continue
                mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                fwd = (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
                valid = mem_t & np.isfinite(fwd) & np.isfinite(X).all(axis=1)
                if valid.sum() < 10:
                    continue
                X_rows.append(X[valid])
                y_rows.append(fwd[valid])
            if not X_rows:
                continue
            X_tr = np.vstack(X_rows)
            y_tr = np.concatenate(y_rows)
            scaler = StandardScaler()
            try:
                Xs = scaler.fit_transform(X_tr)
                model = GradientBoostingRegressor(
                    max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=42
                )
                model.fit(Xs, y_tr)
                X_pred = np.column_stack([fp.loc[rebal_date, tickers].values for fp in feat_panels])
                preds = model.predict(scaler.transform(X_pred))
                score_df.loc[rebal_date, tickers] = preds
            except Exception:
                continue

        score_df = score_df.ffill().fillna(mom_panel)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H374: MLEnsembleRank  (equal-weight blend of OLS + elastic net + GB)
# ---------------------------------------------------------------------------


class MLEnsembleRank(CrossSectionalStrategy):
    """H374: Equal-weight blend of OLS, ElasticNet, and GB cross-sectional predictions.

    Model averaging smooths overfitting from any single model. Implements the
    full ML model ladder (Chen-Pelger-Zhu 2023). Falls back to GP/TA when
    training data insufficient. Quarterly rebalance.
    """

    name = "ml_ensemble_rank"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        sectors: pd.Series,
        train_window: int = 756,
        pred_horizon: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.long_term_debt = long_term_debt
        self.sectors = sectors
        self.train_window = train_window
        self.pred_horizon = pred_horizon
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def _build_feat_panels(self, prices_panel: pd.DataFrame) -> list[pd.DataFrame]:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        return [
            prices_panel.pct_change(252).shift(21).fillna(0.0).clip(-1.0, 5.0),
            prices_panel.pct_change(21).fillna(0.0).clip(-1.0, 2.0),
            (gp / ta_safe).clip(0.0, 5.0).fillna(0.0),
            (ni / be.where(be > 0)).clip(-2.0, 5.0).fillna(0.0),
            -((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0).fillna(0.0),
            -((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0).fillna(0.0),
            (ocf / ta_safe).clip(-1.0, 2.0).fillna(0.0),
            (ltd / ta_safe).clip(0.0, 2.0).fillna(0.0),
        ]

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.linear_model import ElasticNet
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            gp = _align_fund(self.gross_profit, prices_panel)
            ta = _align_fund(self.total_assets, prices_panel)
            gp_ta = (gp / ta.where(ta > 0)).clip(0.0, 5.0)
            daily_w = _sector_neutral_weights(
                gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
            )
            return _periodic_rebalance(daily_w, self.rebalance_period)

        feat_panels = self._build_feat_panels(prices_panel)
        mom_panel = feat_panels[0]
        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        score_df = mom_panel.reindex(index=all_dates, columns=tickers)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos - self.pred_horizon
            train_start = max(0, train_end - self.train_window)
            if train_end - train_start < 40:
                continue
            X_rows, y_rows = [], []
            for tp in range(train_start, train_end - self.rebalance_period, self.rebalance_period):
                t_d = all_dates[tp]
                t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                try:
                    X = np.column_stack([fp.loc[t_d, tickers].values for fp in feat_panels])
                except Exception:
                    continue
                mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                fwd = (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
                valid = mem_t & np.isfinite(fwd) & np.isfinite(X).all(axis=1)
                if valid.sum() < 10:
                    continue
                X_rows.append(X[valid])
                y_rows.append(fwd[valid])
            if not X_rows:
                continue
            X_tr = np.vstack(X_rows)
            y_tr = np.concatenate(y_rows)
            scaler = StandardScaler()
            try:
                Xs = scaler.fit_transform(X_tr)
                m1_coef, _, _, _ = np.linalg.lstsq(
                    np.column_stack([np.ones(len(Xs)), Xs]), y_tr, rcond=None
                )
                m2 = ElasticNet(alpha=0.3, l1_ratio=0.5, max_iter=2000)
                m2.fit(Xs, y_tr)
                m3 = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=42)
                m3.fit(Xs, y_tr)
                X_pred = np.column_stack([fp.loc[rebal_date, tickers].values for fp in feat_panels])
                Xps = scaler.transform(X_pred)
                p1 = np.column_stack([np.ones(len(Xps)), Xps]) @ m1_coef
                p2 = m2.predict(Xps)
                p3 = m3.predict(Xps)
                preds = (p1 + p2 + p3) / 3.0
                score_df.loc[rebal_date, tickers] = preds
            except Exception:
                continue

        score_df = score_df.ffill().fillna(mom_panel)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H375: MacroStateGBRank  (GB with SPY 200d MA macro-state feature)
# ---------------------------------------------------------------------------


class MacroStateGBRank(CrossSectionalStrategy):
    """H375: Gradient Boosting ranker with SPY 200d MA macro-state feature.

    9 features: price+EDGAR features from H373 plus a binary macro-state flag
    (SPY > SPY_200d_MA = 1, else 0). Macro state conditions the entire model.
    Chen-Pelger-Zhu 2023: macro-state conditioning improves asset pricing models.
    Quarterly rebalance.
    """

    name = "macro_state_gb_rank"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        spy_prices: pd.Series,
        sectors: pd.Series,
        train_window: int = 756,
        pred_horizon: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
        max_depth: int = 3,
        n_estimators: int = 100,
        ma_window: int = 200,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.spy_prices = spy_prices
        self.sectors = sectors
        self.train_window = train_window
        self.pred_horizon = pred_horizon
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.ma_window = ma_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            gp = _align_fund(self.gross_profit, prices_panel)
            ta = _align_fund(self.total_assets, prices_panel)
            gp_ta = (gp / ta.where(ta > 0)).clip(0.0, 5.0)
            daily_w = _sector_neutral_weights(
                gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
            )
            return _periodic_rebalance(daily_w, self.rebalance_period)

        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)

        spy_ma = self.spy_prices.reindex(prices_panel.index).ffill()
        macro_state = (
            (spy_ma > spy_ma.rolling(self.ma_window, min_periods=100).mean())
            .astype(float)
            .fillna(0.5)
        )

        feat_panels = [
            prices_panel.pct_change(252).shift(21).fillna(0.0).clip(-1.0, 5.0),
            prices_panel.pct_change(21).fillna(0.0).clip(-1.0, 2.0),
            prices_panel.pct_change(63).rolling(63).std().shift(1).fillna(0.0).clip(0.0, 0.5),
            (gp / ta_safe).clip(0.0, 5.0).fillna(0.0),
            (ni / be.where(be > 0)).clip(-2.0, 5.0).fillna(0.0),
            -((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0).fillna(0.0),
            -((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0).fillna(0.0),
            (ocf / ta_safe).clip(-1.0, 2.0).fillna(0.0),
        ]
        mom_panel = feat_panels[0]
        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        score_df = mom_panel.reindex(index=all_dates, columns=tickers)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos - self.pred_horizon
            train_start = max(0, train_end - self.train_window)
            if train_end - train_start < 40:
                continue
            X_rows, y_rows = [], []
            for tp in range(train_start, train_end - self.rebalance_period, self.rebalance_period):
                t_d = all_dates[tp]
                t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                ms = macro_state.loc[t_d] if t_d in macro_state.index else 0.5
                try:
                    X_base = np.column_stack([fp.loc[t_d, tickers].values for fp in feat_panels])
                    X = np.column_stack([X_base, np.full(len(tickers), ms)])
                except Exception:
                    continue
                mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                fwd = (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
                valid = mem_t & np.isfinite(fwd) & np.isfinite(X).all(axis=1)
                if valid.sum() < 10:
                    continue
                X_rows.append(X[valid])
                y_rows.append(fwd[valid])
            if not X_rows:
                continue
            X_tr = np.vstack(X_rows)
            y_tr = np.concatenate(y_rows)
            scaler = StandardScaler()
            try:
                Xs = scaler.fit_transform(X_tr)
                model = GradientBoostingRegressor(
                    max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=42
                )
                model.fit(Xs, y_tr)
                ms_cur = macro_state.loc[rebal_date] if rebal_date in macro_state.index else 0.5
                X_pred_base = np.column_stack(
                    [fp.loc[rebal_date, tickers].values for fp in feat_panels]
                )
                X_pred = np.column_stack([X_pred_base, np.full(len(tickers), ms_cur)])
                preds = model.predict(scaler.transform(X_pred))
                score_df.loc[rebal_date, tickers] = preds
            except Exception:
                continue

        score_df = score_df.ffill().fillna(mom_panel)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# Shared helper: detect filing event windows from fundamental panel changes
# ---------------------------------------------------------------------------


def _filing_window(fund_panel: pd.DataFrame, lookback_days: int = 5) -> pd.DataFrame:
    """Boolean mask: True in the lookback_days after any filing event.

    Filing event = date when forward-filled fundamental panel changes value.
    Uses past-only window (causal, no lookahead).
    """
    changed = fund_panel.diff().abs().fillna(0.0) > 1e-6
    return changed.rolling(lookback_days, min_periods=1).max().astype(bool)


# ---------------------------------------------------------------------------
# H376: EarningsWindowQualityBoost  (GP quality with post-filing size boost)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H377: EarningsSurpriseDrift  (PEAD scaled by NI change magnitude)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H378: QuietPeriodMomentum  (momentum avoiding post-filing window)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H379: EarningsConfirmedQuality  (GP rank filtered by positive NI surprise)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H380: EventAwareComposite  (batch-6 earnings-event composite)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H381: LowDilutionGovernance  (low share issuance = governance proxy)
# ---------------------------------------------------------------------------


class LowDilutionGovernance(CrossSectionalStrategy):
    """H381: Long stocks with lowest annual share dilution rate.

    dilution = (shares_t - shares_{t-252}) / shares_{t-252}.
    Low dilution = management not extracting value via issuance.
    Governance proxy (Gompers-Ishii-Metrick 2003). Sector-neutral, quarterly.
    """

    name = "low_dilution_governance"

    def __init__(
        self,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
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
        sh = _align_fund(self.shares_outstanding, prices_panel)
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        daily_w = _sector_neutral_weights(
            dilution, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H382: BuybackQualityGovernance  (negative dilution = net buyback quality)
# ---------------------------------------------------------------------------


class BuybackQualityGovernance(CrossSectionalStrategy):
    """H382: Long stocks with declining share count (net buybacks).

    net_buyback_signal = max(0, -dilution) = value when shares decrease.
    Only positive values enter the score; no signal if shares increase.
    Bebchuk-Cohen-Ferrell 2009: management returning capital is measurable.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "buyback_quality_governance"

    def __init__(
        self,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
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
        sh = _align_fund(self.shares_outstanding, prices_panel)
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        net_buyback = (-dilution).clip(lower=0.0)
        daily_w = _sector_neutral_weights(
            net_buyback, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H383: CapitalDisciplineGovernance  (dilution + buyback + accruals composite)
# ---------------------------------------------------------------------------


class CapitalDisciplineGovernance(CrossSectionalStrategy):
    """H383: Governance composite: z(-dilution) + z(net_buyback) + z(-accruals).

    Three independent governance proxies combined. Low dilution and active
    buybacks signal capital return focus; low accruals signal cash-backed quality.
    Edmans 2011, Bebchuk 2009: composite governance beats any single proxy.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "capital_discipline_governance"

    def __init__(
        self,
        shares_outstanding: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.shares_outstanding = shares_outstanding
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
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        net_buyback = (-dilution).clip(lower=0.0)
        avg_ta_safe = ((ta + ta.shift(self.lookback)) / 2).where(ta > 0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        score = (
            _cs_zscore(-dilution).fillna(0.0)
            + _cs_zscore(net_buyback).fillna(0.0)
            + _cs_zscore(-accruals).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H384: GovernanceQualityComposite  (governance x GP quality gate)
# ---------------------------------------------------------------------------


class GovernanceQualityComposite(CrossSectionalStrategy):
    """H384: Capital discipline governance composite filtered by GP/TA quality gate.

    Requires GP/TA > median within membership before ranking by governance score.
    Governance signal only meaningful for fundamentally healthy firms.
    Edmans 2011: governance predicts returns especially when fundamentals strong.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "governance_quality_composite"

    def __init__(
        self,
        shares_outstanding: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        gross_profit: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.shares_outstanding = shares_outstanding
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
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
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(self.lookback)) / 2).where(ta > 0)
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        net_buyback = (-dilution).clip(lower=0.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        gov_score = (
            _cs_zscore(-dilution).fillna(0.0)
            + _cs_zscore(net_buyback).fillna(0.0)
            + _cs_zscore(-accruals).fillna(0.0)
        )
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        mem = membership.reindex_like(gp_ta).fillna(False)
        gp_median = gp_ta.where(mem).median(axis=1)
        quality_gate = gp_ta.ge(gp_median, axis=0) & mem
        gated = gov_score.where(quality_gate, np.nan)
        daily_w = _sector_neutral_weights(
            gated, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H385: GovernanceMDRComposite  (batch-7 governance ensemble)
# ---------------------------------------------------------------------------


class GovernanceMDRComposite(CrossSectionalStrategy):
    """H385: Equal-weight composite of batch-7 governance proxy signals.

    Combines z(-dilution), z(net_buyback), z(-accruals), z(GP/TA) with
    governance composite gate. Tests whether governance signals diversify
    beyond pure quality signals. Sector-neutral top 20%, quarterly.
    """

    name = "governance_mdr_composite"

    def __init__(
        self,
        shares_outstanding: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        gross_profit: pd.DataFrame,
        sectors: pd.Series,
        clip_val: float = 2.0,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.shares_outstanding = shares_outstanding
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.gross_profit = gross_profit
        self.sectors = sectors
        self.clip_val = clip_val
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        sh_lag = sh.shift(252)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        net_buyback = (-dilution).clip(lower=0.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)

        z1 = _cs_zscore(-dilution).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z2 = _cs_zscore(net_buyback).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z3 = _cs_zscore(-accruals).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z4 = _cs_zscore(gp_ta).clip(-self.clip_val, self.clip_val).fillna(0.0)

        score = z1 + z2 + z3 + z4
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H386: SinExclusionPremium  (ex-Consumer Staples equal-weight)
# ---------------------------------------------------------------------------


class SinExclusionPremium(CrossSectionalStrategy):
    """H386: Equal-weight S&P 500 excluding Consumer Staples sector.

    Proxy for sin-stock exclusion (Hong-Kacperczyk 2009): Consumer Staples
    includes tobacco, alcohol, and other socially screened categories.
    Tests whether sector-level exclusion pressure affects large-cap returns.
    Equal-weight, quarterly rebalance.
    """

    name = "sin_exclusion_premium"

    def __init__(
        self,
        sectors: pd.Series,
        excluded_sector: str = "Consumer Staples",
        rebalance_period: int = 63,
    ) -> None:
        self.sectors = sectors
        self.excluded_sector = excluded_sector
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        excluded_tickers = set(self.sectors[self.sectors == self.excluded_sector].index.tolist())
        mem = membership.copy()
        for t in excluded_tickers:
            if t in mem.columns:
                mem[t] = False
        n_inc = mem.sum(axis=1).replace(0, np.nan)
        daily_w = mem.astype(float).div(n_inc, axis=0).fillna(0.0)
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H387: LowCarbonSectorNeutral  (ex-Energy sector-neutral GP quality)
# ---------------------------------------------------------------------------


class LowCarbonSectorNeutral(CrossSectionalStrategy):
    """H387: Sector-neutral GP/TA quality excluding Energy sector.

    Proxy for low-carbon quality portfolio (Bolton-Kacperczyk 2021):
    Energy sector has highest carbon intensity. Excluding it and doing
    sector-neutral GP/TA rank captures carbon-risk-free quality signal.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "low_carbon_sector_neutral"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        excluded_sector: str = "Energy",
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.excluded_sector = excluded_sector
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
        excluded_tickers = set(self.sectors[self.sectors == self.excluded_sector].index.tolist())
        filtered_mem = membership.copy()
        for t in excluded_tickers:
            if t in filtered_mem.columns:
                filtered_mem[t] = False
        daily_w = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H388: GreenProxyQuality  (Utilities + IT sector quality portfolio)
# ---------------------------------------------------------------------------


class GreenProxyQuality(CrossSectionalStrategy):
    """H388: GP/TA quality ranked within Utilities + Information Technology.

    Green proxy: these two sectors have lowest carbon intensity in S&P 500.
    Tests whether a sector-concentrated quality signal within the green proxy
    earns distinct returns from full-universe quality.
    Pastor-Stambaugh-Taylor 2022. Equal-weight top 20%, quarterly.
    """

    name = "green_proxy_quality"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        green_sectors: tuple = ("Utilities", "Information Technology"),
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.green_sectors = green_sectors
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
        green_tickers = set(self.sectors[self.sectors.isin(self.green_sectors)].index.tolist())
        green_mem = pd.DataFrame(False, index=membership.index, columns=membership.columns)
        for t in green_tickers:
            if t in green_mem.columns:
                green_mem[t] = membership[t]
        n_green = green_mem.sum(axis=1).replace(0, np.nan)
        if (n_green < 2).all():
            n = membership.sum(axis=1).replace(0, np.nan)
            return _periodic_rebalance(
                membership.astype(float).div(n, axis=0).fillna(0.0), self.rebalance_period
            )
        daily_w = _equal_weight_top_quantile(
            gp_ta, green_mem, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H389: ControversyProxyAvoid  (avoid high-leverage + dilution + low GP)
# ---------------------------------------------------------------------------


class ControversyProxyAvoid(CrossSectionalStrategy):
    """H389: GP/TA quality excluding worst-decile controversy proxy.

    Controversy proxy = z(leverage) + z(dilution) + z(-GP/TA).
    Avoid bottom decile (most controversial). Long remaining stocks by GP/TA.
    Proxy for ESG exclusion without vendor data (Hong-Kacperczyk 2009 inspired).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "controversy_proxy_avoid"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        exclude_pct: float = 0.10,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.long_term_debt = long_term_debt
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.lookback = lookback
        self.exclude_pct = exclude_pct
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
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        leverage = (ltd / ta_safe).clip(0.0, 2.0)
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        controversy = (
            _cs_zscore(leverage).fillna(0.0)
            + _cs_zscore(dilution).fillna(0.0)
            + _cs_zscore(-gp_ta).fillna(0.0)
        )
        mem = membership.reindex_like(controversy).fillna(False)
        n_members = mem.sum(axis=1).replace(0, np.nan)
        excl_n = (n_members * self.exclude_pct).clip(lower=1)
        ranks_asc = controversy.where(mem).rank(axis=1, ascending=True, na_option="keep")
        excluded = (ranks_asc.ge(ranks_asc.max(axis=1) - excl_n, axis=0)) & ranks_asc.notna()
        filtered_mem = mem & ~excluded
        daily_w = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H390: ESGProxyComposite  (batch-8 ESG proxy ensemble)
# ---------------------------------------------------------------------------


class ESGProxyComposite(CrossSectionalStrategy):
    """H390: Composite of batch-8 ESG/controversy proxy signals.

    Blends: ex-Consumer-Staples quality, ex-Energy quality, controversy-
    avoided quality, and plain GP/TA anchor. Tests whether sector-exclusion
    and controversy-avoidance signals add diversification.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "esg_proxy_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        clip_val: float = 2.0,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.long_term_debt = long_term_debt
        self.shares_outstanding = shares_outstanding
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
        ltd = _align_fund(self.long_term_debt, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        leverage = (ltd / ta_safe).clip(0.0, 2.0)
        sh_lag = sh.shift(252)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)

        # ex-staples GP
        staples_t = set(self.sectors[self.sectors == "Consumer Staples"].index)
        mem_ex_st = membership.copy()
        for t in staples_t:
            if t in mem_ex_st.columns:
                mem_ex_st[t] = False
        daily_w1 = _sector_neutral_weights(
            gp_ta, mem_ex_st, self.sectors, self.within_sector_quantile, ascending=False
        )

        # ex-energy GP
        energy_t = set(self.sectors[self.sectors == "Energy"].index)
        mem_ex_en = membership.copy()
        for t in energy_t:
            if t in mem_ex_en.columns:
                mem_ex_en[t] = False
        daily_w2 = _sector_neutral_weights(
            gp_ta, mem_ex_en, self.sectors, self.within_sector_quantile, ascending=False
        )

        # controversy-avoided GP
        controversy = (
            _cs_zscore(leverage).fillna(0.0)
            + _cs_zscore(dilution).fillna(0.0)
            + _cs_zscore(-gp_ta).fillna(0.0)
        )
        mem = membership.reindex_like(controversy).fillna(False)
        n_members = mem.sum(axis=1).replace(0, np.nan)
        excl_n = (n_members * 0.10).clip(lower=1)
        ranks_asc = controversy.where(mem).rank(axis=1, ascending=True, na_option="keep")
        excluded = (ranks_asc.ge(ranks_asc.max(axis=1) - excl_n, axis=0)) & ranks_asc.notna()
        filtered_mem = mem & ~excluded
        daily_w3 = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )

        # plain GP/TA anchor
        daily_w4 = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )

        combined = daily_w1 + daily_w2 + daily_w3 + daily_w4
        row_sum = combined.sum(axis=1).replace(0, np.nan)
        daily_w = combined.div(row_sum, axis=0).fillna(0.0)
        return _periodic_rebalance(daily_w, self.rebalance_period)


class LowAttentionValuePort(CrossSectionalStrategy):
    """H391: GP/TA quality rank dampened by recent price-activity (attention proxy).

    Attention proxy = trailing 21d mean(|daily_return|) / trailing 63d mean(|daily_return|).
    High ratio = stock is drawing attention; low ratio = neglected.
    Signal = z(GP/TA) * (1 - clip(z(attention_ratio), 0, 2) * 0.3).
    Neglected quality stocks reprice more slowly without event-risk noise.
    Antweiler-Frank 2004. Sector-neutral top 20%, monthly rebalance.
    """

    name = "low_attention_value_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        short_window: int = 21,
        long_window: int = 63,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.short_window = short_window
        self.long_window = long_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)

        daily_abs_ret = prices_panel.pct_change().abs()
        short_act = daily_abs_ret.rolling(self.short_window, min_periods=5).mean()
        long_act = daily_abs_ret.rolling(self.long_window, min_periods=20).mean()
        attention_ratio = (short_act / long_act.replace(0, np.nan)).clip(0.0, 5.0)

        z_gp = _cs_zscore(gp_ta).fillna(0.0)
        z_att = _cs_zscore(attention_ratio).clip(0.0, 2.0).fillna(0.0)
        score = z_gp * (1.0 - 0.3 * z_att)

        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H392: ConsistentMomentumPort  (both 12m and 3m momentum agree = confirmed)
# ---------------------------------------------------------------------------


class ConsistentMomentumPort(CrossSectionalStrategy):
    """H392: 12-1 momentum filtered by 3-month momentum agreement (consistency gate).

    Only hold momentum stocks where both 12m and 3m momentum are positive and
    in the top half of the cross-section. Consistent trend = "crowd-confirmed"
    signal. Chen-De-Hu-Hwang 2014: crowd agreement strengthens anomaly signals.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "consistent_momentum_port"

    def __init__(
        self,
        sectors: pd.Series,
        long_window: int = 252,
        skip_window: int = 21,
        short_window: int = 63,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.sectors = sectors
        self.long_window = long_window
        self.skip_window = skip_window
        self.short_window = short_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        mom_long = prices_panel.pct_change(self.long_window).shift(self.skip_window)
        mom_short = prices_panel.pct_change(self.short_window)
        mem = membership.reindex_like(mom_long).fillna(False)
        med_long = mom_long.where(mem).median(axis=1)
        med_short = mom_short.where(mem).median(axis=1)
        # Gate: both signals must be above their cross-sectional median
        confirmed = mom_long.ge(med_long, axis=0) & mom_short.ge(med_short, axis=0)
        gated_mom = mom_long.where(confirmed, np.nan)
        daily_w = _sector_neutral_weights(
            gated_mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H393: LowVolDisagreementAvoid  (GP quality weighted by inverse realized vol)
# ---------------------------------------------------------------------------


class LowVolDisagreementAvoid(CrossSectionalStrategy):
    """H393: GP/TA quality weighted by inverse realized volatility.

    Position weight inversely proportional to 21d realized vol (high vol =
    high disagreement → reduce weight). Antweiler-Frank 2004: disagreement
    predicts volatility. Penalising high-vol names improves Sharpe.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "low_vol_disagreement_avoid"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        vol_window: int = 21,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.vol_window = vol_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        daily_ret = prices_panel.pct_change()
        realized_vol = daily_ret.rolling(self.vol_window, min_periods=5).std().shift(1)
        inv_vol = (1.0 / realized_vol.replace(0, np.nan)).clip(0.0, 100.0)
        base_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        scaled = (base_w * inv_vol).fillna(0.0)
        row_sum = scaled.sum(axis=1).replace(0, np.nan)
        rescaled = scaled.div(row_sum, axis=0).fillna(0.0)
        return _monthly_rebalance(rescaled, prices_panel)


# ---------------------------------------------------------------------------
# H394: ReturnSpikeRiskFilter  (GP composite with large-return event risk cut)
# ---------------------------------------------------------------------------


class ReturnSpikeRiskFilter(CrossSectionalStrategy):
    """H394: GP/TA quality with 50% weight cut when |return| spikes > 2 sigma.

    Detects event risk via absolute return z-score. If a stock's 5d mean
    absolute return exceeds 2 sigma of its trailing 63d distribution, cut
    position weight by 50%. Reduces exposure to M&A, earnings gaps, short
    squeezes. Sector-neutral top 20%, monthly rebalance.
    """

    name = "return_spike_risk_filter"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        spike_window: int = 5,
        lookback_window: int = 63,
        spike_threshold: float = 2.0,
        cut_factor: float = 0.5,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.spike_window = spike_window
        self.lookback_window = lookback_window
        self.spike_threshold = spike_threshold
        self.cut_factor = cut_factor
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
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0)
        # Sustainability filter (H265-style) then GP rank
        sust = _cs_zscore(-accruals).fillna(0.0) + _cs_zscore(-asset_growth).fillna(0.0)
        mem = membership.reindex_like(gp_ta).fillna(False)
        sust_rank = sust.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        filtered_mem = membership & (sust_rank >= 0.50).fillna(False)
        base_w = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        # Spike detection
        abs_ret = prices_panel.pct_change().abs()
        recent_act = abs_ret.rolling(self.spike_window, min_periods=1).mean()
        hist_mean = (
            abs_ret.rolling(self.lookback_window, min_periods=20).mean().shift(self.spike_window)
        )
        hist_std = (
            abs_ret.rolling(self.lookback_window, min_periods=20).std().shift(self.spike_window)
        )
        z_spike = (recent_act - hist_mean) / hist_std.replace(0, np.nan)
        spike_flag = z_spike.gt(self.spike_threshold).fillna(False)
        scale = spike_flag.astype(float) * (self.cut_factor - 1.0) + 1.0
        scaled = (base_w * scale).clip(lower=0.0)
        row_sum = scaled.sum(axis=1).replace(0, np.nan)
        rescaled = scaled.div(row_sum, axis=0).fillna(0.0)
        return _monthly_rebalance(rescaled, prices_panel)


# ---------------------------------------------------------------------------
# H395: VolatilityAwareComposite  (batch-9 attention ensemble)
# ---------------------------------------------------------------------------


class VolatilityAwareComposite(CrossSectionalStrategy):
    """H395: Equal-weight blend of batch-9 volatility/attention-aware signals.

    Combines: neglected quality (GP dampened by activity), consistent momentum
    (12m+3m agreement), GP x inverse-vol, and spike-filtered GP.
    Tests whether price-based attention proxies diversify vs pure EDGAR quality.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "volatility_aware_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        clip_val: float = 2.0,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.clip_val = clip_val
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
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0)
        daily_abs_ret = prices_panel.pct_change().abs()
        short_act = daily_abs_ret.rolling(21, min_periods=5).mean()
        long_act = daily_abs_ret.rolling(63, min_periods=20).mean()
        attention_ratio = (short_act / long_act.replace(0, np.nan)).clip(0.0, 5.0)
        mom_long = prices_panel.pct_change(252).shift(21)
        mom_short = prices_panel.pct_change(63)
        realized_vol = prices_panel.pct_change().rolling(21, min_periods=5).std().shift(1)
        inv_vol = (1.0 / realized_vol.replace(0, np.nan)).clip(0.0, 100.0)

        mem = membership.reindex_like(gp_ta).fillna(False)
        med_long = mom_long.where(mem).median(axis=1)
        med_short = mom_short.where(mem).median(axis=1)
        confirmed_mom = mom_long.where(
            mom_long.ge(med_long, axis=0) & mom_short.ge(med_short, axis=0), np.nan
        )

        z_gp = _cs_zscore(gp_ta)
        z_att = _cs_zscore(attention_ratio).clip(0.0, 2.0).fillna(0.0)
        neglect_score = (z_gp * (1.0 - 0.3 * z_att)).fillna(0.0)

        sust = _cs_zscore(-accruals).fillna(0.0) + _cs_zscore(-asset_growth).fillna(0.0)
        sust_rank = sust.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        filtered_mem = membership & (sust_rank >= 0.50).fillna(False)

        z1 = _cs_zscore(neglect_score).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z2 = _cs_zscore(confirmed_mom).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z3 = _cs_zscore(gp_ta * inv_vol).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z4 = _cs_zscore(gp_ta.where(filtered_mem)).clip(-self.clip_val, self.clip_val).fillna(0.0)
        score = z1 + z2 + z3 + z4

        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H396: LowTurnoverGP  (threshold rebalancing for GP rank)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H397: SpreadAwarePositionSizer  (GP quality weighted by inverse vol-proxy spread)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H398: LiquidityRegimeCapPort  (ML ensemble with low-vol regime cap)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H399: ExecutionOptimizedComposite  (batch-10 execution ensemble)
# ---------------------------------------------------------------------------


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


# ===========================================================================
# SET 6 — H401–H415: Earnings Quality, Disclosure Quality & Tone Proxies
# ===========================================================================


# ---------------------------------------------------------------------------
# H401: AccrualQualityPort  (Bernard & Thomas PEAD proxy via accruals)
# ---------------------------------------------------------------------------


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
        ta_safe = ta.where(ta > 0)
        avg_ta = ((ta + ta.shift(252)) / 2).where(ta > 0)
        # Accruals: lower is better (ascending=True in selection)
        accruals = ((ni - ocf) / avg_ta).clip(-1.0, 1.0)
        # ascending=True → smallest accruals rank first (best)
        daily_w = _sector_neutral_weights(
            accruals, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H402: EarningsMomentumQuality  (confirmed earnings growth — PEAD proxy)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H403: TorpedoRiskAvoid  (Skinner & Sloan: high P/B + declining NI → avoid)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H404: RevenueEarningsAcceleration  (2nd derivative PEAD proxy)
# ---------------------------------------------------------------------------


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
        rev_safe = rev.where(rev.shift(self.annual_lag) > 0)
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


# ---------------------------------------------------------------------------
# H405: CashEarningsPersistence  (So 2013: analyst error proxy via OCF dominance)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H406: EarningsPersistencePort  (Li 2008 readability → stable NI proxy)
# ---------------------------------------------------------------------------


class EarningsPersistencePort(CrossSectionalStrategy):
    """H406: Long stocks with most-stable NI/TA over 2 years (low earnings variance).

    Stability = 1 / rolling_std(NI/TA, 2 years). Stable earnings = higher-quality
    disclosure (Li 2008 readability proxy) → better future earnings persistence.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "earnings_persistence_port"

    def __init__(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        rolling_std = ni_ta.rolling(self.window, min_periods=self.window // 4).std()
        # Stability = inverse of variance; low std = high stability
        stability = 1.0 / rolling_std.replace(0, np.nan).clip(lower=1e-6)
        daily_w = _sector_neutral_weights(
            stability, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H407: CashFlowReliabilityPort  (Li 2010 FLS durability proxy)
# ---------------------------------------------------------------------------


class CashFlowReliabilityPort(CrossSectionalStrategy):
    """H407: Long stocks where OCF consistently exceeds NI over 2 years.

    Rolling fraction of periods with OCF > NI. High fraction = cash earnings
    reliably dominate accruals → durable financial quality. Proxy for Li (2010):
    positive forward-looking tone firms have more durable earnings.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "cash_flow_reliability_port"

    def __init__(
        self,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ocf_exceeds_ni = (ocf > ni).astype(float)
        reliability = ocf_exceeds_ni.rolling(self.window, min_periods=self.window // 4).mean()
        daily_w = _sector_neutral_weights(
            reliability, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H408: BookValueGrowthPort  (retained earnings quality / FLS proxy)
# ---------------------------------------------------------------------------


class BookValueGrowthPort(CrossSectionalStrategy):
    """H408: Long top-quintile by annual book equity growth rate.

    Book equity growth = book_equity / book_equity.shift(252) - 1. Growing book
    equity = retained earnings quality = management reinvesting profits confidently.
    Proxy for Li (2010) forward-looking positive tone.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "book_value_growth_port"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
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
        be = _align_fund(self.book_equity, prices_panel)
        be_lag = be.shift(self.lookback)
        be_safe_lag = be_lag.where(be_lag.abs() > 0)
        be_growth = ((be - be_lag) / be_safe_lag.abs()).clip(-1.0, 5.0)
        daily_w = _sector_neutral_weights(
            be_growth, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H409: BalanceSheetStrengthPort  (composite BS improvement — FLS trajectory proxy)
# ---------------------------------------------------------------------------


class BalanceSheetStrengthPort(CrossSectionalStrategy):
    """H409: Long top-quintile by composite balance sheet improvement.

    Z-score composite of: (1) debt/assets declining YoY, (2) OCF/TA improving YoY,
    (3) book equity/TA improving YoY. All three signals together proxy for positive
    disclosure trajectory (Li 2010 forward-looking statements content).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "balance_sheet_strength_port"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        operating_cf: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.book_equity = book_equity
        self.long_term_debt = long_term_debt
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
        be = _align_fund(self.book_equity, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        # Ratios
        debt_ratio = (ltd / ta_safe).clip(0.0, 5.0)
        ocf_ratio = (ocf / ta_safe).clip(-2.0, 5.0)
        be_ratio = (be / ta_safe).clip(-1.0, 5.0)
        # YoY improvements
        debt_improve = -(debt_ratio - debt_ratio.shift(self.lookback))  # negative = improving
        ocf_improve = ocf_ratio - ocf_ratio.shift(self.lookback)
        be_improve = be_ratio - be_ratio.shift(self.lookback)
        composite = (
            _cs_zscore(debt_improve).fillna(0.0)
            + _cs_zscore(ocf_improve).fillna(0.0)
            + _cs_zscore(be_improve).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H410: WorkingCapitalHealthPort  (WC efficiency improvement — operational signal)
# ---------------------------------------------------------------------------


class WorkingCapitalHealthPort(CrossSectionalStrategy):
    """H410: Long top-quintile by improving (CA − CL) / Revenue ratio.

    Working capital efficiency = (CA − CL) / Revenue. YoY improvement = better
    operational discipline and cash conversion. Proxy for Li (2010) FLS: improving
    operational signals correlate with positive forward-looking tone.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "working_capital_health_port"

    def __init__(
        self,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.revenue = revenue
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        wc = ca - cl
        wc_ratio = (wc / rev_safe).clip(-2.0, 5.0)
        wc_improve = wc_ratio - wc_ratio.shift(self.lookback)
        daily_w = _sector_neutral_weights(
            wc_improve, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H411: ConsistentGrowthStreak  (Mayew proxy: persistent positive growth streak)
# ---------------------------------------------------------------------------


class ConsistentGrowthStreak(CrossSectionalStrategy):
    """H411: Long stocks with persistent simultaneous revenue + NI growth.

    Rolling fraction of days over 2 years where YoY revenue growth > 0 AND
    YoY NI growth > 0. High fraction = persistent positive management tone proxy
    (Mayew & Venkatachalam 2012): confident management → better future performance.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "consistent_growth_streak"

    def __init__(
        self,
        revenue: pd.DataFrame,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        growth_lookback: int = 252,
        streak_window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.revenue = revenue
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.growth_lookback = growth_lookback
        self.streak_window = streak_window
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
        rev_growth_pos = (rev > rev.shift(self.growth_lookback)).astype(float)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        ni_growth_pos = (ni_ta > ni_ta.shift(self.growth_lookback)).astype(float)
        both_growing = rev_growth_pos * ni_growth_pos
        streak = both_growing.rolling(
            self.streak_window, min_periods=self.streak_window // 4
        ).mean()
        daily_w = _sector_neutral_weights(
            streak, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H412: FundamentalDivergenceAvoid  (torpedo risk: price up / fundamentals down)
# ---------------------------------------------------------------------------


class FundamentalDivergenceAvoid(CrossSectionalStrategy):
    """H412: Long stocks where fundamentals outpace price; avoid the reverse.

    Divergence score = z(NI/TA YoY growth) − z(6m price return). Positive divergence
    = fundamentals running ahead of price (value opportunity). Negative divergence
    = price running ahead of fundamentals (torpedo risk). Monthly rebalance to capture
    price dynamics. Sector-neutral top 20%.
    """

    name = "fundamental_divergence_avoid"

    def __init__(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        fundamental_lookback: int = 252,
        price_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.fundamental_lookback = fundamental_lookback
        self.price_lookback = price_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        ni_growth = (ni_ta - ni_ta.shift(self.fundamental_lookback)).clip(-2.0, 5.0)
        price_ret = prices_panel / prices_panel.shift(self.price_lookback) - 1
        divergence = _cs_zscore(ni_growth).fillna(0.0) - _cs_zscore(price_ret).fillna(0.0)
        daily_w = _sector_neutral_weights(
            divergence, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H413: MarginExpansionPort  (positive management tone proxy via margin dynamics)
# ---------------------------------------------------------------------------


class MarginExpansionPort(CrossSectionalStrategy):
    """H413: Long top-quintile by gross margin expansion + revenue growth confirm.

    GP/Revenue YoY change (margin expansion) with revenue growth > 0 filter.
    Expanding margins with growing revenue = pricing power + competitive advantage.
    Mayew & Venkatachalam proxy: confident management → improving margins.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "margin_expansion_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.revenue = revenue
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
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        gp_margin = (gp / rev_safe).clip(0.0, 2.0)
        margin_expansion = gp_margin - gp_margin.shift(self.lookback)
        rev_growth_pos = (rev > rev.shift(self.lookback)).astype(float)
        # Only consider stocks with positive revenue growth
        score = margin_expansion.where(rev_growth_pos.astype(bool))
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H414: EarningsVariabilityAvoid  (low NI volatility = low uncertainty proxy)
# ---------------------------------------------------------------------------


class EarningsVariabilityAvoid(CrossSectionalStrategy):
    """H414: Long stocks with lowest NI/TA rolling variance (stable earnings).

    Low NI volatility = predictable, stable earnings environment. Proxy for
    Mayew & Venkatachalam (2012): uncertain management (high vocal stress) predicts
    weaker future performance. Low NI variance = low management uncertainty proxy.
    Sector-neutral bottom-20%-variance (ascending=True), quarterly rebalance.
    """

    name = "earnings_variability_avoid"

    def __init__(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        variability = ni_ta.rolling(self.window, min_periods=self.window // 4).std()
        # ascending=True: lowest variability gets selected (best quality)
        daily_w = _sector_neutral_weights(
            variability, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H415: StableMarginMomentum  (stable business model + price confirmation)
# ---------------------------------------------------------------------------


class StableMarginMomentum(CrossSectionalStrategy):
    """H415: Composite of low GP/Revenue volatility + 6m price momentum.

    Stable gross margins (low GP/Revenue rolling std) combined with positive
    6m price momentum. Stocks with both confirmed price trend AND stable business
    model have double confirmation. Proxy for credible, consistent management.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "stable_margin_momentum"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        margin_window: int = 504,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.sectors = sectors
        self.margin_window = margin_window
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        gp_margin = (gp / rev_safe).clip(0.0, 2.0)
        margin_std = gp_margin.rolling(
            self.margin_window, min_periods=self.margin_window // 4
        ).std()
        # Low margin std = stable; negate so high stability = high score
        margin_stability = -margin_std
        # 6m momentum
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        composite = _cs_zscore(margin_stability).fillna(0.0) + _cs_zscore(momentum).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ===========================================================================
# SET 6 — H416–H420: Product-Market & Competitive Position Proxies
# ===========================================================================


class DifferentiatedMarginPort(CrossSectionalStrategy):
    """H416: Long firms with high AND stable gross margins (differentiation proxy).

    z(GP/Revenue level) + z(-std(GP/Revenue, 2yr)). Persistently high, stable
    margins = product differentiation (Hoberg & Phillips 2016 TNIC proxy).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "differentiated_margin_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        stability_window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.sectors = sectors
        self.stability_window = stability_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        gp_margin = (gp / rev_safe).clip(0.0, 2.0)
        margin_std = gp_margin.rolling(
            self.stability_window, min_periods=self.stability_window // 4
        ).std()
        composite = _cs_zscore(gp_margin).fillna(0.0) + _cs_zscore(-margin_std).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CompetitivePressureAvoid(CrossSectionalStrategy):
    """H417: Avoid margin compression + revenue decel; rank rest by GP/TA quality.

    Filter: GP/Revenue declining AND revenue growth decelerating = losing market
    position (Hoberg, Phillips & Prabhala 2014). Remove these, rank survivors.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "competitive_pressure_avoid"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        accel_lag: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.total_assets = total_assets
        self.sectors = sectors
        self.lookback = lookback
        self.accel_lag = accel_lag
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
        rev_safe = rev.where(rev > 0)
        gp_margin = (gp / rev_safe).clip(0.0, 2.0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        margin_declining = gp_margin < gp_margin.shift(self.lookback)
        rev_growth = (rev - rev.shift(self.lookback)) / rev.shift(self.lookback).abs().clip(
            lower=1.0
        )
        rev_decelerating = rev_growth < rev_growth.shift(self.accel_lag)
        under_threat = (margin_declining & rev_decelerating).fillna(False)
        safe_mem = membership & ~under_threat
        daily_w = _sector_neutral_weights(
            gp_ta, safe_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class PeerGroupMomentum(CrossSectionalStrategy):
    """H418: Sector-relative 6m momentum winners with NI growth confirmation.

    Proxy for peer group information spillover: sector leaders with improving
    fundamentals likely benefiting from positive peer earnings signals
    (Hoberg & Phillips 2016). Sector-neutral top 20%, monthly rebalance.
    """

    name = "peer_group_momentum"

    def __init__(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        ni_lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.mom_lookback = mom_lookback
        self.ni_lookback = ni_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        ni_growing = (ni_ta > ni_ta.shift(self.ni_lookback)).fillna(False)
        mom = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        mem_f = membership.reindex_like(mom).fillna(False)
        rel_mom = pd.DataFrame(0.0, index=mom.index, columns=mom.columns)
        for sector in self.sectors.dropna().unique():
            tickers = [t for t in self.sectors[self.sectors == sector].index if t in mom.columns]
            if not tickers:
                continue
            sec_m = mom[tickers].where(mem_f[tickers])
            med = sec_m.median(axis=1)
            rel_mom[tickers] = sec_m.sub(med, axis=0).fillna(0.0).values
        quality_mem = membership & ni_growing
        daily_w = _sector_neutral_weights(
            rel_mom, quality_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CashDefenseProxy(CrossSectionalStrategy):
    """H419: Avoid firms building defensive cash under competitive threat.

    Filter: CA/TA rising AND GP/Revenue declining = defensive cash hoarding.
    Hoberg, Phillips & Prabhala (2014): product market threat triggers cash buildup.
    Long survivors ranked by GP/TA quality. Sector-neutral top 20%, quarterly.
    """

    name = "cash_defense_proxy"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.total_assets = total_assets
        self.current_assets = current_assets
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
        rev = _align_fund(self.revenue, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        rev_safe = rev.where(rev > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        gp_margin = (gp / rev_safe).clip(0.0, 2.0)
        ca_ratio = (ca / ta_safe).clip(0.0, 2.0)
        ca_rising = (ca_ratio > ca_ratio.shift(self.lookback)).fillna(False)
        margin_declining = (gp_margin < gp_margin.shift(self.lookback)).fillna(False)
        defensive_hoarding = ca_rising & margin_declining
        safe_mem = membership & ~defensive_hoarding
        daily_w = _sector_neutral_weights(
            gp_ta, safe_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MarketShareProxy(CrossSectionalStrategy):
    """H420: Long top-quintile sector-relative revenue growth (market share gainers).

    Revenue growth - sector median revenue growth. Proxy for market share gains
    (Hoberg & Phillips 2016 without product text). Sector-neutral top 20%, quarterly.
    """

    name = "market_share_proxy"

    def __init__(
        self,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.revenue = revenue
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        rev_lag = rev.shift(self.lookback)
        rev_growth = ((rev - rev_lag) / rev_lag.abs().clip(lower=1.0)).clip(-1.0, 5.0)
        mem_f = membership.reindex_like(rev_growth).fillna(False)
        sec_rel_rev = pd.DataFrame(0.0, index=rev_growth.index, columns=rev_growth.columns)
        for sector in self.sectors.dropna().unique():
            tickers = [
                t for t in self.sectors[self.sectors == sector].index if t in rev_growth.columns
            ]
            if not tickers:
                continue
            sec_g = rev_growth[tickers].where(mem_f[tickers])
            med = sec_g.median(axis=1)
            sec_rel_rev[tickers] = sec_g.sub(med, axis=0).fillna(0.0).values
        daily_w = _sector_neutral_weights(
            sec_rel_rev, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ===========================================================================
# SET 6 — H426–H430: Innovation & R&D Efficiency Proxies
# ===========================================================================


class AssetEfficiencyGrowth(CrossSectionalStrategy):
    """H426: Long top-quintile by improving Revenue/TA (asset turnover growth).

    Innovation efficiency = output per input. Revenue/TA improvement proxies
    for increasing return on knowledge capital (Hirshleifer et al. 2013).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "asset_efficiency_growth"

    def __init__(
        self,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.revenue = revenue
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
        rev = _align_fund(self.revenue, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        at = (rev / ta_safe).clip(0.0, 10.0)
        at_growth = at - at.shift(self.lookback)
        daily_w = _sector_neutral_weights(
            at_growth, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CapitalProductivityPort(CrossSectionalStrategy):
    """H427: Long top-quintile by GP/TA YoY growth (knowledge capital productivity).

    GP/TA growth rate (different from H239 GP/TA level). Captures improving
    return on capital — proxy for Kogan et al. (2017) patent-value shocks.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "capital_productivity_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
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
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        gp_ta_growth = gp_ta - gp_ta.shift(self.lookback)
        daily_w = _sector_neutral_weights(
            gp_ta_growth, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class GrowthReinvestmentQuality(CrossSectionalStrategy):
    """H428: Quality + growth + retention composite as disciplined reinvestment proxy.

    z(GP/TA) + z(revenue growth) + z(book equity growth). Three factors.
    Proxy for Eberhart et al. (2004): unexpected R&D with balance-sheet strength.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "growth_reinvestment_quality"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        revenue: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.revenue = revenue
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
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        rev_growth = (
            (rev - rev.shift(self.lookback)) / rev.shift(self.lookback).abs().clip(lower=1.0)
        ).clip(-1.0, 5.0)
        be_growth = (
            (be - be.shift(self.lookback)) / be.shift(self.lookback).abs().clip(lower=1.0)
        ).clip(-1.0, 5.0)
        composite = (
            _cs_zscore(gp_ta).fillna(0.0)
            + _cs_zscore(rev_growth).fillna(0.0)
            + _cs_zscore(be_growth).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class SectorRelativeMarginProxy(CrossSectionalStrategy):
    """H429: Sector-relative GP/Revenue premium + 6m momentum composite.

    z(GP/Revenue - sector median) + z(6m momentum). Above-sector-median GP margin
    = innovation/intangible intensity vs peers (Hirshleifer et al. 2013 proxy).
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "sector_relative_margin_proxy"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.gross_profit = gross_profit
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
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        gp_margin = (gp / rev_safe).clip(0.0, 2.0)
        mem_f = membership.reindex_like(gp_margin).fillna(False)
        sec_rel = pd.DataFrame(0.0, index=gp_margin.index, columns=gp_margin.columns)
        for sector in self.sectors.dropna().unique():
            tickers = [
                t for t in self.sectors[self.sectors == sector].index if t in gp_margin.columns
            ]
            if not tickers:
                continue
            sec_m = gp_margin[tickers].where(mem_f[tickers])
            med = sec_m.median(axis=1)
            sec_rel[tickers] = sec_m.sub(med, axis=0).fillna(0.0).values
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        composite = _cs_zscore(sec_rel).fillna(0.0) + _cs_zscore(momentum).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class OperatingLeverageImprovement(CrossSectionalStrategy):
    """H430: Long top-quintile by GP/NI YoY improvement (operating leverage building).

    Rising GP/NI = gross profit growing faster than net income = building scalable
    operating leverage. Proxy for Kogan et al. (2017) firms investing in growth.
    Only defined where NI > 0. Sector-neutral top 20%, quarterly rebalance.
    """

    name = "operating_leverage_improvement"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.net_income = net_income
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
        ni = _align_fund(self.net_income, prices_panel)
        ni_safe = ni.where(ni > 0)
        op_lev = (gp / ni_safe).clip(0.0, 20.0)
        op_lev_change = op_lev - op_lev.shift(self.lookback)
        daily_w = _sector_neutral_weights(
            op_lev_change, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ===========================================================================
# SET 6 — H431–H435: Credit Risk & Distance-to-Default
# ===========================================================================


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


# ===========================================================================
# SET 6 — H436–H440: ETF Crowding & Passive Ownership Proxies
# ===========================================================================


class SectorEWDivergence(CrossSectionalStrategy):
    """H437: Long sector-relative 5d reversal when fundamentals are positive.

    Stocks that underperformed their sector equal-weight return over 5 days
    (ETF-flow-driven dislocation) and have positive NI growth should mean-revert.
    Israeli et al. (2017): ETF flows create non-fundamental dislocations.
    Quality filter: NI/TA YoY growth > 0. Sector-neutral top 20%, weekly rebalance.
    """

    name = "sector_ew_divergence"

    def __init__(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        reversal_window: int = 5,
        ni_lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ) -> None:
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.reversal_window = reversal_window
        self.ni_lookback = ni_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        ni_growing = (ni_ta > ni_ta.shift(self.ni_lookback)).fillna(False)
        ret_5d = prices_panel.pct_change(self.reversal_window)
        # Sector-relative 5d return: stock return minus sector equal-weight
        mem_f = membership.reindex_like(ret_5d).fillna(False)
        sector_rel = pd.DataFrame(0.0, index=ret_5d.index, columns=ret_5d.columns)
        for sector in self.sectors.dropna().unique():
            tickers = [t for t in self.sectors[self.sectors == sector].index if t in ret_5d.columns]
            if not tickers:
                continue
            sec_r = ret_5d[tickers].where(mem_f[tickers])
            sec_mean = sec_r.mean(axis=1)
            sector_rel[tickers] = sec_r.sub(sec_mean, axis=0).fillna(0.0).values
        # Reverse: long most negative sector-relative return (biggest laggards)
        score = (-sector_rel).where(ni_growing, other=np.nan)
        daily_w = _sector_neutral_weights(
            score,
            membership & ni_growing,
            self.sectors,
            self.within_sector_quantile,
            ascending=False,
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MarketCapConcentrationAvoid(CrossSectionalStrategy):
    """H436: GP/TA quality in mid-cap S&P 500 names (exclude mega-cap passive-crowded).

    Market cap = prices * shares. Within each sector, exclude top cap_exclude_pct
    by market cap (most passive-owned) and apply GP/TA quality to remainder.
    Ben-David et al. (2018): high ETF ownership -> higher vol, less idiosyncratic signal.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "market_cap_concentration_avoid"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        cap_exclude_pct: float = 0.10,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.cap_exclude_pct = cap_exclude_pct
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
        # Exclude top cap_exclude_pct within each sector (mega-caps)
        mega_cap = pd.DataFrame(False, index=prices_panel.index, columns=prices_panel.columns)
        for sector in self.sectors.dropna().unique():
            tickers = [
                t for t in self.sectors[self.sectors == sector].index if t in prices_panel.columns
            ]
            if not tickers:
                continue
            sec_cap = mktcap[tickers].where(mem_f[tickers])
            n_mem = mem_f[tickers].sum(axis=1)
            thresh = (n_mem * (1.0 - self.cap_exclude_pct)).clip(lower=1)
            rank = sec_cap.rank(axis=1, ascending=True, na_option="keep")
            mega_cap[tickers] = rank.gt(thresh, axis=0)
        mid_cap_mem = membership & ~mega_cap
        daily_w = _sector_neutral_weights(
            gp_ta, mid_cap_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class InverseIndexWeightMomentum(CrossSectionalStrategy):
    """H438: 6m momentum weighted inversely by market cap rank within sector.

    Smaller S&P 500 names get larger positions (less passive-owned).
    Ben-David et al. (2018): ETF-heavy names have less information advantage.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "inverse_index_weight_momentum"

    def __init__(
        self,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        shares = _align_fund(self.shares_outstanding, prices_panel)
        mktcap = (prices_panel * shares.clip(lower=0.0)).replace(0, np.nan)
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        mem_f = membership.reindex_like(mktcap).fillna(False)
        # Inverse cap rank score within sector
        inv_cap = pd.DataFrame(0.0, index=prices_panel.index, columns=prices_panel.columns)
        for sector in self.sectors.dropna().unique():
            tickers = [
                t for t in self.sectors[self.sectors == sector].index if t in prices_panel.columns
            ]
            if not tickers:
                continue
            sec_cap = mktcap[tickers].where(mem_f[tickers])
            n_mem = mem_f[tickers].sum(axis=1).replace(0, np.nan)
            cap_rank = sec_cap.rank(axis=1, ascending=False, na_option="keep")
            inv_cap[tickers] = (cap_rank / n_mem.values[:, None]).fillna(0.0)
        # Score = momentum * inverse_cap_weight
        score = momentum * inv_cap.where(inv_cap > 0, np.nan)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class SectorCorrelationBreak(CrossSectionalStrategy):
    """H439: GP/TA quality with position sizing reduced when within-sector correlation spikes.

    High rolling within-sector correlation = ETF arbitrage dominating = shrink positions.
    Low correlation = stocks moving independently = full position.
    Israeli et al. (2017): high return synchronicity = ETF crowding.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "sector_correlation_break"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        corr_window: int = 63,
        high_corr_scale: float = 0.50,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.corr_window = corr_window
        self.high_corr_scale = high_corr_scale
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
        daily_rets = prices_panel.pct_change()
        # Mean within-sector pairwise correlation per day
        mem_f = membership.reindex_like(prices_panel).fillna(False)
        sector_corr = pd.Series(0.0, index=prices_panel.index)
        for sector in self.sectors.dropna().unique():
            tickers = [
                t for t in self.sectors[self.sectors == sector].index if t in prices_panel.columns
            ]
            if len(tickers) < 2:
                continue
            sec_ret = daily_rets[tickers].where(mem_f[tickers])
            roll_corr = (
                sec_ret.rolling(self.corr_window, min_periods=20)
                .corr()
                .groupby(level=0)
                .mean()
                .mean(axis=1)
            )
            sector_corr = sector_corr.add(
                roll_corr.reindex(prices_panel.index).fillna(0.0), fill_value=0.0
            )
        # Normalize by n_sectors
        n_sectors = len(self.sectors.dropna().unique())
        mean_corr = (sector_corr / max(n_sectors, 1)).clip(0.0, 1.0)
        # Scale factor: reduce to high_corr_scale when corr > 75th pctile
        corr_thresh = mean_corr.rolling(252, min_periods=63).quantile(0.75)
        high_corr_regime = mean_corr.gt(corr_thresh.fillna(1.0))
        scale = pd.Series(1.0, index=prices_panel.index).where(
            ~high_corr_regime, self.high_corr_scale
        )
        scaled = base_w.mul(scale, axis=0)
        row_sum = scaled.sum(axis=1).replace(0, np.nan)
        rescaled = scaled.div(row_sum, axis=0).fillna(0.0)
        return _periodic_rebalance(rescaled, self.rebalance_period)


class LowIndexBetaQuality(CrossSectionalStrategy):
    """H440: GP/TA quality prioritized for low-beta (less passive-crowded) names.

    Score = GP/TA * (1 / beta) where beta = rolling 252d market beta.
    Low-beta S&P 500 names = less ETF-dominated = stronger fundamentals-to-price link.
    Ben-David et al. (2018): ETF-heavy names have higher index beta.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "low_index_beta_quality"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        spy_rets: pd.Series,
        sectors: pd.Series,
        beta_window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
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
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        daily_rets = prices_panel.pct_change()
        spy = self.spy_rets.reindex(prices_panel.index).fillna(0.0)
        min_p = self.beta_window // 2
        cov = daily_rets.rolling(self.beta_window, min_periods=min_p).cov(spy)
        var = spy.rolling(self.beta_window, min_periods=min_p).var()
        beta = (cov / var.replace(0, np.nan)).clip(0.01, 5.0).shift(1)
        # Low beta = high inv_beta weight bonus
        inv_beta = (1.0 / beta).clip(0.0, 10.0)
        score = gp_ta * inv_beta
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ===========================================================================
# SET 6 — H441–H445: Institutional Horizon Proxies
# ===========================================================================


class ShareholderAlignedPort(CrossSectionalStrategy):
    """H441: Low dilution + positive OCF + 6m momentum = dedicated-holder proxy.

    Shares YoY growth <= 0 (no dilution) AND OCF/TA > 0 AND 6m price > 0.
    All three conditions. Rank qualifiers by GP/TA quality.
    Yan & Zhang (2009): dedicated long-horizon institutions prefer these firms.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "shareholder_aligned_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.shares_outstanding = shares_outstanding
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.lookback = lookback
        self.mom_lookback = mom_lookback
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
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        no_dilution = (shares <= shares.shift(self.lookback)).fillna(False)
        ocf_pos = (ocf > 0).fillna(False)
        mom_pos = (prices_panel > prices_panel.shift(self.mom_lookback)).fillna(False)
        aligned_mem = membership & no_dilution & ocf_pos & mom_pos
        daily_w = _sector_neutral_weights(
            gp_ta, aligned_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class TransientOwnershipAvoid(CrossSectionalStrategy):
    """H442: Avoid high revenue volatility + share issuance (transient ownership signal).

    Filter: rolling std(rev growth) high AND shares YoY increasing -> avoid.
    Bushee (1998): transient institutions pressure short-term EPS -> revenue instability.
    Long survivors ranked by GP/TA quality. Sector-neutral top 20%, quarterly.
    """

    name = "transient_ownership_avoid"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        revenue: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        vol_window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.revenue = revenue
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.lookback = lookback
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
        rev = _align_fund(self.revenue, prices_panel)
        shares = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        rev_safe = rev.where(rev.shift(self.lookback) > 0)
        rev_growth = (
            (rev - rev.shift(self.lookback)) / rev.shift(self.lookback).abs().clip(lower=1.0)
        ).clip(-1.0, 5.0)
        rev_vol = rev_growth.rolling(self.vol_window, min_periods=self.vol_window // 4).std()
        # High revenue volatility: above cross-sectional median
        mem_f = membership.reindex_like(rev_vol).fillna(False)
        vol_median = rev_vol.where(mem_f).median(axis=1)
        high_rev_vol = rev_vol.gt(vol_median, axis=0).fillna(False)
        # Shares increasing (dilution)
        share_diluting = (shares > shares.shift(self.lookback)).fillna(False)
        transient_signal = high_rev_vol & share_diluting
        safe_mem = membership & ~transient_signal
        daily_w = _sector_neutral_weights(
            gp_ta, safe_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class DedicatedOwnerProxy(CrossSectionalStrategy):
    """H443: Consistent buyback + positive NI trend + GP/TA quality = dedicated-owner proxy.

    z(-shares_growth) + z(NI/TA growth) + z(GP/TA level). Three-factor composite.
    Yan & Zhang (2009): dedicated institutions accumulate in these firms.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "dedicated_owner_proxy"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
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
        shares = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        ni_growth = ni_ta - ni_ta.shift(self.lookback)
        shares_lag = shares.shift(self.lookback)
        shares_growth = ((shares - shares_lag) / shares_lag.abs().clip(lower=1.0)).clip(-1.0, 5.0)
        composite = (
            _cs_zscore(-shares_growth).fillna(0.0)
            + _cs_zscore(ni_growth).fillna(0.0)
            + _cs_zscore(gp_ta).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class SmartAccumulationProxy(CrossSectionalStrategy):
    """H444: Price > 200d MA + NI/TA improving + GP/TA improving = smart accumulation.

    All three conditions required. Rank qualifiers by composite improvement score.
    Yan & Zhang (2009): short-term institutional accumulation predicts earnings surprises.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "smart_accumulation_proxy"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        ma_window: int = 200,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.sectors = sectors
        self.ma_window = ma_window
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
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        price_ma = prices_panel.rolling(self.ma_window, min_periods=self.ma_window // 2).mean()
        price_above_ma = (prices_panel > price_ma).fillna(False)
        ni_improving = (ni_ta > ni_ta.shift(self.lookback)).fillna(False)
        gp_ta_improving = (gp_ta > gp_ta.shift(self.lookback)).fillna(False)
        accumulation_mem = membership & price_above_ma & ni_improving & gp_ta_improving
        score = _cs_zscore(ni_ta - ni_ta.shift(self.lookback)).fillna(0.0) + _cs_zscore(
            gp_ta - gp_ta.shift(self.lookback)
        ).fillna(0.0)
        daily_w = _sector_neutral_weights(
            score, accumulation_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class OwnerMixQualityPort(CrossSectionalStrategy):
    """H445: Low dilution + OCF reliability + book equity growth = owner mix quality.

    z(-shares_growth) + z(OCF/NI fraction) + z(BE growth). Three-factor composite.
    Bushee & Yan & Zhang: mix of owner horizons reflected in these fundamental signals.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "owner_mix_quality_port"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        reliability_window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.book_equity = book_equity
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.lookback = lookback
        self.reliability_window = reliability_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        be = _align_fund(self.book_equity, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        shares = _align_fund(self.shares_outstanding, prices_panel)
        shares_lag = shares.shift(self.lookback)
        shares_growth = ((shares - shares_lag) / shares_lag.abs().clip(lower=1.0)).clip(-1.0, 5.0)
        ocf_exceeds_ni = (ocf > ni).astype(float)
        ocf_reliability = ocf_exceeds_ni.rolling(
            self.reliability_window, min_periods=self.reliability_window // 4
        ).mean()
        be_lag = be.shift(self.lookback)
        be_growth = ((be - be_lag) / be_lag.abs().clip(lower=1.0)).clip(-1.0, 5.0)
        composite = (
            _cs_zscore(-shares_growth).fillna(0.0)
            + _cs_zscore(ocf_reliability).fillna(0.0)
            + _cs_zscore(be_growth).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ===========================================================================
# SET 6 — H446–H449: Meta-Ensemble Composites
# ===========================================================================


class EarningsQualityComposite(CrossSectionalStrategy):
    """H448: Accruals + earnings momentum + cash persistence composite.

    z(-accruals) + z(NI+OCF growth) + z(OCF dominance fraction).
    Three PEAD-proxy signals in one composite. Pure earnings quality.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "earnings_quality_composite"

    def __init__(
        self,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        persistence_window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.total_assets = total_assets
        self.sectors = sectors
        self.lookback = lookback
        self.persistence_window = persistence_window
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
        avg_ta = ((ta + ta.shift(self.lookback)) / 2).where(ta > 0)
        # Accruals (low = high quality)
        accruals = ((ni - ocf) / avg_ta).clip(-1.0, 1.0)
        # Earnings + OCF momentum
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        ocf_ta = (ocf / ta_safe).clip(-2.0, 5.0)
        earn_mom = _cs_zscore(ni_ta - ni_ta.shift(self.lookback)).fillna(0.0) + _cs_zscore(
            ocf_ta - ocf_ta.shift(self.lookback)
        ).fillna(0.0)
        # OCF dominance fraction
        ocf_dom = (
            (ocf > ni)
            .astype(float)
            .rolling(self.persistence_window, min_periods=self.persistence_window // 4)
            .mean()
        )
        composite = _cs_zscore(-accruals).fillna(0.0) + earn_mom + _cs_zscore(ocf_dom).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CreditQualityMomentumCombo(CrossSectionalStrategy):
    """H447: Credit health + quality + leverage-adjusted momentum composite.

    z(GP/TA) + z(-LTD/TA) + z(OCF/TA) + z(6m mom where low leverage).
    Credit risk + quality + momentum in one composite. Sector-neutral top 20%.
    Vassalou & Xing (2004) + Avramov et al. (2007) combined signal.
    """

    name = "credit_quality_momentum_combo"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.long_term_debt = long_term_debt
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.mom_lookback = mom_lookback
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
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        composite = (
            _cs_zscore(gp_ta).fillna(0.0)
            + _cs_zscore(-ltd_ta).fillna(0.0)
            + _cs_zscore(ocf_ta).fillna(0.0)
            + _cs_zscore(momentum).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
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


# ===========================================================================
# SET 7 — H451–H460: DuPont Level Signals & Untested Capital Return Ratios
# ===========================================================================


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


# ===========================================================================
# SET 7 — H471–H480: Dollar-Neutral L/S Market-Neutral Extensions
# ===========================================================================


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


# ===========================================================================
# SET 7 — H461–H470: Return Distribution Anomalies Beyond MAX Effect
# ===========================================================================


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


class ReturnAutocorrelationPort(CrossSectionalStrategy):
    """H468: Long stocks with highest positive return autocorrelation (trend-continuers).

    First-order serial autocorrelation of daily returns over trailing 63 days.
    High positive AC = intrinsic momentum persistence within the return series.
    All momentum signals use cumulative returns; this uses the autocorrelation structure.
    Sector-neutral top 20%, monthly rebalance. Full 12-fold.
    """

    name = "return_autocorrelation_port"

    def __init__(
        self,
        sectors: pd.Series,
        ac_window: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.ac_window = ac_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        min_p = self.ac_window // 4
        # Lag-1 autocorrelation via rolling corr of r_t with r_{t-1}
        lag1 = daily_rets.shift(1)
        autocorr = daily_rets.rolling(self.ac_window, min_periods=min_p).corr(lag1)
        # ascending=False: highest autocorr (trend-continuers) selected
        daily_w = _sector_neutral_weights(
            autocorr, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MinReturnMomentumCombo(CrossSectionalStrategy):
    """H469: z(MIN1 safety) + z(6m momentum). All-fold template.

    Combines crash-risk avoidance (high MIN1 = safe) with price momentum.
    MIN1 is negative; higher (closer to 0) = safer. Additive z-score composite.
    H255 TailFilteredMomentum used MAX1 as a filter (exclusion); this uses MIN1
    as an independent equal-weight z-score signal. Monthly rebalance. Full 12-fold.
    """

    name = "min_return_momentum_combo"

    def __init__(
        self,
        sectors: pd.Series,
        min_window: int = 21,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.min_window = min_window
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        min1 = daily_rets.rolling(self.min_window, min_periods=self.min_window // 2).min()
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        # z(min1): higher min1 (safer) gets higher z; z(mom): higher mom gets higher z
        composite = _cs_zscore(min1).fillna(0.0) + _cs_zscore(momentum).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class ReturnConsistencyMomentumCombo(CrossSectionalStrategy):
    """H470: z(fraction positive days) + z(6m momentum). All-fold template.

    Additive z-score of return consistency (252d fraction positive) + 6m momentum.
    H155 SmoothMomentum multiplied consistency × momentum (modifier). This treats
    consistency as an independent equal-weight signal. Monthly rebalance. Full 12-fold.
    """

    name = "return_consistency_momentum_combo"

    def __init__(
        self,
        sectors: pd.Series,
        consistency_window: int = 252,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.consistency_window = consistency_window
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        daily_rets = prices_panel.pct_change()
        min_p = self.consistency_window // 4
        pos_frac = (daily_rets > 0).rolling(self.consistency_window, min_periods=min_p).mean()
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        composite = _cs_zscore(pos_frac).fillna(0.0) + _cs_zscore(momentum).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ===========================================================================
# SET 7 — BATCH 5 (H497, H499, H500): VIX/Vol-Regime CS Signal Selectors
# ===========================================================================


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


# ===========================================================================
# SET 7 — BATCH 6 (H501–H510): Execution-Aware Cost Minimization
# ===========================================================================


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


class PatienceFilteredMomentum(CrossSectionalStrategy):
    """H505: 6m momentum with patience entry filter (3+ consecutive positive weekly returns).

    Entry: stock must be in top 20% by 6m momentum AND have 3+ consecutive positive
    weekly (5-day) returns before a position is initiated. Reduces whipsaw by
    requiring confirmation before entry. Exit when momentum rank falls below 30th pct.
    Monthly rebalance.
    """

    name = "patience_filtered_momentum"

    def __init__(
        self,
        sectors: pd.Series,
        mom_lookback: int = 126,
        patience_weeks: int = 3,
        buy_quantile: float = 0.20,
        hold_quantile: float = 0.70,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.mom_lookback = mom_lookback
        self.patience_weeks = patience_weeks
        self.buy_quantile = buy_quantile
        self.hold_quantile = hold_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        # Patience filter: patience_weeks consecutive positive weekly returns
        weekly_ret = prices_panel.pct_change(5)
        patience = weekly_ret > 0
        for lag in range(1, self.patience_weeks):
            patience = patience & (weekly_ret.shift(5 * lag) > 0)

        # Entry: top 20% by momentum AND patience met
        strict_raw = _sector_neutral_weights(
            momentum, membership, self.sectors, self.buy_quantile, ascending=False
        )
        strict_w = strict_raw * patience.reindex_like(strict_raw).fillna(False).astype(float)
        # Re-normalize strict_w: equal-weight among qualifying stocks per rebalance
        # (keep the weight values from strict_raw for qualifying stocks, zero out others)
        extended_w = _sector_neutral_weights(
            momentum, membership, self.sectors, self.hold_quantile, ascending=False
        )
        return _buy_hold_band(strict_w, extended_w, membership, self.rebalance_period)


class MinTurnoverMomentumBand(CrossSectionalStrategy):
    """H506: 6m momentum with asymmetric buy-hold band to minimize turnover.

    Buy only if ranked above 60th percentile (top 40% by momentum).
    Sell only if ranked below 35th percentile (bottom 35%).
    Large hysteresis band (35th to 60th) prevents churning on small rank fluctuations.
    Monthly rebalance. Tests H293's finding that holding momentum longer preserves alpha.
    """

    name = "min_turnover_momentum_band"

    def __init__(
        self,
        sectors: pd.Series,
        mom_lookback: int = 126,
        buy_quantile: float = 0.40,
        hold_quantile: float = 0.65,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.mom_lookback = mom_lookback
        self.buy_quantile = buy_quantile
        self.hold_quantile = hold_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        strict_w = _sector_neutral_weights(
            momentum, membership, self.sectors, self.buy_quantile, ascending=False
        )
        extended_w = _sector_neutral_weights(
            momentum, membership, self.sectors, self.hold_quantile, ascending=False
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


class SemiAnnualMomentumQualityCombo(CrossSectionalStrategy):
    """H508: z(GP/TA) + z(6m momentum), sector-neutral top 20%, semi-annual rebalance (126 days).

    Same composite signal as H231 QualityMomentum (monthly: 0.870) at semi-annual cadence.
    H292 showed quarterly beats monthly for GP/TA alone. H508 tests whether the quality-momentum
    composite (with faster-decaying momentum component) also benefits from lower turnover.
    """

    name = "semi_annual_momentum_quality_combo"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 126,
    ) -> None:
        self.gross_profit = gross_profit
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
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        composite = _cs_zscore(gp_ta).fillna(0.0) + _cs_zscore(momentum).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
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


# ===========================================================================
# SET 7 — BATCH 8 (H521–H530): Novel Fundamental Combinations
# ===========================================================================


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


class MultiPeriodMomentumBlend(CrossSectionalStrategy):
    """H530: Equal-weight of z(3m), z(6m), z(12-1m) momentum. Sector-neutral, monthly.

    Multi-horizon consensus filter: selects stocks with strong momentum across ALL horizons.
    All prior momentum strategies use a single formation period. H530 tests whether requiring
    confirmation at 3m, 6m, AND 12-1m reduces momentum crash risk via diversification.
    Pure price signal -- full 12-fold coverage.
    """

    name = "multi_period_momentum_blend"

    def __init__(
        self,
        sectors: pd.Series,
        mom_3m: int = 63,
        mom_6m: int = 126,
        mom_12m: int = 252,
        skip_1m: int = 21,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.sectors = sectors
        self.mom_3m = mom_3m
        self.mom_6m = mom_6m
        self.mom_12m = mom_12m
        self.skip_1m = skip_1m
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        m3 = prices_panel / prices_panel.shift(self.mom_3m) - 1
        m6 = prices_panel / prices_panel.shift(self.mom_6m) - 1
        m12 = prices_panel.shift(self.skip_1m) / prices_panel.shift(self.mom_12m) - 1
        composite = (
            _cs_zscore(m3).fillna(0.0) + _cs_zscore(m6).fillna(0.0) + _cs_zscore(m12).fillna(0.0)
        ) / 3.0
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ===========================================================================
# SET 7 — BATCH 9 (H531–H540): Sector-Relative Signals & Novel Architecture
# ===========================================================================


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


class VolScaledMomentumQuality(CrossSectionalStrategy):
    """H533: z(GP/TA) + z(6m momentum) with per-stock vol-targeting. Monthly.

    Quality-momentum composite with per-stock volatility scaling:
    each stock's weight × min(1, vol_target/stock_252d_vol), then renormalized.
    Overweights low-vol quality names; underweights high-vol lottery stocks.
    """

    name = "vol_scaled_momentum_quality"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        vol_target: float = 0.15,
        vol_window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.mom_lookback = mom_lookback
        self.vol_target = vol_target
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
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        composite = _cs_zscore(gp_ta).fillna(0.0) + _cs_zscore(momentum).fillna(0.0)
        raw_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        daily_rets = prices_panel.pct_change()
        stock_vol = daily_rets.rolling(
            self.vol_window, min_periods=self.vol_window // 4
        ).std().shift(1) * np.sqrt(252)
        scale = (self.vol_target / stock_vol.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
        scaled_w = raw_w * scale
        row_sum = scaled_w.sum(axis=1).replace(0, np.nan)
        scaled_w = scaled_w.div(row_sum, axis=0).fillna(0.0)
        return _periodic_rebalance(scaled_w, self.rebalance_period)


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


class LowBetaQualityMomentum(CrossSectionalStrategy):
    """H535: z(-252d beta) + z(GP/TA level) + z(6m momentum). 3-factor composite. Monthly.

    Extends H178 (BAB + momentum: 1.125) by adding GP/TA quality as third factor.
    Beta = rolling cov(stock, SPY) / var(SPY). Negative beta in composite = BAB direction.
    All 12 folds active (price-based beta + momentum; fundamentals add depth post-EDGAR).
    """

    name = "low_beta_quality_momentum"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        spy_returns: pd.Series,
        sectors: pd.Series,
        beta_window: int = 252,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.spy_returns = spy_returns
        self.sectors = sectors
        self.beta_window = beta_window
        self.mom_lookback = mom_lookback
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
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        daily_rets = prices_panel.pct_change()
        spy_r = self.spy_returns.reindex(daily_rets.index).fillna(0.0)
        w = self.beta_window
        min_p = w // 4
        spy_var = spy_r.rolling(w, min_periods=min_p).var()
        spy_mean = spy_r.rolling(w, min_periods=min_p).mean()
        stock_mean = daily_rets.rolling(w, min_periods=min_p).mean()
        cross_mean = (daily_rets.multiply(spy_r, axis=0)).rolling(w, min_periods=min_p).mean()
        cov = cross_mean - stock_mean.multiply(spy_mean, axis=0)
        beta = cov.divide(spy_var.replace(0, np.nan), axis=0).clip(-2.0, 4.0).shift(1)
        composite = (
            _cs_zscore(-beta).fillna(0.0)
            + _cs_zscore(gp_ta).fillna(0.0)
            + _cs_zscore(momentum).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class EarningsQualityGatedMomentum(CrossSectionalStrategy):
    """H536: 6m momentum with strict earnings quality gate. Monthly.

    Quality gate: OCF>0 AND NI>0 AND OCF/NI>0.5 (cash-backed earnings).
    Only gate-passing stocks are eligible for momentum selection.
    Tests if quality gating prevents momentum crashes in low-quality names.
    """

    name = "earnings_quality_gated_momentum"

    def __init__(
        self,
        operating_cf: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.operating_cf = operating_cf
        self.net_income = net_income
        self.sectors = sectors
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ocf = _align_fund(self.operating_cf, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        ni_pos = ni > 0
        ni_safe = ni.where(ni_pos)
        ocf_ni_ok = (ocf / ni_safe) > 0.5
        quality_gate = (ocf > 0) & ni_pos & ocf_ni_ok
        filtered_mem = membership & quality_gate.fillna(False)
        daily_w = _sector_neutral_weights(
            momentum, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MomentumWithPortfolioRatchet(CrossSectionalStrategy):
    """H537: H150 (6m momentum, sector-neutral) with H087-style portfolio ratchet. Monthly.

    When CS momentum portfolio drawdown from peak > dd_trigger, scale by (1-dd/dd_limit).
    Resets when new high-water mark is reached. Pure price signal, all 12 folds.
    Tests if ratchet mechanism reduces momentum MaxDD like H087 reduced H021 MaxDD.
    """

    name = "momentum_with_portfolio_ratchet"

    def __init__(
        self,
        sectors: pd.Series,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
        dd_trigger: float = 0.10,
        dd_limit: float = 0.20,
    ) -> None:
        self.sectors = sectors
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period
        self.dd_trigger = dd_trigger
        self.dd_limit = dd_limit

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        base_w = _sector_neutral_weights(
            momentum, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        base_w_rebal = _periodic_rebalance(base_w, self.rebalance_period)
        port_rets = _portfolio_returns(base_w_rebal, prices_panel)
        cum = (1.0 + port_rets.fillna(0.0)).cumprod()
        peak = cum.expanding(min_periods=1).max()
        dd = (cum / peak - 1.0).fillna(0.0)
        dd_shifted = dd.shift(1).fillna(0.0)
        scale_ratchet = (1.0 + dd_shifted / self.dd_limit).clip(lower=0.0, upper=1.0)
        scale = pd.Series(
            np.where(dd_shifted < -self.dd_trigger, scale_ratchet, 1.0), index=dd.index
        )
        return base_w_rebal.multiply(scale, axis=0)


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


class SectorLeaderMomentumFilter(CrossSectionalStrategy):
    """H539: Top 2 sectors by quality-adjusted momentum; within them, top 20% by 6m mom. Monthly.

    Sector selection: mean(6m_return × GP/TA_zscore) per sector → top 2 sectors.
    Stock selection: within top-2 sectors only, standard 6m momentum top 20%.
    Tests quality-adjusted sector rotation vs price-only (H203: 0.915).
    """

    name = "sector_leader_momentum_filter"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        n_top_sectors: int = 2,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.mom_lookback = mom_lookback
        self.n_top_sectors = n_top_sectors
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
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        gp_ta_z = _cs_zscore(gp_ta).fillna(0.0)
        qa_mom = momentum * gp_ta_z
        sector_list = [s for s in self.sectors.dropna().unique()]
        sector_tickers = {
            s: [t for t in self.sectors[self.sectors == s].index if t in prices_panel.columns]
            for s in sector_list
        }
        mem_f = membership.reindex_like(prices_panel).fillna(False)
        sector_qa_data = {}
        for sec, cols in sector_tickers.items():
            if not cols:
                continue
            qa_sec = qa_mom[cols].where(mem_f[cols])
            sector_qa_data[sec] = qa_sec.mean(axis=1)
        if not sector_qa_data:
            return pd.DataFrame(0.0, index=prices_panel.index, columns=prices_panel.columns)
        sector_qa_df = pd.DataFrame(sector_qa_data)
        sector_ranks = sector_qa_df.rank(axis=1, ascending=False, na_option="bottom")
        in_top_n = sector_ranks <= self.n_top_sectors
        stock_in_top = pd.DataFrame(False, index=prices_panel.index, columns=prices_panel.columns)
        for sec, cols in sector_tickers.items():
            if not cols or sec not in in_top_n.columns:
                continue
            sec_flag = in_top_n[sec]
            stock_in_top[cols] = stock_in_top[cols].values | sec_flag.values[:, None]
        filtered_mem = membership & stock_in_top
        daily_w = _sector_neutral_weights(
            momentum, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
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
