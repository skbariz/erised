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
