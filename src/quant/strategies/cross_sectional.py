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
