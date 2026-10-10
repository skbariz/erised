"""Shared signal and weighting helpers. Private: not part of the public API."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def _periodic_rebalance(weights: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """Hold weights for `period` trading days; rebalance on days 0, period, 2*period, ..."""
    rebal_mask = pd.Series(False, index=weights.index)
    rebal_mask.iloc[::period] = True
    masked = weights.where(rebal_mask)
    return masked.ffill().fillna(0.0)


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


def _filing_window(fund_panel: pd.DataFrame, lookback_days: int = 5) -> pd.DataFrame:
    """Boolean mask: True in the lookback_days after any filing event.

    Filing event = date when forward-filled fundamental panel changes value.
    Uses past-only window (causal, no lookahead).
    """
    changed = fund_panel.diff().abs().fillna(0.0) > 1e-6
    return changed.rolling(lookback_days, min_periods=1).max().astype(bool)
