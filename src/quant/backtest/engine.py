"""Vectorized backtest engine for daily-bar strategies.

Design choices:

1. Positions generated at time t are applied for the period (t, t+1].
   The strategy sees data through t; returns earned are r_{t+1}.
   This is enforced by SHIFTING the position series.

2. Transaction costs are charged on |Δposition|. Default 0.5 bps commission
   + 1 bp slippage = 1.5 bps per unit of turnover. This is conservative for
   SPY but appropriate for a small operation.

3. Returns are computed from adj_close (split/dividend-adjusted) since
   that's the total-return series.

4. Equity curve starts at 1.0 by convention. Multiply by initial capital
   externally if you want dollar P&L.

This is intentionally simple. It does not model:
- Borrow costs for shorts (irrelevant for SPY long-only initially)
- Margin / funding (irrelevant at 1x leverage)
- Partial fills (irrelevant for SPY size at daily frequency)
- Bid-ask asymmetry (folded into the slippage assumption)

When we deploy live, the same target positions will be sent to Alpaca.
The live execution will produce its own fills which we'll reconcile in
post-trade reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Default cost assumptions. Quoted in fractional terms (1 bp = 1e-4).
DEFAULT_COMMISSION_BPS = 0.5
DEFAULT_SLIPPAGE_BPS = 1.0


@dataclass
class BacktestResult:
    """Container for a backtest run."""

    name: str
    positions: pd.Series  # target positions, indexed by date (NOT shifted)
    applied_positions: pd.Series  # positions actually held during each period (shifted)
    gross_returns: pd.Series  # strategy returns before costs
    costs: pd.Series  # cost as a fractional drag per period
    net_returns: pd.Series  # gross - costs
    equity: pd.Series  # cumulative (1 + net_returns).cumprod()
    bench_returns: pd.Series  # buy-and-hold benchmark
    bench_equity: pd.Series

    def excess_returns(self) -> pd.Series:
        """Strategy minus benchmark, period-by-period."""
        return self.net_returns - self.bench_returns


def run_backtest(
    prices: pd.DataFrame,
    positions: pd.Series,
    name: str = "strategy",
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> BacktestResult:
    """Run a vectorized backtest.

    Parameters
    ----------
    prices : DataFrame with at least an adj_close column, indexed by date.
    positions : Series of target positions in [-1, 1], same index as prices.
                Position at time t is the target set AT THE CLOSE of t,
                applied for the period (t, t+1].
    name : label for the result.
    commission_bps, slippage_bps : cost assumptions, in basis points.

    Returns
    -------
    BacktestResult with positions, returns, costs, equity, and benchmark.
    """
    # Align positions to prices and validate.
    positions = positions.reindex(prices.index).fillna(0.0)
    if positions.abs().max() > 1.0 + 1e-9:
        raise ValueError(
            f"[{name}] position magnitude exceeds 1.0 (max={positions.abs().max():.3f}). "
            "Use leverage explicitly in the result, not in target positions."
        )

    # Adjusted-close returns. First bar return is undefined; treat as 0 so
    # equity starts at 1.0 cleanly (we hold no position on the first bar anyway
    # due to the shift below).
    rets = prices["adj_close"].pct_change().fillna(0.0)

    # Apply positions one bar later: position chosen at close of t affects period t+1.
    applied = positions.shift(1).fillna(0.0)

    # Gross strategy return per period
    gross = applied * rets

    # Turnover: |Δposition|. The trade is executed at the open of t+1 in spirit
    # but for daily bars we charge it at the boundary between t and t+1.
    # The cost applies in the period where the position change took effect.
    turnover = positions.diff().abs().fillna(positions.iloc[0].__abs__())
    # Cost rate: commission + slippage, applied to the FRACTION of capital traded
    cost_rate = (commission_bps + slippage_bps) / 1e4
    # Charge the cost in the period the new position becomes active (t+1).
    costs = (turnover * cost_rate).shift(1).fillna(0.0)

    net = gross - costs
    equity = (1 + net).cumprod()

    bench_pos = pd.Series(1.0, index=prices.index)
    bench_applied = bench_pos.shift(1).fillna(0.0)
    bench_rets = bench_applied * rets
    bench_equity = (1 + bench_rets).cumprod()

    return BacktestResult(
        name=name,
        positions=positions,
        applied_positions=applied,
        gross_returns=gross,
        costs=costs,
        net_returns=net,
        equity=equity,
        bench_returns=bench_rets,
        bench_equity=bench_equity,
    )


def assert_no_lookahead(strategy_fn, prices: pd.DataFrame, n_checks: int = 20, seed: int = 0) -> None:
    """Sanity check: chopping off future data must not change today's position.

    Picks random dates t in the middle of the series, computes positions using
    data up to t and data up to t+future, and asserts they agree at t.

    This catches a large class of lookahead bugs in vectorized code.
    """
    rng = np.random.default_rng(seed)
    n = len(prices)
    if n < 500:
        return  # Series too short to test meaningfully
    candidates = rng.choice(np.arange(300, n - 50), size=n_checks, replace=False)
    for t_idx in candidates:
        truncated = prices.iloc[: t_idx + 1]
        full = prices.iloc[: t_idx + 51]
        p_truncated = strategy_fn(truncated)
        p_full = strategy_fn(full)
        # Compare at the truncation point
        date = prices.index[t_idx]
        v_truncated = p_truncated.loc[date]
        v_full = p_full.loc[date]
        if not np.isclose(v_truncated, v_full, equal_nan=True):
            raise AssertionError(
                f"Lookahead detected: position at {date.date()} differs "
                f"depending on future data. Truncated={v_truncated:.6f}, full={v_full:.6f}"
            )
