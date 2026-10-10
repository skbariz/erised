"""Panel (cross-sectional) backtest engine.

Handles portfolios of individual stocks where weights is a 2-D DataFrame
(rows=dates, cols=tickers). Reuses BacktestResult from engine.py so all
downstream metrics (PerformanceReport, bootstrap CI) work without changes.

Cost model: 2.5 bps/turn default (higher than ETF engine to reflect
individual stock slippage; override per-strategy as needed).
"""

from __future__ import annotations

import pandas as pd

from .engine import BacktestResult

DEFAULT_COMMISSION_BPS = 0.5
DEFAULT_SLIPPAGE_BPS = 2.0


def run_panel_backtest(
    prices_panel: pd.DataFrame,
    weights: pd.DataFrame,
    name: str = "strategy",
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    spy_returns: pd.Series | None = None,
) -> BacktestResult:
    """Vectorized backtest for a cross-sectional (multi-stock) strategy.

    Parameters
    ----------
    prices_panel : adj_close panel, rows=dates, cols=tickers.
    weights      : target weights, same index/cols as prices_panel.
                   Long-only: each row sums ≤ 1.0.
                   Market-neutral: gross exposure (abs sum) ≤ 2.0.
                   Weight at date t is applied to the return at t+1 (1-bar lag).
    spy_returns  : optional SPY daily return series for benchmark equity curve.
    """
    # Restrict to tickers present in both; align to weights index
    common_tickers = weights.columns.intersection(prices_panel.columns)
    prices = prices_panel.reindex(index=weights.index, columns=common_tickers)
    w = weights.reindex(columns=common_tickers).fillna(0.0)

    # Per-stock daily returns; NaN where price data missing
    stock_rets = prices.pct_change()
    # Where a stock has no price data, treat return as 0 (forced exit at last known price)
    stock_rets = stock_rets.fillna(0.0)

    # 1-bar lag: weight set at close of t earns return at t+1
    applied = w.shift(1).fillna(0.0)

    # Portfolio gross return
    gross = (applied * stock_rets).sum(axis=1)

    # Transaction costs: total absolute weight change per day × cost rate
    first_row_to = w.abs().sum(axis=1)
    delta_to = w.diff().abs().sum(axis=1)
    turnover = delta_to.copy()
    turnover.iloc[0] = first_row_to.iloc[0]
    cost_rate = (commission_bps + slippage_bps) / 1e4
    costs = (turnover * cost_rate).shift(1).fillna(0.0)

    net = gross - costs
    equity = (1 + net).cumprod()

    # Benchmark
    if spy_returns is not None:
        bench_rets = spy_returns.reindex(w.index).fillna(0.0)
    else:
        bench_rets = pd.Series(0.0, index=w.index)
    bench_equity = (1 + bench_rets).cumprod()

    # Proxy 'positions' = total long gross exposure (used for n_trades counting)
    long_exposure = w.clip(lower=0).sum(axis=1)
    applied_long = applied.clip(lower=0).sum(axis=1)

    return BacktestResult(
        name=name,
        positions=long_exposure,
        applied_positions=applied_long,
        gross_returns=gross,
        costs=costs,
        net_returns=net,
        equity=equity,
        bench_returns=bench_rets,
        bench_equity=bench_equity,
    )


def count_stock_trades(weights: pd.DataFrame, threshold: float = 1e-4) -> int:
    """Count total number of non-trivial stock-level position changes."""
    delta = weights.diff().abs()
    return int((delta > threshold).sum().sum())
