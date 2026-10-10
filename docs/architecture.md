# Architecture

## Package layout

```
src/erised/
├── config.py               DATA_DIR resolution (ERISED_DATA_DIR override)
├── backtest/
│   ├── engine.py           run_backtest, run_multiasset_backtest, assert_no_lookahead[_multi]
│   └── panel_engine.py     run_panel_backtest for stock-level weight panels
├── validation/
│   └── walk_forward.py     make_folds, walk_forward, walk_forward_multi,
│                           walk_forward_cross_sectional
├── metrics/
│   └── performance.py      sharpe, sortino, cagr, max_drawdown, calmar,
│                           bootstrap_sharpe_ci, report -> PerformanceReport
├── data/
│   ├── schema.py           column schema and validate_bars
│   ├── loader.py           load_full, load_research, load_holdout (guarded)
│   ├── ingest.py           yfinance download with Stooq cross-validation
│   ├── stocks.py           S&P 500 constituent prices and point-in-time membership
│   ├── fundamentals.py     SEC EDGAR fundamentals as filing-date panels
│   └── synthetic.py        synthetic_bars for tests and examples
├── strategies/
│   ├── base.py             Strategy, MultiAssetStrategy, and ~150 SPY / ETF strategies
│   └── cross_sectional.py  CrossSectionalStrategy and ~300 stock-selection strategies
└── execution/
    └── paper_trader.py     Alpaca paper-trading bridge (optional "live" extra)
```

Data flows one way: `data` loads validated bars, a strategy turns them into positions or
weights, an engine turns those into a `BacktestResult`, `validation` repeats that per fold,
and `metrics` summarizes the stitched returns.

## Core types

### `BacktestResult` (`erised.backtest.engine`)

| Field | Description |
|---|---|
| `positions` | Target positions as generated (unshifted) |
| `applied_positions` | Positions actually held each period (shifted one bar) |
| `gross_returns` | Returns before costs |
| `costs` | Cost drag per period |
| `net_returns` | Returns after costs; the series every metric uses |
| `equity` | Cumulative equity curve starting at 1.0 |
| `bench_returns`, `bench_equity` | Buy-and-hold over the same dates |

### `WalkForwardResult` (`erised.validation.walk_forward`)

| Field | Description |
|---|---|
| `folds` | `Fold` objects with train/test start and end timestamps (ends exclusive) |
| `fold_results` | One `BacktestResult` per test window |
| `combined_returns` | Stitched out-of-sample net returns |
| `combined_positions` | Stitched out-of-sample positions |
| `combined_report` | `PerformanceReport` over `combined_returns` |

### `PerformanceReport` (`erised.metrics.performance`)

`n_periods, total_return, cagr, annualized_vol, sharpe, sortino, max_drawdown, calmar,
hit_rate, avg_win, avg_loss, turnover, n_trades`. Printing it gives a formatted table;
`as_dict()` gives a plain dict. Sharpe is annualized with a zero risk-free rate.

## Key functions

```python
run_backtest(prices, positions, name="strategy", commission_bps=0.5, slippage_bps=1.0)
run_multiasset_backtest(prices, weights, name="strategy", commission_bps=0.5, slippage_bps=1.0)
run_panel_backtest(prices_panel, weights, name="strategy", commission_bps=0.5,
                   slippage_bps=1.0, spy_returns=None)

make_folds(prices, train_years=5.0, test_years=1.0, step_years=None, min_train_years=5.0)
walk_forward(prices, strategy_factory, folds=None, ...)            # Strategy
walk_forward_multi(prices, strategy_factory, folds=None, ...)      # MultiAssetStrategy
walk_forward_cross_sectional(prices_panel, membership, strategy_factory, folds=None, ...)

bootstrap_sharpe_ci(returns, n_boot=10_000, alpha=0.05, seed=0) -> (low, high)
report(returns, positions=None) -> PerformanceReport
```

`strategy_factory` receives the training slice (and, for cross-sectional, the training
membership) and returns a strategy instance. Parameter-free strategies can ignore it:
`lambda train: RiskParity()`.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/fetch_data.py` | Download ETF and VIX daily bars |
| `scripts/fetch_sp500_members.py` | Build point-in-time S&P 500 membership from Wikipedia change history |
| `scripts/fetch_stocks.py` | Download prices for every historical constituent |
| `scripts/fetch_fundamentals.py` | Download SEC EDGAR XBRL fundamentals (needs `SEC_EDGAR_USER_AGENT`) |
| `scripts/seal_holdout.py` | Split research and holdout files and record hashes |
| `scripts/generate_synthetic_data.py` | Write synthetic files for every symbol (used by CI) |
| `scripts/sanity_check.py` | End-to-end MA(50,200) walk-forward on whatever data is present |

## Tests

`tests/` holds about 1,100 tests: a lookahead and behavioral test for most strategies,
metric correctness checks, data-schema and holdout-guard checks, and smoke tests for the
examples. Some tests import research runners from `research/runners/` to check their
helpers. CI generates synthetic data and seals it before running the suite, so no network
access or real market data is needed.
