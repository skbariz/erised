# Architecture

## Repository layout

```
erised/
├── src/erised/          the installable package (library + CLI)
├── tests/
│   ├── unit/            engine, data, metrics, config, broker
│   ├── strategies/      per-strategy tests (single_asset/, cross_sectional/)
│   ├── integration/     CLI and examples, end to end
│   └── research/        checks on research-archive helpers
├── examples/            small runnable scripts (offline)
├── docs/                user documentation
├── research/            frozen archive of the H001-H670 program (not part of the package)
└── .github/             CI, issue and pull request templates
```

Market data lives outside version control: `./data` in a source checkout by default, or
wherever `ERISED_DATA_DIR` points.

## Package layout

```
src/erised/
├── __init__.py              __version__
├── cli.py                   `erised` command (also `python -m erised`)
├── config.py                DATA_DIR resolution
├── backtest/
│   ├── engine.py            run_backtest, run_multiasset_backtest, assert_no_lookahead[_multi]
│   └── panel_engine.py      run_panel_backtest for stock-level weight panels
├── validation/
│   └── walk_forward.py      make_folds, walk_forward, walk_forward_multi,
│                            walk_forward_cross_sectional
├── metrics/
│   └── performance.py       sharpe, sortino, cagr, max_drawdown, calmar,
│                            bootstrap_sharpe_ci, report -> PerformanceReport
├── data/
│   ├── schema.py            column schema and validate_bars
│   ├── loader.py            load_full, load_research, load_holdout (guarded)
│   ├── holdout.py           seal_holdout: research/holdout split + hash record
│   ├── universe.py          ETF_UNIVERSE and vendor ticker mapping
│   ├── stocks.py            constituent prices, point-in-time membership, ticker validation
│   ├── fundamentals.py      SEC EDGAR fundamentals as filing-date panels
│   ├── synthetic.py         synthetic_bars and write_synthetic_universe
│   └── sources/             network downloaders
│       ├── prices.py        Yahoo Finance bars with Stooq cross-validation; constituents
│       ├── sp500.py         S&P 500 membership from Wikipedia change history
│       └── edgar.py         SEC EDGAR XBRL company facts
├── strategies/
│   ├── base.py              Strategy, MultiAssetStrategy, BuyAndHold
│   ├── trend.py             moving-average and time-series momentum
│   ├── mean_reversion.py    short-horizon reversion and contrarian entries
│   ├── seasonal.py          calendar effects
│   ├── volatility.py        VIX, variance risk premium, vol targeting, drawdown control
│   ├── allocation.py        risk parity and multi-asset rotation
│   ├── macro.py             credit, yield curve, funding stress, breadth overlays
│   ├── ensemble.py          fixed blends of other strategies
│   └── cross_sectional/
│       ├── base.py          CrossSectionalStrategy
│       ├── momentum.py, reversal.py, stat_arb.py, long_short.py
│       ├── quality.py, value.py, earnings.py, risk.py
│       └── proxies.py, composite.py, ml.py
└── execution/
    └── paper_trader.py      Alpaca paper-trading bridge (optional "live" extra)
```

Modules prefixed with `_` (`strategies/_helpers.py`, `strategies/cross_sectional/_helpers.py`)
are private. Import strategies from the package, not the module:

```python
from erised.strategies import RiskParity, MovingAverageCrossover
from erised.strategies.cross_sectional import BettingAgainstBeta
```

Data flows one way: `data` loads validated bars, a strategy turns them into positions or
weights, an engine turns those into a `BacktestResult`, `validation` repeats that per fold,
and `metrics` summarizes the stitched returns.

## Core types

### `BacktestResult` (`erised.backtest`)

| Field | Description |
|---|---|
| `positions` | Target positions as generated (unshifted) |
| `applied_positions` | Positions actually held each period (shifted one bar) |
| `gross_returns` | Returns before costs |
| `costs` | Cost drag per period |
| `net_returns` | Returns after costs; the series every metric uses |
| `equity` | Cumulative equity curve starting at 1.0 |
| `bench_returns`, `bench_equity` | Buy-and-hold over the same dates |

### `WalkForwardResult` (`erised.validation`)

| Field | Description |
|---|---|
| `folds` | `Fold` objects with train/test start and end timestamps (ends exclusive) |
| `fold_results` | One `BacktestResult` per test window |
| `combined_returns` | Stitched out-of-sample net returns |
| `combined_positions` | Stitched out-of-sample positions |
| `combined_report` | `PerformanceReport` over `combined_returns` |

### `PerformanceReport` (`erised.metrics`)

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

## Command-line interface

| Command | Does |
|---|---|
| `erised data fetch [SYMBOL ...]` | Download ETF and VIX bars (`erised.data.sources.prices.download_universe`) |
| `erised data seal [--force]` | Split research/holdout files and write hashes (`erised.data.holdout.seal_holdout`) |
| `erised data synthetic` | Write synthetic bars for every universe symbol |
| `erised data sp500` | Rebuild point-in-time S&P 500 membership |
| `erised data stocks` | Download constituent prices |
| `erised data fundamentals [--user-agent UA]` | Download SEC EDGAR fundamentals |
| `erised strategies [--cross-sectional]` | List strategies with one-line descriptions |
| `erised backtest NAME [--assets ...] [--param k=v ...]` | Walk-forward a single- or multi-asset strategy |

Every command is a thin wrapper over a library function, so anything the CLI does can be
scripted from Python.

## Tests

About 1,100 tests. CI generates and seals synthetic data before running the suite, so no
network access or real market data is needed. Run a single suite with, for example,
`uv run pytest tests/unit`.
