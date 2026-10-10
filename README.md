# Erised

[![tests](https://github.com/skbariz/erised/actions/workflows/tests.yml/badge.svg)](https://github.com/skbariz/erised/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

**A walk-forward backtesting framework for daily-bar equity and multi-asset strategies, built
to make it hard to fool yourself.**

Erised gives you a small, readable toolkit for testing trading ideas honestly: a vectorized
backtest engine with a built-in one-bar execution lag, rolling walk-forward validation,
bootstrap confidence intervals, an automated lookahead check, and a sealed-holdout data
workflow. It ships with 450+ reference strategy classes (trend, calendar, volatility regime,
cross-asset macro, risk parity, cross-sectional factors, statistical arbitrage) and the full
archive of a 670-hypothesis research program that used them.

> **Not investment advice.** Erised is research software. Backtests are not predictive of
> future results, the bundled strategies are not recommendations, and nothing here has been
> validated for live trading. Use at your own risk; see [LICENSE](LICENSE).

## Features

- **No-lookahead by construction.** Positions decided at the close of day *t* earn day *t+1*
  returns; the engine applies the shift, and `assert_no_lookahead()` catches leaks inside
  vectorized signal code.
- **Walk-forward validation.** Rolling train/test folds (default 5-year train, 2-year test),
  with stitched out-of-sample returns and per-fold diagnostics.
- **Statistical keep/kill rules.** Bootstrap 95% confidence intervals on Sharpe, so a strategy
  must beat noise, not just a benchmark.
- **Realistic costs.** 0.5 bp commission + 1.0 bp slippage per unit of turnover by default,
  configurable per run.
- **Sealed holdout.** Research and holdout slices are written once and hash-verified;
  reading the holdout requires an explicit opt-in flag.
- **Single-asset, multi-asset, and cross-sectional engines** sharing one conventions model.
- **Point-in-time data helpers** for S&P 500 membership and SEC EDGAR fundamentals keyed by
  filing date, not period end.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/skbariz/erised.git
cd erised
uv sync --all-extras
uv run python examples/01_quickstart.py   # runs offline on synthetic data
uv run pytest -q                          # ~1,100 tests, about a minute
```

A minimal backtest:

```python
from erised.data.synthetic import synthetic_bars
from erised.metrics.performance import bootstrap_sharpe_ci
from erised.strategies import MovingAverageCrossover
from erised.validation.walk_forward import make_folds, walk_forward

prices = synthetic_bars()  # or erised.data.loader.load_research("SPY")
folds = make_folds(prices, train_years=5, test_years=2)
wf = walk_forward(prices, lambda train: MovingAverageCrossover(50, 200), folds=folds)

lo, hi = bootstrap_sharpe_ci(wf.combined_returns)
print(wf.combined_report)
print(f"Sharpe 95% CI: [{lo:.2f}, {hi:.2f}]")
```

To work with real market data, see [Getting started](docs/getting-started.md).

## Examples

| Script | Shows |
|---|---|
| [`examples/01_quickstart.py`](examples/01_quickstart.py) | Full-sample vs. walk-forward backtest, per-fold Sharpe, confidence interval |
| [`examples/02_custom_strategy.py`](examples/02_custom_strategy.py) | Writing a `Strategy`, running the lookahead check, applying a keep rule |
| [`examples/03_multi_asset_risk_parity.py`](examples/03_multi_asset_risk_parity.py) | `MultiAssetStrategy` weights, risk parity with and without a volatility target |

## Documentation

- [Getting started](docs/getting-started.md): installation, fetching data, sealing the holdout
- [Methodology](docs/methodology.md): conventions, validation protocol, bias checklist
- [Writing strategies](docs/writing-strategies.md): single-asset, multi-asset, and cross-sectional interfaces
- [Architecture](docs/architecture.md): package layout and API reference
- [Data](docs/data.md): sources, storage layout, configuration, and licensing
- [Research findings](docs/research-findings.md): what the 670-hypothesis program found, with caveats

## Repository layout

```
src/erised/        Library: backtest engines, walk-forward, metrics, data, strategies
examples/          Small runnable examples (offline, synthetic data)
scripts/           Data and operations tools: fetch data, seal holdout, sanity check
tests/             pytest suite
docs/              User documentation
research/          Archive of the H001-H670 research program (runners, results, designs)
data/              Local Parquet store (git-ignored except the holdout hash)
```

## Research archive

`research/` contains every experiment from the program that produced this framework: one
runner per hypothesis, a design doc written before each run, and `summary.json` / `stats.md`
results for all 670 hypotheses, including the ones that failed. Start with
[`research/README.md`](research/README.md) and the master leaderboard in
[`research/results/SUMMARY.md`](research/results/SUMMARY.md).

Headline results that survived out-of-sample checks:

| Result | Sharpe | Max drawdown |
|---|---|---|
| SPY buy-and-hold, same walk-forward windows (1998-2022) | 0.51 | -55.9% |
| Stock/bond risk parity (H021), walk-forward | 1.03 | -18.4% |
| Risk parity with 8% volatility target (H086), walk-forward | 1.11 | -9.9% |
| Four-strategy ensemble (H049) on the sealed 2023-2026 holdout | 1.35 | -9.9% |
| SPY on the same holdout | 1.47 | -18.8% |

Later ensembles in the archive report Sharpe ratios near 2.0, but their weights were fit on
the same out-of-sample return streams they are scored on. Read
[Research findings](docs/research-findings.md) before relying on any number in the archive.

## Contributing

Bug reports, documentation fixes, and new strategies are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and the
[Code of Conduct](CODE_OF_CONDUCT.md). To report a security issue, follow
[SECURITY.md](SECURITY.md) rather than opening a public issue.

## License

[MIT](LICENSE)
