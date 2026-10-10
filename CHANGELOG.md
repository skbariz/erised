# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First public release.

### Added

- `erised` package: vectorized single-asset, multi-asset, and cross-sectional backtest
  engines; walk-forward validation; bootstrap Sharpe confidence intervals; lookahead checks;
  sealed-holdout data loaders.
- 450+ reference strategy classes.
- `erised.data.synthetic.synthetic_bars` for offline tests and examples.
- `ERISED_DATA_DIR` to place the data store anywhere.
- Runnable examples in `examples/`.
- User documentation in `docs/`.
- VIX is now fetched and sealed with the ETF universe.
- Research archive for hypotheses H001-H670 under `research/`.

### Changed

- The import package is now `erised` (previously `quant`).
- Experiment runners, results, and design docs moved from `scripts/`, `results/`, and
  `docs/designs/` into `research/`.
- `scripts/fetch_fundamentals.py` reads the SEC contact string from `SEC_EDGAR_USER_AGENT`.

### Fixed

- `PaperBroker.get_position` no longer treats every API error as "no position", which could
  have caused a duplicate full-size order after a transient failure.
