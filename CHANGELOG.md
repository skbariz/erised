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
- `erised` command-line tool: `data fetch|seal|synthetic|sp500|stocks|fundamentals`,
  `strategies`, and `backtest`.
- User documentation in `docs/`.
- VIX is now fetched and sealed with the ETF universe.
- Research archive for hypotheses H001-H670 under `research/`.

### Changed

- The import package is now `erised` (previously `quant`).
- Strategies are split by family into `erised.strategies.<family>` and
  `erised.strategies.cross_sectional.<family>`; import them from the package.
- `scripts/` is replaced by the CLI; downloaders live in `erised.data.sources`, sealing in
  `erised.data.holdout`.
- Tests are organized into `unit`, `strategies`, `integration`, and `research` suites.
- Experiment runners, results, and design docs moved into `research/`; `data/` is no longer
  tracked.
- The SEC EDGAR contact string comes from `SEC_EDGAR_USER_AGENT`.

### Fixed

- `PaperBroker.get_position` no longer treats every API error as "no position", which could
  have caused a duplicate full-size order after a transient failure.
- `PaperBroker.submit_target_position` rejects non-finite or out-of-range targets and
  non-positive prices before contacting the broker.
- Ticker symbols scraped from Wikipedia are validated before being used as file names, so
  a vandalized entry can't direct downloads outside the data directory.

### Security

- Upgraded transitive dependencies with published advisories (pillow, urllib3, soupsieve,
  msgpack, virtualenv). CI now runs `pip-audit` on the lockfile and pins actions to commit
  SHAs.
