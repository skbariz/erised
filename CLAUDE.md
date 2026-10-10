# Erised: guidance for coding agents

Walk-forward backtesting framework for daily-bar SPY, ETF, and S&P 500 stock strategies,
plus the archive of the H001-H670 research program that produced it.

## Layout

- Package: `src/erised/` (import as `erised`; CLI `erised`, defined in `cli.py`)
  - Strategy interfaces: `strategies/base.py`, `strategies/cross_sectional/base.py`
  - Strategy families: `strategies/<family>.py`, `strategies/cross_sectional/<family>.py`
    (see `docs/architecture.md`); import from the package, not the module
  - Engines: `backtest/engine.py`, `backtest/panel_engine.py`
  - Walk-forward: `validation/walk_forward.py`
  - Data: `data/loader.py` (holdout guard), `data/holdout.py`, `data/universe.py`,
    `data/sources/` (downloaders), `config.py` (`ERISED_DATA_DIR`)
- Tests: `tests/unit`, `tests/strategies/{single_asset,cross_sectional}`, `tests/integration`, `tests/research`
- Examples: `examples/`
- Docs: `docs/` (start with `methodology.md` and `writing-strategies.md`)
- Research archive: `research/` (runners, results, design docs; frozen, not linted)

## Commands

- Install: `uv sync --all-extras`
- Offline data for tests: `uv run erised data synthetic && uv run erised data seal`
- Test: `uv run pytest -x -q`
- Lint: `uv run ruff check src tests examples && uv run ruff format src tests examples`
- Type check: `uv run pyright <file>`
- Run a hypothesis: `uv run python research/runners/run_h<NNN>.py` (overwrites its result; restore with git if on synthetic data)

## Reading results

1. `research/results/<NNN>-<name>/summary.json`: read first
2. `research/results/SUMMARY.md`: leaderboard for cross-strategy context
3. `research/results/<NNN>-<name>/stats.md`: full detail only if needed
4. `docs/research-findings.md`: which archived numbers are genuinely out-of-sample

Don't open raw run logs or Parquet files unless asked. Result folders use short names
(`021-risk-parity`); check actual names with `rg --files research/results` before citing.

## Research protocol

- Research period 1993-2022; folds 5-year train / 2-year test, 12 non-overlapping folds.
- Costs 0.5 bp commission + 1.0 bp slippage per unit of turnover.
- SPY benchmark over the same windows: Sharpe 0.51, CAGR 8.35%, max drawdown -55.92%.
- Keep only if the 95% bootstrap CI lower bound is above 0 **and** Sharpe beats 0.51.
- The 2023-2026 holdout was spent in H100. Don't call `load_holdout()` for research.
- Always report: OOS Sharpe, 95% CI, CAGR, max drawdown, H001 correlation, trade count, and
  per-fold Sharpe.
- Sensitivity sweeps (1-D) are required before finalizing a kept strategy, but never re-pick
  canonical parameters from them.
- Ensemble weights fit on stitched OOS returns are in-sample at the portfolio level. Fit on
  each fold's training window if you need an honest ensemble number.

## Bias checklist

- Lookahead: signals use only data available at decision time; no `.shift(-n)`, centered
  windows, or full-sample statistics. Engines already shift positions one bar; don't shift twice.
- Survivorship: use point-in-time S&P 500 membership, never today's constituents.
- Reporting lag: fundamentals keyed by filing date, not period end.
- Data availability: TLT/SHY/IEF/LQD 2002, GLD/VNQ 2004, DBC 2006, HYG 2007 cause zero folds.
- Parameter snooping: canonical parameters fixed before seeing OOS results.

## New hypotheses

1. Write `research/designs/H<NNN>-<name>.md` first: the signal and why it should work,
   expected behavior in 2000-02, 2008-09, 2020, and 2013-17, success criteria, and what
   would falsify it.
2. Implement the class in the matching family module and export it from the package `__init__.py`.
3. Copy `research/runners/run_h021.py` for the runner, including the `write_summary` call.
4. Add a lookahead test and behavioral tests in `tests/strategies/`.
5. Append the result and keep/kill decision to the design doc.

## Coding rules

- Minimum code that tests the idea; match the surrounding style.
- Touch only what the task requires. Mention unrelated bias risks; don't fix them unasked.
- After editing Python: `uv run ruff format <file>` and `uv run ruff check <file>`.
- Never commit data, credentials, `.env`, or machine-specific paths.
