# Research archive (H001-H670)

This directory is the complete record of the research program that produced Erised. It is
kept for transparency and as a worked example of the methodology; it is **not** part of the
library API and its code is not held to the library's lint or stability standards.

Read [Research findings](../docs/research-findings.md) first. It summarizes the results and
explains which reported numbers are out-of-sample and which are not.

## Contents

| Path | What it holds |
|---|---|
| `results/SUMMARY.md` | Master leaderboard and per-batch synthesis for all 670 hypotheses |
| `results/<NNN>-<name>/summary.json` | Compact machine-readable result per hypothesis |
| `results/<NNN>-<name>/stats.md` | Full metrics, per-fold Sharpe, sensitivity, and decision (where present) |
| `designs/H<NNN>-<name>.md` | Design doc written before each run: hypothesis, expected behavior, success and falsification criteria, and the appended result |
| `batch-plans/` | Planning documents for each batch of hypotheses |
| `runners/run_h<NNN>.py` | The script that produced each result |
| `runners/write_summary.py` | Writes `summary.json`; called at the end of every runner |
| `runners/set8_pool.py`, `set8_lev.py` | Shared strategy pool and leverage helpers for the H551+ ensemble sets |
| `log/` | Narrative research log template and early entries |
| `literature/` | Paper notes that seeded later batches, with source links |
| `run-logs/` | Captured console output from selected runs |
| `holdout-seal.txt` | SHA-256 record of the holdout files sealed for this program |

`summary.json` fields: `oos_sharpe`, `ci_low` / `ci_high` (95% bootstrap), `cagr`,
`max_drawdown`, `n_trades`, `h001_corr` (correlation with the MA(50,200) baseline),
`fold_sharpes`, `decision`, and `canonical_params`.

## Running a hypothesis

Runners expect real data in the data directory (see
[Getting started](../docs/getting-started.md)):

```bash
uv run python research/runners/run_h021.py
```

A runner prints its report and overwrites its `results/` entry. Use `git diff` to compare
your run against the archived numbers, and `git restore research/results` to discard it.
Cross-sectional runners (H150 onward) also need constituent prices, membership, and, for
fundamental strategies, EDGAR data.

## Conventions used throughout

- Research period 1993-2022; folds are 5-year train / 2-year test, 12 non-overlapping test
  windows covering 1998-2022.
- Costs 0.5 bp commission + 1.0 bp slippage per unit of turnover.
- Keep rule: 95% CI lower bound above zero **and** Sharpe above SPY's 0.51.
- The 2023-2026 holdout was evaluated once, in H100, and is spent. Later hypotheses use the
  research period only.
- H101-H149 were intentionally skipped.
