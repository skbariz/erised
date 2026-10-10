# quant

AI-native quantitative research platform. Two-person firm. Currently focused on SPY daily-bar swing strategies (3–20 day holding periods).

## Project goals (Z Fellows 2-month sprint)

- Out-of-sample Sharpe ≥ 1.0 on a sealed multi-year holdout
- Max drawdown below SPY's over the same window
- 20+ strategies tested, 2–5 kept, full research log
- 4+ weeks of live paper-trading evidence before check-in

## Methodological commitments

1. **Sealed holdout.** The last ~3 years of data are written once and never touched until final evaluation. Hash recorded in `data/HOLDOUT_HASH.txt`. Any code that reads the holdout file in non-final-eval contexts is a bug.
2. **Walk-forward only.** No single train/test split. All OOS metrics come from rolling walk-forward.
3. **Point-in-time features.** Every feature must be computable from past data only. Same code runs in backtest and live.
4. **Realistic transaction costs.** 0.5 bps commission + 1 bp slippage minimum. Never optimistic.
5. **Research log discipline.** Every hypothesis logged, including dead ones.

## Quickstart

```bash
uv sync
uv run python scripts/fetch_data.py
uv run python scripts/seal_holdout.py
uv run pytest
```

## Structure

```
src/erised/
  data/         # ingestion, validation, loading
  backtest/     # event-driven engine
  features/     # point-in-time feature library
  strategies/   # strategy implementations
  validation/   # walk-forward framework
  execution/    # Alpaca paper trading wiring
  metrics/      # performance metrics
research_log/   # one markdown per hypothesis
scripts/        # entry points
tests/          # pytest
```

## Status

Phase 1: in progress.
