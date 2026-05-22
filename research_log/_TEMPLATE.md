# Hypothesis NN: <one-line description>

**Status:** [ proposed | testing | kept | killed ]
**Date opened:** YYYY-MM-DD
**Date closed:** YYYY-MM-DD
**Strategy family:** [trend / mean-reversion / vol / calendar / regime / ML / cross-asset]
**Instrument(s):** SPY

## Hypothesis

State the claim as a falsifiable prediction. Example:
"On SPY, days following a >2σ down move show positive average next-day returns
because of mean-reverting short-term flow."

## Prior

Why might this work? What evidence (literature, intuition, prior result) suggests it?

## Test design

- Sample period: e.g. 1993–2022 (research set only)
- Validation: walk-forward, train_years=5, test_years=2
- Costs: 0.5 bps commission, 1.0 bps slippage
- Holding horizon: e.g. 1 day
- Position sizing: e.g. binary, full-notional long

## Results

- OOS Sharpe (combined across folds):
- OOS CAGR vs. SPY benchmark over same periods:
- Max drawdown (strategy vs. SPY):
- Hit rate:
- Turnover:
- Bootstrap 95% CI on Sharpe: [ ___, ___ ]

## Stress tests (if kept)

- Parameter sensitivity:
- Cost sensitivity (2x, 5x costs):
- Sub-period stability (per-fold Sharpes):

## Decision

**Decision:** [kept / killed]

**Reasoning:** <a few sentences. If killed, what was the failure mode? If kept,
what role does it play in the ensemble?>

## Code

- Strategy implementation: `src/quant/strategies/<file>.py`
- Notebook / script: `notebooks/<file>.ipynb` or `scripts/<file>.py`
- Commit hash at conclusion: `___`
