# H008 — Absolute Dual Momentum: Stats

**Status:** kept  **Family:** trend / momentum  **Run date:** 2026-05-21

## Canonical parameters: lookback_months=12

| Metric | H008 | SPY | Delta |
|--------|------|-----|-------|
| Sharpe | 0.62 | 0.51 | +0.11 |
| 95% CI | [0.22, 1.02] | — | — |
| CAGR | 8.02% | 8.35% | -0.33 pp |
| Max DD | -31.23% | -55.92% | -24.69 pp |
| Trades | 77 | — | — |

## 1-D sensitivity (lookback_months)

| lookback_months | Sharpe |
|-----------------|--------|
| 6 | 0.59 |
| 9 | 0.65 |
| **12** | **0.62** |

## Per-fold Sharpe
['0.93', '0.05', '0.91', '0.77', '0.47', '0.32', '0.46', '1.49', '0.09', '2.13', '0.59', '0.41']

## Correlation with H001: +0.856

## Decision: kept — strong standalone signal; note: nearly identical to H007 and H001

All three (H001, H007, H008) produce Sharpe ~0.62 with nearly identical drawdown profiles and returns correlations above 0.85. They are measuring the same phenomenon — SPY trend persistence — via MA crossover, skip-month momentum, and trailing return sign respectively. The 9-month lookback variant (0.65) is marginally better. H008 is the simplest implementation of the signal.
