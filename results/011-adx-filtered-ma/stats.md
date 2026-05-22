# H011 — ADX-Filtered MA Crossover: Stats

**Status:** killed  **Family:** trend  **Run date:** 2026-05-21

## Canonical parameters: MA(50,200), ADX(14) > 25

| Metric | H011 | SPY | Delta |
|--------|------|-----|-------|
| Sharpe | 0.27 | 0.51 | -0.24 |
| 95% CI | [-0.14, 0.66] | — | — |
| CAGR | 2.22% | 8.35% | -6.13 pp |
| Max DD | -30.47% | -55.92% | — |
| Trades | 168 | — | — |

## 1-D sensitivity (adx_threshold)

| adx_threshold | Sharpe |
|---------------|--------|
| 20 | 0.63 |
| **25** | **0.27** |
| 30 | 0.15 |

## Per-fold Sharpe
['−0.06', '0.55', '0.30', '0.90', '0.32', '0.56', '0.28', '0.45', '0.90', '1.89', '−0.11', '−0.16']

## Correlation with H001: +0.716

## Kill reason
CI contains zero. The ADX filter at threshold=25 is too aggressive — it keeps the strategy flat during valid trend regimes. adx_threshold=20 recovers to 0.63, very close to H001, confirming the filter adds complexity without value. The ADX approach imposes a timing lag on top of an already-lagging MA signal.
