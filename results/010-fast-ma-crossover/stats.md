# H010 — Fast MA Crossover (10/50): Stats

**Status:** killed  **Family:** trend  **Run date:** 2026-05-21

## Canonical parameters: fast=10, slow=50

| Metric | H010 | SPY | Delta |
|--------|------|-----|-------|
| Sharpe | 0.45 | 0.51 | -0.06 |
| 95% CI | [0.05, 0.85] | — | — |
| CAGR | 4.71% | 8.35% | -3.64 pp |
| Max DD | -30.13% | -55.92% | -25.79 pp |
| Trades | 152 | — | — |

## Grid variants

| Config | Sharpe |
|--------|--------|
| **MA(10,50)** | **0.45** |
| MA(20,100) | 0.47 |
| MA(5,20) | 0.28 |

## Per-fold Sharpe
['0.84', '−0.38', '0.47', '−0.19', '0.18', '0.24', '0.22', '1.06', '0.17', '1.79', '0.48', '0.78']

## Correlation with H001: +0.577

## Kill reason
CI is all-positive [0.05, 0.85] — the signal is real. But Sharpe (0.45) is materially lower than H001 (0.61) with more turnover (152 vs 25 trades). Faster MAs generate more whipsaw in ranging markets and do not survive the additional transaction costs. Dominated by H001 on both Sharpe and turnover.
