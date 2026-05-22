# H004 — RSI Oversold Mean Reversion: Stats

**Status:** killed  **Family:** mean reversion  **Run date:** 2026-05-21

## Canonical parameters: period=14, entry_rsi=30, exit_rsi=50, hold_days=5

| Metric | H004 | SPY | Delta |
|--------|------|-----|-------|
| Sharpe | 0.06 | 0.51 | -0.45 |
| 95% CI | [-0.36, 0.45] | — | — |
| CAGR | 0.17% | 8.35% | -8.18 pp |
| Max DD | -20.63% | -55.92% | — |
| Trades | 71 | — | — |

## 1-D sensitivity (entry_rsi)

| entry_rsi | Sharpe |
|-----------|--------|
| 25 | 0.23 |
| **30** | **0.06** |
| 35 | 0.37 |

## Per-fold Sharpe
['1.00', '−0.28', '0.21', '0.10', '0.11', '−0.08', '0.00', '1.51', '0.13', '0.58', '0.51', '−0.51']

## Correlation with H001: +0.306

## Kill reason
CI contains zero. Very few trades (71), erratic fold performance. entry_rsi=35 variant shows 0.37 but still CI-unclear.
