# H009 — MACD Crossover: Stats

**Status:** killed  **Family:** trend  **Run date:** 2026-05-21

## Canonical parameters: fast=12, slow=26, signal=9

| Metric | H009 | SPY | Delta |
|--------|------|-----|-------|
| Sharpe | 0.37 | 0.51 | -0.14 |
| 95% CI | [-0.04, 0.78] | — | — |
| CAGR | 3.67% | 8.35% | -4.68 pp |
| Max DD | -38.78% | -55.92% | — |
| Trades | 504 | — | — |

## Grid variants

| Config | Sharpe |
|--------|--------|
| **MACD(12,26,9)** | **0.37** |
| MACD(8,21,5) | 0.28 |
| MACD(5,13,3) | 0.00 |

## Per-fold Sharpe
['0.13', '−0.48', '0.54', '0.46', '−0.00', '0.12', '0.51', '1.12', '−0.24', '1.96', '0.69', '1.00']

## Correlation with H001: +0.322

## Kill reason
CI lower bound is −0.04 (just below zero). Higher turnover (504 trades vs H001's 25) than MA crossover for materially lower Sharpe. MACD captures the same trend signal but with more noise. Faster configs collapse to zero Sharpe.
