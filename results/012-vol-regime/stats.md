# H012 — Realized Volatility Regime: Stats

**Status:** killed  **Family:** volatility  **Run date:** 2026-05-21

## Canonical parameters: vol_lookback=20, regime_lookback=252

| Metric | H012 | SPY | Delta |
|--------|------|-----|-------|
| Sharpe | 0.33 | 0.51 | -0.18 |
| 95% CI | [-0.08, 0.73] | — | — |
| CAGR | 2.83% | 8.35% | -5.52 pp |
| Max DD | -49.77% | -55.92% | -6.15 pp |
| Trades | 256 | — | — |

## Grid (vol_lookback x regime_lookback)

| vol_lookback | regime=126 | regime=252 |
|-------------|------------|------------|
| 10 | 0.32 | 0.33 |
| **20** | 0.18 | **0.33** |
| 30 | 0.33 | 0.27 |

## Per-fold Sharpe
['0.36', '−1.34', '0.46', '−0.04', '0.73', '0.53', '1.40', '0.49', '−0.89', '1.32', '0.73', '0.99']

## Correlation with H001: +0.514

## Kill reason
CI contains zero ([-0.08, 0.73]). Max DD (-49.77%) is nearly as bad as SPY (-55.92%), making this a poor risk/return trade. Fold 1 (-1.34) and Fold 8 (-0.89) show the strategy is long in high-volatility regimes it incorrectly classifies as "low vol" relative to the historical median. The low-volatility anomaly is real but not exploitable with a simple regime filter on SPY alone.
