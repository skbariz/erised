# H030: VIXTrendSignal — OOS Walk-Forward Results

## Decision: KILL

| Metric | Value |
|--------|-------|
| OOS Sharpe | −0.09 |
| 95% CI | [−0.50, 0.31] |
| CAGR | −3.65% |
| Max Drawdown | −83.24% |
| N Trades | 940 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | **−0.284** |

## Kill reason

OOS Sharpe −0.09 < 0 AND MaxDD −83.24% (worse than SPY buy-and-hold −55.92%).
The short-SPY-when-VIX-rising signal is conceptually correct but the VIX MA signal
generates extreme whipsaw in ranging conditions. The 940 trades (3+ per week average)
destroys capital via transaction costs.

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 0.30 |
| 1 | −0.08 |
| 2 | −0.24 |
| 3 | 0.14 |
| 4 | −0.12 |
| 5 | −0.15 |
| 6 | 0.14 |
| 7 | 0.21 |
| 8 | 0.04 |
| 9 | −0.17 |
| 10 | −0.27 |
| 11 | −0.15 |

## Sensitivity (ma_period)

| ma_period | Sharpe |
|-----------|--------|
| 10 | −0.32 |
| 20 (canonical) | −0.09 |
| 60 | −0.06 |

## Notes

- **H001 correlation −0.284**: only NEGATIVE correlation in the entire research program — structurally short equities when H001 is long
- Despite negative H001 correlation, Sharpe is negative — so it cannot contribute to ensemble
- MaxDD −83.24% renders the strategy dangerous: short SPY in a strong bull market compounds rapidly against the position
- 940 trades at 1.5 bps each ≈ 1.4% annual drag: costs eat most of the theoretical edge
- Longer MA (60d) reduces trades and Sharpe improves slightly (−0.06) — still negative
- The conceptual signal (rising VIX → falling prices) is correct but timing precision required to profit from it as a long/short strategy is too fine for daily data
- Flagged for re-evaluation as a **long-only** version (flat instead of short when VIX rising) in ensemble batch
