# H032: VIXMeanReversion — OOS Walk-Forward Results

## Decision: KILL

| Metric | Value |
|--------|-------|
| OOS Sharpe | −0.001 |
| 95% CI | [−0.43, 0.40] |
| CAGR | −0.52% |
| Max Drawdown | −35.32% |
| N Trades | 312 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | +0.420 |

## Kill reason

OOS Sharpe ≈ 0 (−0.001) AND 95% CI spans zero symmetrically [−0.43, +0.40].
Indistinguishable from a zero-return strategy. The VIX 2σ spike threshold is
triggered too infrequently to generate meaningful returns, but when triggered,
the strategy is often long during the continuation of the crash that caused the spike.

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 0.27 |
| 1 | −0.51 |
| 2 | 0.24 |
| 3 | 0.13 |
| 4 | 0.14 |
| 5 | −0.22 |
| 6 | 0.08 |
| 7 | −0.07 |
| 8 | 0.13 |
| 9 | −0.33 |
| 10 | 0.31 |
| 11 | 0.20 |

## Sensitivity (n_sigma)

| n_sigma | Sharpe |
|---------|--------|
| 1.5 | 0.12 |
| 2.0 (canonical) | −0.001 |
| 2.5 | 0.03 |

## Notes

- n_sigma=1.5 slightly better (Sharpe 0.12): more frequent signals, but still far below benchmark
- The mean-reversion intuition (VIX high → equities will bounce) fails because: (1) the timing is wrong — VIX spikes precede, not follow, the equity bottom; (2) the bounce may take weeks, during which the strategy may exit
- H001 correlation +0.420: moderate — the strategy is long SPY during VIX extremes which often coincide with different equity phases than H001
- Different from H028 (absolute threshold): H032 normalizes by recent VIX but shows worse OOS performance, suggesting the normalization adds noise
- Overlapping with H028 in concept; both strategies attempt the same contrarian VIX timing and both fail OOS
