# H028: VIXSpikeEntry — OOS Walk-Forward Results

## Decision: KILL

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.36 |
| 95% CI | [−0.07, 0.75] |
| CAGR | 4.34% |
| Max Drawdown | −44.68% |
| N Trades | 50 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | +0.341 |

## Kill reason

OOS Sharpe 0.36 < 0.51 AND 95% CI lower bound −0.07 < 0. The contrarian entry
after VIX spikes captures some bounces but the hold window is too blunt — the
strategy is also long during the continuation of the crash after the initial spike.

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 0.62 |
| 1 | −0.29 |
| 2 | 0.38 |
| 3 | 0.52 |
| 4 | 0.31 |
| 5 | −0.52 |
| 6 | 0.60 |
| 7 | 0.12 |
| 8 | 0.43 |
| 9 | 0.11 |
| 10 | 0.32 |
| 11 | 0.64 |

## Sensitivity (vix_threshold / hold_days)

| vix_threshold | Sharpe |
|---------------|--------|
| 25 | 0.44 |
| 30 (canonical) | 0.36 |
| 40 | 0.16 |

| hold_days | Sharpe |
|-----------|--------|
| 10 | 0.28 |
| 20 (canonical) | 0.36 |
| 30 | 0.39 |

## Notes

- Lower threshold (25) improves Sharpe to 0.44 — still below benchmark; more frequent signals include smaller spikes
- Higher threshold (40) is too rare and loses important GFC/COVID spikes
- MaxDD −44.68% is worse than H001: being long during VIX > 30 events means holding through crisis continuation
- H001 correlation +0.341: meaningfully below 0.50, but Sharpe is too low for standalone or ensemble use
- The strategy is often long during crashes (VIX spikes → hold SPY during the fall)
