# H026: QuarterEndDrift — OOS Walk-Forward Results

## Decision: KILL (but lowest H001 correlation and MaxDD in the program)

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.30 |
| 95% CI | [-0.12, 0.72] |
| CAGR | 1.42% |
| Max Drawdown | -10.13% |
| N Trades | 192 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | **+0.176** |

## Kill reason

OOS Sharpe 0.30 < 0.51 AND 95% CI lower bound -0.12 < 0 — cannot reject the null that this is a zero-return strategy. Standalone Sharpe is insufficient.

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 0.11 |
| 1 | 0.14 |
| 2 | 0.35 |
| 3 | 0.86 |
| 4 | 0.22 |
| 5 | 0.23 |
| 6 | -0.05 |
| 7 | 1.18 |
| 8 | -0.75 |
| 9 | 1.15 |
| 10 | 0.30 |
| 11 | 0.49 |

## Sensitivity (n_days)

| n_days | Sharpe |
|--------|--------|
| 3 | 0.24 |
| 5 (canonical) | 0.30 |
| 10 | 0.12 |

## Notes

- **Lowest H001 correlation in the entire research program: +0.176**
- **Lowest MaxDD in the entire research program: -10.13%** (vs SPY -55.92%)
- Only ~20 trading days active per year → extreme capital efficiency limitation
- Sensitivity is negative: more days (n_days=10) → Sharpe falls to 0.12; effect is
  concentrated in the last 3-5 days before quarter-end
- CI lower bound is negative (-0.12) — the 5,000-bootstrap CI does not rule out
  zero-return; the strategy doesn't pass the statistical bar standalone
- **Ensemble candidate note**: with H001 correlation 0.176 and MaxDD -10.13%, this
  could be a very small-weight overlay in a multi-strategy ensemble even as a "kill"
  — the low correlation is structurally valuable even if standalone Sharpe is weak
