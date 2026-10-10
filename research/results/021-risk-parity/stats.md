# H021: RiskParity — OOS Walk-Forward Results

## Decision: KEEP ★ First OOS Sharpe ≥ 1.0 ★

| Metric | Value |
|--------|-------|
| OOS Sharpe | 1.03 |
| 95% CI | [0.62, 1.44] |
| CAGR | 7.69% |
| Max Drawdown | -18.41% |
| N Trades | 4845 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | +0.407 |

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 0.00 (no TLT data) |
| 1 | 0.00 (no TLT data) |
| 2 | 1.55 |
| 3 | 0.98 |
| 4 | 0.93 |
| 5 | 0.32 |
| 6 | 2.12 |
| 7 | 1.15 |
| 8 | 1.12 |
| 9 | 2.24 |
| 10 | 1.44 |
| 11 | 0.80 |

## Sensitivity (vol_lookback)

| vol_lookback | Sharpe |
|--------------|--------|
| 20d | 1.00 |
| 60d (canonical) | 1.03 |
| 120d | 1.00 |

## Notes

- **First strategy to achieve OOS Sharpe ≥ 1.0 in this research program.**
- 95% CI lower bound is 0.62 — clearly positive even at the pessimistic tail
- MaxDD -18.41% vs SPY -55.92%; dramatically better downside protection
- Per-fold: 8 of 10 active folds above Sharpe 0.80; minimum active fold is 0.32 (fold 5)
- High turnover (4845 trades) — daily vol-rebalancing; costs at 1.5 bps/trade manageable
- Correlation with H001 is 0.407 — above the 0.30 ensemble threshold but substantially
  less correlated than H022 (0.912) or H019 (0.788); meaningful partial diversification
- Sensitivity: very stable across vol_lookback (all 1.00-1.03); signal is robust
- Mechanism: TLT's negative equity correlation reduces portfolio vol during equity
  selloffs, allowing larger effective position sizing per unit of risk
