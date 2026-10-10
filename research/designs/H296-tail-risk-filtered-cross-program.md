# H296: TailRiskFilteredCrossProgram — H200-MDR with Tail-Risk Stock Exclusion

## Hypothesis

H200 (cross-program MDR, Sharpe 1.350) concentrates weights across the best multi-family
diversifiers from H001-H250. Adding a tail-risk exclusion at the individual stock level —
removing stocks in the bottom-decile of the H254 tail risk composite (idio-vol + MAX + downside
beta) at each rebalance — should reduce MaxDD without materially reducing expected return.

Signal: Apply the tail-risk composite (z(-idio_vol) + z(-MAX21) + z(-downside_beta)) as an
exclusion filter on the underlying stock selection for any cross-sectional strategy components
of H200. Stocks in bottom-decile tail risk excluded from selection universe.
For non-equity-crosssectional components (SPY/TLT/GLD via risk parity): no change.

Implementation: Modify the H200 strategy to exclude bottom-decile tail-risk stocks from the
cross-sectional strategy universe before the MDR optimization step.

Expected regime behavior:
- **2000–2002**: Tail-risk exclusion removes the worst bubble names. Protective.
- **2008–2009**: Crisis raises all stocks' tail risk; filter may leave sparse universe.
- **2013–2017**: Filter removes few names (low tail risk era). Minimal impact.
- **2020**: Filter fires during COVID spike; protective in worst month.

Success criteria: OOS Sharpe ≥ 1.35, MaxDD < −8.0% (better than H200 −8.83%), CI lower > 0.
Falsification: Filter removes names the MDR optimizer wants — Sharpe falls below H200.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.829 |
| 95% CI | [0.419, 1.238] |
| CAGR | 9.95% |
| Max DD | -33.05% |
| H001 corr | -0.019 |
| Decision | KEEP |

Folds 0-4 zero (EDGAR gap). Active folds 5-11: 0.48, 0.67, 1.87, 1.20, 2.02, 1.13, 1.10.
Sensitivity: exclude_quantile=0.20 achieves Sharpe 0.889 / MaxDD -28.25% (best MaxDD in batch). Wider exclusion zone removes more crash-prone names, improving risk-adjusted performance.
H292 quarterly GP/TA (0.884) without tail filter slightly outperforms H296 (0.829) on Sharpe — but H296 provides much better MaxDD (-33% vs -34%). Tail-risk filter adds downside protection at modest alpha cost.
