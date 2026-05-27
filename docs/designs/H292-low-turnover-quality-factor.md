# H292: LowTurnoverQualityFactor — GP/TA with Quarterly Rebalance

## Hypothesis

Gross profitability (GP/TA) is the most stable cross-sectional signal in this program
(H227: 0.878, H268 top-decile: 0.922). Stable signals don't need monthly rebalancing —
quarterly rebalancing should maintain most of the gross alpha while cutting turnover by ~75%.

Signal: Gross profitability rank from GP/TA. Same sector-neutral top-quintile construction
as H227. Rebalance quarterly instead of monthly (every ~63 trading days via _periodic_rebalance).

Benchmark: H227 monthly Sharpe 0.878. Target: quarterly version ≥ 0.85 net.

Distinction from H227: Identical signal, different rebalance frequency. Tests whether
GP/TA signal IC decays within 1-3 months or remains stable.

Expected regime behavior:
- **2000–2002**: Quality names hold well; quarterly rebalance sufficient.
- **2008–2009**: GP/TA stable in crisis (balance sheet based); quarterly fine.
- **2013–2017**: Stable signal; quarterly equivalent to monthly.
- **2020**: COVID disrupts GP; quarterly rebalance may be slow to exit deteriorating names.

Success criteria: OOS Sharpe ≥ 0.85, CI lower > 0, within 5% of H227 Sharpe.
Falsification: Sharpe drops > 10% vs H227 — signal IC decays faster than quarterly cadence.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.884 |
| 95% CI | [0.472, 1.296] |
| CAGR | 10.99% |
| Max DD | -34.14% |
| H001 corr | -0.017 |
| Decision | KEEP |

Folds 0-4 zero (EDGAR gap). Active folds 5-11: 0.66, 0.75, 2.06, 1.02, 2.04, 1.22, 1.18.
Sensitivity: quarterly (63d) is the sweet spot — monthly 0.852, bi-monthly 0.859, quarterly 0.884, semi-annual 0.864. GP/TA signal IC is stable at 3-month horizon; quarterly reduces turnover ~75% at marginal alpha improvement.
Quarterly beats monthly rebalance: GP/TA is sufficiently persistent that lower-frequency rebalancing actually improves net Sharpe after costs.
