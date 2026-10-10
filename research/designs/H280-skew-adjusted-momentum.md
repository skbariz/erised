# H280: SkewAdjustedMomentum — Momentum Minus Coskewness and CVaR Penalties

## Hypothesis

Momentum (12-1 month) is the strongest price-based signal in the cross-section, but momentum
crashes occur precisely when high-coskewness names are crowded at the top of the momentum
ranking. H280 builds a composite score that rewards momentum while penalizing names with
high crash sensitivity (coskewness) and large left-tail loss (CVaR).

Signal: score = mom_pct − 0.4 × coskewness_pct − 0.3 × cvar_pct
where each term is a cross-sectional percentile rank [0, 1]:
- mom_pct: 12-1 month momentum percentile (higher = better)
- coskewness_pct: coskewness percentile (higher = more crash-sensitive = worse)
- cvar_pct: CVaR_5% percentile (higher = larger left tail = worse)

Long top-quintile by score. Sector-neutral, monthly rebalance.

Distinction from H275 (MomentumVolatilityScore): H275 penalizes vol and MAX (crowding/attention
proxies). H280 penalizes coskewness and CVaR (tail risk proxies). Both adjust momentum but
via different dimensions of risk. Expected H275 corr < 0.75 if these measure distinct risks.

Expected regime behavior:
- **2000–2002**: High-coskewness momentum names (tech) are removed; residual names perform.
- **2008–2009**: All momentum names have high coskewness in crash; signal limited.
- **2013–2017**: Quiet — penalties fire infrequently; momentum dominates.
- **2020**: Penalty fires in Feb-Mar; removes the crash-sensitive momentum names.

Success criteria: OOS Sharpe ≥ 1.00, CI lower > 0, beats H157 vol-scaled momentum (1.079).
Falsification: Penalties reduce Sharpe vs. raw momentum — tail risk penalization removes the
best momentum names (which legitimately carry tail risk as compensation for momentum).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.953 |
| 95% CI | [0.550, 1.365] |
| CAGR | 17.58% |
| Max DD | -51.62% |
| H001 corr | +0.002 |
| Decision | KEEP |
