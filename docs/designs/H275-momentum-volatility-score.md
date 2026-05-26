# H275: MomentumVolatilityScore — Momentum Minus Vol and MAX Penalties

## Hypothesis

The momentum lifecycle (Lee & Swaminathan 2000) predicts that momentum with high attention
state reverses faster. We operationalize this with a composite score that rewards momentum
and penalizes two proxies for crowding/late-stage risk: realized volatility and MAX return.

Signal: score = mom_pct − 0.5 × vol_pct − 0.3 × max_pct
where each term is a cross-sectional percentile rank [0, 1]:
- mom_pct: 12-1 month momentum percentile (higher = better momentum)
- vol_pct: 21d realized vol percentile (higher = more crowded/noisy)
- max_pct: MAX21 percentile (higher = more lottery-like)

Long top-quintile by score. Sector-neutral, monthly rebalance.

Adapted from plan's VolumeMomentumLifeCycle (turnover unavailable). Vol and MAX are the
best available price proxies for the overcrowding mechanism.

Expected regime behavior:
- **2000–2002**: Strong — vol + MAX penalties remove late-stage bubble names from momentum.
- **2008–2009**: Mixed — crash reduces all momentum names' scores.
- **2013–2017**: Potentially the best period — uncrowded names compound steadily.
- **2020**: Mixed — penalties fire broadly in March spike.

Success criteria: OOS Sharpe ≥ 1.00, CI lower > 0, beats H157 vol-scaled momentum (1.079).
Falsification: Vol/MAX penalties reduce Sharpe vs. raw momentum — penalties remove the best
momentum names along with the crowded ones.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
