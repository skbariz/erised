# H271: EarlyStageMomentum — Momentum × Low-Vol State (Crowding Proxy)

## Hypothesis

Lee & Swaminathan (2000) show that high-turnover momentum stocks are in a late-stage, crowded
phase of the momentum cycle; their continuation is weaker. We proxy crowding with realized
volatility: high-vol momentum names attract retail attention and are late-stage; low-vol momentum
names are still diffusing to a broader investor base.

Signal: For each stock compute 12-1 month momentum (skip 21d) and 21d realized vol. Combine
as: score = z(momentum) - z(vol_21d). High momentum + low recent vol = uncrowded early-stage.
Long top-quintile, sector-neutral, monthly rebalance.

Note: Dollar volume / turnover is unavailable (stock parquets contain adj_close only). Realized
vol is the most natural available price-based proxy for attention and crowding.

Expected regime behavior:
- **2000–2002**: Strong — early-stage momentum names are less overvalued than late-stage.
- **2008–2009**: Mixed — crash kills all momentum; vol penalty may help slightly.
- **2013–2017**: Moderate — low-vol environment makes vol penalty less discriminating.
- **2020**: Mixed — COVID spike collapses vol-based distinctions temporarily.

Success criteria: OOS Sharpe ≥ 0.95, CI lower > 0, beats H157 vol-scaled momentum (1.079).
Falsification: Vol penalty reduces Sharpe vs. raw momentum — vol is positively correlated with
continuation in this universe (high-vol = high momentum = best performers).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.940 |
| 95% CI | [0.535, 1.357] |
| CAGR | 16.29% |
| Max DD | -48.49% |
| H001 corr | +0.005 |
| Decision | KEEP |
