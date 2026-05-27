# H284: LowCoverageMomentumProxy — Momentum × Idio-Vol Trajectory Coverage Proxy

## Hypothesis

Hong, Lim & Stein (2000) show momentum is stronger in firms with low analyst coverage,
where information diffuses slowly. Without analyst count data, we use the idiosyncratic
volatility trajectory as a coverage proxy: a stock whose 63d idio-vol has risen sharply
relative to its 252d idio-vol is "attracting attention" (information environment improving).
Conversely, stable-low-idio-vol stocks are in a quiet information phase — low-coverage proxy.

Signal: Compute idio-vol (residual from 63d OLS on SPY daily return). Compute
idio_vol_ratio = ivol_63d / ivol_252d.where(ivol_252d > 0). Low ratio = stable, low-attention.
Score = z(momentum_12_1) - z(idio_vol_ratio). High momentum + low idio-vol spike = uncrowded.
Long top-quintile, sector-neutral, monthly rebalance.

Distinction from H271 (EarlyStageMomentum): H271 uses total realized vol. H284 uses
idiosyncratic vol (market-risk-stripped), and computes a ratio (spike vs baseline) rather
than level.

Expected regime behavior:
- **2000–2002**: Volatile-idio tech names penalized; steady-momentum names preferred.
- **2008–2009**: Broad idio-vol spike; signal disrupted in crisis.
- **2013–2017**: Low-vol, low-idio regime; less discrimination between names.
- **2020**: COVID spikes idio-vol broadly; sector-neutral helps isolate signal.

Success criteria: OOS Sharpe ≥ 0.85, CI lower > 0, H271 corr < 0.85.
Falsification: H271 corr > 0.85 — idio-vol is not distinct from total vol for this purpose.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
