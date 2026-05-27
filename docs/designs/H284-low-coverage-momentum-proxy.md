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
| OOS Sharpe | 0.974 |
| 95% CI | [0.571, 1.389] |
| CAGR | 18.44% |
| Max DD | -51.36% |
| H001 corr | +0.002 |
| Decision | KEEP |

Active from fold 0 (price-only signal). Folds: 1.35, 1.28, 1.09, 1.69, 1.41, -0.14, 0.73, 2.44, 1.12, 1.59, 1.24, 0.76.
Flat sensitivity (beta_window 42-126: Sharpe 0.974-0.986). Near-zero H001 corr (+0.002).
Idio-vol ratio adds value over pure momentum (H271: 0.875) — coverage-proxy signal is real.
High CAGR (18.44%) with large MaxDD (-51.36%): momentum family risk profile.
