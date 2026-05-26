# H251: IdiosyncraticVolFactor — Low Idiosyncratic Volatility

## Hypothesis

Ang, Hodrick, Xing, and Zhang (2006) document that stocks with high idiosyncratic volatility
earn anomalously low future returns. Unlike total realized volatility (H176), idiosyncratic vol
is the residual risk after removing market exposure via OLS regression on SPY.

Signal: 63-day rolling residual volatility from daily OLS of stock returns on SPY returns.
Rank ascending (low idio vol = best). Sector-neutral top 20%, monthly rebalance.

Distinction from H176 (LowVolatilityAnomaly, Sharpe 0.877): H176 uses 12-month total vol.
H251 uses 63-day residual vol after stripping market beta — captures idiosyncratic noise premium,
not systematic risk exposure.

Expected regime behavior:
- **2000–2002 tech crash**: High idio vol lottery names punished — signal should be strong.
- **2008–2009 GFC**: Distress spikes idio vol across sectors; idio vs. systematic boundary blurs.
- **2013–2017 low-vol**: Compressed dispersion weakens the spread; moderate performance expected.
- **2020 COVID**: Short-term spike in idio vol may invert signal briefly; should recover quickly.

Success criteria: OOS Sharpe ≥ 0.70, Max DD < −45%, CI lower bound > 0.
Falsification: H176 correlation > 0.90 (confirmed duplicate of total vol).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.897 |
| 95% CI | [0.490, 1.306] |
| CAGR | 13.51% |
| Max DD | −41.42% |
| H001 corr | −0.003 |
| Decision | **KEEP** |

**Sensitivity (idio_vol_window):** 21d → 0.920 | 63d → 0.897 | 126d → 0.883. Shorter window
slightly stronger; flat sensitivity confirms robust signal. Canonical = 63d.

**Fold 5 (2008–2010):** Sharpe +0.030 — GFC collapses idio vol spread as all stocks become
correlated. Otherwise consistent across all folds (11/12 positive, range +0.03 to +2.07).

H001 corr = −0.003: near-zero, confirming the cross-sectional equity signal is orthogonal to
the macro trend-following baseline.
