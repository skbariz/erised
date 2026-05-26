# H216 — SectorResidualMomentum

## Hypothesis

Strip both market beta (SPY) and the equal-weight sector-average market-residual, leaving a pure idiosyncratic momentum signal orthogonal to both factors. H163 removes market beta; this strategy additionally removes the sector-average component of that residual, isolating stock-specific momentum that cannot be explained by either the market or the sector trend. The intuition: if a stock's CAPM residual is driven by its whole sector moving together, it is sector momentum, not stock momentum. By removing the sector mean from the residuals we get a cleaner idiosyncratic signal with near-zero exposure to both the equity market and sector rotation.

**Mechanics**: (1) Regress daily returns on SPY to get market-residual. (2) Each day, compute the equal-weight mean of those residuals across all stocks in the same GICS sector. (3) Subtract the sector mean → pure idiosyncratic residual. (4) Sum over 252-21d lookback. (5) Sector-neutral top 20%.

## Expected behavior by regime

- **2000-2002 tech crash**: Sector-average crash is removed, so stocks within tech that held up relative to their sector peers score positively — limits crash exposure.
- **2008-2009 GFC**: Systematic de-risking flattens sector averages; idiosyncratic scores become noisier but the equal-weight equal-weight long-only portfolio should protect via diversification.
- **2020 COVID**: Rapid reversal; stocks that recovered faster within their sector score highly.
- **Low-vol 2013-2017**: Idiosyncratic momentum should be durable in a steady trending regime.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- H001 corr < 0.10 (the point of double-stripping is decorrelation).
- Meaningful improvement in Sharpe over H163 baseline, or near-zero H001 corr as a portfolio diversifier.

## Falsification

If OOS Sharpe < 0.51 or CI lower ≤ 0, the second-level sector demeaning adds noise without signal. If H001 corr remains high, the sector stripping is ineffective.

## Results (2026-05-25)

**Decision: KEEP**

| Metric | Value |
|---|---|
| OOS Sharpe | 0.983 |
| 95% CI | [0.581, 1.395] |
| CAGR | 19.92% |
| Max DD | -51.97% |
| H001 corr | +0.002 |

**Per-fold Sharpes**: 1.622, 1.047, 1.222, 1.607, 1.228, -0.126, 0.736, 2.391, 0.991, 1.613, 1.136, 0.909

Near-zero H001 correlation (+0.002) confirms the double-stripping effectively removes market and sector exposure. Strong and consistent fold performance (11/12 positive). H001 corr rivals H078 — valuable ensemble diversifier.
