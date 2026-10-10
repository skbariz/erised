# H163: ResidualMomentum — CAPM-Adjusted Market-Neutral Momentum

## Signal
For each stock over the formation period (days t-252 to t-21), compute daily CAPM residuals: `ε_t = r_stock,t − β × r_SPY,t`, where β is estimated over the same window. Cumulate these residuals to produce a market-neutral momentum score. Long top quintile. Equal-weight, monthly rebalance.

**Why it should work:** Standard 12-1m momentum (H150) inadvertently loads on high-beta stocks during bull runs and low-beta during bear runs, because high-beta names have the highest returns in up-markets. By removing the market component, residual momentum isolates stock-specific alpha — the component due to firm-level fundamentals, news, or investor attention rather than market exposure. Blitz et al. (2011) showed residual momentum earns higher risk-adjusted returns than total-return momentum in international markets.

## Expected regime behavior
- **Bull markets**: Total momentum overweights high-beta; residual momentum doesn't. Slightly less CAGR, but more consistent.
- **Momentum crashes (2001, 2009)**: Traditional momentum crashes occur partly because high-beta momentum leaders sell off hardest. Removing beta loading should reduce crash severity.
- **Low-vol 2013-2017**: Should perform well — a sustained bull market with few large reversals.

## Success criteria
- OOS Sharpe > 0.51
- 95% CI lower bound > 0
- H001 correlation < H150's 0.71 — beta removal should reduce market-timing character

## Falsification
- If OOS Sharpe < 0.40 with wide CI, the CAPM residual is too noisy at monthly rebalance frequency
- If H001 corr is similar to H150, beta removal had negligible effect in practice
- If fold variance is high (max - min fold Sharpe > 3.0), the beta estimation is unstable

## Implementation notes
- Rolling beta estimated over the same 252-21 day formation window
- `β = cov(r_stock, r_SPY) / var(r_SPY)` computed per-stock per-date
- Cumulative residual: `sum(r_stock,t − β̂ × r_SPY,t)` for t in [t-252, t-21]
- Vectorized: compute daily residuals panel-wide, then rolling sum
