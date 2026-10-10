# H278: LeftTailLossFactor — Conditional Value at Risk (CVaR) Signal

## Hypothesis

Agarwal and Naik (2004) show that CVaR (expected shortfall) is a more complete tail risk
measure than VaR. In the cross-section, stocks with large left-tail expected losses during
normal times are overpriced relative to their true risk — investors underweight tail scenarios.
We test whether a pure CVaR-based sort (avoid high CVaR names) produces alpha.

Signal: CVaR_5% = mean of worst 5% of 252d daily returns for each stock. Rank ascending
(least negative CVaR = lowest tail loss = rank 1). Long top-quintile (lowest CVaR names).
Sector-neutral, monthly rebalance.

Note: CVaR here is the empirical expected shortfall (no parametric assumption). The 5% cutoff
means we average the bottom 12-13 daily returns out of 252. Requires 252-day warm-up.

Distinction from H276/H277: H276 is skewness (shape), H277 is coskewness (market copula).
H278 is the magnitude of the left tail — how bad are the worst days on average.

Expected regime behavior:
- **2000–2002**: High-CVaR tech names experience large left-tail events; avoiding helps.
- **2008–2009**: All names have large CVaR; cross-sectional sort still informative within sector.
- **2013–2017**: Low dispersion in CVaR; signal weaker in benign environment.
- **2020**: COVID creates large left-tail events; sector-neutral helps isolate stock-specific CVaR.

Success criteria: OOS Sharpe ≥ 0.70, CI lower > 0.
Falsification: CVaR rank correlates ≥ 0.85 with 21d realized vol rank — CVaR is just vol.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.911 |
| 95% CI | [0.506, 1.325] |
| CAGR | 13.99% |
| Max DD | -39.02% |
| H001 corr | +0.000 |
| Decision | KEEP |
