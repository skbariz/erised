# H276: NegativeSkewnessAvoidance — Avoid Stocks with Negative Return Skew

## Hypothesis

Harvey and Siddique (2000) show that stocks with more negative skewness earn higher expected
returns as compensation — investors dislike left-tail risk. Conversely, stocks with positive
skewness are lottery-like (Conrad, Dittmar, Ghysels 2013) and are overpriced by retail
attention. We test whether avoiding negative-skew names and loading on positive-skew names
produces a durable alpha signal in the cross-section.

Signal: Compute 252d rolling daily return skewness for each stock. Rank ascending (most
positive skew = rank 1). Long top-quintile by skewness (most positive skew). Sector-neutral,
monthly rebalance.

Note: Positive-skew stocks (right-tailed) have modest average returns but rare large gains.
Negative-skew stocks (left-tailed) are priced to compensate for crash risk. The anomaly:
overpriced lottery/positive-skew names underperform as attention fades.

Expected regime behavior:
- **2000–2002**: Positive-skew tech/internet names were the expensive lottery tickets —
  avoiding them may actually help or have mixed effect depending on timing.
- **2008–2009**: Crisis creates broad left skew; fewer positive-skew names to select.
- **2013–2017**: Low-vol environment; skewness signal noisier with small daily moves.
- **2020**: COVID crash creates universal left-tail events; signal disrupted for 1-2 months.

Success criteria: OOS Sharpe ≥ 0.65, CI lower > 0. Check correlation with H197 (DownsideBetaAvoidance).
Falsification: H197 correlation > 0.85 — skewness and downside beta measure the same tail risk.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
