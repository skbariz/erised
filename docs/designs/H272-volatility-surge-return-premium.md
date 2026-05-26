# H272: VolatilitySurgeReturnPremium — Attention Spike + Positive Return

## Hypothesis

Gervais, Kaniel, and Mingelgrin (2001) show that abnormal trading volume predicts positive
returns over the next month (visibility theory). Without volume data, we use vol surge as
an attention proxy: a stock with an unusually large recent realized vol spike AND positive
recent return is attracting investor attention.

Signal: vol_ratio = std(21d returns) / std(252d returns). Flag stocks where vol_ratio > 1.5
AND 21-day return > 0. Long flagged stocks (equal-weight), sector-neutral, monthly rebalance.

Note: Earnings-day volume spike exclusion is not feasible (no earnings date data in this
framework). VIX regime filter replaced by SPY realized vol threshold as described in H279.

Distinction from H251 (IdiosyncraticVolFactor): H251 ranks by level of idiosyncratic vol.
H272 uses a spike in total vol relative to baseline, combined with a positive return context.

Expected regime behavior:
- **2000–2002**: Strong if flagged stocks are trending; risky if bubble names flagged.
- **2008–2009**: Weak — crisis vol spikes are cross-sectional; signal noisy.
- **2013–2017**: Limited signal — low baseline vol means few spikes.
- **2020**: COVID spike universal; flag fires on all stocks simultaneously.

Success criteria: OOS Sharpe ≥ 0.75, CI lower > 0.
Falsification: Premium disappears after removing high-distress names (pure distress correlation).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
