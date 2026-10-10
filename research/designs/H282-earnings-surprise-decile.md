# H282: EarningsSurpriseDecile — Top-Decile NI Improvers (Concentrated)

## Hypothesis

Post-earnings announcement drift (PEAD) is strongest for the largest positive surprises.
H281 tests top-quintile NI improvement. H282 concentrates on the top-decile — only the
stocks with the largest year-over-year earnings improvement. Top-decile concentration has
proven superior in this program (H268 beat H227: top-decile GP/TA beats quintile).

Signal: Same annual NI change as H281: (NI - NI.shift(252)) / abs(NI.shift(252)).clip(-5, 5).
Rank descending. Long top-decile (10% concentration) instead of top-quintile (20%).
Sector-neutral, monthly rebalance.

Distinction from H281: H282 tests whether concentration in the largest NI improvers
amplifies the signal. Expected higher Sharpe but higher volatility vs H281's quintile approach.

Expected regime behavior:
- **2000–2002**: Strong top-decile concentration may over-fit to tech names pre-crash.
- **2008–2009**: Fewer qualifying stocks; those that qualify should outperform significantly.
- **2013–2017**: Steady earnings growth; concentrated bet on best improvers.
- **2020**: Strong recovery signal in H2 2020 from COVID-trough improvement.

Success criteria: OOS Sharpe ≥ 0.75, CI lower > 0, beats H281 (top-quintile).
Falsification: Top-decile concentration hurts Sharpe vs H281 — too few stocks, too noisy.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
