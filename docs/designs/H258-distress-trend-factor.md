# H258: DistressTrendFactor — QoQ Change in Distress Score

## Hypothesis

A rising distress score predicts further deterioration more than the level alone. Balance-sheet
health trends are persistent: firms on a deteriorating trajectory tend to continue declining
before any fundamental reversal.

Signal: QoQ change in the H256 O-score proxy (current quarter minus previous quarter).
Rank ascending (score falling = improving = best). Sector-neutral top-quintile, monthly rebalance.

Explicit test of level-vs-change dynamics within the distress family. Prior program finding
(feedback_level_vs_change_signals) shows level typically beats change. This tests whether
distress is an exception — balance-sheet deterioration may be more trend-persistent than
profitability metrics.

Expected regime behavior:
- **2000–2002**: Moderate — accounting lags equity; change signal may be noisy.
- **2008–2009**: Weak — simultaneous distress across all names compresses spread.
- **2013–2017**: Moderate — few meaningful distress changes; low signal activity.
- **2020**: Weak — sudden shock makes QoQ changes noisy.

Success criteria: OOS Sharpe ≥ 0.60, CI lower bound > 0.
Falsification: Level signal (H256) materially outperforms change signal — confirming prior program finding.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
