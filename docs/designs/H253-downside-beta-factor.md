# H253: DownsideBetaFactor — Low Crash Sensitivity

## Hypothesis

Harvey and Siddique (2000) show that higher-order moments (skewness, coskewness) matter for
asset pricing. Downside beta — OLS slope computed only on days when SPY returns are negative —
captures crash sensitivity more precisely than unconditional beta.

Signal: rolling 252-day downside beta computed as cov(stock_ret, spy_ret | spy_ret < 0) /
var(spy_ret | spy_ret < 0) via regression-through-origin on masked returns. Rank ascending
(low downside beta = best). Sector-neutral top 20%, monthly rebalance.

Distinction from H175 (BettingAgainstBeta, Sharpe 1.088): H175 uses unconditional beta across
all market days. H253 uses only negative-SPY-return days — captures crash sensitivity specifically.

Expected regime behavior:
- **2008–2009 GFC**: Strong — high downside-beta names severely punished in drawdown.
- **2013–2017 low-vol**: Weak — down-market days rare; noisy signal with few observations.
- **2020 COVID**: Sharp March 2020 spike should benefit; signal active during drawdown.
- **2000–2002**: Moderate — gradual bear market gives signal time to work.

Success criteria: OOS Sharpe ≥ 0.70, Max DD < −45%, CI lower bound > 0.
Falsification: H175 correlation > 0.88 (conditional beta indistinguishable from unconditional).
