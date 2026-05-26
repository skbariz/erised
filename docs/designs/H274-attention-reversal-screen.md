# H274: AttentionReversalScreen — Avoid MAX + Vol Spike + Weak Quality

## Hypothesis

Barber & Odean (2008) show that retail investors are net buyers of attention-grabbing stocks
(extreme price moves, high volume), creating temporary overpricing. Without volume, we identify
"attention froth" via a combination of extreme MAX return AND vol spike AND weak fundamentals.

Signal: Flag as lottery/attention froth if:
- top-decile MAX21 (largest single-day return over 21d) — extreme price spike
- top-decile 21d realized vol — heightened recent volatility
- bottom-half GP/TA (weak gross profitability) — no fundamental support

Long stocks that pass all three screens (NOT flagged as attention froth). Equal-weight, sector-neutral,
monthly rebalance.

Standalone test of the screen before deploying as an overlay on other strategies.

Expected regime behavior:
- **2000–2002**: Strong — bubble stocks flagged and avoided; quality names preferred.
- **2008–2009**: Moderate — crisis vol spikes flag many quality names incorrectly; GP filter helps.
- **2013–2017**: Moderate — few truly flagged names in low-vol regime.
- **2020**: Mixed — vol spike flags broadly; GP filter distinguishes signal from noise.

Success criteria: OOS Sharpe ≥ 0.65, CI lower > 0.
Falsification: MAX21 + vol spike alone drive all the predictive power (GP quality filter adds nothing).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
