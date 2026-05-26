# H252: MaxReturnLotteryAvoid — MAX21 Lottery Premium Avoidance

## Hypothesis

Bali, Cakici, and Whitelaw (2011) show that the maximum single daily return over the past month
(MAX) negatively predicts future returns. Retail investors overpay for lottery-like upside
optionality, depressing subsequent returns for high-MAX stocks.

Signal: rolling max(daily_return) over 21 trading days. Rank ascending (low MAX = best).
Sector-neutral top 20%, monthly rebalance.

Distinction from H197 (LowSkewnessAnomaly, Sharpe 0.879): H197 uses 252-day distributional
skewness. H252 uses a 21-day peak single-day return — a shorter, event-driven signal targeting
specific lottery spikes rather than the full return distribution shape.

Expected regime behavior:
- **2000–2002 tech crash**: Lottery demand peaked pre-crash; strong signal.
- **2008–2009 GFC**: Crisis creates extreme MAX events across all stocks; signal may be noisy.
- **2013–2017 low-vol**: Weak — few extreme single-day events in this period.
- **2020 COVID**: Systematic MAX spike in March; difficult to distinguish lottery from crisis.

Success criteria: OOS Sharpe ≥ 0.65, Max DD < −50%, CI lower bound > 0.
Falsification: Near-zero incremental IC over H197 (skewness) or H251 (idio vol).
