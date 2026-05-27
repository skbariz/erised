# H293: CostAwareMomentumLongHold — Bi-Monthly Momentum Rebalance

## Hypothesis

Momentum's biggest net-alpha leak is turnover — monthly rebalancing generates ~180% one-way
annual turnover. Extending the rebalance period from monthly to bi-monthly (every ~42 trading
days) reduces turnover by ~40% with modest gross alpha reduction. The question is whether the
momentum IC decay is fast enough that bi-monthly rebalancing materially hurts OOS returns.

Signal: 12-1 month momentum (same as H157/H271). Sector-neutral top-quintile. Rebalance
every 42 business days (bi-monthly) instead of 21 days (monthly).

Benchmark: H157 vol-scaled momentum Sharpe 1.079 monthly. Target: bi-monthly ≥ 1.00.

Distinction from H090 (monthly risk parity, Sharpe 1.020 vs H021 daily): H090 showed rebalance
frequency doesn't matter for risk parity (low-signal-decay). Momentum has faster IC decay —
this tests whether momentum IC decays within 42 days.

Expected regime behavior:
- **2000–2002**: Momentum crash is fast (daily timescale); bi-monthly may miss the exit.
- **2008–2009**: Momentum crash; bi-monthly rebalance may be slower to rotate.
- **2013–2017**: Steady momentum; bi-monthly sufficient.
- **2020**: Fast reversal in March; bi-monthly may miss COVID rotation.

Success criteria: OOS Sharpe ≥ 1.00, CI lower > 0. Compare with monthly momentum.
Falsification: Sharpe drops > 10% vs monthly momentum — bi-monthly is too slow for momentum.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
