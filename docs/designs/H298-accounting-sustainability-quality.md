# H298: AccountingSustainabilityQualityComposite — GP/TA + Low Accruals + Low NOA

## Hypothesis

Accounting sustainability adds the cash-backing verification that gross profitability alone
lacks. This is the "cash-confirmed quality" factor: high GP/TA (productive), low accruals
(cash-backed), low NOA (lean balance sheet). Three related but independent quality dimensions.

Signal: score = z(gp_ta) + z(-accruals) + z(-noa)
where gp_ta = GP/TA, accruals = (NI-OCF)/avg_TA, noa = (TA-CL)/TA.
Long top-quintile by composite. Sector-neutral, monthly rebalance.

This is similar to H264 (SustainableEarningsComposite) but uses GP/TA as the primary factor
instead of NOA/accruals alone. GP/TA is the dominant signal; accruals and NOA are confirmatory.

Note: Per program history, 3-factor composites sit at the dilution boundary. H264 (3-factor
accruals composite) came in at 0.811 vs H265 (2-factor sustainability-filtered GP) at 0.952.
This is a direct test of whether GP-led 3-factor beats the 2-factor filtered approach.

Expected regime behavior:
- **2000–2002**: Triple filter removes accrual-rich tech names. Protective.
- **2008–2009**: Accruals spike in crisis as receivables accumulate.
- **2013–2017**: Clean accounting environment; all three factors positive.
- **2020**: OCF disrupted; accruals component noisy.

Success criteria: OOS Sharpe ≥ 0.88, MaxDD < −40%, CI lower > 0.
Falsification: Sharpe < H265 (0.952) — 3-factor dilution confirmed.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
