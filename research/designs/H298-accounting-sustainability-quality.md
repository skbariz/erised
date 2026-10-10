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
| OOS Sharpe | 0.856 |
| 95% CI | [0.450, 1.272] |
| CAGR | 10.57% |
| Max DD | -38.49% |
| H001 corr | -0.018 |
| Decision | KEEP |

Folds 0-4 zero (EDGAR gap). Active folds 5-11: 0.91, 0.72, 2.28, 1.20, 1.91, 1.11, 1.02.
Sensitivity: top-decile (0.10 quantile) gives Sharpe 0.924 vs top-quintile 0.856. Concentration wins here.
3-factor composite (0.856) below H265 2-factor filtered quality (0.952) — dilution confirmed again. Adding NOA and accruals to GP/TA produces marginal negative dilution, consistent with program-wide 3-factor boundary finding. H265 (2-factor) is superior.
