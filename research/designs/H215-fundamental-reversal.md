# H215: FundamentalReversal

**Signal:** Monthly price reversal gated by NI>0 AND OCF>0
**Date:** 2026-05-25

## Hypothesis

Buy oversold stocks (bottom quintile by 21d return) that remain fundamentally stable (positive earnings AND positive operating cash flow). Avoids value traps: pure reversal strategies often buy deteriorating businesses; the quality gate screens for temporary price dislocations in solid companies.

## Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.718 |
| 95% CI | [0.318, 1.122] |
| CAGR | 9.44% |
| Max DD | −43.10% |
| H001 corr | −0.015 |

Sensitivity (reversal_window): 5d 0.706, 10d 0.712, 21d 0.718, 42d 0.720. Longer windows very slightly better; 21d canonical is robust.

OCF coverage enabled fold 5 (2008-2010) to activate (+0.459). H001 corr −0.015. MaxDD −43.10% is worse than H211-H212-H214 due to reversal exposure during equity drawdowns (oversold stocks keep falling in 2008). The quality gate provides meaningful filtering but doesn't eliminate reversal crash risk entirely.
