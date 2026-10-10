# H229 — OperatingLeverage

## Hypothesis

Operating leverage improvement: gross profit growth minus revenue growth. When GP grows faster than revenue, the company is expanding its gross margin — it is gaining pricing power, reducing COGS, or benefiting from scale. This is distinct from both revenue momentum (H228) and gross margin level (H214 GrossMarginMomentum) — it specifically measures the rate of margin expansion: how much faster is the top line converting to gross profit?

Economically: companies with positive operating leverage are in a virtuous cycle — volume growth + margin expansion simultaneously. The market tends to underreact to this combination because it requires combining two separate signals.

**Mechanics**: gp_growth_yoy - rev_growth_yoy, clipped [-2, 2]. Sector-neutral top 20%. Monthly rebalance.

## Expected behavior by regime

- **2000-2002 tech crash**: Tech had massive COGS (hardware, infrastructure); GP growth often lagged revenue in bubble years. Post-crash survivors had high operating leverage.
- **2008-2009 GFC**: Companies that cut costs faster than revenue declined had positive op leverage post-2009.
- **2020 COVID**: Cloud software and e-commerce scaled aggressively with high operating leverage; strong signal.
- **Low-vol 2013-2017**: Mature compounders with slow-but-steady margin improvement; consistent signal.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- Meaningfully distinct from H228 RevenueGrowth (the signal is about margin expansion, not just growth).
- H001 corr near zero.

## Falsification

If correlation with H228 (revenue growth) is very high, the operating leverage signal is just a noisy version of revenue momentum. If CI lower ≤ 0, GP growth - rev growth does not predict returns.

## Results (2026-05-25)

**Decision: KEEP**

| Metric | Value |
|---|---|
| OOS Sharpe | 0.764 |
| 95% CI | [0.350, 1.178] |
| CAGR | 9.00% |
| Max DD | -34.79% |
| H001 corr | -0.016 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.377, 2.036, 0.990, 1.825, 1.183, 0.999

Folds 0-5 zero (EDGAR gap). Folds 6-11 average 1.235. Max DD -34.79% is the second-best in batch (near H227). The 504d window shows improvement (0.834) — longer lookback better captures structural operating leverage. Fold 10 (1.183) and 11 (0.999) are particularly strong, suggesting the signal remained relevant through the 2017-2022 expansion period.
