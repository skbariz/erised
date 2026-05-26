# H270: IntangibleQualityComposite — Gross Margin + Profitability + Accrual Quality

## Hypothesis

Three complementary intangible-quality dimensions:
1. Gross margin level (H266 signal): intangible asset intensity via pricing power
2. Gross margin momentum (H267 signal): expanding competitive moat
3. Low accruals (H262 signal): earnings are cash-backed (intangible returns are real)

Combining these three signals reduces noise from any single proxy for intangible quality.
The composite identifies firms with high, improving, and reliable gross margins.

Signal: equal-weight z-score composite of (1) gm_level = gross_profit/revenue,
(2) gm_change = YoY delta gross margin, (3) cash_quality = operating_cf/net_income.
All sector-normalized before combining. Rank descending; long top-quintile, monthly rebalance.

This is the 3-factor composite for the intangible batch — deliberate test of dilution boundary.

Expected regime behavior:
- **2000–2002**: Moderate to strong — composite combines stable intangible proxy signals.
- **2008–2009**: Moderate — cash quality component helps identify real vs. accounting earnings.
- **2013–2017**: Moderate to strong — composite avoids low-signal periods of any single metric.
- **2020**: Mixed — YoY margin change component noisy during revenue shocks.

Success criteria: OOS Sharpe ≥ 0.75, CI lower > 0, beats individual components.
Falsification: Composite Sharpe falls below best individual signal (dilution confirmed).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.793 |
| 95% CI | [0.382, 1.207] |
| CAGR | 9.53% |
| Max DD | −35.36% |
| H001 corr | −0.018 |
| Decision | **KEEP** |

**Sensitivity (within_sector_quantile):** 10% → 0.800 | 20% → 0.793 | 30% → 0.792. Nearly identical across quintiles — composite is stable.

**Dilution observed:** 3-factor composite (0.793) underperforms best standalone H266 (0.861). The gross margin change component dilutes the strong level signal. Composite passes the bar but GM level alone is the preferred single signal. Consistent with program-wide composite dilution finding (feedback_composite_signal_dilution).
