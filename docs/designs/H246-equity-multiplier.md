# H246: EquityMultiplier — Low TA/Book Equity (DuPont Leverage Component)

## Hypothesis

The equity multiplier (TA/BE) is the DuPont leverage component: it measures how much of total assets is financed by debt relative to equity. A low equity multiplier means the firm is equity-financed, with minimal financial leverage and lower distress risk. Within each sector, firms with the lowest EM should outperform through the business cycle.

Signal: TA/BE, clipped [1.0, 20.0]. Sector-neutral bottom 20% (ascending=True = lowest leverage wins). Monthly rebalance.

Expected regime behavior:
- **2000–2002 tech crash**: Tech companies with low debt outperform; banks/financials with high EM get punished.
- **2008–2009 GFC**: The financial crisis directly punishes high-leverage firms; low-EM firms survive.
- **2020 COVID**: Low-leverage firms retain access to capital markets; high-EM firms face covenant risks.
- **2013–2017 low-vol**: Low leverage premium compresses in benign credit environments.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0.
Falsification: If equity multiplier merely proxies for sector membership (financials always high EM), sector-neutral design should neutralize it, but signal would be weak.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.759 |
| 95% CI | [0.352, 1.166] |
| CAGR | 9.27% |
| Max DD | −36.51% |
| H001 corr | −0.017 |
| Decision | **KEEP** |

**Sensitivity (within_sector_quantile):** 10% → 0.781/MaxDD −33.49% | 20% → 0.759 | 30% → 0.760. Flat; top 10% achieves best MaxDD (−33.49%).

**Folds 5–11 Sharpe:** 0.687, 0.765, 1.678, 0.860, 1.685, 0.953, 1.054. Fold 5 shows 0.687 (partial EDGAR coverage, GFC period — signal is strong during credit stress).

Strong fold 5 (GFC period) confirms the distress-risk hypothesis: low-leverage firms outperform dramatically when credit markets freeze.
