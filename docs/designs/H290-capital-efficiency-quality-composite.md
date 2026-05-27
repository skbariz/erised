# H290: CapitalEfficiencyQualityComposite — ROIC + Investment Efficiency + Low NOA

## Hypothesis

Capital discipline has three dimensions: (1) how well management earns on existing assets
(ROIC), (2) whether new investment is productive (investment efficiency), (3) whether the
balance sheet is lean without hidden obligations (low NOA). A composite of these three
dimensions represents the most complete capital quality assessment.

Signal: score = z(ROIC) + z(efficiency_ratio) + z(-NOA)
where ROIC = NI/(TA-CL), efficiency = rev_growth/(capex_intensity+0.01), NOA = (TA-CL)/TA.
Long top-quintile by composite. Sector-neutral, monthly rebalance.

Note: This is a 3-factor composite at the known dilution boundary (program finding: composites
above 2-3 factors begin diluting). All three components must have positive IC in most folds.

Distinction from H288 (ROIC+capex): H290 adds NOA (balance sheet leanness) to the mix.
NOA captures off-balance-sheet obligations that reduce true returns.

Expected regime behavior:
- **2000–2002**: Triple quality screen filters out empire-builders and accrual-heavy names.
- **2008–2009**: Lean balance sheets (low NOA) outperform in credit stress.
- **2013–2017**: All three dimensions positive in economic expansion.
- **2020**: NOA filter protects in COVID stress; ROIC disrupted temporarily.

Success criteria: OOS Sharpe ≥ 0.85, MaxDD < −38%, beats best standalone component, CI lower > 0.
Falsification: Sharpe below best standalone component — dilution at 3-factor boundary.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
