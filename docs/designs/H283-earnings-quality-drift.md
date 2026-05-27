# H283: EarningsQualityDrift — NI Improvement + OCF Confirmation

## Hypothesis

PEAD is strongest when earnings improvement is backed by operating cash flow, not accruals.
Firms whose NI improvement is confirmed by an OCF/NI ratio > 1 (cash earnings exceed reported
earnings) have genuine earnings quality. Filtering the NI improvement signal by cash backing
removes accrual-inflated "surprise" names.

Signal: Compute NI improvement = (NI - NI.shift(252)) / abs(NI.shift(252)).clip(-5, 5).
Compute OCF quality = (OCF / NI.where(NI.abs() > 1e-6)).clip(-5, 5).
Long top-quintile NI improvement AND OCF/NI > median (cash-backed improvement only).
Sector-neutral, equal-weight survivors, monthly rebalance.

Distinction from H281/H282: H281/H282 rank NI improvement only. H283 adds the cash-backing
filter, removing firms where earnings improvement is driven by accruals rather than cash.

Expected regime behavior:
- **2000–2002**: Accrual-heavy tech names filtered; quality improvers kept.
- **2008–2009**: OCF disruption in crisis; filter may fire broadly.
- **2013–2017**: Clean earnings environment; both conditions met by best names.
- **2020**: COVID cash flow disruption; OCF filter may remove otherwise good names.

Success criteria: OOS Sharpe ≥ 0.78, CI lower > 0, beats H281 (NI only).
Falsification: OCF filter removes the best NI improvers — cash earnings and accrual earnings
predict equally well in S&P 500 large caps.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
