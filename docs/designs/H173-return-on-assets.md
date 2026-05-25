# H173: ReturnOnAssets

## Hypothesis
Long top quintile by return on assets = net_income / total_assets (NI/TA), sector-neutral.
NI/TA captures total earnings efficiency across all stakeholders (equity + debt).
Distinct from H165 (GP/TA): NI is net of interest, taxes, SGA — capturing total operational efficiency.
Hypothesis: higher NI/TA → better capital allocation → higher expected returns.

## Expected regime behavior
- High NI/TA firms: efficiently convert asset base to earnings
- Tax-heavy periods: NI may differ from GP for reasons unrelated to economic performance
- Financial firms: NI/TA has different interpretation (asset-heavy by nature)

## Success criteria
- OOS Sharpe > 0.51 AND CI lower bound > 0
- Non-trivial correlation with H165 (expected; both are return-on-assets variants)

## What would falsify this
- Near-zero post-2009 Sharpe (NI is noisy vs GP; accounting adjustments destroy signal)
- Worse than H165 (GP/TA) due to noise in NI
