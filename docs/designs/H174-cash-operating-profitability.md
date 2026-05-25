# H174: CashOperatingProfitability

## Hypothesis
Long top quintile by cash operating profitability = operating_cf / book_equity (OCF/BE), sector-neutral.
Ball et al. (2016): cash-based operating profitability (without accruals) is more predictive of
future returns than standard earnings-based OP. OCF directly measures cash generation capacity.
Combination of H171's denominator (BE) and H166's spirit (cash vs accruals).

## Expected regime behavior
- High OCF/BE: strong cash generation relative to equity book value
- 2008-2009: cash flow is king in crises; high OCF/BE firms survive credit crunch
- 2013-2017: consistently profitable cash generators should outperform
- General: OCF is harder to manipulate than NI, so signal should be durable

## Success criteria
- OOS Sharpe > 0.65 (expect better than H171 due to reduced accruals noise)
- CI lower bound > 0
- Lower noise in post-2009 folds than H171

## What would falsify this
- OCF/BE similar to GP/BE (H171) — accruals filtering not adding value
- Negative OCF firms distort signal (OCF can be negative unlike GP)
