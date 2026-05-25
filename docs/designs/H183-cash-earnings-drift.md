# H183: CashEarningsDrift

## Hypothesis

**Signal:** Quality-filtered PEAD. Use the same YoY NI growth (SUE proxy) as H180, but only for stocks where operating cash flow significantly backs the earnings: `cash_ratio = OCF / |NI| > threshold`. Long top-20% NI growers within sector, restricted to high-quality (cash-backed) earners.

**Why it should work:** Sloan (1996) "Do Stock Prices Fully Reflect Information in Accruals?" shows that accrual-heavy earnings are less persistent and predict future earnings disappointments. The cash ratio filter screens out "accrual earnings" — companies that report NI growth via favorable accounting treatments without corresponding cash generation. High cash_ratio > 0.8 means the earnings quality is good: what they report, they actually collected.

**Two-step filter:**
1. Cash quality screen: `cash_ratio = OCF / |NI| > 0.8`
2. Within qualified stocks, rank by YoY NI growth, long top 20% sector-neutral

**Expected regime behavior:**
- **2000-2002 tech crash:** Many tech companies had high accruals during the bubble. Filter should eliminate them, potentially providing early defensive rotation.
- **2008-2009 GFC:** Banks and financials typically show low cash ratios during stress. Filter protects here.
- **2020 COVID:** Airlines/Hotels had negative OCF but still reported positive NI in some cases. Filter eliminates these accurately.
- **Low-vol 2013-2017:** Well-run companies with consistent FCF generation should dominate.

**Data constraint:** Requires both `net_income` AND `operating_cash_flow` from EDGAR. Same pre-2009 coverage gap.

## Success criteria

- OOS Sharpe > H180 Sharpe (quality filter should improve signal)
- 95% CI lower bound > 0
- MaxDD improvement over H180 (quality filter reduces blow-up risk)

## Falsification

If cash-backed earnings drift produces equal or worse Sharpe than H180, the accrual quality filter adds no value within the S&P 500 universe. High-quality S&P 500 membership may already screen out the worst accrual manipulators.

## Implementation

- `CashEarningsDrift` in `src/quant/strategies/cross_sectional.py`
- Signal: `sue` where `cash_ratio = ocf / |ni| > cash_ratio_min`
- Sensitivity sweep: cash_ratio_min ∈ {0.5, 0.8, 1.2}

## OOS Results

*(to be filled after backtest)*
