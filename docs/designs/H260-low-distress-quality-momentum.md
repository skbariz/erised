# H260: LowDistressQualityMomentum — Three-Way Quality+Momentum+Distress Screen

## Hypothesis

Novy-Marx (2013) gross profitability and price momentum are empirically orthogonal.
Adding a distress filter prevents late-cycle blowups where momentum and quality signals
both fire but the underlying business is deteriorating.

Signal: Three-way intersection:
- Top-third gross profitability (gross_profit / total_assets, sector-normalized)
- Top-half 12-1 month momentum (prices shifted 21 days)
- Bottom-half CHS distress composite (H257 signal)

Equal-weight stocks passing all three screens. Sector-neutral, monthly rebalance.

Each individual filter is well-tested; the three-way intersection targets a high-quality
subset with confirmed momentum and clean balance sheets.

Expected regime behavior:
- **2000–2002**: Strong — distress guard removes bubble names that pass momentum/quality screens.
- **2008–2009**: Good — removes financials with deteriorating balance sheets.
- **2013–2017**: Potentially strong — clean intersection in low-vol regime.
- **2020**: Mixed — distress spike may temporarily exclude too many names.

Success criteria: OOS Sharpe ≥ 0.90, Max DD < −40%, CI lower > 0.
Falsification: Three-way intersection leaves N < 15 per rebalance on average.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
