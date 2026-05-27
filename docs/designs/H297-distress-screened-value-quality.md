# H297: DistressScreenedValueQuality — GP/TA + Earnings Yield × CHS < Median

## Hypothesis

The value-quality blend (gross profitability + earnings yield) is the most robust fundamental
combination in this program. The distress screen prevents the value leg from becoming a
value-trap graveyard — low-price stocks with deteriorating fundamentals.

Signal: gp_ta = GP/TA (gross profitability).
earnings_yield = NI / (price × shares) (earnings-to-price).
CHS distress score = z(leverage) + z(-ROA) + z(-momentum) + z(vol12m) + z(-cash_ratio) + z(-log_size).
Filter: CHS distress rank < median (keep only bottom-half distress names).
score = z(gp_ta) + z(earnings_yield) among passing stocks.
Long top-quintile within filtered universe. Sector-neutral, monthly rebalance.

Distinction from H260 (LowDistressQualityMomentum): H260 uses GP/TA + momentum + low distress.
H297 uses GP/TA + earnings yield + low distress. Replaces momentum with earnings yield.

Expected regime behavior:
- **2000–2002**: Value trap avoidance critical; distress filter removes zombie survivors.
- **2008–2009**: Distress spikes broadly; filter may be too restrictive.
- **2013–2017**: Clean fundamental environment; distress filter removes few names.
- **2020**: COVID distress spike; sector-neutral critical.

Success criteria: OOS Sharpe ≥ 0.90, CI lower > 0, beats H232 standalone value-quality (0.864).
Falsification: Distress screen near-empty in S&P 500 — 99% of names pass, no added value.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
