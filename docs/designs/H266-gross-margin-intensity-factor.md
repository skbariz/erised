# H266: GrossMarginIntensityFactor — Intangible Proxy via Gross Margin Level

## Hypothesis

Chan, Lakonishok, and Sougiannis (2001) show that firms investing heavily in R&D earn
excess returns not captured by traditional value metrics. Since `rd_expense` is not in
VALID_FIELDS, we proxy intangible intensity via gross margin (gross_profit / revenue).

High-gross-margin firms — particularly in tech, healthcare, and services — embed value
from intangible assets (software, IP, brand) in their revenue premium over COGS.
Unlike R&D expense (which is expensed), gross margin reflects the realized output of
intangible investment. Firms that convert intangible capital into pricing power show up
as persistently high-margin.

Signal: gross_profit / revenue (gross margin ratio). Rank descending; long top-quintile
(high gross margin = strong intangible returns). Sector-neutral, monthly rebalance.

Expected regime behavior:
- **2000–2002**: Mixed — software firms had high GM but were overvalued.
- **2008–2009**: Moderate — high-margin firms more resilient to credit contraction.
- **2013–2017**: Strong — high-GM tech/healthcare performed well.
- **2020**: Positive — high-GM firms less exposed to supply chain disruptions.

Success criteria: OOS Sharpe ≥ 0.65, CI lower bound > 0.
Falsification: High correlation with gross profitability (H227 GP/assets > 0.88) — not distinct.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
