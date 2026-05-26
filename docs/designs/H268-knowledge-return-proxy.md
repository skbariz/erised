# H268: KnowledgeReturnProxy — Gross Profit per Asset as Intangible ROA

## Hypothesis

Gross profit / total assets measures the return earned on all assets including intangibles.
Unlike book-equity-based ROE (H208) or net income based ROA, GP/assets captures returns
before SG&A and D&A — it reflects the raw economic return on both tangible and intangible
capital.

This is distinct from H227 (gross profitability factor) in framing: H227 uses GP/assets
as a quality signal. H268 uses it explicitly as a knowledge capital ROA proxy —
focusing on sector-relative performance and high-decile concentration to identify firms
earning exceptional returns on their intangible capital base.

Signal: gross_profit / total_assets, but select top-decile (10%) rather than top-quintile (20%)
to concentrate in the highest-conviction intangible-ROA names. Sector-neutral, monthly rebalance.

Expected regime behavior:
- **2000–2002**: Mixed — tech firms had high GP/assets but also high multiples.
- **2008–2009**: Positive — capital-light high-GP firms avoid credit stress.
- **2013–2017**: Strong — tech/healthcare dominate high GP/assets.
- **2020**: Positive — asset-light firms weathered COVID better.

Success criteria: OOS Sharpe ≥ 0.70, CI lower > 0, distinct from H227 (corr < 0.85).
Falsification: H268–H227 correlation > 0.90 (top-decile concentration adds nothing over quintile).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
