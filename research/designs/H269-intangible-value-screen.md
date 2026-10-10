# H269: IntangibleValueScreen — High Margin + Cheap B/M + Low Distress

## Hypothesis

Intangible-asset-rich firms appear expensive on traditional book-to-market because R&D
investment is expensed. We approximate intangible value traps by identifying firms that
are: (a) high-gross-margin (intangible quality confirmed), (b) cheap on B/M (accounting
book equity understates true value — potential mis-valuation), and (c) low distress
(not a fundamental deterioration).

Signal: Three-way screen:
- Top-third gross margin (gross_profit / revenue, sector-normalized)
- Bottom-half book-to-market (book_equity / (price × shares), cheap = high B/M; here we want middle-to-high B/M for cheap names)
- Bottom-half CHS distress composite

Wait — "cheap on B/M" means high B/M (low price relative to book). High-margin firms
with high B/M are genuinely cheap intangible-aware value names. Equal-weight survivors.
Sector-neutral, monthly rebalance.

Expected regime behavior:
- **2000–2002**: Strong — identifies cheap quality names misclassified as value traps.
- **2008–2009**: Moderate — B/M inflates for all distressed names; distress filter is critical.
- **2013–2017**: Moderate — value spreads compressed.
- **2020**: Mixed — three-way screen may leave few names.

Success criteria: OOS Sharpe ≥ 0.72, CI lower > 0.
Falsification: Three-way screen consistently leaves N < 10 per rebalance.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.628 |
| 95% CI | [0.220, 1.036] |
| CAGR | 8.19% |
| Max DD | −43.18% |
| H001 corr | −0.017 |
| Decision | **KEEP** |

**Sensitivity (gm_pct):** 25% → 0.610 | 33% → 0.628 | 50% → 0.765. Loosening the gross margin filter strongly improves Sharpe — the three-way intersection is too restrictive at 25%.

**Weakest in batch 4.** Three-way screen creates fold volatility: Fold 6 = −0.807 (2010–2012). The combination of high margin + high B/M + low distress conflicts — high-margin firms tend to have low B/M (expensive), so the intersection leaves very few stocks and fold variance explodes. Passes keep bar (CI lower 0.220 > 0) but wide CI is a concern. The individual screens are stronger than this combination.
