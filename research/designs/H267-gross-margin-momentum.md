# H267: GrossMarginMomentum — Improving Gross Margins

## Hypothesis

Improving gross margins signal rising pricing power or falling COGS — both are indicators
of growing intangible competitive advantage. A firm expanding its gross margin is converting
more intangible investment into realized revenue premium.

This is the change signal to H266's level signal. Prior program finding (feedback_level_vs_change_signals)
shows level typically beats change, but gross margin improvement specifically may carry
information about competitive moat expansion not captured by level alone.

Signal: YoY change in gross margin = (gross_profit_t / revenue_t) − (gross_profit_{t-252} / revenue_{t-252}).
Rank descending (improving = best). Sector-neutral top-quintile, monthly rebalance.

Expected regime behavior:
- **2000–2002**: Moderate — margin expansion was unsustainable for many tech firms.
- **2008–2009**: Mixed — revenue collapse means margin change is noisy.
- **2013–2017**: Moderate — operational improvements rewarded.
- **2020**: Mixed — revenue-shock environment makes YoY margin changes unreliable.

Success criteria: OOS Sharpe ≥ 0.68, CI lower bound > 0.
Falsification: Change signal underperforms level signal (H266) — confirms general program finding.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.768 |
| 95% CI | [0.354, 1.182] |
| CAGR | 9.04% |
| Max DD | −34.53% |
| H001 corr | −0.016 |
| Decision | **KEEP** |

**Sensitivity (window):** 126d → 0.760 | 252d → 0.768 | 504d → 0.831. Longer lookback better — moat expansion is a slow-moving process. Longer window candidates for sensitivity canonical.

**Change underperforms level** (H267: 0.768 vs H266: 0.861) — confirms general program finding. The level of gross margin is more predictive than YoY improvement. Passes the keep bar; useful as a composite component (H270).
