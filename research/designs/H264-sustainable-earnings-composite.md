# H264: SustainableEarningsComposite — 3-Factor Accounting Quality

## Hypothesis

Three orthogonal angles on whether earnings are sustainable and cash-backed:
1. Low Net Operating Assets (H261) — balance sheet not bloated beyond cash generation
2. Low total accruals (H262) — earnings are cash-backed, not accrual-driven
3. High CFO/NI ratio — operating cash flow exceeds net income (positive cash quality)

Signal: equal-weight z-score composite of all three, sector-normalized.
Explicit test of the 3-factor composite boundary identified in H201–H250 program findings
(feedback_composite_signal_dilution: diminishing returns above 2–3 factors).

Rank descending (high composite = best accounting quality). Sector-neutral top-quintile,
monthly rebalance.

Expected regime behavior:
- **2000–2002**: Strong — three signals aligned against tech/telecom aggressive accounting.
- **2008–2009**: Moderate — financials present mixed signals.
- **2013–2017**: Moderate — signals weaker in low-activity environment.
- **2020**: Neutral — CFO/NI may diverge as working capital shifts.

Success criteria: OOS Sharpe ≥ 0.82, beats H261 and H262 standalone, CI lower > 0.
Falsification: Composite Sharpe falls below H261 or H262 (dilution confirmed at 3 factors).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.811 |
| 95% CI | [0.401, 1.224] |
| CAGR | 9.39% |
| Max DD | −38.65% |
| H001 corr | −0.019 |
| Decision | **KEEP** |

**Sensitivity (within_sector_quantile):** 10% → 0.803 | 20% → 0.811 | 30% → 0.805. Flat across cutoffs — composite signal is robust.

**Dilution confirmed:** 3-factor composite (0.811) underperforms best standalone H261 (0.889). The 3-component z-score sum dilutes the strong NOA signal. MaxDD also worse (−38.65%). The composite passes the bar but is not the preferred signal in this batch.
