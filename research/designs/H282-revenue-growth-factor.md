# H282: RevenueGrowthFactor — YoY Revenue Dollar Growth

## Hypothesis

Revenue growth is a distinct signal from earnings growth (H180/H281) and earnings quality
(H183). Top-line growth captures market expansion and competitive positioning independent of
cost management. A firm growing revenue faster than peers is gaining market share or expanding
its addressable market — a durable advantage.

Signal: rev_change = (Revenue - Revenue.shift(252)) / abs(Revenue.shift(252)).clip(-5, 5).
Rank descending. Long top-quintile (largest revenue dollar improvement). Sector-neutral, monthly rebalance.

Distinction from existing strategies:
- H180: NI change (bottom-line earnings growth)
- H181: Revenue AND NI double growth (requires both conditions)
- H266: GP/revenue level (gross margin, not revenue growth)
- H267: YoY gross margin change
- H281: GP dollar growth (gross profit, not total revenue)

Pure revenue growth tests top-line momentum independent of whether that growth is profitable.

Expected regime behavior:
- **2000–2002**: Revenue collapse in tech; defensive sectors (staples, healthcare) dominate.
- **2008–2009**: Revenue contraction broadly; sparse selection.
- **2013–2017**: Steady organic growth drives consistent positive selection.
- **2020**: COVID shock; recovery leaders (tech, healthcare) generate strong signal in H2 2020.

Success criteria: OOS Sharpe ≥ 0.70, CI lower > 0, H181 corr < 0.85.
Falsification: Revenue growth alone has no incremental signal after controlling for earnings.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.756 |
| 95% CI | [0.353, 1.166] |
| CAGR | 8.83% |
| Max DD | -36.62% |
| H001 corr | -0.023 |
| Decision | KEEP |

Folds 0-5 zero (EDGAR data gap). Active folds 6-11: 0.44, 2.13, 0.84, 1.59, 1.01, 1.14.
Flat sensitivity (quantile 0.15-0.25: Sharpe 0.756-0.763). Slightly below H281 (GP change: 0.775).
Revenue growth provides an incremental but weaker signal than gross profit improvement.
