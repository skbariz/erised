# H285: EarningsAttentionComposite — NI Change + GP Change + Price Momentum

## Hypothesis

Earnings drift is driven by the combination of fundamental improvement (NI surprise),
profitability quality improvement (GP improvement), and price confirmation (6m momentum).
A composite of these three signals captures the "earnings attention" mechanism even without
explicit announcement dates. The original plan used Friday earnings + volume + ADV attention;
we replace with fundamental-change signals and price momentum.

Signal: score = z(NI_change) + z(GP_change) + z(price_6m)
where:
- NI_change = (NI - NI.shift(252)) / abs(NI.shift(252)).clip(-5, 5) (annual NI improvement)
- GP_change = (GP - GP.shift(252)) / abs(GP.shift(252)).clip(-5, 5) (annual GP improvement)
- price_6m = prices / prices.shift(126) - 1 (6-month price momentum, shorter than standard 12-1)

Long top-quintile by composite score. Sector-neutral, monthly rebalance.

Note: This tests whether combining earnings-related fundamental improvements with medium-term
price momentum captures PEAD-type drift better than any single signal.

Expected regime behavior:
- **2000–2002**: NI/GP declines dominate; composite points defensively short → residual long
  names are genuinely improving.
- **2008–2009**: Fundamental deterioration; price momentum also weak. Few qualifying names.
- **2013–2017**: Steady fundamental improvement + uptrend. Composite strong.
- **2020**: Sharp V-shaped recovery in fundamentals + prices from Q3 2020.

Success criteria: OOS Sharpe ≥ 0.82, CI lower > 0, beats H281 (NI standalone).
Falsification: Composite Sharpe below H281 — dilution from adding GP/price to NI signal.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
