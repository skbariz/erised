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
| OOS Sharpe | 0.740 |
| 95% CI | [0.331, 1.158] |
| CAGR | 8.87% |
| Max DD | -36.76% |
| H001 corr | -0.011 |
| Decision | KEEP |

Folds 0-5 zero (EDGAR data gap). Active folds 6-11: 0.32, 1.97, 0.73, 1.88, 1.08, 1.01.
Flat sensitivity (mom_window 63-252: Sharpe 0.730-0.745). Composite dilution confirmed:
H284 standalone (0.974) > H285 composite (0.740). Adding fundamental signals to price momentum hurts.
Three-factor composite exhibits same dilution pattern as H251-H270 batch.
