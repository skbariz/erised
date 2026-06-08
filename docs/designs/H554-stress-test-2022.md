# H554: Stress-Test Evaluation — 2021-2022 Inflation Regime

## Hypothesis

The 12-fold walk-forward covers 1993–2022, but the final fold ends 2022-12-31.
The 2021-2022 inflation/rate-hike regime is the most hostile environment ever seen
for risk-parity strategies: SPY fell ~20%, TLT fell ~31%, and the SPY-TLT
correlation flipped strongly positive (+0.6 vs historical −0.3). This scenario
is NOT tested independently in our OOS protocol.

H554 evaluates every Target-bar survivor (H300, H350, H517, H518, H543–H547, H549,
H550, H551 champion, H552 champion) on the 2021-01-01 → 2022-12-31 isolated window
using the same fold structure but a single 2yr test period with 5yr lookback
(2016–2020 train, 2021–2022 test).

This is a **post-hoc stress probe**, not a new strategy. It gates continued
inclusion in the ensemble pool: if any strategy has Sharpe < 0 or MaxDD < −20%
in this window, it is flagged as inflation-regime-fragile.

## Expected behavior

- **Risk parity core (H300-era)**: SPY+TLT correlation flip is the primary threat.
  Strategies with VIX-regime or yield-curve overlays (H097, H079, H080) should
  partially protect by shifting to TLT at the wrong time but limiting equity exposure.
- **Calendar strategies (H043, H078)**: May-Oct in 100% TLT (H078) is devastated
  in 2022. H043's bond confirm should keep it out of TLT when bonds fall.
- **CS strategies**: High-beta CS strategies (momentum, reversal) will track equity
  drawdowns; low-beta quality may outperform.

## Success criteria (pass/fail for each strategy)

- PASS: Sharpe > 0 AND MaxDD > −15% in the 2021–2022 window
- CONDITIONAL PASS: Sharpe > 0 AND MaxDD > −20% (fragile but usable)
- FAIL: Sharpe < 0 OR MaxDD < −20%

## Implementation

No new strategy classes. Pure evaluation script:
1. Load 2016–2022 data
2. Fit each strategy on 2016–2020 (train), evaluate on 2021–2022 (test)
3. Report Sharpe, Sortino, MaxDD, CAGR for the 2021–2022 window only
4. Flag any strategy that fails the stress criteria

## Results

*To be filled after H551/H552 complete.*
