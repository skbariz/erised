# H190: SectorPairsTrading

## Hypothesis

**Signal:** Gatev, Goetzmann, Rouwenhorst (2006) minimum-distance pairs. During the training period, normalize each stock's price to start at 1.0, then find the top-5 pairs per GICS sector by minimum sum-of-squared-distances. During test: monitor the log-spread z-score (vs 126-day rolling history); enter when |z| > 2.0, long the underperformer, short the overperformer.

**Why it should work:** The Gatev et al. paper is one of the most replicated in empirical finance. Stocks that historically moved together (similar sector, similar business) tend to mean-revert when they diverge. The excess spread is explained by transient liquidity shocks or news overreaction to one leg. Within-sector pairing ensures pairs are economically related (same customers, suppliers, macro drivers).

**Long-short structure:** True self-financing pairs trade — long underperformer, short overperformer. Dollar-neutral by construction. Market beta ≈ 0. H001 correlation should be near zero (the strategy is indifferent to market direction).

**Expected regime behavior:**
- **2000-2002 tech crash:** Sector pairs within tech may have very wide spreads as companies diverged in their crash timing. Pairs that diverged without reverting in the training window would be problematic.
- **2008-2009 GFC:** Broad simultaneous crash means spreads may widen and stay wide — the liquidity mechanism fails when systemic risk dominates. Expect the worst fold performance here.
- **2020 COVID:** Sector-specific divergences (airlines diverged within sector due to varying balance sheets). Should have mean-reverted.
- **Low-vol 2013-2017:** Most favorable: tight range-bound markets produce genuine mean-reverting pairs.

**Cost consideration:** L/S with weekly-ish turnover. Both legs charged 2.5 bps. Gross exposure = 1.0, so cost = turnover × 2.5 bps.

## Success criteria

- OOS Sharpe > 0.51
- 95% CI lower bound > 0
- H001 corr < 0.30 (pairs is market-neutral by design)

## Falsification

If Sharpe ≤ 0.51 or CI clips zero: within-sector pairs in S&P 500 don't reliably mean-revert. The most likely reason: S&P 500 large-caps are too efficiently priced — the spread correction happens within 1-2 days (inside our monitoring frequency), or the spread widens persistently in trending regimes.

## Implementation

- `SectorPairsTrading` in `src/quant/strategies/cross_sectional.py`
- Training: `_form_min_distance_pairs()` — vectorized SSD over normalized prices
- Sensitivity sweep: entry_z ∈ {1.5, 2.0, 2.5}

## OOS Results

**Decision: kill**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.206 |
| 95% CI | [-0.183, 0.608] |
| CAGR | 1.27% |
| Max DD | -38.35% |
| H001 corr | +0.042 |

CI spans zero. Fold Sharpes erratic: 7 of 12 negative including fold 7 at -1.191. H001 corr +0.042 confirms true market-neutrality — but alpha is absent. Classic Gatev pairs is too efficiently arbitraged in S&P 500 large-caps. Sensitivity flat and universally weak. Short-term reversal (H185) is a strictly superior mean-reversion vehicle.
