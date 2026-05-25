# H184: EarningsMomentumComposite

## Hypothesis

**Signal:** Blend fundamental momentum (SUE proxy) and price momentum. Composite = `(z_score(SUE) + z_score(12-month price momentum)) / 2`. Long top-20% composite score within sector.

**Why it should work:** Chan, Jegadeesh, and Lakonishok (1996) show that earnings momentum and price momentum are complementary — stocks with high SUE and high price momentum have the most persistent drift. Using both signals reduces the noise inherent in either alone: price momentum can be driven by noise/sentiment, while earnings momentum can lag actual price discovery. Their conjunction is more robust.

**This creates H150 × H180:** H150 (MomentumLongOnly) captures price momentum; H180 captures earnings momentum. H184 combines them. If the combination improves on both, it validates the Chan et al. finding in our dataset.

**Expected regime behavior:**
- **2000-2002 tech crash:** Price momentum was strongly negative for tech in 2001-2002; NI growth also collapsed. Double negative filter likely reduces exposure heavily. Defensive.
- **2008-2009 GFC:** Both signals negative. Near-zero exposure expected.
- **2020 COVID:** Tech/healthcare: strong price momentum AND strong NI growth. Double-positive. Should capture the COVID recovery well.
- **Low-vol 2013-2017:** Broad-based momentum with earnings support. Consistent positive alpha expected.

**Data constraint:** Price momentum available for all periods; SUE (EDGAR) only from ~2009. Composite degrades gracefully to pure price momentum in early folds.

## Success criteria

- OOS Sharpe > max(H180 Sharpe, H150 Sharpe) — must beat both parents
- 95% CI lower bound > 0
- H001 corr moderate (price momentum component adds equity correlation)

## Falsification

If Sharpe ≤ max(H180, H150), the signals are either redundant or the combination creates noise. Earnings momentum may be already reflected in price by the time EDGAR data arrives.

## Implementation

- `EarningsMomentumComposite` in `src/quant/strategies/cross_sectional.py`
- Signal: `(z(sue) + z(mom_12m)) / 2`, sector-neutral top 20%
- Sensitivity sweep: mom_window ∈ {126, 189, 252} days

## OOS Results

*(to be filled after backtest)*
