# H153: SectorNeutralMomentum — 12-1m Momentum Within GICS Sectors

## Hypothesis

Raw cross-sectional momentum (H150) has a structural flaw: when one sector is in a strong trend (e.g., tech 2020, energy 2022), that sector dominates the top quintile. The momentum crash that follows is partly a sector reversal rather than a stock-specific reversal. Sector-neutral momentum isolates the stock-specific momentum component by ranking within each of the 11 GICS sectors separately and selecting the top 30% of each sector. This gives equal sector allocation regardless of which sector is trending, eliminating sector concentration risk.

**Signal:** `mom = price(t-21) / price(t-252) - 1`. Rank descending **within each GICS sector**. Select top 30% per sector. Equal sector weight: each sector receives weight = 1 / num_active_sectors. Equal stock weight within each sector.

## Expected Regime Behavior

- **2000-2002 tech crash:** Sector neutralization helps — tech is only one sector (≈9% weight), not the entire top quintile. Non-tech sectors provide buffer. Expected better than H150 in fold 0-1.
- **2008-2009 GFC:** Financials concentrated in the bottom quintile in H150 (short in H151). In H153, financials get their sector allocation even in a bear market — prevents the worst momentum crashes caused by sector reversal.
- **2013-2017 low-vol bull:** Broad sector participation in uptrend. Sector neutralization doesn't hurt much — multiple sectors are doing well simultaneously. Slight Sharpe drag vs H150 (can't concentrate in the best sector).
- **2020 COVID:** Healthcare and tech led. H150 would overweight both; H153 gives each sector equal weight. Slight underperformance vs H150 in the recovery, but less crash exposure.
- **2022 bear:** Energy outperformed. H153 limits energy to 1/11 of portfolio; H150 would overweight energy. Sector neutralization benefits in this type of single-sector-rotation year.

## Success Criteria

- OOS Sharpe ≥ H150 (sector neutralization should improve risk-adjusted returns)
- Max DD better than H150 (the main benefit: lower concentration in reversing sectors)
- 95% CI lower bound > 0
- Fold consistency: fewer negative folds than H150

## Falsification

- Sharpe < H150 and Max DD worse → sector neutralization constrains the portfolio away from the strongest signals without improving drawdowns; the constraint is purely harmful
- H001 corr > +0.80 → sector neutralization doesn't change the market exposure character of the strategy

## Implementation

- Class: `SectorNeutralMomentum(CrossSectionalStrategy)` in `src/erised/strategies/cross_sectional.py`
- Params: `sector_map` (pd.Series: ticker → sector), `long_days=252`, `skip_days=21`, `top_quantile=0.30`, `monthly_rebalance=True`
- Equal sector weight: each active sector gets 1/N_sectors total allocation, then equal-weight within sector
- Sector map loaded from `data/sp500_sectors.parquet` via `load_sp500_sectors()`
- Runner: `scripts/run_h153.py`

## Results

*(to be filled after backtest)*
