# H155: SmoothMomentum — Path-Consistent 12-1 Momentum (Da et al. 2014)

## Hypothesis

Raw 12-1 momentum (H150) treats a stock up 40% via two gap-up days identically to one up 40% via a steady daily grind. Da, Gurun & Warachka (2014) show the latter is more durable: investors underreact more to continuous gradual information than to discrete events ("frog in the pan"). Multiplying the 12-1 return by the fraction of positive days in the formation window selects stocks with smooth upward paths and penalizes episodic/event-driven winners.

**Signal:** `smooth_score = mom_12_1 × fraction_positive_days`. Rank descending. Long top quintile, equal-weighted. Monthly rebalance.

## Expected Regime Behavior

- **2000–2002 tech crash:** Tech stocks in 1999–2000 had many gap-up days from analyst upgrades and M&A rumors — precisely the event-driven pattern H155 penalizes. Expected: lower tech overweight vs H150 entering the crash → better fold 0–1 performance.
- **2008–2009 GFC:** Financial stocks crashed with large daily moves, then recovered explosively (few positive days in a smooth uptrend). H155 should have lower financial weighting at the GFC peak → modestly better crash protection.
- **2013–2017 low-vol bull:** This is the ideal regime — broad-based steady grind upward. Fraction-positive close to 60% for most large-cap names. H155 ≈ H150 in this regime.
- **2020 COVID:** Crash then V-recovery. The V-recovery is driven by large gap-up days, not smooth paths — H155 may underselect the strongest COVID-recovery names. Could trail H150 in fold 11 (2019–2021).
- **2022 bear:** Energy names had smooth upward paths in 2021–2022 (driven by persistent supply disruptions, not event gaps). H155 should overweight energy appropriately → potentially better 2022 performance than H150.

## Success Criteria

- OOS Sharpe > H150 Sharpe − 0.05 (smooth path should not add cost without benefit)
- Max DD materially better than H150 (the key expected benefit: fewer gap-driven blow-ups)
- 95% CI lower bound > 0
- H001 corr within ±0.05 of H150 (signal modification, not market-exposure change)

## Falsification

- Sharpe ≤ H150 − 0.10 AND Max DD no better → the fraction-positive weighting adds noise without reducing crashes; the jumpy stocks are genuinely better performers, not fragile
- H001 corr > +0.85 → the signal modification is too small to matter; H155 is statistically identical to H150

## Implementation

- Class: `SmoothMomentum(CrossSectionalStrategy)` in `src/quant/strategies/cross_sectional.py`
- Params: `long_days=252`, `skip_days=21`, `top_quantile=0.20`, `monthly_rebalance=True`
- Score: `mom × fraction_positive_days` over same formation window (skip_days to long_days)
- Runner: `scripts/run_h155.py`

## Results

*(to be filled after backtest)*
