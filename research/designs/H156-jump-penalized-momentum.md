# H156: JumpPenalizedMomentum — Trimmed-Return Momentum

## Hypothesis

A stock's 12-1 momentum can be dominated by a few extreme daily returns (earnings gaps, acquisition announcements, short squeezes). These extreme days often represent one-time information shocks that are fully priced in — subsequent drift is weaker than for stocks whose returns are spread across many smaller moves. Removing the top-5 and bottom-5 daily returns from the formation window and computing momentum on the remaining "trimmed" path isolates the steady-trend component. This is a complement to H155: H155 reweights by path smoothness; H156 removes the extreme observations entirely.

**Signal:** `trimmed_mom = sum of daily returns in formation window excluding top-5 and bottom-5`. Rank descending. Long top quintile, equal-weighted. Monthly rebalance.

## Expected Regime Behavior

- **2000–2002 tech crash:** Dot-com stocks had extreme gap-up days on IPO lock-up expirations, analyst upgrades, and earnings beats. Trimming removes these from the momentum score → H156 should underweight the most bubble-inflated names at peak.
- **2008–2009 GFC:** Financial sector had extreme negative and positive days during the crisis. Trimming removes both extremes, making the financial sector's "recovered momentum" signal cleaner. Could improve fold 5 vs H150.
- **2013–2017 low-vol bull:** Few extreme daily returns — trimming removes little signal. H156 ≈ H150 in this regime.
- **2020 COVID:** March 2020 had many extreme down days; recovery had many extreme up days. Trimming removes the worst COVID crash exposure from the signal and also tempers the explosive recovery names. Net effect ambiguous.
- **2022 bear:** Energy had steady positive drift (not gap-driven). Trimming doesn't significantly alter energy's momentum ranking. H156 ≈ H150 in 2022.

## Success Criteria

- OOS Sharpe > H150 Sharpe − 0.05 (trimming should not remove useful signal)
- CI lower bound > 0
- Sensitivity: robust to trim_days ∈ {3, 5, 10}

## Falsification

- Sharpe ≤ H150 − 0.10 → trimming removes genuine momentum signal, not just noise; the extreme-day moves ARE the momentum
- High correlation with H155 (> 0.95) → both signals measure the same thing; not worth including both

## Implementation

- Class: `JumpPenalizedMomentum(CrossSectionalStrategy)` in `src/erised/strategies/cross_sectional.py`
- Params: `long_days=252`, `skip_days=21`, `top_quantile=0.20`, `trim_days=5`, `monthly_rebalance=True`
- Score: sum of daily returns excluding top and bottom `trim_days` in the formation window
- Runner: `scripts/run_h156.py`

## Results

*(to be filled after backtest)*
