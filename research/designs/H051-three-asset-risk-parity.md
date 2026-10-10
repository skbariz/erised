# H051: Three-Asset Risk Parity (SPY + TLT + GLD)

## Hypothesis

H021 (SPY+TLT risk parity, Sharpe 1.03, Max DD -18.41%) is the program's star. The mechanism is well-understood: TLT provides genuine crisis diversification because its negative equity beta reduces portfolio vol during equity selloffs, allowing larger effective position sizing per unit of risk.

GLD provides a *different* crisis hedge: inflation and stagflation regimes (2021-2022) where real rates fall and both SPY and TLT suffer. Adding GLD as a third asset creates a more complete all-weather portfolio: SPY covers growth regimes, TLT covers deflation/risk-off regimes, and GLD covers inflation/stagflation regimes.

**Signal:** Inverse-vol weighting across SPY, TLT, and GLD using 60d realized vol. Each asset weight = (1/vol_i) / sum(1/vol_j). Normalized to sum to 1.0. Daily rebalancing. Falls back to SPY+TLT (H021 behavior) when GLD data is unavailable (folds 0-2).

## Expected Regime Behavior

- **2000-2002 tech crash:** No TLT/GLD data → flat (zero fold). Same as H021.
- **2008-2009 GFC:** SPY -50%, TLT +25%, GLD -5% → TLT dominates weighting; GLD minor contribution. Similar to H021 but with slight GLD drag.
- **2013-2017 low vol bull:** SPY outperforms both → SPY gets highest weight (lowest vol). GLD slightly reduces SPY allocation vs H021.
- **2020 COVID:** SPY -34% then +70%, TLT +20%. Rapid parity shift toward TLT then back to SPY.
- **2021-2022 inflation bear:** SPY -20%, TLT -35%, GLD +5%. GLD weight increases as SPY/TLT vol rises. This is where H051 should outperform H021 most clearly.

## Success Criteria

- OOS Sharpe > 0.90 (should approach H021's 1.03)
- 95% CI lower bound > 0.40
- Max DD < -20% (should beat or match H021's -18.41%)
- H001 corr < 0.45 (lower than H021's +0.407 due to GLD's orthogonal regime)

## Falsification

- Sharpe < H021 (0.95) → GLD adds noise in most regimes, reducing the clean SPY-TLT parity signal
- Max DD worse than H021 → GLD weight hurts in regimes where all three assets fall together
- H001 corr higher than H021 → GLD doesn't add regime diversity, just follows SPY

## Implementation

- Class: `ThreeAssetRiskParity(MultiAssetStrategy)`
- Uses `walk_forward_multi()`
- `generate_weights()` returns DataFrame with columns ["SPY", "TLT", "GLD"]
- Canonical params: `vol_lookback=60`
- Requires: `tlt_close`, `gld_close` columns in prices DataFrame

## Results

*(to be filled after backtest)*
