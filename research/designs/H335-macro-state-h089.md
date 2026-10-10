# H335: MacroStateH089 -- H089 (H043+H021 50/50) with macro scaler on H021 half

## Hypothesis
H089 (H043+H021 50/50 blend) achieved Sharpe 1.074 and MaxDD -9.62%, one of the best simple blends in the program. Its strength comes from H043's seasonal TLT anchor (May-Oct flat, winter bond-confirmed) blended with H021's dynamic vol-weighting. This strategy leaves the H043 half untouched (binary, seasonal — should not be modulated by macro because it already has its own macro gate via TLT trend) and applies a macro stress scaler only to the H021 half. The scaler uses the H306 3-signal stress composite (VIX percentile + credit stress + vol acceleration) to modulate H021 exposure: H021_half × (0.7 + 0.6 × (1 - stress)) ∈ [0.70×, 1.30×]. Low stress → amplify H021; high stress → dampen H021. The H043 half acts as a stress-independent anchor.

**Expected behavior:**
- 2000-2002: High stress → macro scaler near 0.70 → H021 half reduced → net portfolio more defensive
- 2008-2009: Maximum stress → H021 half at 0.70× weight → near-full protection from H043 TLT anchor
- 2013-2017 low vol: Low stress throughout → macro scaler near 1.30 → slight H021 amplification
- 2020 COVID: Stress spikes in Feb-March → H021 half trimmed → partial protection

**Success criteria:** OOS Sharpe ≥ H089 (1.074); MaxDD ≤ -9.62%; CI lower > 0.
**Falsification:** Macro scaler on H021 half creates noise — the signal fires at wrong times or the 30% swing range is too small to make a difference vs H089.

## Results
*[To be filled after backtest]*
Decision:
