# H332: VRPEnhancedHalloween -- HalloweenMomentumConfirm (H046) + VRP scaler

## Hypothesis
H046 (HalloweenMomentumConfirm) produces a binary SPY weight within Oct-Apr: 1.0 if SPY 6m momentum is positive, 0.5 if momentum is negative. The May-Sep window holds 100% TLT. This strategy adds a VRP-based scaler to the active window weights: when VRP > 0, multiply the H046 SPY weight by 1.10 (cap 1.0); when VRP ≤ 0, multiply by 0.90. The intuition is that within the favorable seasonal window, VRP conditions still matter — a positive VRP means market participants are over-paying for protection, implying realized vol will likely come in below implied, making equity exposure more attractive. Conversely, negative VRP means the market under-compensates for variance risk.

**Expected behavior:**
- 2000-2002: Negative VRP common → H046 Oct-Apr weights scaled down 10% → mild protection
- 2008-2009: Severely negative VRP → 0.90 scaler applied to H046 weights → partial reduction
- 2013-2017 low vol: VRP consistently positive → 1.10 scaler → modest return boost in Oct-Apr
- 2020 COVID: VRP turns severely negative Feb-March → scaler cuts Oct-Apr SPY by 10%

**Success criteria:** OOS Sharpe > H046 (need baseline; likely 0.6-0.8); MaxDD improvement vs H046; CI lower > 0.
**Falsification:** 10% scaler is too small to materially differentiate; or VRP fires at wrong times within Oct-Apr (positive during crash, negative during bull phase) making it noise.

## Results
*[To be filled after backtest]*
Decision:
