# H323: CappedEqualWeightEnsemble -- Top-10 H001-H100 strategies equal-weighted

## Hypothesis
H048 equal-weights all 35 kept H001-H100 strategies, including low-Sharpe strategies that add noise. H095 MDR concentrates heavily but uses a complex optimizer. A simple middle ground: select the top 10 strategies by OOS Sharpe, equal-weight them at 10% each. This "winners with equal weight" approach concentrates on the strategies with the strongest standalone performance while avoiding optimizer instability. The rationale is that high-Sharpe strategies are high-Sharpe for structural reasons (e.g., H086 LowVolTargetPortfolio at 1.109, H089 H043+H021 blend at 1.074, H070 MultiAssetDynamicVolTarget at 1.066) and that equal-weighting prevents post-hoc concentration bias from MDR optimization. This is the simplest possible "take the best" approach and tests whether optimizer complexity (H095) adds value over naive top-N selection.

**Expected behavior:**
- vs H095 (Sharpe 1.133, MaxDD -7.16%): expect lower Sharpe if MDR's fractional calendar weighting (e.g., 24% H043) adds genuine value beyond simple top-N selection; MaxDD likely worse since top-Sharpe strategies are not necessarily low-MaxDD
- vs H048 equal-weight all (Sharpe 0.800): expect Sharpe improvement since excluding bottom-25 low-Sharpe strategies removes noise
- 2000-2002 tech crash: top-10 likely includes H086/H087 (vol-target/ratchet) which protect well; also H089 (H043+H021 blend) which has low equity exposure in May-Oct
- 2008-2009 GFC: H086/H087/H070 vol-target family dominates the top-10; strong drawdown protection
- 2013-2017 low vol: vol-scaling strategies reduce exposure in calm periods; slight return drag but consistent

**Success criteria:** OOS Sharpe >= 1.00 (demonstrates top-N adds value over equal-weight); MaxDD < -15%; CI lower > 0. If Sharpe >= 1.10 that would be competitive with H095, suggesting optimizer adds little.
**Falsification:** Sharpe < H048 (0.800) would mean top-N selection on OOS Sharpe is overfitting (strategies are top-Sharpe partly by luck). Sharpe between 0.800-0.900 would confirm MDR optimization adds substantial value.

## Results
(to be filled after running)
