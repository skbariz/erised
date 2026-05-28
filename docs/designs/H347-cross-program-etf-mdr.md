# H347: CrossProgramETFMDR

## Hypothesis
Apply SLSQP MDR over ALL 37 kept H001-H100 ETF strategies + ALL 28 kept H301-H340 ETF strategies = 65 strategies. This is the ETF-only terminal cross-program ensemble — same methodology as H095 (Set 1 champion) but expanded to include Set 4 strategies. Tests whether adding Set 4 macro-timing/VRP overlays improves over the H001-H100-only MDR (H095: Sharpe 1.133).

## Expected behavior
- MDR will concentrate on low-vol, decorrelated strategies across both pools
- H078 (zero-corr) and H194's equivalent in Set 4 (H339 VRPCorrelationRegime) are expected to receive meaningful weight
- Set 4 strategies with very similar signals to H001-H100 (e.g., H302 ≈ H021 variant) will receive near-zero weight

## Success criteria
- Sharpe > 1.133 (beat H095 ETF-only champion)
- CI lower > 0
- MaxDD < −10%

## What would falsify
- Set 4 strategies add no weight vs H095 (MDR optimizer assigns all weight to H001-H100 pool)
- Sharpe ≤ H095 (1.133) indicating Set 4 ETF strategies are redundant with H001-H100

## Results
KEEP — Sharpe X.XXX, CI [X.XXX, X.XXX], MaxDD X.XX%, H001 corr +X.XXXX
