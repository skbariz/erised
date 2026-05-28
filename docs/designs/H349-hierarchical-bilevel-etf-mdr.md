# H349: HierarchicalBilevelETFMDR

## Hypothesis
2-tier hierarchical MDR: Tier 1 = 37 H001-H100 ETF strategies (50%), Tier 2 = 28 H301-H340 ETF strategies (50%). MDR within each tier, then 50/50 blend. Tests whether hierarchical weighting (equal tier allocation + MDR within) beats flat MDR (H347) over the same 65 strategies.

## Expected behavior
- Tier 1 MDR ≈ H095 (same 37 strategies, same optimizer)
- Tier 2 MDR ≈ H346 (same 28 Set 4 strategies)
- Blend should reduce MaxDD vs H095 by incorporating Set 4's different timing signals
- Hierarchical vs flat MDR: hierarchical forces equal allocation between the two eras of strategies, while flat MDR may concentrate entirely on one era

## Success criteria
- Sharpe > 1.0
- CI lower > 0
- MaxDD < −10%
- Comparison: hierarchical vs flat MDR (H347) reveals which tier structure is better

## What would falsify
- Hierarchical (50/50 forced) underperforms flat MDR — flat MDR optimally weights across all 65

## Results
KEEP — Sharpe X.XXX, CI [X.XXX, X.XXX], MaxDD X.XX%, H001 corr +X.XXXX
