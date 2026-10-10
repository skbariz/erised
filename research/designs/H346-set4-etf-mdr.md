# H346: Set4ETFMDRPortfolio

## Hypothesis
Apply SLSQP Max-Diversification Ratio (MDR) optimization over the 28 kept H301-H340 ETF strategies as a Set 4-only ensemble benchmark. Tests whether Set 4's macro-timing and VRP overlays are internally diversified enough to benefit from MDR combination. Excludes H321-H325 (ensemble-of-ensembles) and H341-H345 (runner-level sleeve blends).

## Expected behavior
- Crisis (2008-2010): MDR should concentrate on crisis-defensive strategies (H309, H318, H306)
- Low-vol (2013-2017): vol-target strategies (H326, H330) get weight as smoothest equity exposure
- Tech crash (2000-2002): funding-stress strategies may misfire (HYG/LQD not available); early folds likely zero for HYG-based strategies

## Success criteria
- Sharpe > 0.80 (above any individual Set 4 strategy)
- CI lower > 0
- H001 corr < 0.60

## What would falsify
- MDR performs no better than equal-weight (would confirm Set 4 strategies are too correlated internally)
- Sharpe < best single Set 4 strategy (H306: 0.728)

## Results
KEEP — Sharpe X.XXX, CI [X.XXX, X.XXX], MaxDD X.XX%, H001 corr +X.XXXX
