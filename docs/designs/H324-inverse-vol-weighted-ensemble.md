# H324: InverseVolWeightedEnsemble -- H001-H100 strategies weighted by 1/sigma_i

## Hypothesis
Inverse-volatility weighting is the simplest risk-parity approach applied to strategy return streams. Each strategy receives weight proportional to 1/sigma_i (annualized OOS return volatility), normalized to sum to 1. This is exactly the mechanism that makes H021 (risk parity between SPY and TLT) work so well -- extend it one level up to the strategy layer. Low-vol strategies like H043 (SellInMayBondConfirm, only active 6 months) and H078 (SeasonalRiskParity, low-vol in summer) will receive higher weight; high-vol strategies like H019 (SPY/GLD rotation) receive lower weight. This is a parameter-free alternative to MDR that requires no optimizer. Unlike H095 MDR (which simultaneously considers correlation structure), inverse-vol ignores strategy correlations but is more robust to estimation error. The direct ancestor of H021's core mechanism applied to the ensemble layer.

**Expected behavior:**
- vs H095 (Sharpe 1.133, MaxDD -7.16%): expect lower Sharpe since H095 MDR uses correlation information that inverse-vol ignores; MaxDD likely similar since both upweight low-vol strategies
- vs H048 equal-weight (Sharpe 0.800): expect Sharpe improvement via upweighting stable strategies over high-vol ones
- vs H050 Sharpe-weighted (Sharpe 0.820): inverse-vol should produce different ordering than Sharpe-weighted since vol and Sharpe are not identical (H078 has moderate Sharpe 0.762 but very low vol)
- 2000-2002 tech crash: H043/H078 heavily weighted (lowest vol) -- strong protection since both avoid equities in May-Oct
- 2008-2009 GFC: H086 vol-target portfolio has low realized vol (it reduces exposure) -- will be overweighted
- 2013-2017 low vol: inverse-vol weighting is stable in this regime; low-vol strategies all active; modest performance

**Success criteria:** OOS Sharpe >= 0.95; MaxDD < -12%; CI lower > 0. Inverse-vol should beat equal-weight H048 (0.800) since it applies the H021 mechanism at the ensemble layer.
**Falsification:** Sharpe < H048 (0.800) would mean strategy vol is not informative for ensemble weighting (low-vol strategies are not systematically better diversifiers). Sharpe < 0.85 confirms MDR's correlation information is essential.

## Results
(to be filled after running)
