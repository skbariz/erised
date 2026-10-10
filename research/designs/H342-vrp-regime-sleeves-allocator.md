# H342: VRPRegimeSleevesAllocator — Three-sleeve blend (H086 / H021 / H078) driven by VRP percentile

## Hypothesis
H086 (LowVolTargetPortfolio, Sharpe 1.109) is the best individual strategy. H021 (RiskParity, Sharpe 1.029) is the workhorse. H078 (SeasonalRiskParity, Sharpe ~0.82) is the orthogonal diversifier. This strategy treats VRP percentile rank (rolling 252d window) as a regime classifier. When VRP is high (>67th pctile), the options market prices a large risk premium — favorable conditions for strategies that harvest carry-like returns. Weight 60% H086 (most aggressive vol-targeting strategy), 30% H021, 10% H078. When VRP is low (<33rd pctile), the market expects less reward for risk. Weight 10% H086, 40% H021, 50% H078 (shift toward the defensive seasonal strategy). Normal regime: 30% H086, 50% H021, 20% H078. The use of a percentile (rather than a fixed threshold) ensures the regime signal adapts to structural changes in VRP over time, avoiding fixed-threshold regime flip bias.

**Expected behavior:**
- 2000-2002 tech crash: VRP plunges below 33rd pctile; 50% H078 positioning; H078 is in TLT during May-Oct — significant crash protection during the extended 2001-2002 downturn
- 2008-2009 GFC: extreme low-VRP regime for months; persistent H078-heavy positioning; expected MaxDD improvement over H086 standalone
- 2020 COVID: VRP spikes then normalizes within weeks; brief H086 reduction then rapid return to high-VRP regime; expected small cost, fast recovery
- 2013-2017 low vol: VRP persistently in normal-to-high range; 30-60% H086 allocation; H086's vol-scaling is nearly inactive in low-vol periods anyway — blend should approximate H021 behavior

**Success criteria:** OOS Sharpe >= 0.95; MaxDD < -15%; CI lower > 0.
**Falsification:** VRP percentile is too noisy a daily regime signal; the three-way blend does not meaningfully differ from a fixed 35% H086 / 45% H021 / 20% H078 blend; Sharpe is not statistically distinguishable from fixed weights.
