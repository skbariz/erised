# H338: DoubleCheckH080 -- RiskParityYieldCurveOverlay (H080) + credit spread double-confirm

## Hypothesis
H080 (RiskParityYieldCurveOverlay) reduces SPY weight by 50% when SHY 63d return > TLT 63d return (curve inverted). This can produce false positives — transient yield curve movements that do not reflect genuine recession risk. Adding a credit spread confirmation (HYG 63d < LQD 63d) as a second gate creates three distinct regimes: (1) both curve inverted AND credit stress → strong cut (60% of H021 SPY weight); (2) only curve inverted, no credit stress → partial cut (85% of H021 SPY weight); (3) only credit stress, no curve inversion → no change (H021 full). The credit spread is an independent real-economy signal — when both yield curve and credit markets signal stress simultaneously, the recession risk is substantially higher than either signal alone.

**Expected behavior:**
- 2000-2002: Curve may invert; credit stress follows → both fire → strong SPY cut; protection
- 2008-2009: Credit stress (HYG devastated) + curve inversion → both fire → 60% SPY weight
- 2013-2017 low vol: Curve normal, credit healthy → H021 full; zero friction
- 2020 COVID: Brief credit stress; curve may not invert → credit-only = no change; natural containment

**Success criteria:** OOS Sharpe > H080 (≥ 0.89); MaxDD ≤ H080; CI lower > 0.
**Falsification:** Adding the double-confirm requirement misses protective opportunities (curve inverts without credit stress → only gets partial cut instead of full cut) → worse MaxDD than H080 despite the nuance.

## Results
*[To be filled after backtest]*
Decision:
