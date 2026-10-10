# H329: VolRegimeRiskParity -- 3-state vol regime using structural mean comparison with asymmetric exposure

## Hypothesis
H086 and H326-H328 all define the vol regime relative to the current observation's own distribution. A complementary framing uses the structural long-run mean: if today's 21d vol is well below its 252-day average (< 0.70x), the market is structurally calm and a modest exposure boost (1.20x) is warranted; if it is well above (> 1.30x), apply a tighter 7% vol-target rather than H086's 8% to be extra conservative. The transition boundaries (0.70 and 1.30) are symmetric around 1.0, so the strategy is not biased toward any regime. The tighter 7% target in the high regime operationalizes the "uncertainty amplification" concept from Bloom (2009), who shows that period of high vol uncertainty warrant disproportionately conservative positioning. The 1.20x scale in the low regime is motivated by the low-vol anomaly (Baker, Bradley, Wurgler 2011): low-realized-vol periods are statistically followed by above-average risk-adjusted returns, justifying modest leverage.

**Expected behavior:**
- 2000-2002 tech crash: 21d vol climbs to > 1.30x the 252d mean; 7% target fires; tighter than H086's 8%; better protection
- 2008-2009 GFC: extreme high-vol regime triggers 7% target throughout; modest MaxDD improvement over H086
- 2020 COVID: rapid spike to high regime; immediate 7% target; then low regime post-crisis triggers 1.20x bonus; recovery acceleration
- 2013-2017 low vol: persistently below 0.70x mean; 1.20x multiplier for extended stretches; material CAGR improvement vs H086

**Success criteria:** OOS Sharpe >= H086 (1.109); MaxDD <= -12%; CI lower > 0.
**Falsification:** The 0.70/1.30 boundaries are arbitrary and the regime signal is noisy; Sharpe is flat across vol_target_high values (0.06-0.08) in sensitivity sweep; or the 1.20x leverage in calm periods adds tail risk that dominates the CAGR benefit.
