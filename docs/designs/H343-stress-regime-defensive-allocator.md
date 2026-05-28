# H343: StressRegimeDefensiveAllocator — Three-sleeve blend (H021 / H043 / H078) driven by VIX level

## Hypothesis
VIX is the most direct measure of market stress. H043 (SellInMayBondConfirm, MaxDD -10.11%) is the best individual drawdown-protected strategy. H078 (SeasonalRiskParity, H001 corr +0.008) is the zero-correlation anchor. H021 (RiskParity) is the workhorse. This strategy replaces a continuous VRP signal with a simple three-state VIX level: VIX > 28 (crisis), VIX 20-28 (elevated), VIX < 20 (normal). In crisis, nearly all capital goes to the two most defensive seasonal strategies (80% H078 + 20% H043). In elevated stress, a balanced blend (40% H021 + 30% H043 + 30% H078). In normal conditions, the workhorse dominates (70% H021 + 30% H078). H043 contributes because its bond-confirmation mechanism specifically protects in the "elevated but not panic" zone (VIX 20-28). VIX 28 is selected as the crisis threshold based on H079 (crisis protocol risk parity) which showed this level is approximately where the realized-vs-implied vol premium turns negative and crash dynamics dominate.

**Expected behavior:**
- 2000-2002 tech crash: VIX spends extended time 25-45; elevated and crisis states dominate; H078+H043 heavy positioning; expected MaxDD significantly below H021
- 2008-2009 GFC: VIX >28 for months (Sep 2008 - Mar 2009); 80% H078 positioning throughout; H078 is in TLT during winter months — near-perfect defensive positioning
- 2020 COVID: VIX spikes to 80+ briefly; crisis state; 80% H078; rapid VIX normalization returns to normal by June 2020; recovery exposure restored
- 2013-2017 low vol: VIX persistently <20; 70% H021 positioning; expect H021-like returns with 30% H078 drag (H078 underperforms H021 in fully invested low-vol periods)

**Success criteria:** OOS Sharpe >= 0.90; MaxDD < -15%; CI lower > 0.
**Falsification:** VIX regime cuts are too coarse; wrong state during most crisis transitions; Sharpe below H021 (1.029) due to defensive drag in bull markets.
