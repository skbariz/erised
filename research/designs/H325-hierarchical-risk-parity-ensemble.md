# H325: HierarchicalRiskParityEnsemble -- 2-tier MDR: H001-H100 (50%) + H150-H299 (50%)

## Hypothesis
H300 (Sharpe 1.525) combines all 52 components in a single MDR pass. This risks the optimizer over-concentrating on whichever strategies happen to have the lowest correlation to the full pool -- in H300 this meant H194 LongShortReversal got 23% weight, dominating the allocation. A hierarchical approach first allocates 50% to each of two tiers (H001-H100 macro/ETF strategies and H150-H299 cross-sectional equity strategies), then runs MDR within each tier. This enforces structural balance between the macro and cross-sectional worlds. The cross-sectional strategies (H150-H299) are high-CAGR (~15-25%) but high-MaxDD (~-40-55%); the macro strategies (H001-H100) are lower-CAGR (~5-9%) but much lower-MaxDD (~-7-23%). A 50/50 structural split prevents the MDR optimizer from discovering that 70% cross-sectional gives the highest DR and then concentrating there. Compare directly to H300 (flat MDR, Sharpe 1.525, MaxDD -6.86%).

**Expected behavior:**
- vs H300 (Sharpe 1.525, MaxDD -6.86%): hierarchical structure may improve MaxDD further (by forcing 50% into low-DD macro strategies) but likely sacrifices some Sharpe vs H300's unconstrained allocation
- vs H095 (Sharpe 1.133, MaxDD -7.16%): H325 adds the cross-sectional tier (50%) on top of H095-style macro allocation; cross-sectional diversification should lift Sharpe while 50% macro floor holds MaxDD
- 2000-2002 tech crash: macro tier H095-style heavily in calendars/TLT; CS tier (H194 LongShortReversal, reversal strategies) uncorrelated to tech crash; balanced allocation protects
- 2008-2009 GFC: macro tier moves to TLT heavily; CS tier short-reversal strategies tend to profit in high-vol regimes; good structural alignment
- 2020 COVID: macro tier protects on the way down; CS tier (momentum-reversal) recovers quickly; 50/50 split keeps total drawdown modest
- 2013-2017 low vol: macro tier earns steady 5-7% from risk parity; CS tier earns high alpha from quality/momentum; 50/50 split captures both

**Success criteria:** OOS Sharpe >= 1.35 (between H095 and H300); MaxDD < -12% (better than H300's -6.86% is ambitious given cross-sectional MaxDD -40-55%, but the macro floor should help); CI lower > 0.
**Falsification:** Sharpe < H095 (1.133) would mean the hierarchical structure destroys value by forcing the cross-sectional allocation even when macro strategies dominate. MaxDD worse than -20% would confirm 50% CS allocation introduces too much equity drawdown.

## Results
(to be filled after running)
