# H341: TwoRegimeSleeveAllocator — Two-sleeve blend (H021 / H078) driven by VRP + term spread

## Hypothesis
H021 (RiskParity) and H078 (SeasonalRiskParity) are the two most important individual strategies in the program: H021 is the workhorse all-weather strategy; H078 has H001 correlation +0.008 — the lowest of all 300 strategies, making it an orthogonal diversifier. The question is whether dynamically tilting between these two sleeves based on a macro regime signal (VRP + term spread) beats a static 50/50 blend. The regime: when VRP > 0 AND TLT 252d momentum > SHY 252d momentum (bonds outperforming cash, i.e. duration is rewarded), go 70% H021 / 30% H078 (favorable for the risk-parity engine). When both signals are negative (VRP < 0 AND bond momentum negative), go 30% H021 / 70% H078 (defensive: lean on H078's seasonal TLT-heavy positioning). In mixed states, 50/50. The combined_returns from each component WF are blended daily with a 1-day lagged regime signal to avoid lookahead.

**Expected behavior:**
- 2000-2002 tech crash: VRP turns negative, bonds outperform cash initially then diverge; mixed/unfavorable signal rotates toward H078 (TLT-heavy); expected to reduce equity drawdown
- 2008-2009 GFC: strong unfavorable signal (VRP deeply negative, bond momentum positive but H078 is already in TLT); 70% H078 positioning; significant protection
- 2020 COVID: brief unfavorable spike then rapid recovery; signal lags the instantaneous crash but normalizes quickly; small cost vs upside recovery
- 2013-2017 low vol: VRP persistently positive, bonds gentle positive momentum; persistent 70% H021 weighting; should capture H021's sustained outperformance in calm markets

**Success criteria:** OOS Sharpe >= 0.90; MaxDD < -20%; CI lower > 0; H001 corr ideally < 0.50 (exploiting H078's near-zero correlation).
**Falsification:** Dynamic regime signal adds no value over static 50/50; Sharpe approximately equal to naive blend 0.5*H021 + 0.5*H078. Or: regime signal fires too rarely to matter (most days are "mixed" 50/50).
