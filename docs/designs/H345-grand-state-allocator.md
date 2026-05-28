# H345: GrandStateAllocator — Five-state macro regime blend (H021 / H086 / H078) driven by VRP + VIX + term spread

## Hypothesis
The H300 grand MDR (Sharpe 1.525) showed that multi-strategy blending is the highest-value approach in this program. H345 takes this further by using a richer five-state macro regime to drive sleeve weights across three core strategies: H021 (RiskParity), H086 (LowVolTargetPortfolio), H078 (SeasonalRiskParity). The five states are: Risk-On (VRP>0 AND VIX<20): highest H021/H086, minimal H078; Growth (VRP>0 AND VIX 20-28): moderate H021/H086 tilt; Neutral (VIX 20-28 AND VRP near zero): balanced; Caution (VRP<0 AND VIX 20-28): shift toward H078; Crisis (VIX>28 OR term spread inverted): 50% H078, maximum defense. The VRP + VIX combination is more powerful than either alone: VRP can be negative while VIX is low (quiet complacency), or VIX can be elevated while VRP is positive (recovering market). The term spread inversion (SHY 252d > TLT 252d, i.e. short bonds outperforming long bonds) provides a leading indicator of recession that VIX alone misses. This operationalizes the cross-program lesson that correlation regime (H097) + vol regime (H086) + seasonal regime (H078) are the three irreplaceable signal types.

**Expected behavior:**
- 2000-2002 tech crash: VRP negative, VIX 25-45; Caution/Crisis states; 40-50% H078; H078 in TLT during bear market; expected MaxDD < -15%
- 2008-2009 GFC: term spread inverts in late 2007; Crisis state early (before VIX explodes); term spread signal provides earlier warning than VIX; strong pre-positioning
- 2020 COVID: VIX>28 triggers Crisis state; 50% H078; VIX normalizes by June 2020; transitions to Risk-On by Q3 2020 (VRP positive, VIX<20); full recovery participation
- 2013-2017 low vol: VRP positive, VIX persistently <20; Risk-On state dominates; 60% H021 + 20% H086 + 20% H078; expect highest CAGR of all H341-H345

**Success criteria:** OOS Sharpe >= 1.00 (match or beat H021 at 1.029); MaxDD < -12%; CI lower > 0; the five-state machine provides meaningful state diversity (no single state dominates >70% of trading days).
**Falsification:** Five states are too many for the signal resolution; most days collapse into 2-3 states; the added complexity of term spread adds noise rather than signal (Sharpe < H341's two-state version).
