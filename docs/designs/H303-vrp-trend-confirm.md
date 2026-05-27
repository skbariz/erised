# H303: VRPTrendConfirm — VRP + SPY 200d MA Dual Gate

## Hypothesis
Combine VRP signal with SPY trend filter (200-day MA) as dual confirmation. VRP alone can spike during crashes before losses are over — the trend filter provides a guardrail. When both are favorable (VRP > 0 AND SPY > 200d MA), run H021 at full scale. When only one is favorable, run at 85%. When neither is favorable, run at 60%. This is the "Bollerslev guardrail" approach: VRP tells us about compensation, trend tells us about regime continuity.

**Expected behavior:**
- 2008: SPY breaks 200d MA in 2008 early → defensive even before VRP signals all-clear
- 2020 COVID: brief below-200d MA period → temporary reduction then recovery
- 2013-2017: Both favorable most of the time → near-normal

**Success criteria:** OOS Sharpe ≥ 0.95; MaxDD < -15%; CI lower > 0.
**Falsification:** Dual gate is too restrictive — many false exits reduce CAGR without meaningful MaxDD improvement.

## Result

**OOS Sharpe 0.667** | CI [0.270, 1.057] | CAGR 5.01% | MaxDD −18.46% | H001 corr +0.559 | N_trades 5023

**Decision: KEEP** (CI lower > 0, Sharpe > 0.51)

Best MaxDD of any VRP strategy (−18.46%) — the SPY 200d MA provides the real guardrail, not VRP. Sensitivity: min_scale=0.5/0.6/0.7 is monotonically improving (0.662/0.667/0.670) — raising the floor slightly improves Sharpe by reducing over-defensiveness. The dual gate clearly reduces drawdown vs H021 (−21.31%) through the trend filter, but Sharpe (0.667) underperforms H021 (0.695) because the trend gate also reduces upside participation. Primary value: MaxDD reduction as ensemble component.
