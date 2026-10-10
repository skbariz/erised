# H304: VRPVolTarget — H086 Vol-Scale Amplified by VRP

## Hypothesis
H086 (Sharpe 1.109) vol-targets H021 at 8% annualized. We extend it with a VRP amplifier: scale = min(max_scale, vol_scale × (1 + vrp_amplifier × vrp_zscore_clipped)). When VRP is high, we allow slightly more risk-taking on top of the volatility target. When VRP is negative, we dampen below the vol target floor. The idea is that a high VRP justifies accepting slightly more variance. Source: Bollerslev-Tauchen-Zhou and Martin (2017) who both argue that the VRP is a risk compensation signal.

**Expected behavior:**
- High VRP + low vol: vol_scale near 1.0 × amplification → modest leverage
- High VRP + high vol: vol_scale < 1.0, amplified slightly → still defensive
- Low VRP + high vol: double dampening → very defensive

**Success criteria:** OOS Sharpe ≥ H086 (1.109); MaxDD ≤ -10%; CI lower > 0.
**Falsification:** VRP amplification adds volatility to the vol-target without improving risk-adjusted returns.

## Result

**OOS Sharpe 0.646** | CI [0.253, 1.042] | CAGR 4.73% | MaxDD −23.35% | H001 corr +0.532 | N_trades 5023

**Decision: KEEP** (CI lower > 0, Sharpe > 0.51)

H086 baseline in the OOS walk-forward context: 0.662 / MaxDD −21.17%. H304 (0.646) underperforms its H086 baseline — the VRP amplifier weakens the vol-target. Sensitivity fully insensitive: vrp_amplifier=0.10/0.15/0.20 all produce Sharpe 0.645-0.646 — the amplifier has negligible effect, suggesting VRP magnitude is too small relative to the vol-scale to matter. Hypothesis falsified as improvement over H086; the vol-target is already a stronger signal than VRP.
