# H314: MacroCombinationOverlay — Ensemble of 3 Macro Signals for H021 Exposure

## Hypothesis
Rapach-Zhou (2010) show forecast combinations outperform individual predictors. We combine three signal families: (1) VRP signal (VRP > 0 = bullish, ≤ 0 = bearish), (2) term spread (TLT 252d > SHY 252d = bullish, else bearish), (3) credit spread (HYG 63d > LQD 63d = bullish, else bearish; 0 points if unavailable). Signal sum maps to H021 exposure tiers: sum ≥ 2 → 120% scale; sum = 1 → 105%; sum = 0 → 90%; sum ≤ -1 → 75%. With HYG unavailable, only 2 signals available; tiers adjust: ≥ 1 → 115%; 0 → 95%; -1 → 75%. This ensemble is more stable than any single signal.

**Expected behavior:**
- 2008: Term spread and credit signal likely bearish simultaneously → 75% scale → protection
- 2013-2017: VRP + term spread both bullish → 120% scale → modest leverage during calm
- 2020: All three bearish briefly → 75% scale → crash protection then recovery

**Success criteria:** OOS Sharpe ≥ 1.00; MaxDD < -15%; CI lower > 0.
**Falsification:** Signal combination doesn't outperform H021 (1.029); ensemble too noisy to add value.

## Result

**OOS Sharpe 0.641** | CI [0.243, 1.035] | CAGR 5.22% | MaxDD −23.14% | H001 corr +0.533 | N_trades 5023

**Decision: KEEP** (CI lower > 0, Sharpe > 0.51)

Hypothesis falsified as improvement: H314 (0.641) underperforms its best constituent signal (H306: 0.728) and H021 (0.695). Macro composite dilution confirmed at the portfolio-timing level — the same dilution pattern that affects cross-sectional factor composites also applies to macro signal combinations. Combining VRP + term spread + credit spread weakens all three because: (1) the signals have different best operating frequencies (VRP=daily, term spread=252d, credit=63d), and (2) when they disagree, the compromise exposure is suboptimal for all three. Three signals do not equal three times the information.
