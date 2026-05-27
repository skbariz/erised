# H315: ConstrainedMacroOverlay — H314 with Campbell-Thompson Economic Constraints

## Hypothesis
Campbell-Thompson (2008) show that applying weak economic constraints to equity premium forecasts improves OOS performance. We apply their constraints to H314's macro combination: (1) SPY weight is never reduced below 50% of H021's computed SPY weight (floor constraint); (2) SPY weight is never increased above 130% of H021 weight (cap constraint); (3) exposure increases (leverage) only when at least 2-of-3 signals are bullish (sign constraint on leverage). The floor prevents the strategy from going too defensive on false signals; the cap prevents over-leverage. These constraints are parameter-free by design — they follow economic logic, not optimization.

**Expected behavior:**
- All three constrained: 50% SPY floor → worst case is half H021 SPY exposure
- Bullish regime: 2+ signals needed for leverage → avoids premature leverage in mixed signals
- Neutral: defaults to H021 unmodified (no signal = no intervention)

**Success criteria:** OOS Sharpe ≥ H314; MaxDD ≤ H314; CI lower > 0.
**Falsification:** Constraints are too conservative; H314 without constraints performs better because false signals are rare.

## Result

**OOS Sharpe 0.631** | CI [0.237, 1.026] | CAGR 5.00% | MaxDD −22.60% | H001 corr +0.534 | N_trades 5023

**Decision: KEEP** (CI lower > 0, Sharpe > 0.51)

Hypothesis partially falsified: H315 (0.631) underperforms H314 (0.641) — the Campbell-Thompson constraints add friction without benefit. Sensitivity: spy_floor=0.4/0.5/0.6 produces identical Sharpe (0.631) and identical MaxDD (−22.60%) — the floor never binds (H314's SPY weight never goes below 50% in the test windows). The leverage cap (130%) similarly never binds. When constraints are never active, they cannot help or hurt. The sole evidence of constraints: slight Sharpe reduction (0.641→0.631) from the 2-of-3 leverage gate, which is marginally too conservative.
