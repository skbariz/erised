# H307: LiquiditySpiralProtection — Triple-Confirm Crisis Override with Hysteresis

## Hypothesis
Liquidity spirals (Brunnermeier-Pedersen 2009) occur when multiple stress conditions align simultaneously. We require all three conditions to be true (lagged 1 day) to trigger crisis mode: (1) VIX > 35, (2) HYG 63d return < LQD 63d return, (3) SPY 21d vol > 2× SPY 63d vol. When crisis flag fires, override to 90% TLT / 10% SPY. Exit after 10 consecutive days where VIX < 30 AND vol spike has normalized. When HYG/LQD unavailable, fall back to VIX>35 + vol spike (2-of-2) as the trigger. The triple-confirm design prevents false alarms from a single stressed indicator.

**Expected behavior:**
- 2008: All three conditions likely in Oct-Nov 2008 → crisis mode → MaxDD protection
- 2020: Fast VIX spike may trigger on condition 1+3; HYG helps confirm condition 2
- 2013-2017: Conditions almost never all three true → H021 runs normally

**Success criteria:** OOS Sharpe ≥ 0.85; MaxDD < -15%; CI lower > 0.
**Falsification:** Triple-confirm is too rare to help (too few activations); or VIX+hysteresis alone (H079) is already capturing the same signal.

## Result

**OOS Sharpe 0.695** | CI [0.298, 1.085] | CAGR 6.02% | MaxDD −21.31% | H001 corr +0.541 | N_trades 5023

**Decision: KEEP** (CI lower > 0, Sharpe > 0.51)

Hypothesis falsified as MaxDD improvement — H307 is identical to H021 (Sharpe 0.695, MaxDD −21.31%). The triple-confirm flag (VIX>35 AND HYG<LQD AND 21d_vol>2×63d_vol) never triggered in any OOS test window. Sensitivity across resume_days=5/10/20 produces exactly the same results — confirming the flag is dormant. This validates the falsification condition: triple-confirm is too rare. H079 (crisis protocol, Sharpe 0.952) already captures this type of event more effectively with a simpler mechanism (drawdown + VIX only). KEEP status is because CI > 0 but the strategy adds no unique value.
