# H320: DefensiveMacroStateMachine — 3-state Risk-On/Neutral/Risk-Off machine with hysteresis

## Hypothesis
Risk-off regimes are asymmetric: they arrive quickly and take time to end. A pure signal-following approach that exits Risk-Off as soon as VIX drops below 28 for one day will generate whipsaw costs during volatile recoveries. This strategy implements a formal state machine with hysteresis: Risk-Off requires 5 consecutive non-triggering days before returning to Neutral. The three regime signals are: (1) VRP (options market premium), (2) SPY 252d momentum, and (3) relative momentum of SPY vs TLT over 63 days (bond trend breakout signal). Risk-On requires all three favorable. Risk-Off triggers if VIX > 28 OR SPY 252d < 0. The state machine with calm_days hysteresis prevents false exits after brief VIX spikes (e.g., August 2015, Q4 2018). In Risk-Off, SPY weight is cut to 0.65x of H021 with the freed weight moved to TLT — explicitly defensive without abandoning equities.

**Expected behavior:**
- 2000-2002 tech crash: SPY 252d turns negative quickly; Risk-Off state entered and sustained; 0.65x SPY / elevated TLT; bonds rally strongly in this period, offsetting equity losses substantially
- 2008-2009 GFC: VIX >> 28, SPY 252d deeply negative; Risk-Off the entire Lehman episode; hysteresis prevents exit on bear-market rallies; expect MaxDD materially below H021
- 2020 COVID: VIX spikes above 28 in March 2020 → Risk-Off; calm_days hysteresis delays re-entry until sustained calm; misses some initial recovery but avoids whipsaw
- 2013-2017 low vol: VIX < 28, SPY up, VRP positive, SPY > TLT; Risk-On state for extended periods; 1.15x scale adds modest return

**Success criteria:** OOS Sharpe >= 0.95; MaxDD < -20%; CI lower > 0; H001 corr < 0.75.
**Falsification:** Hysteresis delay costs more return than it saves in drawdown reduction; risk_off_spy_scale sensitivity shows 0.50/0.65/0.80 all produce the same result (scale is not the binding constraint). Or: state machine provides no improvement over H021's natural vol-scaling.
