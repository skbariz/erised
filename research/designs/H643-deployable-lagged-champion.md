# H643 — DeployableLaggedChampion

**Signal:** The H622 VIX-momentum asymmetric mechanism with the regime signal **lagged 1 day** (decisions
use only VIX known at the prior close → realistic next-day execution). The "what you'd actually realize"
champion.

**Why it should work:** Set10's H638 showed the VIX-momentum edge loses ~0.5 Sharpe under a 1-day lag —
much of the headline relied on same-day VIX. Building the lag in from the start yields an honest, deployable
strategy whose reported Sharpe is achievable in production.

**Success:** Sharpe > 2.0 with the lag baked in (deployable, not a same-day artifact). Report the gap vs
the unlagged H622 (2.642). **Falsify:** Sharpe < SPY bar after lag → the edge is not executable.
**Result:** appended after run.
