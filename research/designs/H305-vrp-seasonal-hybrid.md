# H305: VRPSeasonalHybrid — H078 Seasonal + VRP Overlay in Nov-Apr Window

## Hypothesis
H078 (SeasonalRiskParity, Sharpe 0.762, H001 corr +0.008) allocates H021 risk parity during Nov-Apr and 100% TLT during May-Oct. We layer a VRP overlay only during the active Nov-Apr window: when VRP > 0, run H021 at full scale; when VRP ≤ 0, run at 75% scale (more TLT, less SPY). The May-Oct TLT period remains unchanged. This combines the zero-correlation seasonal anchor with a modest VRP timing signal during the equity-active season.

**Expected behavior:**
- High VRP in Nov-Apr: full H021 risk parity (unchanged from H078)
- Low or negative VRP in Nov-Apr: reduced equity, more TLT
- May-Oct: pure TLT regardless of VRP (H078 anchor)
- 2008 winter: VRP may signal well; 2020 March: VRP spike helps if in favorable season

**Success criteria:** OOS Sharpe ≥ H078 (0.762); H001 corr ≤ +0.15; CI lower > 0.
**Falsification:** VRP overlay during Nov-Apr hurts H078 returns — the seasonal anchor is sufficient.

## Result

**OOS Sharpe 0.333** | CI [−0.068, 0.719] | CAGR 2.68% | MaxDD −22.17% | H001 corr +0.217 | N_trades 2539

**Decision: KILL** (CI lower bound −0.068 < 0)

Hypothesis falsified. VRP overlay inside the Nov-Apr H078 window is worse than H078 alone (0.375 baseline). The unfavorable_scale sensitivity is all identical (0.321/0.333/0.342 for 0.6/0.75/0.9) — raising the floor toward H078 uniformly improves Sharpe, confirming VRP conditioning hurts during the seasonal active window. The Nov-Apr equity window in H078 is already a favorable regime; adding VRP conditioning introduces false exits that reduce returns without protecting against actual drawdowns. H078's zero-correlation value is diminished: H001 corr rises to +0.217 vs H078 +0.008.
