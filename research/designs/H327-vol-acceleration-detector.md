# H327: VolAccelerationDetector -- H086 vol-target with acceleration/deceleration penalty/bonus

## Hypothesis
H086 responds to the level of realized vol but ignores its direction of change. Research by Ang and Chen (2002) and subsequent literature shows that vol itself is predictable in the short term: a rapid rise in short-window vol (21d) relative to a medium-window baseline (63d) signals the onset of a stress regime that typically persists for days or weeks. This strategy adds a momentum layer on top of H086's vol-scale: when 21d_vol > 1.5 x 63d_vol (vol accelerating), it multiplies the H086 scale by 0.75 as an extra penalty; when 21d_vol < 0.75 x 63d_vol (vol calming), it multiplies by 1.15 (capped at 1.0) as a bonus. The acceleration signal should trigger at the start of major vol spikes, reducing exposure before the worst losses accumulate; the deceleration bonus recaptures upside after vol normalizes. The mechanism is additive over H086 rather than replacing it: the base vol-target already handles the level, and this layer handles the rate-of-change.

**Expected behavior:**
- 2000-2002 tech crash: 21d vol accelerates sharply above 63d baseline; penalty fires, cutting exposure further than H086; fewer whipsaw costs during multi-week drawdown
- 2008-2009 GFC: prolonged high-vol regime with frequent acceleration spikes; penalty dominant; scale well below H086 through October 2008 - March 2009; MaxDD should improve
- 2020 COVID: extreme 21d/63d ratio in Feb-March 2020; penalty fires at peak dislocation; deceleration bonus kicks in April-May 2020 as vol falls; faster recovery than H086
- 2013-2017 low vol: 21d < 63d most of the time; deceleration bonus applies frequently but is capped at 1.0; scale equivalent to H086 on average; minimal CAGR difference

**Success criteria:** OOS Sharpe >= H086 (1.109); MaxDD <= -12%; CI lower > 0.
**Falsification:** Accel/decel signals are too noisy to improve risk-adjusted returns over H086; sensitivity sweep shows Sharpe is non-monotone in accel_threshold; or accel penalty fires too early and cuts exposure before the worst losses, but then misses recovery.
