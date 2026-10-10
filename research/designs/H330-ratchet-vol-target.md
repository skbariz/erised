# H330: RatchetVolTarget -- Combined vol-target (H086) and ratchet drawdown (H087) via min() operator

## Hypothesis
H086 (vol-target) and H087 (ratchet drawdown) each protect from a different type of risk. H086 reacts to the instantaneous vol environment; H087 reacts to the accumulated loss from peak. In a fast crash (2020 COVID), H086 fires first because vol spikes before large drawdowns accumulate. In a slow-moving bear market (2000-2002), drawdown builds gradually and H087 fires progressively as losses mount while vol may be only modestly elevated. Taking min(vol_scale, dd_scale) means: either protection mechanism can cut exposure, whichever is more conservative at any given moment. This "dual guard" design improves on both components: it does not require the vol spike and the drawdown to co-occur for the system to respond. The drawdown_sensitivity=3.0 parameter means the ratchet fully exits at 33% drawdown (dd_scale = 0) but has a floor of 0.10 to avoid complete cash-out. The drawdown scale resets to 1.0 when the portfolio sets a new cumulative high. This is motivated by Grossman and Zhou (1993) who show optimal dynamic strategies under drawdown constraints exhibit exactly this reset structure.

**Expected behavior:**
- 2000-2002 tech crash: early vol spikes cut via H086 mechanism; later cumulative drawdown triggers ratchet; dual mechanism more aggressive than either alone
- 2008-2009 GFC: both mechanisms fire; vol-scale drives early cuts; ratchet then holds exposure down through prolonged drawdown; MaxDD substantially better than H086
- 2020 COVID: vol-scale fires fast (Feb-March 2020); short sharp drawdown; new high reached by August 2020 -> ratchet resets; recovery exposure restored quickly
- 2013-2017 low vol: portfolio near all-time highs; ratchet scale = 1.0 (no effect); vol low; vol_scale near 1.0; behavior equivalent to H086; no cost to the mechanism

**Success criteria:** OOS Sharpe >= H086 (1.109); MaxDD <= -12% (ideally < -10%); CI lower > 0.
**Falsification:** Min(vol_scale, dd_scale) is overly conservative and cuts too much exposure during recoveries, sacrificing CAGR without material MaxDD improvement over H086; or drawdown_sensitivity sweep shows flat Sharpe (the ratchet is inactive in OOS folds).
