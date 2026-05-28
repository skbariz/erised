# H316: QuarterlyMacroStateOverlay — Slow-updating composite macro signal (quarterly resample)

## Hypothesis
Business-cycle research (Lettau-Ludvigson 2001; Fama-French 1989) shows macro state variables predict equity returns at low frequencies. Daily re-estimation of fast signals introduces noise; resampling to quarterly cadence enforces the slow-moving nature of macro regimes. This strategy scores three signals at quarterly frequency: (1) term spread (TLT 252d return vs SHY 252d return — bonds pricing growth), (2) equity momentum (SPY 252d return positive/negative — trend confirmation), and (3) growth/value spread (IVW 252d vs IVE 252d — risk appetite within equities). The quarterly sum [-3, +3] sets H021's exposure scale; forward-filling the quarterly read daily forces the strategy to commit to slow regime calls rather than react to daily noise.

**Expected behavior:**
- 2000-2002 tech crash: equity momentum and growth/value spread turn negative; composite drops below -1 quickly; scale falls to 0.80x, reducing drawdown
- 2008-2009 GFC: all three signals turn negative (bond gains, negative equity momentum, value outperforms); max bearish reading, scale 0.80x throughout
- 2020 COVID: brief spike negative, followed by rapid recovery in equity momentum and growth/value as tech leads; scale recovers to 1.20x by Q3 2020
- 2013-2017 low vol: SPY and IVW both trend positively; term spread favorable; persistent 1.20x scale adds incremental equity exposure

**Success criteria:** OOS Sharpe >= 0.90; MaxDD < -25%; CI lower > 0; H001 corr < 0.80.
**Falsification:** Composite at quarterly frequency adds no information over daily H021; Sharpe <= 0.85 or CI clips zero. Alternatively, all three quarterly scales produce the same Sharpe (signal is flat).
