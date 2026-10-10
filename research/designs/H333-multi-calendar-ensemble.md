# H333: MultiCalendarEnsemble -- Equal-weight blend of 5 calendar strategies

## Hypothesis
Calendar strategies (H023, H025, H043, H046, H078) each capture different seasonal dimensions: raw Halloween effect (H023), January Barometer predictor (H025), bond-confirmed seasonal (H043), momentum-gated seasonal (H046), and risk-parity seasonal (H078). Equal-weighting all five should smooth out the idiosyncratic fold variance of individual calendar strategies — some fire in winters others are defensive, creating partial diversification within the calendar family. Unlike MDR (H095) which concentrates on best strategies, equal-weighting the entire calendar family tests whether breadth within one signal family adds value. H078's zero-equity May-Oct TLT position dominates the ensemble defensive character; H025's annual binary signal adds low-frequency timing.

**Expected behavior:**
- 2000-2002: H078 + H043 both defensive (100% TLT / flat); H023 active; H025 can flip to flat if Jan return negative → ensemble naturally defensive
- 2008-2009: H043 bond-confirm likely flat in winter 2008 (TLT inverted); H046 momentum gate cuts SPY; H078 100% TLT May-Oct; ensemble defensive
- 2013-2017 low vol: All strategies mostly long; equal-weight ≈ long SPY; H078 adds TLT in summer; low-cost diversification
- 2020 COVID: May-Oct H078 TLT; H025 depends on Jan return; H043/H046 momentum-gated

**Success criteria:** OOS Sharpe > 0.51; MaxDD < -30%; CI lower > 0; H001 corr < 0.50.
**Falsification:** Calendar strategies are too correlated to diversify within the family (they all fire or all sit out simultaneously); equal-weight dilutes H078's unique signal.

## Results
*[To be filled after backtest]*
Decision:
