# H334: TripleOverlayCalendar -- H078 + VRP scaler + funding stress in Nov-Apr

## Hypothesis
H078 applies H021 weights in Nov-Apr and 100% TLT in May-Oct. H305 already added a single VRP overlay. This strategy stacks two independent overlays on the Nov-Apr window: (1) VRP z-score scaler bounded to [0.80, 1.20] — same mechanism as H301; (2) funding stress reduction multiplier (0.80) when VIX is above its 20th percentile AND credit spreads are widening (HYG 63d < LQD 63d). The May-Oct TLT anchor is unchanged. The three signals (seasonal structure, VRP, funding stress) are orthogonal: seasonal is purely calendar, VRP reflects options market pricing, funding stress reflects realized credit and vol conditions. The product of three independent scalers should produce a more nuanced SPY weight than any single modifier.

**Expected behavior:**
- 2000-2002: Negative VRP (0.80 scaler) + credit stress (0.80 stress mult) → Nov-Apr SPY weight × 0.64 → substantial protection
- 2008-2009: Both stress signals fire aggressively → SPY weight in Nov-Apr heavily reduced
- 2013-2017 low vol: VRP positive (1.20×) + no stress → SPY weight boosted slightly; small return gain
- 2020 COVID: Brief but severe stress → March 2020 both signals fire → protection during COVID crash

**Success criteria:** OOS Sharpe > H305 (target ≥ 0.90); MaxDD < H078 (-18%); CI lower > 0.
**Falsification:** Triple stacking creates over-dampening — the product of three scalers becomes too small, exiting even mild winters and sacrificing returns without proportional drawdown benefit.

## Results
*[To be filled after backtest]*
Decision:
