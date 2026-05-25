# H167: PiotroskiHighBar

## Hypothesis
Variation of H164 (PiotroskiValueScreen) with stricter F-score gate: min_fscore=7 instead of 6.
Hypothesis: higher quality gate further reduces risk within the value universe at the cost of breadth.
Uses same PiotroskiValueScreen class, just different threshold parameter.

## Expected regime behavior
- Higher threshold → fewer stocks → more concentrated portfolio
- Risk/return tradeoff: stricter gate may improve stock-level quality but reduce diversification
- The 8-point F-score distribution peaks around 5-6, so 7+ is a tight filter

## Success criteria
- OOS Sharpe > 0.51 AND CI lower bound > 0
- Lower MaxDD than H164 (fewer low-quality value stocks)

## What would falsify this
- Sharpe worse than H164 (over-filtering destroys the value signal)
- MaxDD worse despite stricter gate (less diversification outweighs quality gain)

## Results (2026-05-25)
- Sharpe: 0.591  CI: [0.201, 0.997]  CAGR: 9.88%  MaxDD: -55.68%
- H001 corr: +0.3720
- Decision: **KEEP** (barely — CI lo=0.201 > 0, Sharpe=0.591 > 0.51)
- Notes: MaxDD worse than H164 (tighter gate → fewer stocks → less diversification).
  Hypothesis partially falsified: stricter gate hurts rather than helps.
  Lower H001 corr than H164 (+0.372 vs +0.500) — interesting diversification property.
