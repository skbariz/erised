# H348: CrossProgramETFEqualWeight

## Hypothesis
Equal-weight all 65 kept ETF strategies (37 H001-H100 + 28 H301-H340). Compares EW vs MDR at the same 65-strategy pool. The MDR result (H347) should outperform; the EW baseline quantifies how much MDR adds beyond simple diversification. Also compares to H048 (EW H001-H100 only: Sharpe 0.800) to measure Set 4 marginal contribution.

## Expected behavior
- EW over 65 strategies should have lower variance than EW over 35 (H048)
- Many Set 4 strategies are variants of H021/H086, so marginal diversification benefit is limited
- Sharpe probably near H048 (0.800) ± 0.05

## Success criteria
- Sharpe > 0.80 (at least as good as H048)
- CI lower > 0
- Shows clear gap below H347 MDR (confirms MDR adds value)

## What would falsify
- EW over 65 beats H347 MDR (would suggest MDR over-concentrates and EW is more robust)

## Results
KEEP — Sharpe X.XXX, CI [X.XXX, X.XXX], MaxDD X.XX%, H001 corr +X.XXXX
