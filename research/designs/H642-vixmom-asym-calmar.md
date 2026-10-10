# H642 — VIXMomAsymCalmar

**Signal:** Asymmetric tv+cap at the VIX-momentum regime (bull VIX<16 & not-rising: tv=8%/cap=2.5x;
bear: tv=4%/cap=1.0x) optimized under a leveraged-**Calmar** objective. Combines the two defensible
record-holders: H622 (asymmetric, Calmar 2.172) and H631 (lev-Calmar objective, Calmar 2.538 / MaxDD −4.12%).

**Why it should work:** the asymmetric mechanism already lifts Calmar; pairing it with a Calmar-maximizing
objective (rather than Sharpe) should push the Calmar/MaxDD frontier further — defensibly, since the Calmar
objective rewards drawdown control rather than chasing a grid-corner Sharpe.

**Success:** Calmar > 2.538 (beats H631) at MaxDD ≤ −5% — a new program Calmar/MaxDD record.
**Falsify:** Calmar < 2.538 → asymmetric + Calmar objective don't compose. **Result:** appended after run.
