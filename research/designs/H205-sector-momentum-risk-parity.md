# H205: SectorMomentumRiskParity

**Signal:** Moskowitz & Grinblatt (1999) — H021 variant — top equity ETF replaces SPY in risk-parity pair
**Date:** 2026-05-25

## Results

**Decision: kill**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.399 |
| 95% CI | [−0.004, 0.802] |
| CAGR | 4.22% |
| Max DD | −54.16% |
| H001 corr | +0.047 |

CI lower bound clips zero (−0.004) — fails keep bar. MaxDD (−54.16%) near SPY's −55.92%, offering no downside protection. Concentrating the equity leg to a single ETF vs H021's SPY+TLT loses the diversification of the risk-parity baseline entirely. The H021 baseline achieved 0.695 Sharpe with −21.31% MaxDD using the same framework. Sector rotation adds noise, not signal, to a risk-parity pair structure.
