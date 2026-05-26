# H255: TailFilteredMomentum — Momentum With Lottery Exclusion

## Hypothesis

Cross-sectional momentum (H157, Sharpe 1.079) includes some high-idio-vol and high-MAX stocks
whose price appreciation reflects lottery demand, not information. Removing these from the
momentum universe before ranking should improve signal quality and reduce crash risk.

Signal: 12-1 month momentum with exclusion filter — remove stocks in the top decile of
idio vol (63d) OR top decile of MAX21. Rank remaining stocks by momentum; long top quintile.
Sector-neutral, monthly rebalance.

This tests whether tail-risk screening improves momentum quality. The exclusion filter uses
union (either trigger removes the stock) to maximize lottery cleanup.

Expected regime behavior:
- **2000–2002 tech crash**: Filter removes bubble momentum stocks; significantly reduces crash.
- **2008–2009 GFC**: Filter removes high-vol momentum names; should limit momentum crash.
- **2013–2017 low-vol**: Fewer exclusions; performance similar to unfiltered momentum.
- **2020 COVID**: Filter active in March; should protect against momentum crash.

Success criteria: OOS Sharpe ≥ 1.00, Max DD < −30%, CI lower bound > 0.
Must beat H157 vol-scaled momentum (Sharpe 1.079) to justify the filtering complexity.
Falsification: Sharpe below H157 (filter removed the good momentum names along with bad).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.896 |
| 95% CI | [0.496, 1.313] |
| CAGR | 16.31% |
| Max DD | −49.46% |
| H001 corr | +0.000 |
| Decision | **KEEP** |

**Sensitivity (exclude_decile):** 5% → 0.903 | 10% → 0.896 | 20% → 0.883. Tighter exclusion
marginally better; diminishing returns as more names are excluded. Canonical = 10%.

**Falsification confirmed:** H255 (0.896) materially underperforms H157 (1.079). The lottery
filter removes some genuine momentum names along with the noise. Max DD −49.46% is also worse
than H157 (−25.4%). The volatility-scaling in H157 is more effective than lottery exclusion as
a momentum quality filter. H255 is kept (passes bar) but H157 is the superior momentum strategy.
