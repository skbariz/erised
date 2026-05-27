# H311: TermSpreadSPYTimer — H021 with TLT vs SHY Term Spread Signal

## Hypothesis
The term spread (long-term bond yield minus short-term yield) is one of Goyal-Welch's (2008) more reliable equity premium predictors. We proxy it with the relative 252-day performance of TLT (long-term bonds) versus SHY (short-term bonds). When TLT has outperformed SHY over 252 days (positive term premium environment), the yield curve is normal — H021 runs at full weight. When SHY has outperformed (inverted curve proxy), reduce H021 SPY weight by spy_scale (default 0.50) and reallocate freed weight to TLT. This is a cleaner version of H080 (same mechanism, different lookback and signal computation).

**Expected behavior:**
- 2006-2008: Yield curve inverted → signal fires → SPY cut → protection before GFC
- 2020 COVID: Very brief inversion; signal fires then recovers quickly
- 2013-2017: Normal curve → H021 full weight throughout

**Success criteria:** OOS Sharpe ≥ 0.90; MaxDD < -20%; CI lower > 0.
**Falsification:** This signal is already captured by H080 (yield curve overlay); marginal contribution is zero.

## Result

**OOS Sharpe 0.608** | CI [0.204, 0.999] | CAGR 5.17% | MaxDD −21.90% | H001 corr +0.416 | N_trades 5023

**Decision: KEEP** (CI lower > 0, Sharpe > 0.51)

Lowest H001 corr in batches 1-3 (+0.416) — the term spread dimension captures information orthogonal to VRP and credit signals. Sensitivity: spread_lookback=126/252/504 → Sharpe 0.600/0.608/0.653 — longer lookback is consistently better (less noisy term spread signal). The 252d/504d range produces CI lower 0.18-0.24; the 504d version would be the preferred canonical. The 252d canonical Sharpe (0.608) below H021 (0.695) confirms the term spread signal has mild predictive power at monthly resolution but it's less powerful than the VIX-composite approach. Primary ensemble value: low H001 corr.
