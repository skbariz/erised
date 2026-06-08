# H552: MaxDD-Constrained MDR (Expanded Pool)

## Signal
MDR (Maximum Diversification Ratio) optimizer over an expanded 58-strategy pool: H543's 55 strategies
(H001–H098 AA + H150–H473 CS keepers) plus three near-target Set7 strategies excluded from H543 solely
by the MaxDD > -10% bar: H537 (MomentumWithPortfolioRatchet), H538 (ReversionQualityCombo), H540
(FundamentalReversalWeekly). A soft MaxDD penalty term `maxdd_penalty * max(0, dd - (-0.10))` is added
to the MDR objective so the SLSQP optimizer is nudged away from weight configurations that push the
aggregate portfolio beyond -10% drawdown.

Two variants are solved and compared:
- **Baseline**: pure unpenalized MDR on the expanded 58-strategy pool (same objective as H543).
- **Penalized**: MDR + soft penalty (penalty strength grid-searched: 1, 2, 5, 10).

## Why it should work
H543 achieved Sharpe 1.600 but CAGR only 5.65%. The MDR objective maximizes the ratio of weighted
volatility to portfolio volatility — it is diversification-oriented and volatility-minimizing, which
suppresses CAGR. H537/H538/H540 are all-fold CS strategies with Sharpe ≈ 1.0–1.14 and H001 corr
< 0.02; they have higher expected CAGR than the H043/H078 calendar anchors that dominate H543 weights.
Adding them to the pool and penalizing extreme drawdown-concentration may allow the optimizer to select
a configuration that is less calendar-heavy, producing higher CAGR while preserving the MaxDD floor.

## Expected behavior across regimes
- **2000–2002 tech crash**: CS reversal strategies (H538/H540) are market-neutral and should be
  unaffected; H537 momentum ratchet may reduce exposure during the drawdown.
- **2008–2009 GFC**: Calendar anchor H078 (0 corr) and H043 still expected in top weights; CS reversal
  strategies maintain near-zero market beta.
- **2020 COVID**: Rapid mean-reversion in reversal strategies; momentum ratchet cuts exposure promptly.
- **Low-vol 2013–2017**: CS strategies contribute positive alpha; penalty does not bind (no large DDs).

## Success criteria
- OOS Sharpe ≥ 1.5 (matches H543 champion)
- CAGR ≥ 7.0% (vs H543 5.65%)
- MaxDD ≤ -10%
- Calmar ≥ 0.80
- CI lower bound > 0

## Falsification
If CAGR stays ≤ 6% despite adding higher-CAGR components, the MDR objective's volatility-minimization
dominance is too strong for any soft penalty to overcome — pool expansion alone cannot fix the CAGR
problem within the MDR framework.

---
*Results will be appended below after backtest.*
