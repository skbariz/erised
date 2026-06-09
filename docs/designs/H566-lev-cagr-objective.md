# H566: LevCAGRObjective

## Signal
SLSQP maximizing annualized CAGR of the vol-targeted portfolio directly, instead of Sharpe or
Calmar. The objective is `-(1 + r_lev).prod()^(252/T) - 1`. MDR warm-start. H559 leverage
params: 6.5%/2.0x/20d.

## Why should it work?
H558 showed that optimizing Calmar_lev beats optimizing Cal_base. By extension, optimizing CAGR
on the deployed leveraged stream should outperform post-hoc CAGR maximization (H556's implicit
structure). CAGR is a geometric-mean objective; SLSQP with an additive portfolio linear in
weights finds weight combinations that compound fastest after leverage costs. The optimizer has
no volatility penalty, so it may accept higher vol if that produces higher geometric mean. This
is a CAGR-first mandate: the test answers whether leverage amplifies CAGR more than it amplifies
volatility for this pool, and whether such a portfolio is still statistically stable (CI > 1.3).

## Expected behavior in key regimes
- 2000-2002 tech crash: CAGR optimizer may over-weight crash-sensitive strategies since their
  long-term CAGR is still positive. Expect worse DD than H559.
- 2008-2009 GFC: similar concern — the optimizer does not explicitly protect from fat-tails.
  Vol-targeting provides the only crash protection.
- 2020 COVID: short-duration crash; CAGR optimizer may handle this well since the recovery
  is swift and the geometric return impact is small.
- 2013-2017 low-vol: high leverage fully deployed; CAGR optimizer maximally benefits here.

## Success criteria
- CAGR_lev > 16% (CAGR-first mandate)
- Sh_lev > 1.5 (minimum statistical quality bar)
- CI lower bound > 1.3 (stricter than usual due to CAGR overfitting risk)
- At least 10 of 12 OOS folds positive (fold consistency check)

## Keep bar override
Standard bar is CI lower > 0. For H566, require CI lower > 1.3 due to CAGR overfitting risk.

## Falsification
If CAGR_lev ≤ 16%, direct CAGR maximization via SLSQP is not more efficient than lev-Sharpe
for compounding. If CI lower ≤ 1.3 at Sh_lev > 1.5, the strategy is statistically fragile
and KEEP is denied despite meeting the Sharpe bar.

## Results (fill in after running)
- Decision:
- Sh_lev:
- CI:
- CAGR_lev:
- DD_lev:
- Cal_lev:
- Notes:
