# H555: Leveraged Evaluation at 1.5× Notional

## Hypothesis

All champion strategies are unleveraged (max equity allocation = 100% SPY).
At 1.5× notional leverage, CAGR approximately doubles for the leveraged fraction.
Since Calmar = CAGR/|MaxDD| and both scale proportionally with leverage, Calmar
is roughly leverage-invariant. The benefit of leverage is in raw CAGR:
a strategy with CAGR 6% and MaxDD 7% becomes CAGR ~9% and MaxDD ~10.5% at 1.5×,
meeting the institutional ≥ 8% CAGR threshold.

H555 evaluates the H551/H552/H553 winners at 1.0×, 1.25×, 1.5×, and 2.0× to
find the leverage point where:
- CAGR ≥ 8% (institutional minimum)
- MaxDD ≤ −15% (institutional tolerance)
- Sharpe ≥ 1.5 (preserved — Sharpe is leverage-invariant net of borrowing costs)

Borrowing cost: assume 50 bps/year on the leveraged fraction (Libor/SOFR proxy).

## Expected behavior

At 1.5× leverage:
- CAGR: ~6% × 1.5 − 0.5% borrowing = ~8.5%
- MaxDD: ~7% × 1.5 = ~10.5% (just within tolerance)
- Sharpe: unchanged (both returns and vol scale equally)
- Calmar: (8.5%) / (10.5%) = 0.81 — roughly unchanged

At 2× leverage:
- CAGR: ~6% × 2 − 1% borrowing = ~11%
- MaxDD: ~7% × 2 = ~14%
- Sharpe: unchanged
- Calmar: 11/14 = 0.79 — also roughly unchanged

## Success criteria

- 1.5× CAGR ≥ 8.0%
- 1.5× MaxDD ≤ −15%
- All leverage levels Sharpe ≥ 1.4 (net of borrowing cost)

## Implementation

Pure calculation (no new walk-forward needed):
1. Take the OOS return series from H551/H552/H553 champions
2. Compute leveraged returns: `r_lev = r_strategy * leverage − borrow_cost/252`
3. Compute metrics at each leverage level
4. Find the optimal leverage level satisfying all constraints

## Results

*To be filled after H551/H552/H553 complete.*
