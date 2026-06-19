# H652 — DeployableSparse652

**Signal:** The deployable Sharpe portfolio (H650 construction: weights fit on the 1-day-lagged signal,
interior grid) but constrained to a **sparse top-K weight vector** (K ∈ {3,5,8,12}). After the full
optimization, keep only the K largest sleeves and renormalize; finalize on the smallest K that holds within
0.05 Sharpe of the full optimum.

**Why it should work:** fewer active strategy sleeves means lower operational cost and turnover to run the
book in practice. If a handful of sleeves carries most of the deployable Sharpe, a sparse blend is cheaper
to trade with little performance loss — a direct operational-cost lever on the deployable frontier.

**Success:** a small K (≤ 8) that retains deployable Sharpe within ~0.05 of the full blend (≈ H650 1.98).
**Falsify:** Sharpe degrades sharply as K shrinks (no sparse subset is competitive) → the edge needs the
full diversified blend.

**Result (KEEP).** Config t15w20, chosen **K=5** sleeves. Deployable **Sharpe 1.933** [1.54, 2.33],
Calmar 1.253, Sortino 2.020, CAGR 9.15%, MaxDD −7.30%, H001 corr +0.385, folds 1.33–2.66. K-sweep
{3: 1.498, 5: 1.933, 8: 1.972, 12: 1.980}: just **5 of 55 sleeves retain Sharpe within 0.05** of the full
blend (K=3 collapses to 1.498). Confirms the deployable edge is carried by a handful of sleeves — a real
operational-cost saving.
