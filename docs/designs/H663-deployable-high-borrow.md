# H663 — DeployableHighBorrow663

**Signal:** The deployable Sharpe champion (H650 construction: weights fit on the 1-day-lagged signal,
interior grid), but **re-fit and evaluated under a realistic higher financing cost** — a **200bps** annual
borrow spread on the leveraged (>1×) portion, vs the 50bps default the program has used since H559/H570.
Sweep the interior grid; finalize on the best config under the high-borrow regime; report the Sharpe drag
vs the 50bps H650.

**Why it should work / what it tests:** the whole leveraged frontier assumes a 50bps financing spread. A
retail or small-institutional mandate borrows wider. This re-optimizes *and* re-scores the deployable
champion at 200bps to answer: does the edge survive real-world borrow, and does the optimal allocation shift
toward lower-leverage sleeves when financing is expensive? It is the deployable, re-fit counterpart of the
same-day borrow-stress checks (H588/H616), which only re-scored fixed weights.

**Expected behavior:** higher borrow costs the avg-1.5× leveraged book ~ (avg_lev−1)·150bps ≈ 75bps/yr of
extra drag, so Sharpe should fall a few hundredths and CAGR ~0.5–1.0pp; the re-fit may modestly favor
sleeves that need less leverage to hit target vol.

**Success:** deployable Sharpe still clears the keep bar (and ideally ≥ 1.5) at 200bps — i.e. the edge is
not an artifact of cheap financing. **Falsify:** the high-borrow Sharpe collapses below the keep bar → the
leveraged edge is a financing-cost mirage.

**Result:** appended after run.
