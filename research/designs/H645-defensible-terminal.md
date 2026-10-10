# H645 — DefensibleTerminal645

**Signal:** The terminal *defensible* champion. Selects the best VIX-momentum config from an **interior-only**
grid — threshold {15,16,17} × window {20} (plain and asymmetric tv+cap) — explicitly excluding the overfit
short-window / high-threshold corner that produced H640's inflated 3.04. Runs the full champion battery:
5-start weight stability, threshold/window sensitivity, and a 1-day-lag robustness check.

**Why it should work:** Set10 proved the threshold×window grid has no interior optimum; the honest champion
is the interior config that is both strong and lag-robust. This codifies that as the program's defensible
terminal strategy (vs the curve-fit H640).

**Success:** an interior config with Sharpe ~2.5–2.65, weight-stable (5/5 restarts tight), and a lag gap
smaller than H640's. **Falsify:** no interior config clears Sharpe 2.0 lag-robust → the whole family is fragile.
**Result:** appended after run.
