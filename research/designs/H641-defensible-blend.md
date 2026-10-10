# H641 — DefensibleBlend

**Signal:** Equal-weight blend of the leveraged return streams of the four *defensible* Set10 champions
(H622 VIX-mom asymmetric, H625 VIX-mom downside-vol, H630 vol-of-vol scaling, H631 VIX-mom lev-Calmar),
each re-derived by optimizing its own weights. Deliberately excludes the overfit grid-corner configs
(H640/H621).

**Why it should work:** Set10 showed raw Sharpe-maximization drifts to an overfit grid corner. Blending
*mechanism-diverse but all-interior* champions should retain most of the Sharpe while improving robustness
(lower variance across folds, better worst-fold) — a diversification benefit across leverage mechanics.

**Success:** Sharpe in the defensible band (~2.3–2.6) with Calmar > 2.0 and MaxDD no worse than the
weakest component. **Falsify:** blend Sharpe < min(components) → mechanisms are redundant, not diversifying.
**Result:** appended after run.
