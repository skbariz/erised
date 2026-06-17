# H647 — DeployableLowCorr

**Signal:** Asymmetric VIX-momentum regime with the signal **lagged 1 day**, optimized under an
H001-correlation-penalized leveraged-Sharpe objective (obj = lev-Sharpe − λ·corr(port, H001); λ swept).
The deployable institutional diversifier.

**Why it should work:** prior low-corr strategies (H596/H632) used same-day signals. An equity investor
wants a *tradeable* low-correlation overlay; building in the lag gives an honest corr/Sharpe frontier
point that can actually be held alongside an equity book.

**Success:** H001 corr < 0.30 at deployable Sharpe > 1.5. **Falsify:** can't reach corr < 0.30 without
Sharpe < SPY bar under lag. **Result:** appended after run.
