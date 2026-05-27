# H313: EarningsYieldOverlay — H021 with SPY/TLT Long-Run Relative Valuation Signal

## Hypothesis
Lettau-Ludvigson (2001) and Goyal-Welch (2008) show that valuation-based predictors contain equity premium information. We use a tractable proxy: the ratio of SPY's recent return to its long-run average return, compared to TLT's recent return. Specifically: compute SPY 252d return minus SPY 1260d (5-year) annualized CAGR. When the short-term excess CAGR is high (recent run-up), market is extended → reduce SPY weight by 15%. When negative (market below trend), → neutral or slightly increase. This is a slow-moving contrarian valuation overlay, not a momentum signal.

**Expected behavior:**
- 1998-1999 tech bubble: SPY 1yr CAGR far exceeds 5yr → reduce SPY weight → modest protection
- 2007 pre-crisis: Strong 2006-2007 rally over long-run → reduce → some protection
- 2013-2017: Steady bull market; 1yr ≈ 5yr CAGR → near-neutral (no big signal)

**Success criteria:** OOS Sharpe ≥ 0.80; H001 corr < +0.45; CI lower > 0.
**Falsification:** Contrarian valuation signal too slow; mean-reversion at index level doesn't survive costs.

## Result

**OOS Sharpe 0.679** | CI [0.281, 1.070] | CAGR 5.85% | MaxDD −21.31% | H001 corr +0.517 | N_trades 5023

**Decision: KEEP** (CI lower > 0, Sharpe > 0.51)

Valuation overlay passes the bar but does not add Sharpe vs H021 (0.695). MaxDD identical to H021 (−21.31%) — the valuation signal barely fires (threshold insensitive: 0.75/1.0/1.5 all produce 0.679, identical MaxDD). The z-score computed on expanding window rarely crosses 1.0 threshold in steady bull markets, making the signal essentially dormant for long periods. The SPY 1yr vs 5yr CAGR comparison is structurally sensible but the signal is too infrequent and small (15% cut only) to have material impact on the portfolio.
