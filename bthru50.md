# H013–H050 Algorithm Development Plan

Research goals and batch structure for the next 38 hypotheses.

**STATUS: PROGRAM COMPLETE — All 50 hypotheses tested, target ensemble identified.**

---

## The problem we solved

After 12 hypotheses, best OOS Sharpe was **0.62** against a target of **1.0**. All 3 kept strategies (H001, H007, H008) had return correlations of 0.86–0.89 with each other. No ensemble was possible.

The path to Sharpe ≥ 1.0 required genuinely uncorrelated return streams. Three structural changes drove H013–H050:

1. **Long/short** — going short when bearish captures the bear side
2. **Multi-asset** — SPY-only strategies can never escape equity beta
3. **Non-price signals** — calendar and volatility signals are structurally uncorrelated with price momentum

---

## Final program results

| Goal | Target | Achieved |
|------|--------|----------|
| OOS Sharpe | ≥ 1.0 | **1.029** (H021 RiskParity) |
| Max drawdown | < −55.92% (SPY) | **−10.11%** (H043 SellInMayBondConfirm) |
| Total strategies tested | 50 | **50** ✓ |
| Strategies kept | 8–10 diverse signals | **16 individual + 3 ensemble = 19 total** |
| Working ensemble Sharpe | ≥ 1.0 | **0.959** (H049 MinCorrEnsemble, CI [0.543, 1.357]) |

---

## Batch 2 — H013–H017: Long/Short Structures ✓ COMPLETE

**Result:** All 5 killed. Long/short degrades all signals vs long-only. Shorting adds whipsaw costs without capturing meaningful bear-market alpha. H016 (VolScaledTrend) kept — best Max DD in program at time (−26.30%).

| # | Name | OOS Sharpe | Status | Key finding |
|---|------|-----------|--------|-------------|
| H013 | MA(50,200) Long/Short | 0.36 | killed | CI negative lower; shorting hurts |
| H014 | TS Momentum Long/Short | 0.44 | killed | below SPY benchmark; shorting degrades H007 |
| H015 | Dual MA L/S (20/100) | 0.08 | killed | near-zero; whipsaw dominates |
| H016 | Volatility-Scaled Trend | **0.55** | **kept** | best Max DD at time; no ensemble value |
| H017 | Trend L/S + VIX Gate | 0.43 | killed | VIX gate too rare to help |

---

## Batch 3 — H018–H022: Multi-Asset Rotation ✓ COMPLETE

**Result:** 4 of 5 kept. Multi-asset strategies consistently beat SPY benchmark. H021 (RiskParity) is the star of the entire program at Sharpe 1.029 — first strategy to clear the Sharpe ≥ 1.0 goal.

| # | Name | OOS Sharpe | Status | Key finding |
|---|------|-----------|--------|-------------|
| H018 | SPY/TLT Dual Momentum | 0.57 | **kept** | beats benchmark; high H001 corr limits ensemble value |
| H019 | SPY/GLD Rotation | 0.63 | **kept** | highest CAGR in rotation batch; anchor of target ensemble |
| H020 | Best-of-3 Momentum | 0.47 | killed | Sharpe < 0.51; early folds starved of TLT/GLD data |
| H021 | SPY/TLT Risk Parity | **1.029** | **★ kept** | FIRST OOS Sharpe ≥ 1.0; lowest MaxDD at time (−18.41%) |
| H022 | Defensive Equity Hedge | 0.68 | **kept** | best trend Sharpe; near-zero diversification value |

---

## Batch 4 — H023–H027: Seasonality & Calendar Effects ✓ COMPLETE

**Result:** 3 of 5 kept. Calendar signals confirmed as structurally low H001 corr (0.50–0.71). Sell-in-May (H023) and January Barometer (H025) replicate academic anomalies. Key discovery: calendar signals have lower H001 corr than any technical signal family.

| # | Name | OOS Sharpe | Status | Key finding |
|---|------|-----------|--------|-------------|
| H023 | Sell in May | 0.53 | **kept** | confirmed Bouman & Jacobsen; lowest corr of any kept trend/momentum |
| H024 | Turn of Month | 0.41 | killed | Sharpe < 0.51; n_days=5 version narrowly beats benchmark |
| H025 | January Barometer | 0.53 | **kept** | second-best Max DD (−29.53%); avoids bear years via Jan filter |
| H026 | Quarter-End Drift | 0.30 | killed | CI negative; excellent Max DD (−10.13%) and H001 corr (+0.176) |
| H027 | Seasonality + Trend | 0.61 | **kept** | matches H001 Sharpe; too correlated for ensemble |

---

## Batch 5 — H028–H032: VIX-Based Strategies ✓ COMPLETE

**Result:** 1 of 5 kept. VIX contrarian signals fail — they enter during crash continuation, not after capitulation. VIX regime filter reduces trades too aggressively. Only H031 (RealizedVsImpliedVol) passed, capturing the vol risk premium rather than fear/greed.

| # | Name | OOS Sharpe | Status | Key finding |
|---|------|-----------|--------|-------------|
| H028 | VIX Spike Entry | 0.36 | killed | long during crash continuation |
| H029 | VIX Regime Filter | 0.43 | killed | CI barely positive; below benchmark |
| H030 | VIX Trend L/S | −0.09 | killed | negative Sharpe; first negative H001 corr (−0.284) in program |
| H031 | Realized vs Implied Vol | **0.58** | **kept** | vol risk premium filter; best CAGR in batch |
| H032 | VIX Mean Reversion | −0.001 | killed | near-zero; indistinguishable from zero |

---

## Batch 6 — H033–H037: Cross-Asset Macro Timing ✓ COMPLETE

**Result:** 2 of 5 kept. Cross-asset signals identify crisis regimes but struggle to consistently beat benchmark. H034 (SPY-TLT Correlation Regime) and H035 (Gold Fear Gauge) kept. H037 (Dual Safe-Haven Confirmation) achieved program-low H001 corr (+0.264) but was too restrictive — 5 zero folds.

| # | Name | OOS Sharpe | Status | Key finding |
|---|------|-----------|--------|-------------|
| H033 | Bond Trend Equity Timer | 0.44 | killed | below benchmark; excellent Max DD (−14.15%) |
| H034 | SPY-TLT Correlation Regime | 0.55 | **kept** | Max DD = SPY (−55.92%) — held long during 2008 |
| H035 | Gold Fear Gauge | 0.57 | **kept** | anchor of target ensemble; best H001 corr in batch (+0.384) |
| H036 | Equity vs Safe-Haven Momentum | 0.49 | killed | just misses benchmark; good Max DD |
| H037 | Dual Safe-Haven Confirmation | 0.22 | killed | lowest H001 corr in program (+0.264); too restrictive |

---

## Batch 7 — H038–H042: Contrarian & Mean Reversion ✓ COMPLETE

**Result:** All 5 killed. Contrarian strategies achieved the lowest H001 correlations in the program (+0.108 to +0.165) but all had insufficient trade count for reliable OOS estimation. This revealed the fundamental **correlation/activity tradeoff**: strategies decorrelated from H001 trade only in crises (N=6–18 trades over 30 years) — too sparse to clear the keep bar.

| # | Name | OOS Sharpe | H001 corr | Status | N trades |
|---|------|-----------|-----------|--------|---------|
| H038 | Long-Term Reversal | 0.243 | +0.165 | killed | ~7 zero folds |
| H039 | Deep Drawdown Crisis Buyer | 0.061 | +0.133 | killed | N=6 |
| H040 | Short-Term Weekly Reversal | 0.481 | +0.480 | killed | active but just below benchmark |
| H041 | Post-Crash Recovery | 0.405 | +0.108 | killed | N=18; CI clips zero |
| H042 | Vol-of-Vol Calm | 0.106 | +0.338 | killed | highly inconsistent folds |

---

## Batch 8 — H043–H047: Calendar Refinements + Cross-Asset Hybrids ✓ COMPLETE

**Result:** 2 of 5 kept. Adding cross-asset filters to calendar signals selectively improved both Sharpe and Max DD. H043 (SellInMayBondConfirm) achieved the **best Max DD in the program at −10.11%** by skipping bond risk-off winters. H046 (HalloweenMomentumConfirm) achieved the **best Sharpe in the calendar family at 0.707**.

| # | Name | OOS Sharpe | Status | Key finding |
|---|------|-----------|--------|-------------|
| H043 | Sell in May + Bond Confirm | 0.577 | **★ kept** | best Max DD in program (−10.11%); anchor of low-DD strategy set |
| H044 | January Barometer + Gold Filter | 0.374 | killed | CI clips zero; N=14 trades; gold filter too restrictive |
| H045 | Turn-of-Month + Low Vol Gate | 0.490 | killed | just below benchmark; inconsistent sensitivity |
| H046 | Halloween + Momentum Confirm | **0.707** | **kept** | best Sharpe in calendar family; consistent folds |
| H047 | Pre-Earnings Drift | 0.234 | killed | CI contains zero; no sensitivity improvement |

---

## Batch 9 — H048–H050: Ensembles ✓ COMPLETE

**Result:** All 3 kept. The ensemble batch validated the core hypothesis: diversification through min-corr selection significantly lifts Sharpe. Equal-weight (H048) lifts from avg 0.61 to 0.800 through noise reduction alone. Min-corr selection (H049) at 0.959 is the target portfolio.

**Key insight:** Sharpe-weighting (H050: 0.820) barely beats equal-weight (H048: 0.800). The dominant effect is variance reduction through diversification, not quality-weighting. Selecting fewer, more orthogonal strategies (H049) beats weighting more correlated ones.

| # | Name | OOS Sharpe | Max DD | Status | Key finding |
|---|------|-----------|--------|--------|-------------|
| H048 | Equal-Weight Ensemble | 0.800 | −23.35% | **kept** | 16 components; noise cancellation lifts Sharpe from 0.614 to 0.800 |
| H049 | Min-Corr Ensemble | **0.959** | **−18.26%** | **★★ kept** | 4 components; best ensemble |
| H050 | Sharpe-Weighted Ensemble | 0.820 | −22.94% | **kept** | marginal improvement over equal-weight |

**H049 target ensemble components:**

| Strategy | OOS Sharpe | Role |
|----------|-----------|------|
| H021 RiskParity | 1.029 | Anchor — highest Sharpe, low corr via bond allocation |
| H046 HalloweenMomentumConfirm | 0.707 | Calendar + momentum; low corr with risk parity |
| H019 SPYGLDRotation | 0.630 | Gold flight-to-safety signal; uncorrelated with bond-based H021 |
| H035 GoldFearGauge | 0.570 | Fear detection complement to H019 |

All pairwise OOS correlations < 0.50. Ensemble H001 corr: +0.697.

---

## Highest-conviction hypotheses — retrospective

Original predictions vs outcomes:

| Original pick | Prediction | Actual outcome |
|---------------|------------|----------------|
| H036 Equity vs Safe-Haven Momentum | Strong decorrelation | killed (Sharpe 0.49) |
| H038 Long-Term Reversal | Strongest decorrelation candidate | killed (N=7 zero folds) |
| H039 Deep Drawdown Crisis Buyer | Negative H001 corr | killed (N=6 trades) |
| H043 Sell in May + Bond Confirm | H001 corr < 0.30 | kept (0.334 — best among kept) |
| H049 Min-Correlation Ensemble | Destination portfolio | **KEPT ★★ — Sharpe 0.959** |

The prediction for H049 was correct. The key surprise: contrarian decorrelation (H038, H039) failed due to trade sparsity, not signal quality.
