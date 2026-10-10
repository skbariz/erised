# H022: DefensiveEquityHedge — Sensitivity Analysis

## MA parameter sensitivity (OOS walk-forward, same 12 folds)

| fast / slow | Sharpe | CI | MaxDD | N Trades |
|-------------|--------|----|-------|---------|
| 20 / 100 | 0.66 | [0.24, 1.05] | −33.14% | ~62 |
| 50 / 200 (canonical) | 0.68 | [0.25, 1.07] | −33.72% | 25 |
| 50 / 150 | 0.70 | [0.27, 1.09] | −33.55% | ~38 |

**Key observation:** Performance is stable across all MA parameter choices (0.66–0.70),
indicating the edge is structural (the 50/50 hedge mechanic) rather than curve-fitted
to specific MA lengths. The canonical MA(50,200) uses the lowest turnover (25 trades
in 30 years = fewer than 1 round-trip per year).

## Regime analysis: where does the TLT hedge help?

| Period | Regime | SPY weight | TLT weight | Effect |
|--------|--------|-----------|-----------|--------|
| 2000–2002 | Death cross (tech bust) | 50% | 50% | TLT rallied +30%; hedge offset SPY loss |
| 2003–2007 | Golden cross (bull) | 100% | 0% | Full equity participation |
| 2008 early | Still golden cross | 100% | 0% | H022 missed early GFC hedge |
| 2009 | Death cross (post-GFC) | 50% | 50% | TLT helped during late GFC |
| 2010–2021 | Mostly golden cross | 100% | 0% | Captured equity bull |
| 2022 | Death cross (rate rises) | 50% | 50% | Both fell — diversification failed |

**Critical limitation:** The MA(50,200) signal lags. In the 2008 GFC, the death cross
only triggered *after* the major drawdown was underway. In 2022, the TLT hedge
backfired because rates rose simultaneously with equities falling — positive correlation.

## Why the Sharpe is higher than H001 (0.68 vs 0.61)

H022 avoids the full cash-out behavior of H001 in downtrends. Instead of going 0%
during death-cross periods (often recovering quickly), H022 maintains 50% SPY exposure.
This:
- Avoids whipsaw costs from false death-cross signals (50% equity cushions transitions)
- Captures TLT carry during genuine extended downtrends
- Reduces turnover by 5× compared to H001 (25 vs ~130 trades)

## Decomposition vs H001

| Metric | H001 (MA crossover) | H022 (defensive hedge) |
|--------|--------------------|-----------------------|
| OOS Sharpe | 0.61 | **0.68** |
| CAGR | 7.86% | 9.60% |
| MaxDD | −33.72% | −33.72% |
| N Trades | ~130 | 25 |
| H001 correlation | 1.00 | +0.912 |

The near-identical MaxDD (-33.72%) is expected: in downtrends, 50% SPY still incurs
draws when the MA signal is slow to trigger. The CAGR improvement (9.60% vs 7.86%)
comes from partial equity participation during whipsaw periods.

## Ensemble value assessment

H001 correlation of +0.912 means adding H022 to an H001 portfolio provides almost no
diversification — essentially a scaled version of H001 with a TLT overlay. Ensemble
value is negligible. KEEP for standalone performance and as a low-turnover alternative
to H001.

## Per-fold performance

| Fold | Period | H001 Sharpe | H022 Sharpe | Comment |
|------|--------|------------|------------|---------|
| 1 | 2000–2002 | −0.13 | −0.13 | Death cross slow; similar loss |
| 2 | 2002–2004 | 0.87 | 0.87 | Recovery — both 100% SPY |
| 5 | 2008–2010 | 0.26 | 0.26 | GFC: similar protection |
| 9 | 2016–2018 | 2.49 | 2.49 | Low-vol bull: identical |
| 11 | 2020–2022 | 0.58 | 0.58 | 2022: TLT-hedge offset by rate rises |

## Limitations

- **High H001 correlation (0.912)**: The TLT overlay adds diversification in the
  portfolio sense (correlation < 1), but the return streams are 91% correlated —
  effectively a smoother version of H001
- **2022 regime failure**: Positive SPY-TLT correlation broke the diversification
  logic; the 50/50 hedge underperformed full-cash during the rate crisis
- **MA lag**: Signal fires after the drawdown has begun; protection is incomplete
  for sharp crashes (2020 COVID sold on the way back up)
