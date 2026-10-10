# Hypothesis 002: Short-term mean reversion on SPY after large down moves

**Status:** killed

**Date opened:** 2026-05-21

**Date closed:** 2026-05-21

**Strategy family:** mean reversion

**Instrument(s):** SPY


## Hypothesis

On days following a >2σ down move (return normalized against a 20-day rolling
distribution), the next 1–5 days show positive average returns due to short-term
flow imbalances. The signal should be uncorrelated with the MA(50,200) trend
baseline, providing ensemble diversification.


## Prior

Short-term mean reversion after large down moves is well-documented in the
academic literature (e.g., De Bondt & Thaler 1985, Lehmann 1990). Empirically,
oversold equities tend to bounce within 1–5 trading days as selling pressure
exhausts and market makers widen then close spreads. The mechanism weakens as
more participants exploit it — several practitioners report the signal fading
significantly post-2010 as systematic vol-selling and retail order flow
normalized the pattern.


## Signal design

**Parameters (canonical):** lookback=20, threshold=2.0, hold_days=5

At close of day t, compute:

```
r_t = (adj_close_t / adj_close_{t-1}) - 1
mu_{t-1}  = rolling_mean(r, 20)[t-1]   # shifted: does not include today
sig_{t-1} = rolling_std(r, 20)[t-1]    # shifted: does not include today
z_t       = (r_t - mu_{t-1}) / sig_{t-1}

entry_t   = 1  if z_t < -2.0  else 0
position_t = rolling_max(entry, 5)[t]   # hold for 5 bars after any entry
```

The `.shift(1)` on rolling stats is the key lookahead-avoidance discipline: the
distribution we compare today's return against is built from yesterday and
earlier only. Position sizing is binary (0 or 1, long-only). The backtester
applies a one-bar shift before computing returns, so position_t earns r_{t+1}.


## Test design

- Sample: 1993-01-29 to 2022-12-30 (research set; holdout untouched)
- Validation: walk-forward, 5-year train / 2-year test, 12 folds,
  identical fold boundaries to H001 (enables direct OOS correlation computation)
- Costs: 0.5 bps commission + 1.0 bps slippage = 1.5 bps per unit turnover
- Position sizing: binary 0/1
- No-lookahead verification: `assert_no_lookahead` utility + 3 deterministic
  unit tests (signal fires on correct bar, hold expires on correct bar,
  flat during warm-up)


## Results

### Canonical OOS (lookback=20, threshold=2.0, hold_days=5)

Combined out-of-sample (6,036 periods across 12 folds):

| Metric         | H002       | SPY benchmark | Difference |
|----------------|------------|---------------|------------|
| Sharpe         | 0.24       | 0.51          | −0.27      |
| CAGR           | 1.85%      | 8.35%         | −6.50 pp   |
| Ann vol        | 9.67%      | 19.66%        | −10.0 pp   |
| Max drawdown   | −28.04%    | −55.92%       | −27.88 pp  |
| Calmar         | 0.07       | 0.15          | −0.08      |
| Hit rate       | 47.67%     | 54.54%        | −6.87 pp   |
| Total trades   | 370        | 0             |            |
| Turnover/day   | 6.13%      | n/a           |            |

**Bootstrap 95% CI on OOS Sharpe: [−0.17, 0.65]** (10,000 i.i.d. resamples)

The CI includes zero substantially. There is no statistically reliable evidence
of a positive risk-adjusted return from this signal at these parameters.

### Per-fold OOS Sharpe (regime analysis)

| Fold | Test window                | H002  | SPY   |
|------|----------------------------|-------|-------|
| 0    | 1998-01-29 to 2000-01-29   | +0.69 | +0.94 |
| 1    | 2000-01-29 to 2002-01-28   | −0.69 | −0.31 |
| 2    | 2002-01-28 to 2004-01-28   | +1.28 | +0.20 |
| 3    | 2004-01-28 to 2006-01-27   | −0.05 | +0.77 |
| 4    | 2006-01-27 to 2008-01-27   | +0.17 | +0.33 |
| 5    | 2008-01-27 to 2010-01-26   | +0.35 | −0.06 |
| 6    | 2010-01-26 to 2012-01-26   | +0.12 | +0.67 |
| 7    | 2012-01-26 to 2014-01-25   | +1.61 | +1.51 |
| 8    | 2014-01-25 to 2016-01-25   | +0.18 | +0.46 |
| 9    | 2016-01-25 to 2018-01-24   | +1.29 | +2.40 |
| 10   | 2018-01-24 to 2020-01-24   | −0.22 | +0.73 |
| 11   | 2020-01-24 to 2022-01-23   | −0.03 | +0.75 |

**Regime observation:** The per-fold Sharpe is highly erratic — not a clean
pre-/post-2010 decay pattern. Strong folds (2, 7, 9) interleave with negative
ones (1, 3, 10). The concern about post-2010 weakening is partially borne out
(folds 10 and 11 both negative or flat), but the pre-2010 performance is also
inconsistent. This is noise, not a signal with a regime shift.

### Parameter sensitivity (1-D, others at canonical)

Varying **lookback** (threshold=2.0, hold_days=5):

| lookback | OOS Sharpe |
|----------|-----------|
| 10       | 0.30      |
| **20**   | **0.24**  |
| 30       | 0.13      |

Varying **threshold** (lookback=20, hold_days=5):

| threshold | OOS Sharpe |
|-----------|-----------|
| 1.5       | 0.36      |
| **2.0**   | **0.24**  |
| 2.5       | 0.33      |

Varying **hold_days** (lookback=20, threshold=2.0):

| hold_days | OOS Sharpe |
|-----------|-----------|
| 3         | 0.06      |
| **5**     | **0.24**  |
| 7         | 0.24      |

The canonical (20, 2.0, 5) is neither the best nor worst; Sharpes shift by
±0.1–0.3 across the grid without any meaningful structure — consistent with
noise rather than a signal with exploitable curvature.

### Full grid stability summary (27 combinations)

| Metric                  | Value               |
|-------------------------|---------------------|
| N combinations          | 27                  |
| Median OOS Sharpe       | 0.24                |
| IQR                     | [0.14, 0.34]        |
| Count Sharpe > 0.5      | 0 / 27              |
| Min / Max               | 0.06 / 0.44         |

**Not a single parameter combination in the 27-point grid produces OOS Sharpe
above 0.5.** The canonical (0.24) is median across the grid. This rules out
the possibility that we chose bad canonical parameters — the whole space is weak.

### Correlation with H001 (MA 50/200, same OOS folds)

| Measure                        | Value  |
|--------------------------------|--------|
| Pearson (net returns)          | +0.546 |
| Pearson (positions)            | +0.090 |
| Ensemble diversification check | FAIL   |

Returns correlation is +0.546, well above the 0.30 threshold. Position
correlation is low (+0.09), reflecting different signal mechanics, but returns
are driven by shared equity beta exposure when both strategies are long. This
signal fails on both of the criteria needed for ensemble inclusion: it is
neither strong on its own nor uncorrelated with H001.


## Stress tests

Not performed. The canonical result is too weak (Sharpe 0.24, CI contains zero,
0/27 grid combinations above 0.5) to justify further parameter stress testing.


## Decision

**Decision:** Killed. Two independent kill conditions met:

1. **Weak signal.** OOS Sharpe 0.24 with 95% CI [−0.17, 0.65]. Zero of 27
   parameter combinations reaches Sharpe 0.5. The signal is not reliably
   positive over the research period.

2. **High returns correlation with H001.** Return Pearson correlation +0.546
   exceeds the 0.30 ensemble threshold. Adding H002 to an ensemble with H001
   does not provide genuine diversification at the returns level.

### What we learned

The mean reversion intuition is directionally plausible but the 5-day hold
period is likely too long: post-bounce, the strategy overstays and gives back
gains in the subsequent drift-down. An impulse-response analysis (not done
here) might show reversion peaks at 1–2 days. A future variant could use
a shorter hold (1–2 days) at the cost of higher turnover.

The high returns correlation with H001 is structural: both are long-only SPY
strategies. Any long-only SPY strategy will have correlated returns with any
other long-only SPY strategy unless it has near-zero invested time. This argues
for testing strategies that are short-capable or use assets uncorrelated with
SPY equity beta (e.g., bonds, vol) as the next ensemble dimension.

### Suggested follow-ons

- **Hypothesis 003:** Shorter hold (1–2 day) mean reversion — test whether
  reversion is faster than the 5-day assumption here.
- **Hypothesis 004 or later:** Strategies on a different beta (bonds, gold,
  vol) to escape the long-equity-correlation trap.
