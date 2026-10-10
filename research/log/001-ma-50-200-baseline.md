\# Hypothesis 001: MA(50,200) crossover as long-only trend baseline



\*\*Status:\*\* kept (as baseline)

\*\*Date opened:\*\* 2026-05-21

\*\*Date closed:\*\* 2026-05-21

\*\*Strategy family:\*\* trend

\*\*Instrument(s):\*\* SPY



\## Hypothesis



A simple 50-day / 200-day moving average crossover, going long when the

50-day MA is above the 200-day MA and flat otherwise, produces positive

risk-adjusted excess return versus SPY buy-and-hold by avoiding sustained

drawdowns.



\## Prior



The "golden cross / death cross" is one of the oldest documented trend

signals. It works in theory because equity drawdowns tend to be persistent

(the worst losses cluster after the market has already turned), so a slow

trend filter exits before the bottom of typical bear markets. Realistic

expectations from the literature: roughly equal CAGR to buy-and-hold, with

materially lower volatility and drawdown.



\## Test design



\- Sample period: 1993-01-29 to 2022-12-30 (research set; holdout untouched)

\- Validation: walk-forward, 5-year train / 2-year test, 12 folds spanning

&#x20; 1998-01-29 through 2022-01-23

\- Costs: 0.5 bps commission + 1.0 bps slippage = 1.5 bps per unit turnover

\- Holding horizon: indefinite within signal regime

\- Position sizing: binary, full-notional long (+1.0) or flat (0.0)



\## Results



\*\*Combined out-of-sample (6,036 periods across 12 folds):\*\*



| Metric | Strategy | SPY benchmark | Difference |

|---|---|---|---|

| Sharpe | 0.61 | 0.51 | \*\*+0.10\*\* |

| CAGR | 7.86% | 8.35% | −0.49 pp |

| Annualized vol | 13.91% | 19.66% | −5.75 pp |

| Max drawdown | −33.72% | −55.92% | \*\*−22.20 pp\*\* |

| Calmar | 0.23 | 0.15 | +0.08 |

| Hit rate | 55.20% | 54.54% | +0.66 pp |

| Total trades | 25 | 0 | low turnover |

| Turnover/day | 0.40% | n/a | |



\*\*Bootstrap 95% CI on OOS Sharpe: \[0.18, 1.02]\*\* (10,000 i.i.d. resamples)



\*\*In-sample reference (full research period 1993–2022, 7,536 periods):\*\*

Sharpe 0.71 vs SPY 0.58, max DD −33.72% vs −55.19%. Modestly better than OOS,

consistent with mild parameter-fit on the choice of 50/200 (these are

canonical numbers, so the "fitting" is conventional rather than empirical

in our hands).



\## Stress tests



Not performed for this baseline. The 50/200 parameter pair is taken as a

convention, not optimized. Sensitivity to alternative pairs will be tested

as a separate hypothesis if/when we explore the parameter space.



\## Decision



\*\*Decision:\*\* Kept as the long-o

