# Strategy Results — Master Leaderboard

Updated after each hypothesis run. All metrics are **walk-forward out-of-sample**
(research period 1993–2022). H100 holdout (2023–2026) evaluated once — sealed period now spent.
Folds: 5-year train / 2-year test, 12 non-overlapping folds. Costs: 0.5 bps commission + 1.0 bps slippage.

SPY benchmark (same OOS windows): Sharpe 0.51 | CAGR 8.35% | Max DD −55.92%

---

## Leaderboard

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr | Notes |
|---|------|--------|--------|-----------|--------|------|--------|-----------|-------|
| 001 | MA(50,200) crossover | trend | kept | 0.61 | [0.18, 1.02] | 7.86% | −33.72% | — | baseline |
| 002 | Mean reversion 2σ SPY | mean rev | killed | 0.24 | — | — | — | +0.55 | CI contains zero |
| 003 | Short-hold mean rev (2d) | mean rev | killed | 0.01 | — | — | — | — | near-zero |
| 004 | RSI oversold entry | mean rev | killed | 0.06 | — | — | — | — | CI contains zero |
| 005 | Consecutive down days | mean rev | killed | 0.44 | — | — | — | +0.40 | CI positive; fails ensemble |
| 006 | VIX-filtered reversion | mean rev | killed | 0.01 | — | — | — | — | VIX filter adds no value |
| 007 | 12-1m TS momentum | momentum | kept | 0.62 | [0.21, 1.01] | 8.48% | −33.72% | +0.892 | H001 corr too high |
| 008 | Absolute dual momentum | momentum | kept | 0.62 | [0.22, 1.02] | 8.02% | −31.23% | +0.863 | H001 corr too high |
| 009 | MACD crossover (12,26,9) | trend | killed | 0.37 | — | — | — | — | CI lower negative |
| 010 | Fast MA crossover (10/50) | trend | killed | 0.45 | — | — | — | — | dominated by H001 |
| 011 | ADX-filtered MA crossover | trend | killed | 0.27 | — | — | — | — | CI contains zero |
| 012 | Realized vol regime | vol | killed | 0.33 | — | — | — | — | CI contains zero |
| 013 | MA(50,200) long/short | trend L/S | killed | 0.36 | [−0.04, 0.77] | 5.32% | −40.16% | +0.707 | shorting hurts vs flat; CI negative lower |
| 014 | TS momentum long/short | momentum L/S | killed | 0.44 | [0.04, 0.83] | 6.85% | −40.54% | +0.647 | below SPY benchmark; shorting degrades H007 |
| 015 | Dual MA L/S (20/100) | trend L/S | killed | 0.08 | [−0.33, 0.50] | −0.28% | −51.00% | +0.213 | near-zero; whipsaw dominates; lowest H001 corr |
| 016 | Volatility-scaled trend | risk mgmt | kept | 0.55 | [0.14, 0.95] | 5.33% | −26.30% | +0.788 | best max DD in program; no ensemble value |
| 017 | Trend L/S + VIX gate | trend / hybrid | killed | 0.43 | [0.03, 0.83] | 5.92% | −39.10% | +0.701 | below SPY benchmark; VIX gate too rare to help |
| 018 | SPY/TLT dual momentum | multi-asset rotation | kept | 0.57 | [0.17, 0.98] | 7.34% | −33.72% | +0.687 | beats benchmark; high H001 corr limits ensemble value |
| 019 | SPY/GLD rotation | multi-asset rotation | kept | 0.63 | [0.22, 1.03] | 9.68% | −33.72% | +0.788 | highest CAGR in rotation batch; too correlated for ensemble |
| 020 | Best-of-3 momentum | multi-asset rotation | killed | 0.47 | [0.06, 0.86] | 6.04% | −31.29% | +0.329 | Sharpe < 0.51; early folds starved of TLT/GLD data |
| 021 | SPY/TLT risk parity | risk parity | **kept ★** | **1.03** | **[0.62, 1.44]** | 7.69% | **−18.41%** | +0.407 | **FIRST OOS Sharpe ≥ 1.0; lowest MaxDD in program** |
| 022 | Defensive equity hedge | multi-asset trend | kept | 0.68 | [0.25, 1.07] | 9.60% | −33.72% | +0.912 | best trend Sharpe; near-zero diversification value |
| 023 | Sell in May | calendar | kept | 0.53 | [0.12, 0.93] | 6.71% | −34.82% | +0.505 | confirmed Bouman & Jacobsen; lower corr than all prior kept strategies |
| 024 | Turn of Month (3d) | calendar | killed | 0.41 | [0.02, 0.82] | 3.72% | −34.41% | +0.363 | Sharpe < 0.51; n_days=5 version (0.54) narrowly beats benchmark |
| 025 | January Barometer | calendar | kept | 0.53 | [0.12, 0.94] | 5.47% | −29.53% | +0.565 | second-best MaxDD (−29.53%); avoids bear years via Jan filter |
| 026 | Quarter-End Drift (5d) | calendar | killed | 0.30 | [−0.12, 0.72] | 1.42% | **−10.13%** | **+0.176** | lowest MaxDD and H001 corr in program; CI negative — standalone fails |
| 027 | Seasonality + MA trend | hybrid calendar | kept | 0.61 | [0.19, 1.03] | 5.70% | −33.72% | +0.712 | matches H001 Sharpe; too correlated for ensemble |
| 028 | VIX spike entry (30, 20d) | VIX contrarian | killed | 0.36 | [−0.07, 0.75] | 4.34% | −44.68% | +0.341 | below benchmark; long during crash continuation |
| 029 | VIX regime filter (<20) | VIX regime | killed | 0.43 | [0.01, 0.84] | 3.39% | −25.53% | +0.562 | below benchmark; CI barely positive; good MaxDD |
| 030 | VIX trend (L/S, MA 20d) | VIX L/S | killed | −0.09 | [−0.50, 0.31] | −3.65% | −83.24% | **−0.284** | negative Sharpe; first negative H001 corr in program |
| 031 | Realized vs implied vol | VIX regime | **kept** | **0.58** | **[0.16, 0.96]** | 8.52% | −51.43% | +0.693 | vol risk premium filter; best CAGR in batch |
| 032 | VIX mean reversion (2σ, 60d) | VIX contrarian | killed | −0.001 | [−0.43, 0.40] | −0.52% | −35.32% | +0.420 | near-zero Sharpe; indistinguishable from zero |
| 033 | Bond trend equity timer | cross-asset macro | killed | 0.44 | [0.04, 0.85] | 2.96% | −14.15% | +0.442 | Sharpe below benchmark; excellent MaxDD −14.15% |
| 034 | SPY-TLT correlation regime | cross-asset macro | kept | 0.55 | [0.13, 0.95] | 7.96% | −55.92% | +0.562 | MaxDD = SPY (−55.92%) — held long during 2008; limited ensemble value |
| 035 | Gold fear gauge | cross-asset macro | **kept** | 0.57 | [0.15, 0.97] | 5.51% | −41.55% | **+0.384** | best H001 corr of batch; sensitivity drops past 50d MA |
| 036 | Equity vs safe-haven momentum | cross-asset macro | killed | 0.49 | [0.10, 0.91] | 3.79% | −22.02% | +0.477 | just misses benchmark; good MaxDD |
| 037 | Dual safe-haven confirmation | cross-asset macro | killed | 0.22 | [−0.17, 0.62] | 0.83% | −11.70% | **+0.264** | lowest H001 corr in program; too restrictive — 5 zero folds |
| 038 | Long-term reversal | contrarian | killed | 0.243 | [−0.179, 0.627] | 2.37% | −46.86% | **+0.165** | 7 zero folds; very low H001 corr but inactive; 1512d lookback best |
| 039 | Deep drawdown crisis buyer | contrarian | killed | 0.061 | [−0.353, 0.450] | −0.04% | −56.47% | **+0.133** | N=6 trades, 9 zero folds; too few trades to be viable |
| 040 | Short-term weekly reversal | contrarian | killed | 0.481 | [0.061, 0.880] | 6.28% | −43.70% | +0.480 | just misses benchmark; p15 threshold lifts Sharpe to 0.532 |
| 041 | Post-crash recovery | contrarian | killed | 0.405 | [−0.001, 0.806] | 3.55% | −31.27% | **+0.108** | CI clips zero; N=18; vix=35 gives CI > 0 but Sharpe 0.485 |
| 042 | Vol-of-vol calm | contrarian | killed | 0.106 | [−0.300, 0.525] | 0.43% | −17.49% | +0.338 | highly inconsistent fold-by-fold; sensitive to vov_window |
| 043 | Sell-in-May + bond confirm | calendar-hybrid | **kept** | 0.577 | [0.168, 0.976] | 2.67% | **−10.11%** | +0.334 | best MaxDD of any kept strategy (−10.11%); lower corr than H023 (+0.505) |
| 044 | January barometer + gold filter | calendar-hybrid | killed | 0.374 | [−0.027, 0.781] | 3.10% | −29.53% | +0.399 | CI clips zero; N=14 trades; gold filter too restrictive |
| 045 | Turn-of-month + low vol gate | calendar-hybrid | killed | 0.490 | [0.080, 0.903] | 3.08% | −15.63% | +0.371 | just below benchmark; vix<35 gives 0.555 but sensitivity inconsistent |
| 046 | Halloween + momentum confirm | calendar-hybrid | **kept** | **0.707** | **[0.300, 1.119]** | 6.04% | −23.65% | +0.579 | best Sharpe in calendar family; consistent folds; H001 corr too high for ensemble |
| 047 | Pre-earnings drift | calendar | killed | 0.234 | [−0.177, 0.638] | 1.09% | −25.66% | **+0.151** | CI contains zero; no sensitivity shows improvement |
| 048 | Equal-weight ensemble | ensemble | **kept** | 0.800 | [0.378, 1.207] | 7.43% | −23.35% | +0.914 | 16 components equal-weight; noise cancellation lifts Sharpe from 0.614 to 0.800 |
| 049 | Min-corr ensemble | ensemble | **kept ★★** | **0.959** | **[0.543, 1.357]** | 7.59% | **−18.26%** | +0.697 | **Best ensemble: 4 components (H021+H046+H019+H035); all pairwise corr < 0.50** |
| 050 | Sharpe-weighted ensemble | ensemble | **kept** | 0.820 | [0.401, 1.228] | 7.50% | −22.94% | +0.914 | Sharpe weighting negligible improvement over equal-weight |
| 051 | 3-asset risk parity (SPY+TLT+GLD) | risk parity | **kept** | 0.626 | [0.227, 1.038] | 7.33% | −47.67% | +0.477 | GLD adds little vs H021; MaxDD worse due to no-cash early folds |
| 052 | Commodity momentum overlay | cross-asset macro | killed | 0.271 | [−0.152, 0.668] | 2.80% | −46.01% | +0.369 | DBC data from 2006; 4 zero folds; CI contains zero |
| 053 | Credit spread regime (HYG vs LQD) | cross-asset macro | killed | 0.337 | [−0.064, 0.754] | 2.57% | −42.98% | +0.401 | HYG data from 2007; fold 4 Sharpe −0.941; CI contains zero |
| 054 | Yield curve proxy (TLT vs SHY) | cross-asset macro | killed | 0.409 | [−0.002, 0.810] | 5.02% | −41.31% | +0.474 | CI lower bound just −0.002; 42d lookback gives CI > 0 |
| 055 | Five-asset rotation | multi-asset rotation | killed | 0.402 | [−0.007, 0.807] | 5.78% | −50.14% | +0.549 | CI clips zero; 5-asset dilution hurts vs focused rotation |
| 056 | Yield curve rotation (SPY/TLT) | multi-asset rotation | killed | 0.419 | [−0.002, 0.822] | 5.65% | −52.77% | +0.377 | CI lower −0.002; 126d lookback gives 0.540 [0.127, 0.919] |
| 057 | Bond + credit dual confirm | cross-asset macro | killed | 0.499 | [0.090, 0.904] | 3.17% | **−12.34%** | +0.430 | Sharpe 0.499 just misses 0.51 benchmark; MaxDD excellent; 150d MA gives 0.577 |
| 058 | Inverted curve treasury ladder | multi-asset rotation | killed | 0.400 | [−0.015, 0.796] | 5.34% | −56.56% | +0.469 | CI clips zero; MaxDD worse than SPY; ladder to TLT+IEF doesn't protect |
| 059 | Macro fear composite (4-signal) | cross-asset macro | killed | 0.321 | [−0.078, 0.723] | 2.56% | −47.79% | +0.547 | CI contains zero; composite noise outweighs signal aggregation benefit |
| 060 | Carry+momentum SPY/TLT | multi-asset rotation | **kept** | **0.531** | **[0.137, 0.934]** | 5.21% | −23.37% | **+0.294** | **First kept strategy below +0.30 H001 corr; 126d lookback gives 0.632** |
| 061 | Value/growth rotation (IVW/IVE) | equity factor | killed | 0.470 | [0.045, 0.850] | 7.70% | −58.45% | +0.688 | Sharpe below benchmark; MaxDD worse than SPY; curve regime signal too coarse |
| 062 | Small-cap cyclical (IWM) | equity factor | killed | 0.494 | [0.066, 0.882] | 8.36% | −56.78% | +0.696 | Sharpe 0.494 just misses 0.51; credit+momentum filters still mostly SPY-correlated |
| 063 | Defensive sector overlay (XLP) | equity factor | **kept** | **0.577** | **[0.170, 0.972]** | 8.39% | −39.66% | +0.752 | VIX+SPY-MA regime overlay; MaxDD -39.66% significant improvement; robust across params |
| 064 | Tech/defensive momentum (QQQ/XLU) | equity factor | killed | 0.406 | [−0.006, 0.800] | 6.05% | −45.59% | +0.640 | CI clips zero; canonical 252d lookback only N=10 trades; 126d gives 0.641 [0.270, 1.014] |
| 065 | Equal-weight defensive factor | equity factor | **kept** | **0.554** | **[0.136, 0.948]** | 8.19% | −49.19% | +0.671 | Static 25/25/25/25 SPY+XLP+XLU+IVE; no timing; N=3 trades (buy-and-hold) |
| **066** | **VIX risk parity enhanced** | **vol regime** | **kept** | **0.990** | **[0.585, 1.391]** | 7.64% | **−18.99%** | **+0.318** | H021 + VIX spike override (VIX>35→80/20 TLT/SPY); near-H021 Sharpe with low H001 corr |
| 067 | VIX percentile regime | vol regime | killed | 0.305 | [−0.098, 0.702] | 1.51% | −18.92% | +0.365 | CI contains zero; quantile=40% gives 0.518 but post-OOS param; fold variance extreme |
| **068** | **VRP + calendar hybrid** | **vol regime** | **kept** | **0.575** | **[0.166, 0.997]** | 6.07% | −27.13% | **+0.481** | H031 VRP × H043 Nov-Apr calendar; lower H001 corr than H031 (+0.693) as expected |
| 069 | VIX term structure proxy | vol regime | killed | 0.476 | [0.065, 0.868] | 4.73% | −51.41% | +0.361 | Sharpe below 0.51 benchmark; MaxDD worse than SPY; slope signal too noisy as proxy |
| **070** | **Multi-asset dynamic vol target** | **vol regime** | **kept ★★** | **1.066** | **[0.663, 1.476]** | 7.18% | **−12.28%** | +0.414 | **New program star: H021 weights × portfolio vol-scaling; beats H021 Sharpe+MaxDD** |
| 071 | Triple confirm calendar | calendar | killed | 0.508 | [0.096, 0.918] | 2.24% | **−10.34%** | +0.319 | Sharpe 0.508 misses 0.51 by 0.002; CI solidly positive; near-H043 MaxDD record |
| 072 | Halloween + credit confirm | calendar | killed | 0.491 | [0.079, 0.897] | 2.42% | **−9.82%** | +0.366 | Sharpe misses; MaxDD −9.82% (program best individual!); credit=90d gives 0.669 but post-OOS |
| 073 | Sell-in-May + yield curve confirm | calendar | killed | 0.351 | [−0.058, 0.751] | 0.62% | **−7.22%** | **+0.126** | CI clips zero; **MaxDD −7.22% would be program best**; yield curve filter too restrictive |
| 074 | Pre-holiday drift | calendar | killed | 0.158 | [−0.262, 0.505] | 0.67% | −11.64% | **+0.098** | **Lowest H001 corr in program (+0.098)**; CI contains zero; holiday alpha eaten by costs |
| 075 | Earnings season drift | calendar | killed | 0.288 | [−0.110, 0.679] | 2.35% | −48.57% | +0.262 | CI contains zero; index-level earnings drift doesn't survive after costs |
| **076** | **Four-asset risk parity** | **multi-asset** | **kept** | **0.666** | **[0.252, 1.070]** | 4.65% | −19.27% | +0.338 | SPY+TLT+GLD+DBC inverse-vol parity; folds 0-2 zero (pre-GLD/DBC era) |
| **077** | **Risk parity + momentum filter** | **multi-asset** | **kept** | **0.683** | **[0.287, 1.085]** | 8.01% | −27.58% | +0.478 | H021 with per-asset 6m momentum gate; goes cash when both negative |
| **078** | **Seasonal risk parity** | **calendar** | **kept ★★** | **0.762** | **[0.360, 1.161]** | 7.55% | −20.66% | **+0.008** | **Program record H001 corr +0.008**; H021 Nov-Apr + 100% TLT May-Oct |
| **079** | **Crisis protocol risk parity** | **multi-asset** | **kept** | **0.952** | **[0.554, 1.350]** | 7.71% | −21.18% | +0.254 | H021 + drawdown+VIX hysteresis override to 90/10 TLT/SPY in crises |
| **080** | **Risk parity yield curve overlay** | **multi-asset** | **kept** | **0.891** | **[0.472, 1.295]** | 6.99% | −22.58% | +0.257 | H021 + SHY>TLT inversion cuts SPY weight by 50%; spy_scale=0.75 gives 0.975 |
| 081 | Dual-frequency momentum | momentum | killed | 0.238 | [−0.164, 0.638] | 1.78% | −24.61% | +0.552 | CI negative lower; 1m filter adds whipsaw vs H007; mom_short=42 gives 0.567 but post-OOS |
| **082** | **Risk-adjusted momentum** | **momentum** | **kept** | **0.616** | **[0.217, 1.013]** | 9.10% | −38.33% | +0.475 | Highest risk-adj score (return/vol) wins SPY/TLT/GLD; MaxDD high but Sharpe/CI pass |
| 083 | Contrarian multi-asset rotation | momentum | killed | 0.326 | [−0.071, 0.744] | 2.61% | −28.66% | **+0.073** | CI clips zero; H001 corr +0.073 (2nd lowest for kept class); fold 8 kills: −1.413 |
| **084** | **Momentum + carry blend** | **momentum** | **kept** | **0.548** | **[0.140, 0.941]** | 6.15% | −29.43% | +0.337 | 50% top 12-1m momentum + 50% TLT carry (TLT>SHY return); stable across blends |
| 085 | Momentum + macro gate | momentum | killed | 0.445 | [0.044, 0.855] | 6.24% | −41.59% | +0.364 | Sharpe below SPY benchmark; yield curve gate forces TLT at wrong times; MaxDD poor |
| **086** | **Low-vol target portfolio** | **vol regime** | **kept ★★** | **1.109** | **[0.707, 1.519]** | 7.04% | **−9.93%** | +0.413 | H021 × vol scale (min(1, 8%/port_vol)); **program MaxDD record −9.93%**; robust across vol targets |
| **087** | **Ratchet drawdown protection** | **drawdown control** | **kept** | **1.010** | **[0.608, 1.427]** | 5.94% | **−10.93%** | +0.417 | H021 × ratchet scale (1+dd/limit); cuts exposure proportional to portfolio drawdown depth |
| **088** | **H001+H021 direct blend** | **blend** | **kept** | **0.871** | **[0.453, 1.271]** | 7.86% | −21.61% | +0.924 | 50% H001 momentum + 50% H021 risk parity; high H001 corr from MA signal dominance |
| **089** | **H043+H021 regime blend** | **blend** | **kept ★** | **1.074** | **[0.667, 1.478]** | 5.23% | **−9.62%** | +0.477 | 50% H043 seasonal + 50% H021 risk parity; **program MaxDD record −9.62%**; stable across blends |
| **090** | **Monthly rebalancing risk parity** | **multi-asset** | **kept** | **1.020** | **[0.605, 1.425]** | 7.76% | −17.89% | +0.426 | H021 rebalanced monthly (N=231 trades); near-identical Sharpe to H021; robust across lookbacks |
| **091** | **Enhanced min-corr ensemble v2** | **ensemble** | **kept** | **0.984** | **[0.571, 1.391]** | 7.08% | −15.38% | +0.784 | Greedy min-corr (0.50) over all 35 keepers; selects H086+H046+H019+H035+H060 (5 components) |
| **092** | **Regime-adaptive ensemble** | **ensemble** | **kept** | **0.964** | **[0.553, 1.370]** | 6.94% | −19.97% | +0.750 | H049 components with daily VIX+curve regime weights; marginal over H049 equal-weight (0.959) |
| **093** | **Ensemble anti-momentum rebalance** | **ensemble** | **kept** | **0.852** | **[0.440, 1.262]** | 7.67% | −21.93% | +0.696 | H049 components; monthly anti-momentum (upweight 24m underperformers); below H049 baseline |
| **094** | **H049 + best new component** | **ensemble** | **kept ★** | **1.084** | **[0.668, 1.493]** | 7.70% | **−15.03%** | +0.726 | H049 + H078 (SeasonalRiskParity, lowest avg corr to H049 = +0.209); 5-component equal weight |
| **095** | **Max diversification portfolio** | **ensemble** | **kept ★★** | **1.133** | **[0.727, 1.557]** | 5.81% | **−7.16%** | +0.665 | **Program MaxDD record −7.16%**; SLSQP MDR over 35 strategies; concentrates on H043+H078+H035 |
| 096 | PCA risk regime | regime | killed | 0.323 | [−0.083, 0.719] | 3.49% | −34.45% | **+0.140** | CI contains zero; binary PCA switcher; strong 2020 fold but inconsistent across others |
| **097** | **Correlation risk parity hybrid** | **regime** | **kept** | **0.992** | **[0.591, 1.402]** | 7.78% | −18.48% | +0.329 | H021 + positive SPY-TLT corr crisis override to 90% TLT; robust across corr lookback params |
| **098** | **Reversion entry risk parity** | **regime** | **kept** | **0.879** | **[0.468, 1.282]** | 7.14% | −21.24% | **+0.196** | H021 + heavy-TLT (80%) on SPY drawdown >10%; crisis_tlt=0.70 gives Sharpe 0.956/MaxDD −17.38% |
| **099** | **Batch 19 augmented min-corr** | **ensemble** | **kept** | **0.958** | **[0.578, 1.368]** | 6.61% | −17.97% | +0.724 | Greedy min-corr (0.50) over 38 candidates (35+H096/H097/H098); selects H086+H046+H019+H035+H060+H096 |
| 100 | Holdout evaluation (H049) | holdout | special | 1.346 | [0.246, 2.392] | 11.39% | −9.91% | +0.879 | **FINAL HOLDOUT** H049 (2023-01-03 – 2026-05-20, 848 days); SPY holdout: 1.465/23.46%/−18.76%; H049 MaxDD −9.91% vs SPY −18.76% |

---

## Goal trackers

| Target | Current best | Status |
|--------|-------------|--------|
| OOS Sharpe ≥ 1.0 | **1.133 (H095)** / **1.109 (H086)** / **1.084 (H094)** / **1.03 (H021)** | **MET ★** |
| Max DD < SPY (−55.92%) | **−7.16% (H095)** / **−9.62% (H089)** / **−9.93% (H086)** / **−10.11% (H043)** | **MET ★★** |
| Holdout evaluation | **H049 Sharpe 1.346 / MaxDD −9.91%** (SPY: 1.465 / −18.76%) | **COMPLETE ★** — H049 holds up; MaxDD advantage maintained |
| Strategies tested ≥ 100 | **100** | Batch 19 complete (4/5 individual kept — H096 killed, H097/H098 kept; H099 ensemble kept; H100 holdout) |
| Strategies kept (ensemble) | 47 (+ H097, H098, H099 from Batch 19; H096 killed) | Program complete — 100 hypotheses tested |

---

## H150–H200 Program: Cross-Sectional Equity + Terminal Ensemble

H101–H149 intentionally skipped. Program pivoted to cross-sectional equity (H150–H199) then cross-program ensemble (H200).
All metrics are **walk-forward OOS (1993–2022)** using same fold structure. H001 corr = correlation with H001 MA(50,200) crossover.
Note: cross-sectional strategies record `n_trades=1` as artifact; actual rebalance counts are ~350/yr (monthly) or ~1500/yr (weekly) × ~100 stocks per rebalance.

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr | Notes |
|---|------|--------|--------|-----------|--------|------|--------|-----------|-------|
| 150 | Momentum long-only | CS momentum | **kept** | 0.981 | [0.578, 1.396] | 20.38% | −52.12% | +0.714 | 12-1m momentum top quintile, equal-weight, monthly rebalance |
| 151 | Momentum market-neutral | CS momentum L/S | killed | 0.099 | [−0.301, 0.508] | 0.46% | −70.66% | +0.235 | L/S 50bps/yr borrow + 2.5bps/turn; ~zero market beta; collapses to noise |
| 152 | High-water momentum | CS momentum | **kept** | 0.850 | [0.447, 1.253] | 13.60% | −46.03% | +0.710 | price/52w-high anchoring signal (George & Hwang 2004) |
| 153 | Sector-neutral momentum | CS momentum | **kept** | 0.941 | [0.534, 1.354] | 18.31% | −51.46% | +0.678 | 12-1m momentum ranked within GICS sector; top 30% per sector |
| 154 | Trend-gated momentum | CS momentum | **kept** | 0.962 | [0.568, 1.367] | 14.84% | −27.40% | +0.723 | H150 gated by SPY MA200/MA100; MaxDD −27% vs −52% for H150 |
| 155 | Smooth momentum | CS momentum | **kept** | 0.976 | [0.573, 1.389] | 20.23% | −52.08% | +0.715 | 12-1m × fraction positive days (path-consistent, Da et al. 2014) |
| 156 | Jump-penalized momentum | CS momentum | **kept** | 0.949 | [0.552, 1.365] | 20.34% | −54.70% | +0.706 | 12-1m trimmed return (exclude top/bottom 5 daily returns) |
| **157** | **Vol-scaled momentum** | **CS momentum** | **kept ★** | **1.079** | **[0.685, 1.495]** | 14.15% | **−25.39%** | +0.700 | H150 + vol-target overlay (H086 mechanism on CS momentum); N=5868 |
| **158** | **Drawdown circuit breaker** | **CS momentum** | **kept ★** | **1.077** | **[0.681, 1.487]** | 17.44% | **−27.55%** | +0.748 | H150 + hysteresis drawdown circuit breaker (H087 mechanism); N=119 |
| **159** | **Smooth vol-target momentum** | **CS momentum** | **kept ★** | **1.072** | **[0.676, 1.485]** | 14.05% | **−25.23%** | +0.701 | H155 smooth signal + H157 vol-target; dual protection |
| 160 | Large-cap value | CS value | **kept** | 0.730 | [0.333, 1.138] | 9.76% | −43.51% | +0.605 | B/M + E/P composite, point-in-time EDGAR XBRL; coverage ~2009+ |
| 161 | Five-year reversal | CS contrarian | **kept** | 0.850 | [0.446, 1.253] | 18.32% | −55.02% | +0.562 | Buy 5-year losers (De Bondt-Thaler 60-month formation) |
| 162 | Composite value+momentum | CS contrarian | **kept** | 0.905 | [0.503, 1.319] | 17.60% | −48.51% | +0.680 | 50/50 12-1m momentum + negated 36m reversal (Asness-Moskowitz-Pedersen) |
| 163 | Residual momentum | CS momentum | **kept** | 0.961 | [0.563, 1.380] | 21.24% | −60.08% | +0.649 | Cumulative CAPM residual; removes market-beta loading from signal |
| 164 | Near 52-week low | CS contrarian | **kept** | 0.884 | [0.489, 1.294] | 17.07% | −48.17% | +0.564 | Buy stocks nearest to 52-week low (price/rolling_min) |
| 165 | Gross profitability rank | CS quality | **kept** | 0.878 | [0.469, 1.287] | 11.03% | −34.46% | +0.620 | Long top 20% by GP/A within GICS sector (Novy-Marx 2013) |
| 166 | Low accruals quality | CS quality | **kept** | 0.824 | [0.425, 1.238] | 10.54% | −34.52% | +0.634 | Long bottom 20% by accruals = (NI−OCF)/|TA| (Sloan 1996) |
| 167 | Piotroski high bar | CS value | **kept** | 0.591 | [0.201, 0.997] | 9.88% | −55.68% | +0.372 | Top B/M quintile filtered by F-score ≥ 7; too restrictive |
| **168** | **Composite quality score** | **CS quality** | **kept ★★** | **1.181** | **[0.773, 1.594]** | 18.55% | **−40.44%** | +0.660 | QMJ-inspired: profitability (GP/A, ROE, cash-ROE) + growth + safety |
| 169 | Quality+value composite | CS quality+value | **kept** | 0.839 | [0.433, 1.250] | 10.32% | −37.11% | +0.628 | (1/3)*z(B/M) + (1/3)*z(GP/A) + (1/3)*z(−accruals) |
| 170 | FF5 investment factor | CS investment | **kept** | 0.721 | [0.322, 1.131] | 8.43% | −38.41% | +0.575 | Long bottom 20% by 1-yr total asset growth; FF5 CMA factor |
| 171 | Operating profitability FF5 | CS quality | **kept** | 0.864 | [0.447, 1.271] | 10.83% | −36.29% | +0.621 | Long top 20% by GP/BE within sector; FF2015 RMW factor |
| 172 | FF5 composite | CS multi-factor | **kept** | 0.757 | [0.353, 1.170] | 9.45% | −39.03% | +0.617 | (1/3)*z(B/M) + (1/3)*z(GP/BE) + (1/3)*z(−ΔCTA); FF5 HML+RMW+CMA |
| 173 | Return on assets | CS quality | **kept** | 0.785 | [0.382, 1.188] | 9.31% | −35.82% | +0.634 | Long top 20% by NI/TA within sector |
| 174 | Cash operating profitability | CS quality | **kept** | 0.766 | [0.365, 1.176] | 9.32% | −38.86% | +0.626 | Long top 20% by OCF/BE within sector; cash-based OP filters accruals |
| **175** | **Betting against beta** | **CS low-beta** | **kept ★** | **1.088** | **[0.681, 1.497]** | 16.32% | **−40.75%** | +0.638 | Long bottom 20% by 252d beta vs SPY; Frazzini-Pedersen (2014) BAB |
| 176 | Low-volatility anomaly | CS low-vol | **kept** | 0.877 | [0.466, 1.293] | 13.35% | −40.48% | +0.660 | Long bottom 20% by 252d realized vol within sector |
| 177 | Vol-targeted momentum | CS momentum | **kept** | 0.937 | [0.535, 1.346] | 17.86% | −49.53% | +0.727 | 12-1m momentum with per-stock vol-targeting |
| **178** | **Defensive quality momentum** | **CS quality+momentum** | **kept ★** | **1.125** | **[0.721, 1.539]** | 19.11% | **−46.95%** | +0.664 | (1/2)*z(GP/A + low-beta) + (1/2)*z(12-1m momentum) |
| **179** | **Low-beta quality** | **CS low-beta+quality** | **kept ★★** | **1.198** | **[0.788, 1.609]** | 18.16% | **−40.44%** | +0.644 | **(1/2)*z(−beta) + (1/2)*z(GP/A), sector-neutral; BAB + Novy-Marx composite** |
| 180 | Earnings trend long | CS earnings drift | **kept** | 0.715 | [0.311, 1.125] | 8.31% | −37.19% | +0.581 | SUE proxy = (NI_t − NI_{t−252})/|NI_{t−252}|; EDGAR filing-date PIT |
| 181 | Revenue+earnings double growth | CS earnings drift | **kept** | 0.762 | [0.355, 1.172] | 8.70% | −36.80% | +0.586 | Both NI and revenue grew YoY (Jegadeesh-Livnat proxy) |
| 182 | Filing-date gap continuation | CS earnings / event | **kept** | 0.623 | [0.224, 1.027] | 8.69% | −42.49% | +0.538 | Long stocks with >2% price reaction on EDGAR filing date, hold 21d |
| 183 | Cash earnings drift | CS earnings / quality | **kept** | 0.687 | [0.287, 1.095] | 7.86% | −36.66% | +0.578 | NI YoY growth filtered to OCF/NI > 0.5 (Sloan 1996 accruals filter) |
| 184 | Earnings momentum composite | CS earnings+momentum | **kept** | 0.992 | [0.594, 1.400] | 18.66% | −46.21% | +0.687 | (z_SUE + z_mom_63d)/2 sector-neutral; combines PEAD + price momentum |
| **185** | **Weekly reversal sector-neutral** | **CS short-term reversal** | **kept ★** | **1.103** | **[0.703, 1.517]** | 24.88% | **−50.76%** | +0.613 | Long prior 5-day losers within sector, weekly rebalance (Lehmann 1990) |
| **186** | **Residual weekly reversal** | **CS short-term reversal** | **kept ★** | **1.095** | **[0.696, 1.513]** | 24.20% | **−51.71%** | +0.613 | H185 on idiosyncratic 5d return only (strips SPY beta) |
| **187** | **Liquidity-filtered reversal** | **CS short-term reversal** | **kept ★** | **1.068** | **[0.666, 1.483]** | 25.64% | **−53.17%** | +0.600 | H185 applied to top-50% by realized vol (Amihud illiquidity proxy) |
| 188 | Monthly reversal long | CS short-term reversal | **kept** | 0.949 | [0.550, 1.363] | 20.82% | −55.66% | +0.589 | Prior-month losers sector-neutral, monthly rebalance |
| **189** | **Residual OU mean reversion** | **CS stat arb** | **kept ★** | **1.025** | **[0.626, 1.439]** | 24.58% | **−49.77%** | +0.548 | Avellaneda-Lee (2010) OU z-score on 5d idiosyncratic return; N=3 |
| 190 | Sector pairs trading | CS pairs / stat arb | killed | 0.205 | [−0.183, 0.608] | 1.27% | −38.35% | +0.043 | Gatev et al. (2006) minimum-SSD pairs; CI contains zero |
| **191** | **PCA residual reversal** | **CS PCA stat arb** | **kept ★** | **1.135** | **[0.733, 1.553]** | 24.75% | **−48.11%** | +0.619 | K=10 PCA, strip factor exposure, 5d residual reversal, weekly rebalance |
| **192** | **PCA OU mean reversion** | **CS PCA stat arb** | **kept ★★** | **1.175** | **[0.776, 1.592]** | 27.53% | **−50.69%** | +0.567 | K=10 PCA, OU z-score on 5d cumulative residuals; entry z<−1.5 |
| 193 | Long-short momentum | CS market-neutral | killed | −0.157 | [−0.546, 0.245] | −1.22% | −41.27% | +0.116 | Market-neutral L/S momentum; L/S costs destroy long-only signal |
| 194 | Long-short reversal | CS market-neutral | **kept** | 0.829 | [0.439, 1.214] | 4.39% | −18.33% | **+0.146** | Market-neutral L/S reversal; H001 corr +0.146 — key cross-program diversifier |
| 195 | 52-week high momentum | CS momentum | **kept** | 0.878 | [0.475, 1.286] | 14.43% | −45.83% | +0.692 | George-Hwang (2004) price/52w-high ratio as momentum signal |
| 196 | Idiosyncratic momentum | CS momentum | **kept** | 0.930 | [0.526, 1.343] | 18.08% | −50.97% | +0.690 | 12-1m stock return minus sector-average return |
| 197 | Low skewness anomaly | CS risk anomaly | **kept** | 0.879 | [0.477, 1.287] | 16.97% | −51.84% | +0.612 | Long low-skewness stocks (Bali, Cakici, Whitelaw 2011) |
| 198 | Earnings acceleration | CS earnings drift | **kept** | 0.681 | [0.282, 1.088] | 7.60% | −36.08% | +0.584 | Second derivative of earnings: sue_t − sue_{t−63d} |
| 199 | Multi-signal composite | CS composite | **kept** | 1.020 | [0.622, 1.427] | 20.68% | −51.39% | +0.668 | Equal-weight z-score: momentum + reversal + earnings growth + quality |
| **200** | **Cross-program MDR portfolio** | **cross-program ensemble** | **kept ★★** | **1.350** | **[0.941, 1.737]** | 5.36% | **−8.83%** | +0.622 | **Terminal champion. SLSQP MDR over 46 strategies (37 AA + 9 CS). H194 weight 29.8%.** |

---

## H150–H200 Summary

| Target | Result |
|--------|--------|
| CS strategies tested | 50 (H150–H199) |
| CS strategies kept | 47 (killed: H151, H190, H193) |
| Terminal champion (H200) | Sharpe 1.350, MaxDD −8.83%, CAGR 5.4% |
| Best CS individual | H179 LowBetaQuality — Sharpe 1.198, MaxDD −40.4% |
| Key cross-program diversifier | H194 (market-neutral L/S reversal) — H001 corr +0.146, MaxDD −18.3% |
| H200 top component weights | H194 29.8%, H043 19.7%, H078 13.9%, H035 10.5%, H060 9.7% |

---

## H201–H250 — Sector/Industry Momentum, Fundamental Factors, Behavioral Finance

### Batch 1: Industry/Sector Momentum (H201–H205)


| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr | Notes |
|---|------|--------|--------|-----------|--------|------|--------|-----------|-------|
| 201 | Sector ETF momentum | sector momentum (AA) | **kept** | 0.654 | [0.273, 1.057] | 7.24% | −38.99% | +0.068 | Top-3 equity ETFs by 126d return, risk-parity weighted; TLT fallback |
| 202 | Industry-confirmed momentum | sector/industry momentum (CS) | **kept** | 0.908 | [0.507, 1.313] | 17.06% | −47.16% | +0.005 | 12-1m momentum gated by sector 6m trend; H001 corr near zero |
| 203 | Sector-leader long-only | sector/industry momentum (CS) | **kept** | 0.915 | [0.518, 1.325] | 15.30% | −46.69% | +0.001 | Top-2 GICS sectors by 126d return, equal-weight within; H001 corr ~0 |
| 204 | Industry relative strength | sector/industry momentum (CS) | **kept** | 0.824 | [0.421, 1.233] | 15.43% | −52.86% | +0.004 | 21d stock return minus sector-avg 21d return; sector-neutral top 20% |
| 205 | Sector-momentum risk parity | sector momentum (AA) | killed | 0.399 | [−0.004, 0.802] | 4.22% | −54.16% | +0.047 | Top-1 ETF + TLT risk parity; CI clips zero, MaxDD −54% unacceptable |

### Batch 2: Asset Growth & Net Share Issuance (H206–H210)

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr | Notes |
|---|------|--------|--------|-----------|--------|------|--------|-----------|-------|
| 206 | Low asset growth | asset growth (CS) | **kept** | 0.709 | [0.306, 1.123] | 7.97% | −39.25% | −0.019 | 2-yr TA growth; folds 0-5 zero (EDGAR ~2009+); H001 corr negative |
| 207 | Shareholder yield | capital discipline (CS) | **kept** | 0.768 | [0.360, 1.170] | 8.92% | −38.07% | −0.022 | Share count reduction (buyback) YoY; best MaxDD in batch; H001 corr −0.022 |
| 208 | Conservative quality | capital discipline (CS) | **kept** | 0.762 | [0.362, 1.174] | 9.19% | −35.97% | −0.020 | z(-asset_growth)+z(-net_issuance)+z(OCF/TA); fold 5 active, best MaxDD |
| **209** | **Buyback momentum** | **capital discipline+momentum (CS)** | **kept ★** | **0.930** | **[0.527, 1.344]** | 18.03% | −49.88% | **+0.002** | **z(-net_issuance)+z(12-1m mom); full 12-fold coverage; batch star** |
| 210 | Capital discipline | capital discipline (CS) | **kept** | 0.733 | [0.329, 1.141] | 8.44% | −37.64% | −0.019 | z(-asset_growth)+z(-net_issuance)+z(-leverage_change); 6 active folds |

### Batch 3: Accruals & Earnings Quality (H211–H215)

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr | Notes |
|---|------|--------|--------|-----------|--------|------|--------|-----------|-------|
| 211 | Balance-sheet accruals | accruals / earnings quality (CS) | **kept** | 0.800 | [0.396, 1.206] | 9.22% | −37.35% | −0.021 | Sloan (1996) BS approach: delta(NWC)/avg_TA; distinct from H166 CF approach |
| 212 | Dual momentum+earnings | accruals / earnings quality (CS) | **kept** | 0.587 | [0.184, 0.991] | 7.14% | −36.26% | −0.009 | AND condition: top 30% price rank AND NI YoY rank; lower Sharpe vs composite |
| **213** | **Earnings quality momentum** | **accruals / earnings quality (CS)** | **kept ★** | **0.907** | **[0.496, 1.316]** | 17.43% | −49.88% | **+0.001** | **z(-BS_accruals)+z(12-1m mom); full 12-fold; batch star** |
| 214 | Gross margin momentum | accruals / earnings quality (CS) | **kept** | 0.690 | [0.282, 1.099] | 8.52% | −34.55% | −0.014 | 12-1m mom gated by improving gross margin; 504d window stronger (0.907) |
| 215 | Fundamental reversal | accruals / earnings quality (CS) | **kept** | 0.718 | [0.318, 1.122] | 9.44% | −43.10% | −0.015 | Monthly reversal gated by NI>0 AND OCF>0; avoids value traps |

### Batch 4: Residual Momentum Variants (H216–H220)

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr | Notes |
|---|------|--------|--------|-----------|--------|------|--------|-----------|-------|
| **216** | **Sector residual momentum** | **residual momentum (CS)** | **kept ★** | **0.983** | **[0.581, 1.395]** | 19.92% | −51.97% | **+0.002** | **Strip market + sector beta; matches H163 Sharpe; batch star** |
| 217 | Residual mom vol-scaled | residual momentum (CS) | **kept** | 0.936 | [0.532, 1.353] | 17.76% | −51.41% | +0.001 | H163 + idiosyncratic vol-target; MaxDD sensitivity flat across target_vol |
| 218 | Intangible return reversal | residual momentum (CS) | **kept** | 0.897 | [0.494, 1.307] | 19.11% | −56.47% | −0.001 | Fade price>fundamental; MaxDD −56.47% slightly worse than SPY |
| 219 | Fundamental momentum | residual momentum (CS) | **kept** | 0.729 | [0.329, 1.139] | 8.56% | −39.05% | −0.021 | Pure accounting growth: no price signal; H001 corr −0.021 |
| 220 | Residual mom quality | residual momentum (CS) | **kept** | 0.958 | [0.558, 1.369] | 19.11% | −51.94% | +0.001 | H163 + low-accruals gate; slightly below H216 but stable sensitivity |

### Batch 5: Quality Factors (H221–H225)

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr | Notes |
|---|------|--------|--------|-----------|--------|------|--------|-----------|-------|
| 221 | ROE momentum | quality factors (CS) | **kept** | 0.717 | [0.309, 1.135] | 8.41% | −38.34% | −0.020 | YoY ROE improvement; folds 0-5 zero; H001 corr −0.020 |
| **222** | **Cash conversion quality** | **quality factors (CS)** | **kept ★** | **0.829** | **[0.423, 1.240]** | 10.24% | **−35.83%** | **−0.020** | **OCF/gross_profit ratio; fold 5 active; best MaxDD + Sharpe in batch** |
| 223 | Leverage reduction | quality factors (CS) | **kept** | 0.716 | [0.305, 1.118] | 8.31% | −38.22% | −0.022 | YoY decrease in LTD/TA; folds 0-5 zero; H001 corr −0.022 |
| 224 | Piotroski F-score | quality factors (CS) | **kept** | 0.752 | [0.346, 1.162] | 8.53% | −37.02% | −0.022 | 9-component binary quality score; folds 0-5 zero; consistent folds 6-11 |
| 225 | Earnings consistency | quality factors (CS) | **kept** | 0.692 | [0.288, 1.104] | 7.73% | −37.42% | −0.017 | z(1yr NI growth)+z(2yr NI growth); folds 0-5 zero; consistent growers |

### Batch 6: Value & Efficiency Factors (H226–H230)

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr | Notes |
|---|------|--------|--------|-----------|--------|------|--------|-----------|-------|
| 226 | Book-to-market momentum | value factors (CS) | **kept** | 0.766 | [0.364, 1.176] | 9.77% | −38.18% | −0.015 | z(BE/price)+z(12-1m mom); fold 5 active; value+momentum composite |
| **227** | **Gross profitability** | **value factors (CS)** | **kept ★** | **0.879** | **[0.469, 1.287]** | 11.03% | **−34.46%** | **−0.017** | **Novy-Marx (2013) GP/TA; top 10% quantile 0.922/−29.29%; batch star** |
| 228 | Revenue growth | value factors (CS) | **kept** | 0.760 | [0.357, 1.170] | 8.93% | −36.62% | −0.023 | YoY revenue growth; folds 0-5 zero; H001 corr −0.023 (strongest in batch) |
| 229 | Operating leverage | value factors (CS) | **kept** | 0.764 | [0.350, 1.178] | 9.00% | −34.79% | −0.016 | GP_growth−rev_growth; 504d window stronger (0.834); MaxDD −34.79% |
| 230 | Asset turnover | value factors (CS) | **kept** | 0.741 | [0.334, 1.149] | 8.52% | −36.32% | −0.021 | YoY rev/TA improvement; DuPont efficiency component; folds 0-5 zero |

### Batch 7: Composite Quality+Momentum Signals (H231–H235)

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr | Notes |
|---|------|--------|--------|-----------|--------|------|--------|-----------|-------|
| **231** | **Quality momentum** | **composite factors (CS)** | **kept ★** | **0.870** | **[0.460, 1.282]** | 11.24% | **−35.71%** | **−0.014** | **z(GP/TA)+z(12-1m mom); flat across mom windows 126/252/504d; batch star** |
| 232 | Value quality | composite factors (CS) | **kept** | 0.864 | [0.451, 1.274] | 11.00% | −35.61% | −0.019 | z(BE/price)+z(GP/TA); QMJ-style; top 10% MaxDD -32.40% |
| 233 | Growth quality | composite factors (CS) | **kept** | 0.803 | [0.386, 1.220] | 9.41% | −34.38% | −0.019 | z(rev_growth)+z(GP/TA); best MaxDD in batch; 504d window 0.815 |
| 234 | Multi-factor composite | composite factors (CS) | **kept** | 0.857 | [0.442, 1.266] | 10.48% | −35.76% | −0.013 | z(mom)+z(GP/TA)+z(-accruals)+z(-asset_growth); 4-factor composite |
| 235 | Efficient growth | composite factors (CS) | **kept** | 0.766 | [0.362, 1.170] | 8.79% | −34.51% | −0.022 | z(rev_growth)+z(OCF/TA); cash-backed growth; fold 11 strong 1.284 |

### Batch 8: Solvency & Profitability Factors (H236–H240)

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr | Notes |
|---|------|--------|--------|-----------|--------|------|--------|-----------|-------|
| 236 | Liquidity momentum | solvency factors (CS) | **kept** | 0.815 | [0.404, 1.229] | 9.51% | −36.93% | −0.016 | YoY current ratio (CA/CL) improvement; folds 0-5 zero; fold 7 Sharpe 2.14 |
| 237 | Debt coverage | solvency factors (CS) | **kept** | 0.729 | [0.329, 1.139] | 9.04% | −36.18% | −0.020 | OCF/LTD solvency ratio; zero-debt firms NaN; fold 7 Sharpe 2.06 |
| 238 | Net profit margin | solvency factors (CS) | **kept** | 0.696 | [0.293, 1.095] | 8.00% | −37.21% | −0.020 | YoY NI/revenue improvement; bottom-line efficiency trend; folds 0-5 zero |
| **239** | **Gross margin level** | **solvency factors (CS)** | **kept ★** | **0.861** | **[0.457, 1.269]** | 10.78% | **−34.59%** | **−0.012** | **Persistent GP/revenue level; pricing power/moat proxy; batch star** |
| 240 | ROA momentum | solvency factors (CS) | **kept** | 0.751 | [0.339, 1.167] | 8.66% | −36.54% | −0.020 | YoY ROA (NI/avg_TA) improvement; DuPont: margin × turnover; folds 0-5 zero |

### Batch 9: Market Value & Cash Flow Factors (H241–H245)

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr | Notes |
|---|------|--------|--------|-----------|--------|------|--------|-----------|-------|
| 241 | FCF yield | market value factors (CS) | **kept** | 0.758 | [0.358, 1.166] | 9.98% | −42.31% | −0.021 | OCF/market_cap; fold 5 active; top 10% stronger (0.833); MaxDD worst in batch |
| 242 | Cash return on equity | market value factors (CS) | **kept** | 0.766 | [0.365, 1.176] | 9.32% | −38.86% | −0.021 | OCF/book_equity; cash ROE avoids NI accruals; fold 5 active |
| **243** | **Gross profit growth** | **market value factors (CS)** | **kept ★** | **0.791** | **[0.381, 1.202]** | 9.47% | **−34.90%** | **−0.015** | **YoY absolute GP growth / TA; flat window sensitivity; best MaxDD; batch star** |
| 244 | Altman Z proxy | market value factors (CS) | **kept** | 0.764 | [0.362, 1.168] | 9.48% | −36.64% | −0.017 | z(NWC/TA)+z(NI/TA)+z(BE/TL)+z(rev/TA); fold 7 Sharpe 2.25; accounting-only |
| 245 | OCF momentum | market value factors (CS) | **kept** | 0.741 | [0.334, 1.146] | 8.64% | −37.99% | −0.018 | YoY OCF/TA improvement; cash-flow ROA analog; flat window sensitivity |

### Batch 10: Structural & Terminal Composite Factors (H246–H250)

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr | Notes |
|---|------|--------|--------|-----------|--------|------|--------|-----------|-------|
| 246 | Equity multiplier | structural factors (CS) | **kept** | 0.759 | [0.352, 1.166] | 9.27% | −36.51% | −0.017 | Low TA/BE = low financial leverage; DuPont component; top 10% MaxDD -33.49% |
| 247 | Retained earnings proxy | structural factors (CS) | **kept** | 0.783 | [0.376, 1.191] | 9.28% | −37.85% | −0.017 | YoY BE growth / TA; profitable AND retaining earnings; folds 0-5 zero |
| **248** | **Intangible quality** | **structural factors (CS)** | **kept ★** | **0.864** | **[0.446, 1.271]** | 10.83% | **−36.29%** | **−0.019** | **GP/book_equity; intangible asset intensity; fold 9 Sharpe 2.09; batch star** |
| 249 | Financial health composite | structural factors (CS) | **kept** | 0.588 | [0.180, 1.013] | 7.71% | −35.73% | −0.019 | z(OCF/TA)+z(-LTD/TA)+z(curr_ratio)+z(NI/TA); fold 5 negative; weakest batch |
| 250 | Grand fundamental composite | structural factors (CS) | **kept** | 0.742 | [0.329, 1.144] | 8.61% | −35.08% | −0.020 | 5-factor: z(GP/TA)+z(-accruals)+z(-asset_growth)+z(OCF/TA)+z(-LTD/TA); fold 6 low |

---

## H201–H250 Synthesis

| Target | Result |
|--------|--------|
| CS fundamental strategies tested | 50 (H201–H250) |
| CS fundamental strategies kept | **50/50 (100% keep rate)** |
| Best individual Sharpe | H216 SectorResidualMomentum — 0.983 |
| Best MaxDD | H243 GrossProfitGrowth — −34.90% |
| Worst MaxDD | H216 SectorResidualMomentum — −51.97% (full equity exposure) |
| H001 corr range | −0.012 to −0.023 (universally near-zero / slightly negative) |
| EDGAR active folds | Folds 6–11 only (2010–2022); folds 0–5 = 0.000 for all pure fundamental strategies |
| GitHub issues | #149–#207 |

### Signal family quality (approximate avg Sharpe, active folds)

| Family | Best strategy | Best Sharpe | Notes |
|--------|--------------|-------------|-------|
| Residual momentum (B4) | H216 SectorResidualMomentum | 0.983 | Full 12-fold coverage (price-based) |
| Gross profit family (B6+B8+B10) | H248 IntangibleQuality | 0.864 | GP/TA, GP/revenue, GP/BE all 0.861–0.879 |
| Composite quality+momentum (B3+B7) | H213 EarningsQualityMomentum | 0.907 | 2-factor composites consistently strong |
| Capital discipline (B2) | H209 BuybackMomentum | 0.930 | Only full-12-fold fundamental strategy |
| Solvency + profitability (B8) | H239 GrossMarginLevel | 0.861 | Level beats change in same family |
| Market value + cash flow (B9) | H243 GrossProfitGrowth | 0.791 | Flat window sensitivity = structural signal |
| Structural + terminal (B10) | H248 IntangibleQuality | 0.864 | 5-factor terminal composite underperforms |

### Key design lessons (H201–H250)

1. **Residual momentum is the strongest price-based signal.** H216 (0.983): strip market and sector beta before ranking. Full 12-fold coverage vs 6-fold for pure fundamentals. Best cross-sectional strategy in H201–H250.

2. **Buyback momentum (H209) is the only fundamental strategy with full 12-fold coverage.** Combining shares_outstanding change (buyback signal) with 12-1m price momentum means the price component fills pre-EDGAR folds. Any composite adding a price momentum component gains full coverage.

3. **Level-based profitability signals outperform change-based signals in the same family.** GP/revenue level (H239: 0.861) beats GP/revenue change gate (H214: 0.690). GP/TA level (H227: 0.879) outperforms operating leverage change (H229: 0.764). Persistent quality is a more stable predictor than recent improvement.

4. **Composite signals degrade above 2–3 factors.** 2-factor composites (H231 QualityMomentum: 0.870, H213 EarningsQualityMomentum: 0.907) consistently outperform 4–5-factor composites (H250 GrandFundamentalComposite: 0.742). EDGAR timing differences across sources cause fold-6 near-zero in multi-source composites.

5. **Flat sensitivity table = structural signal.** When Sharpe barely changes across 126/252/504d windows (e.g., H243 GrossProfitGrowth: 0.813/0.791/0.814), the signal is not parameter-sensitive and the canonical choice is justified without tuning risk.

6. **Market-cap linked signals have higher MaxDD.** FCFYield (H241, OCF/market_cap): MaxDD −42.31% vs GrossMarginLevel (H239, accounting-only): MaxDD −34.59%. Combining price with fundamentals introduces valuation-driven drawdowns absent from pure accounting signals.

---

## H251–H300 — Set 3: Tail Risk, Distress Avoidance, Accounting Sustainability

### Batch 1: Tail Risk and Lottery Avoidance (H251–H255)

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr | Notes |
|---|------|--------|--------|-----------|--------|------|--------|-----------|-------|
| 251 | IdiosyncraticVolFactor | tail risk | kept | 0.897 | [0.490, 1.306] | 13.51% | −41.42% | −0.003 | Residual vol from SPY OLS; 11/12 folds positive |
| 252 | MaxReturnLotteryAvoid | tail risk | kept | 0.927 | [0.522, 1.345] | 15.14% | −46.02% | −0.000 | Highest standalone Sharpe in batch; shorter window stronger |
| 253 | DownsideBetaFactor | tail risk | kept | 0.976 | [0.577, 1.392] | 19.50% | −46.74% | +0.002 | Best standalone signal; longer window monotonically better |
| 254 | TailRiskComposite | tail risk | kept | 0.911 | [0.506, 1.327] | 14.76% | −44.60% | −0.000 | 4-signal composite; dilutes H253; best MaxDD in batch |
| 255 | TailFilteredMomentum | tail risk | kept | 0.896 | [0.496, 1.313] | 16.31% | −49.46% | +0.000 | Momentum with lottery exclusion; underperforms H157 (1.079) |

**Batch 1 summary:** All 5 KEEP. H253 DownsideBetaFactor (0.976) is the standout — strongest
standalone signal, CI lower 0.577, robust to longer windows. H252 MaxReturnLotteryAvoid (0.927)
is second. H254 composite (0.911) shows dilution vs. H253 but still passes bar. H255 falsified:
lottery exclusion filter hurts momentum quality vs. H157 vol-scaling. H001 correlations
near-zero (−0.003 to +0.002) across all five — these cross-sectional tail-risk signals are
orthogonal to the macro trend-following universe.

---

_See `results/<id>-<name>/stats.md` for full metrics on each hypothesis._
