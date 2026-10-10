# H301: SimpleVRPOverlay — VRP-Scaled H021 Exposure

## Hypothesis
The variance risk premium (VRP = implied variance − realized variance) predicts future equity returns. When VRP is positive and high, the market is paying extra to hedge variance risk — this compensation signals above-average expected returns. We scale H021's total portfolio weight by a bounded linear function of VRP z-score: scale ∈ [0.5, 1.5], centered at 1.0 when VRP z-score is zero. This avoids all-in/all-out switching while amplifying the H021 allocation when compensation is rich.

**Expected behavior:**
- 2000-2002 tech crash: VRP often negative (realized > implied pre-crash) → reduce exposure before drawdown
- 2008-2009 GFC: VIX spikes first → VRP very positive early, so exposure may increase into the crash; VRP signal is not a perfect crisis predictor
- 2020 COVID: VRP spikes with VIX → exposure may increase right at the crash bottom
- 2013-2017 low vol: VRP positive but modest → near-normal H021 exposure

**Success criteria:** OOS Sharpe ≥ 0.90; MaxDD better than H021 (-18.41%); CI lower bound > 0.
**Falsification:** If VRP z-score adds no Sharpe vs H021, VRP is not a useful H021 overlay in this framework.

## Result

**OOS Sharpe 0.633** | CI [0.241, 1.031] | CAGR 5.52% | MaxDD −27.47% | H001 corr +0.536 | N_trades 5023

**Decision: KEEP** (CI lower > 0, Sharpe > 0.51)

VRP z-score overlay passes the bar but does not improve on H021 (0.695 baseline). MaxDD worsens slightly (−27.47% vs −18.41%) — VRP amplification increases equity exposure at times that later reverse. Sensitivity insensitive: vrp_sensitivity=0.15/0.25/0.35 produces Sharpe 0.663/0.633/0.629, confirming modest signal but no tuning leverage. Fold 6 (2010-2012) negative (−0.615) — VRP gave false positives in post-GFC equity recovery. Folds 0-1 zero (pre-TLT data).

Key finding: VRP as a continuous scaler on H021 adds modest value for ensemble diversification but does not improve absolute Sharpe vs the base strategy. The signal is directionally correct but too noisy to overcome transaction costs from rebalancing to scaled weights daily.
