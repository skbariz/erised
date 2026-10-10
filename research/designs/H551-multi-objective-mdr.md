# H551: Multi-Objective MDR Ensemble

## Hypothesis

Replace the pure Max Diversification Ratio (MDR) objective with a composite score
`w1×Sharpe + w2×Calmar + w3×Sortino` optimized via SLSQP over the same 55-strategy
pool used by H543. The MDR objective maximizes diversification but is agnostic to
the shape of returns — it does not directly target drawdown control. By explicitly
incorporating Calmar (CAGR/MaxDD) and Sortino (downside-risk-adjusted return) into
the optimizer's objective, we expect the optimizer to find weights that trade a small
fraction of diversification for meaningfully lower drawdowns.

Pool is identical to H543: 37 AA strategies + 18 CS strategies (55 total).
Four weight vectors are grid-searched; the best-Calmar vector achieving Calmar ≥ 0.90
while keeping Sharpe as high as possible is used for the final summary.

## Expected behavior by regime

- **2000-2002 tech crash**: Calendar strategies (H043, H078) and TLT-heavy risk parity
  provide crash protection. Calmar objective should further tilt toward these.
- **2008-2009 GFC**: Drawdown-sensitive objective should increase TLT allocation during
  crisis; H079 (crisis protocol) and H097 (correlation regime) likely receive higher weight.
- **2020 COVID**: Short sharp drawdown — Calmar sensitive to this. Expect optimizer to
  favor H086 (vol-target) and H087 (ratchet) which proved robust here.
- **Low-vol 2013-2017**: Pure Sharpe environment. Calmar/Sortino objectives may slightly
  under-allocate to trend strategies that underperform in low-vol regimes.

## Success criteria

- OOS Sharpe ≥ 1.5 (match H543 1.578 approximately)
- CI lower bound > 0
- MaxDD ≤ −10%
- Calmar ≥ 0.90 (currently ~0.84–0.91 on champion strategies)
- Sortino ≥ 1.3

## Falsification

If all four weight vectors produce lower Sharpe than H543 (1.578) without a compensating
Calmar improvement (≥ 0.90), then the multi-objective approach degrades the MDR portfolio
and pure MDR remains superior. The hypothesis is falsified if the best-Calmar vector
achieves < 0.90 Calmar AND < 1.50 Sharpe simultaneously.

## Grid

| w1 (Sharpe) | w2 (Calmar) | w3 (Sortino) | Label              |
|-------------|-------------|--------------|---------------------|
| 1.0         | 0.0         | 0.0          | Pure Sharpe         |
| 0.5         | 0.5         | 0.0          | Sharpe-Calmar       |
| 0.4         | 0.4         | 0.2          | Primary target      |
| 0.33        | 0.33        | 0.33         | Equal weights       |

## Results

*(To be filled after run)*

Decision: TBD
