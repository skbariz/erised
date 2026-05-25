# H158: DrawdownCircuitBreaker — Momentum with Drawdown-Based Exposure Cutoff

## Hypothesis

Momentum crashes are characterized by rapid, severe drawdowns — the portfolio loses 15–25% in weeks. A circuit breaker that reduces exposure when the momentum portfolio is in deep drawdown prevents "doubling down" behavior. H087 (RatchetDrawdownProtection, Sharpe 1.010, MaxDD -10.93%) proved this mechanism on H021. H158 applies the same drawdown-triggered exposure reduction to the cross-sectional momentum portfolio.

**Signal:** Build H150 (12-1 momentum, top quintile). Track portfolio's trailing 252-day drawdown. Exposure: 1.0 when dd > -5%; 0.5 when -15% < dd ≤ -10%; 0.25 when dd ≤ -15%. Resume full only when drawdown recovers above -5% (hysteresis).

## Expected Regime Behavior

- **2000–2002 tech crash:** Momentum portfolio enters deep drawdown in 2000. Circuit breaker fires at -10% → cuts to 50%, then -15% → cuts to 25%. Misses some of the recovery rally too (slow to reopen). Net: better Max DD.
- **2008–2009 GFC:** Portfolio drawdown hits -15%+ by late 2008 → near-quarter exposure through the worst months. Recovery in 2009 is slow (drawdown takes time to recover to -5%). Net: better crash protection, slightly slower recovery capture.
- **2013–2017 low-vol bull:** No significant drawdowns → circuit breaker never fires → H158 ≈ H150 in this regime.
- **2020 COVID:** Sharp -15%+ drawdown in March 2020 → circuit breaker fires. Recovers by May–June 2020. Brief improvement vs H150.
- **2022 bear:** Momentum was actually profitable in 2022 (energy). Drawdown stays moderate → circuit breaker rarely fires → H158 ≈ H150.

## Success Criteria

- OOS Sharpe > H150 (circuit breaker should not drag Sharpe significantly)
- Max DD materially better than H150 (the primary expected benefit)
- CI lower bound > 0
- Hysteresis test: circuit breaker fires in 2008 and 2020 (verify via fold 5 and fold 11 improvement)

## Falsification

- Sharpe ≤ H150 − 0.10 → the circuit breaker fires during normal drawdowns and misses the recovery; it's cutting exposure at market bottoms
- Max DD no better → the circuit breaker is too slow to fire before the worst losses occur

## Implementation

- Class: `DrawdownCircuitBreaker(CrossSectionalStrategy)` in `src/quant/strategies/cross_sectional.py`
- Params: `long_days=252`, `skip_days=21`, `top_quantile=0.20`, `dd_soft=-0.10`, `dd_hard=-0.15`, `dd_resume=-0.05`, `scale_soft=0.5`, `scale_hard=0.25`, `lookback=252`, `monthly_rebalance=True`
- Mechanism: uses unscaled H150 portfolio returns to track drawdown; scales weights accordingly with hysteresis
- Runner: `scripts/run_h158.py`

## Results

*(to be filled after backtest)*
