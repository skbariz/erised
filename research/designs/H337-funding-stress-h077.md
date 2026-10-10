# H337: FundingStressH077 -- RiskParityMomentumFilter (H077) + funding stress override

## Hypothesis
H077 (RiskParityMomentumFilter) applies H021 when SPY 12m momentum is positive, else switches to 100% TLT. This momentum gate handles secular bear markets well (2000-2002, 2008 — momentum turns negative early) but may be slow to respond to sudden liquidity crises where SPY momentum is still positive when the shock hits. This strategy adds a second, faster-firing crisis signal: if VIX > 30 AND HYG 63d return < LQD 63d return (credit stress), force the portfolio to 80% TLT / 20% SPY regardless of the momentum gate. The momentum gate and the stress override address different tail scenarios — the gate filters slow bear markets; the override catches sudden liquidity crises. Pre-HYG era (pre-2007): override requires only VIX > 30.

**Expected behavior:**
- 2000-2002: SPY 12m momentum turns negative → H077 already in TLT; stress override redundant but harmless
- 2008-2009: Momentum turns negative in late 2008; stress override fires earlier when VIX>30 + HYG stress → earlier protection
- 2013-2017 low vol: VIX rarely > 30; no stress; momentum positive → identical to H077
- 2020 COVID: VIX spikes to 80+ in March; HYG stress; override fires → 80% TLT regardless of momentum

**Success criteria:** OOS Sharpe ≥ H077 (target ≥ 0.7); MaxDD improvement especially in 2008 and 2020; CI lower > 0.
**Falsification:** Stress override fires too late (after momentum gate has already triggered) or fires falsely during temporary vol spikes, degrading returns in normal markets.

## Results
*[To be filled after backtest]*
Decision:
