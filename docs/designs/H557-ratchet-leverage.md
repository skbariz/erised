# H557: Ratchet-Conditional Leverage on H551

## Hypothesis

H556 proved vol-targeting leverage works (avg 1.23x realized, Sharpe 1.790, CAGR 9.61%).
H557 tests a different leverage mechanism: **drawdown-conditional leverage** (ratchet).
Instead of scaling leverage by volatility, scale it by how far the portfolio is below its peak:

```
leverage_t = 1.0 + (max_lev - 1.0) * max(0, 1 - dd_{t-1} / dd_limit)
```

At peak (dd=0): full max_lev. At dd_limit: back to 1.0x. Beyond: stays at 1.0x.
Drawdown tracked on the BASE (unleveraged H551) portfolio to avoid feedback circularity.

**Why this should work differently from H556:** Vol-targeting de-levers when realized vol
spikes — which often happens mid-crash, after the first big down day. The ratchet de-levers
based on cumulative loss, which fires earlier when losses are steady but vol is moderate
(e.g., a slow grind-down like 2000-2002). The two mechanisms are complementary:
- H556 (vol-target): fast response to vol spikes, slower to accumulating quiet losses
- H557 (ratchet): cumulative-loss response, regardless of vol

**Success bar:** Sharpe > 1.70, MaxDD < −6%, CAGR ≥ 8%. Should beat fixed 1.5x from H555
and ideally approach H556's Sharpe 1.790.

## Expected behavior by regime

- **2000-2002 tech crash**: Slow equity grind. H551 loses gradually (TLT offsets but can't
  fully neutralize). Ratchet fires early as base-portfolio dd accumulates → leverage cut
  proportionally → MaxDD stays contained.
- **2008-2009 GFC**: Sharp drop then TLT rally. Base-portfolio dd likely small due to TLT hedge.
  Ratchet may barely fire — which is correct (H551 was largely unaffected by GFC).
- **2020 COVID**: Brief but deep spike. Ratchet fires hard for a few weeks, then recovers.
  Similar to vol-target but with less whipsaw.
- **2013-2017 low-vol**: Base portfolio at or near peak consistently → full max_lev throughout.

## Parameters

Sensitivity grid:
- `dd_limit`: [0.03, 0.05, 0.07, 0.10, 0.15]
- `max_lev`: [1.5, 2.0]

Borrow cost: 50 bps/yr on leveraged fraction (same as H556).

## What would falsify this

- MaxDD worse than fixed 1.5x (ratchet fires too late or too slowly)
- Sharpe below H556 (vol-targeting is the dominant mechanism; ratchet adds no incremental value)
- Ratchet fires so frequently it hurts CAGR (over-triggered in normal drawdowns of a low-DD portfolio)

---

## Results

*(To be filled after run)*
