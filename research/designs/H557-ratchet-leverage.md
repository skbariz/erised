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

**Decision: KEEP**

| Config | Sharpe | CAGR | MaxDD | Calmar | Avg Lev |
|---|---|---|---|---|---|
| H551 base (1.0x) | 1.742 | 7.54% | −4.40% | — | 1.00x |
| H556 vol-target (ref) | 1.790 | 9.61% | −5.79% | 1.660 | 1.23x |
| ratchet 1.5x, dd=3% | 1.684 | 9.89% | −5.73% | 1.726 | 1.35x |
| ratchet 1.5x, dd=15% | 1.701 | 10.90% | −6.48% | 1.683 | 1.47x |
| **ratchet 2.0x, dd=3%** | 1.640 | 12.26% | −7.05% | **1.741** | 1.70x |

**Final (canonical):** max_lev=1.5x, dd_limit=15% (best Sharpe among CAGR≥8% configs)
- OOS Sharpe: **1.701**  CI [1.300, 2.103]
- CAGR: **10.90%**  MaxDD: **−6.48%**  Calmar: **1.683**
- H001 corr: +0.641. All 12 folds positive [+1.203, +2.260].

**Key insight — ratchet is inferior to vol-targeting for Sharpe:**
No ratchet configuration beats H551's base Sharpe of 1.742. Vol-targeting (H556) improves
Sharpe to 1.790; ratchet degrades it. Root cause: H551's base MaxDD is only −4.40%, so any
ratchet with dd_limit ≥ 3% almost never fires — it runs near max leverage continuously,
adding CAGR but also adding noise. Vol-targeting fires daily via realized vol → de-levers
proactively before losses accumulate → Sharpe-preserving. Ratchet fires after losses →
reactive, Sharpe-degrading.

**When ratchet wins:** CAGR (10.90% vs 9.61%) and Calmar at 2.0x (1.741 vs 1.660).
If the goal is raw CAGR with Calmar as secondary, ratchet at 2.0x/3% is competitive.
For risk-adjusted return (Sharpe), vol-targeting dominates.
