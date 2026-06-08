# H556: Dynamic Leverage via Vol-Targeting on H551

## Hypothesis

H555 confirmed that every top ensemble passes institutional bars at **fixed 1.5x leverage**.
The problem: fixed leverage amplifies drawdowns in stressed regimes — exactly when you don't want it.
H556 applies a vol-targeting overlay to H551's portfolio so leverage scales **inversely with realized volatility**:

```
leverage_t = min(max_lev, target_vol / realized_vol_{t-1})
```

In calm regimes (realized vol ≈ 4.3%), leverage ≈ 1.5x (capped). In crises (realized vol ≈ 10%),
leverage drops to ~0.65x automatically. This is the H086 mechanism (which improved H021's MaxDD by 8.5 pp)
applied one level up — to the already-optimized H551 ensemble.

**Why H551's weights, not a fresh optimization?** H551's optimal allocation (H065: 27%, H076: 26%,
H473: 17%, H157: 15%) is already determined out-of-sample. H556 tests a pure leverage modification —
keeping the component selection fixed isolates the vol-targeting effect cleanly.

## Expected behavior by regime

- **2000-2002 tech crash**: SPY vol spikes to 25-30%. Portfolio vol rises; leverage drops to 0.2-0.3x.
  Result: de-risked exposure preserves capital better than fixed 1.5x.
- **2008-2009 GFC**: Similar spike. H551 already has heavy TLT (via H076/H065); vol-targeting would
  further reduce exposure. Possible over-de-risking (TLT was up 25% in 2008).
- **2020 COVID**: Brief vol spike, rapid recovery. Dynamic leverage cuts and quickly restores.
- **2013-2017 low-vol**: Portfolio vol near 3-4%. leverage = min(1.5, target/3%) = 1.5x (capped).
  Full 1.5x benefit in calm bull markets.

## Parameters

Sensitivity grid:
- `target_vol`: [5%, 6%, 7%, 8%] — calibrates the "normal" leverage level
- `max_lev`: [1.25, 1.5, 2.0] — hard cap on leverage
- `vol_lookback`: 20 trading days (fixed, matches H086)

At target_vol=6.5%, max_lev=1.5: E[leverage] ≈ 1.5 in calm regimes.

## Success criteria

- OOS Sharpe > 1.60 (must beat fixed-lev approximation from H555)
- MaxDD < −10% (leverage must not materially worsen H551's -4.41%)
- CAGR ≥ 8% (the institutional bar)
- CI lower bound > 0

## What would falsify this

- MaxDD worse than fixed 1.5x (vol-targeting isn't cutting leverage fast enough)
- Sharpe drops vs H551 unleveraged (financing costs eat the leverage benefit)
- No configuration improves on H551 at fixed 1.5x across Sharpe AND MaxDD simultaneously

---

## Results

**Decision: KEEP**

| Config | Sharpe | CAGR | MaxDD | Calmar | Avg Lev |
|---|---|---|---|---|---|
| H551 base (1.0x) | 1.742 | 7.54% | -4.40% | — | 1.00x |
| tv=6.5%, cap=1.25x | 1.756 | 8.26% | -4.94% | 1.672 | 1.08x |
| tv=6.5%, cap=1.50x | 1.771 | 8.89% | -5.40% | 1.647 | 1.15x |
| **tv=6.5%, cap=2.00x** | **1.790** | **9.61%** | **-5.79%** | **1.658** | **1.23x** |
| tv=8.0%, cap=2.00x | 1.774 | 10.39% | -6.48% | 1.603 | 1.31x |

**Final (canonical):** target_vol=6.5%, max_lev=2.0x, lookback=20d
- OOS Sharpe: **1.790**  CI [1.393, 2.189]
- CAGR: **9.61%**  MaxDD: **-5.79%**  Calmar: **1.658**
- H001 corr: +0.6121
- All 12 folds positive (range: +1.222 to +2.450)

**Key insight:** avg realized leverage is 1.23x despite 2.0x cap — vol-targeting actively de-levers in stressed regimes, so MaxDD worsens only 1.39pp over base while CAGR gains 2.07pp. The mechanism works exactly as designed: leverage when calm, de-risk when volatile.
