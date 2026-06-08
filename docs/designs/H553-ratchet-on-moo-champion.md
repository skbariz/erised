# H553: Ratchet Overlay on Multi-Objective Champion

## Hypothesis

Apply H087's ratchet drawdown protection mechanism to the H551 multi-objective MDR
champion portfolio. H547 already demonstrated that a ratchet overlay on H543 (pure MDR)
adds +0.068 Calmar (0.840 → 0.908) with no Sharpe cost. If H551's multi-objective
optimizer raises the base Calmar to ≥ 0.90, adding a ratchet overlay should push
Calmar above 1.0 — the institutional-grade threshold we are targeting.

Mechanism: at each day, the portfolio's scale = 1 − max(0, current_drawdown / dd_limit),
where dd_limit is typically 0.20 (20% drawdown → zero exposure). This is applied to
the H551 combined return series.

## Expected behavior by regime

- **2000-2002 tech crash**: Ratchet will de-lever during the drawdown. Net effect:
  reduced losses, better Calmar. Sharpe may drop slightly if the recovery is missed.
- **2008-2009 GFC**: Drawdown was deep (~18% for our best strategies). Ratchet fires
  early, reducing MaxDD. Risk: if de-levering happens before the bottom, recovery
  misses some upside.
- **2020 COVID**: Sharp 10-20% drawdown then rapid recovery. Ratchet fires and
  partially de-levers; recovery speed determines net benefit.
- **Low-vol 2013-2017**: Ratchet almost never fires. Near-identical to H551.

## Success criteria

- OOS Sharpe ≥ 1.5 (no worse than H551 champion)
- Calmar ≥ 1.0 (primary target — first time we'd achieve institutional-grade Calmar)
- MaxDD ≤ −8% (improvement over H551 champion)
- Sortino ≥ 1.3

## Falsification

If H553 Calmar is not materially above H551 champion's Calmar (< +0.05 gain),
then the ratchet mechanism is already baked into the multi-objective optimizer's
solution and applying it redundantly adds no value.

## Implementation note

H553 is NOT a new strategy class. It is a post-processing overlay on H551's combined
return series. The runner:
1. Re-runs H551 (same pool, same champion weight vector from H551 results)
2. Applies the ratchet scale: `scale_t = 1 + min(0, drawdown_t / dd_limit)`
3. Reports scaled returns as H553

dd_limit parameter: grid over [0.15, 0.20, 0.25]. Use canonical 0.20.

## Results

*To be filled after H551 completes and reports champion weight vector.*
