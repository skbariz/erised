# H159: SmoothVolTargetMomentum — Smooth Signal + Vol-Target Composite

## Hypothesis

H155 (smooth momentum) improves stock selection by avoiding jumpy event-driven names. H157 (vol-scaled momentum) improves portfolio-level exposure management by cutting risk in high-vol regimes. These address different failure modes and should compound each other's benefits. The full Barroso & Santa-Clara + Da et al. combination: select stocks by smooth momentum signal AND scale total exposure by portfolio vol-target multiplier.

**Signal:** Build portfolio using H155 smooth momentum ranks (stock selection). Apply H157 vol-target multiplier to total exposure (portfolio management). Monthly stock selection, daily scaling.

## Expected Regime Behavior

- **2000–2002 tech crash:** Dual protection — H155 reduces tech overweight (event-driven names penalized) AND H157 cuts total exposure as portfolio vol spikes. Expected: best fold 1 performance in momentum family.
- **2008–2009 GFC:** Same dual protection. H155 selects fewer financial stocks (fewer extreme-day-driven rankings). H157 cuts exposure as crisis vol spikes. Expected best Max DD in momentum family.
- **2013–2017 low-vol bull:** H155 selects the steady compounders (which dominate this regime). H157 multiplier near 1.0 (low vol). Expected: close to or better than H150 Sharpe.
- **2020 COVID:** H157 fires during March crash, H155 selects quality recovery names (steady post-May 2020 ramp). Combined effect should be better than either H155 or H157 alone.
- **2022 bear:** H155 correctly weights energy (smooth uptrend), H157 runs at partial exposure (elevated market vol). Net effect positive vs H150.

## Success Criteria

- OOS Sharpe > max(H155, H157) (combination should beat each component)
- Max DD ≤ min(H155 Max DD, H157 Max DD) (dual protection)
- CI lower bound > 0
- Sensitivity: robust to vol_target ∈ {8%, 12%, 16%}

## Falsification

- Sharpe < min(H155, H157) → combining the two overlays introduces negative interaction (the smooth signal selects stocks that have higher vol, fighting the vol-target)
- Combination Sharpe < H150 + 0.05 → the compounding of two protective overlays adds complexity without proportional reward

## Implementation

- Class: `SmoothVolTargetMomentum(CrossSectionalStrategy)` in `src/quant/strategies/cross_sectional.py`
- Params: `long_days=252`, `skip_days=21`, `top_quantile=0.20`, `vol_target=0.12`, `vol_window=21`, `vol_min=0.2`, `vol_max=1.5`, `monthly_rebalance=True`
- Stock selection: H155 smooth momentum signal
- Portfolio scaling: H157 vol-target multiplier applied to unscaled portfolio returns
- Runner: `scripts/run_h159.py`

## Results

*(to be filled after backtest)*
