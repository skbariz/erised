"""H092: RegimeAdaptiveEnsemble -- standalone runner.

H049's 4 components (H021+H046+H019+H035) with regime-dependent weights.
Regime determined daily by VIX level and yield curve (SHY vs TLT 63d return).

  calm_bull (VIX<15, curve not inverted): H021=0.20, H046=0.30, H019=0.30, H035=0.20
  normal (15<=VIX<=25): H021=0.25, H046=0.25, H019=0.25, H035=0.25
  risk_off (VIX>25 OR curve inverted): H021=0.40, H046=0.20, H019=0.10, H035=0.30

Run: uv run python scripts/run_h092.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from quant.strategies.base import (
    GoldFearGauge,
    HalloweenMomentumConfirm,
    MultiAssetStrategy,
    RiskParity,
    SPYGLDRotation,
)
from quant.validation.walk_forward import make_folds, walk_forward_multi

SEP = "=" * 64

# H049 canonical components
H049_COMPONENTS = {
    "H021": RiskParity(),
    "H046": HalloweenMomentumConfirm(),
    "H019": SPYGLDRotation(),
    "H035": GoldFearGauge(),
}

# Regime weight vectors: [H021, H046, H019, H035]
W_CALM = np.array([0.20, 0.30, 0.30, 0.20])
W_NORMAL = np.array([0.25, 0.25, 0.25, 0.25])
W_RISKOFF = np.array([0.40, 0.20, 0.10, 0.30])


def run_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    from quant.validation.walk_forward import walk_forward

    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def compute_regime_returns(
    returns_df: pd.DataFrame,
    prices_all: pd.DataFrame,
    vix_calm: float = 15.0,
    vix_riskoff: float = 25.0,
    curve_lookback: int = 63,
) -> pd.Series:
    """Combine 4-component returns using daily VIX+curve regime signals."""
    vix_lag = prices_all["vix_close"].shift(1)
    tlt_ret = prices_all["tlt_close"].pct_change(curve_lookback).shift(1)
    shy_ret = prices_all["shy_close"].pct_change(curve_lookback).shift(1)
    inverted_curve = tlt_ret < shy_ret

    calm = (vix_lag < vix_calm) & ~inverted_curve
    riskoff = (vix_lag > vix_riskoff) | inverted_curve
    normal = ~calm & ~riskoff

    # Align signals to OOS return dates
    idx = returns_df.index
    calm_s = calm.reindex(idx).fillna(False).astype(float)
    normal_s = normal.reindex(idx).fillna(False).astype(float)
    riskoff_s = riskoff.reindex(idx).fillna(False).astype(float)

    # Normalise (edge case: all signals zero on a row)
    row_sum = calm_s + normal_s + riskoff_s
    row_sum = row_sum.replace(0.0, 1.0)
    calm_s /= row_sum
    normal_s /= row_sum
    riskoff_s /= row_sum

    comp_vals = returns_df[["H021", "H046", "H019", "H035"]].values
    weights = (
        calm_s.values[:, None] * W_CALM
        + normal_s.values[:, None] * W_NORMAL
        + riskoff_s.values[:, None] * W_RISKOFF
    )
    ens_vals = (comp_vals * weights).sum(axis=1)
    return pd.Series(ens_vals, index=idx, name="ensemble")


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})

    prices_all = (
        spy.join(tlt, how="left").join(gld, how="left").join(vix, how="left").join(shy, how="left")
    )
    for col in ["tlt_close", "gld_close", "vix_close", "shy_close"]:
        prices_all[col] = prices_all[col].ffill()

    folds = make_folds(spy, train_years=5, test_years=2)
    print(f"H092: RegimeAdaptiveEnsemble (H049 components, VIX/curve regime weights)\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")

    print("\nRunning H049 component strategies...")
    component_returns = {}
    for name, strat in H049_COMPONENTS.items():
        wf = run_wf(prices_all, strat, folds)
        component_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    returns_df = pd.DataFrame(component_returns)

    # H049 baseline: equal weight
    h049_rets = returns_df.mean(axis=1)
    print(
        f"\nH049 equal-weight baseline: Sharpe {sharpe(h049_rets):.3f}  MaxDD {max_drawdown(h049_rets):.2%}"
    )

    # Regime distribution
    vix_lag = prices_all["vix_close"].shift(1)
    tlt_ret = prices_all["tlt_close"].pct_change(63).shift(1)
    shy_ret = prices_all["shy_close"].pct_change(63).shift(1)
    inverted_curve = tlt_ret < shy_ret
    calm = (vix_lag < 15.0) & ~inverted_curve
    riskoff = (vix_lag > 25.0) | inverted_curve
    idx = returns_df.index
    n = len(idx)
    n_calm = calm.reindex(idx).fillna(False).sum()
    n_riskoff = riskoff.reindex(idx).fillna(False).sum()
    n_normal = n - n_calm - n_riskoff
    print("\nRegime distribution over OOS period:")
    print(f"  calm_bull: {n_calm} days ({n_calm / n:.1%})")
    print(f"  normal:    {n_normal} days ({n_normal / n:.1%})")
    print(f"  risk_off:  {n_riskoff} days ({n_riskoff / n:.1%})")

    print("\nCanonical (calm<15, riskoff>25, curve_lookback=63):")
    ens_rets = compute_regime_returns(returns_df, prices_all)
    lo, hi = bootstrap_sharpe_ci(ens_rets, n_boot=5000)
    s = sharpe(ens_rets)
    c = cagr(ens_rets)
    dd = max_drawdown(ens_rets)
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")

    print("\nSensitivity -- VIX thresholds:")
    for v_calm, v_risk in [(12, 20), (15, 25), (18, 30)]:
        er = compute_regime_returns(returns_df, prices_all, vix_calm=v_calm, vix_riskoff=v_risk)
        lo_v, hi_v = bootstrap_sharpe_ci(er, n_boot=1000)
        print(
            f"  calm<{v_calm} riskoff>{v_risk}  Sharpe {sharpe(er):.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {max_drawdown(er):.2%}"
        )

    from quant.strategies.base import MovingAverageCrossover
    from quant.validation.walk_forward import walk_forward

    wf_h001_ma = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    corr = ens_rets.corr(wf_h001_ma.combined_returns.reindex(ens_rets.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (ens_rets.index >= fold.test_start) & (ens_rets.index < fold.test_end)
        fold_ret = ens_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    write_summary(
        hypothesis_id="092",
        name="regime-adaptive-ensemble",
        strategy_class="RegimeAdaptiveEnsemble",
        family="ensemble",
        sharpe=s,
        ci_low=lo,
        ci_high=hi,
        cagr=c,
        max_drawdown=dd,
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=[float(fs) for fs in fold_sharpes],
        decision=decision,
        notes="H049 components with daily regime weights: VIX+SHY/TLT curve signal. Calm: equity-heavy; risk-off: H021+H035 heavy.",
        canonical_params={"vix_calm": 15, "vix_riskoff": 25, "curve_lookback": 63},
    )


if __name__ == "__main__":
    main()
