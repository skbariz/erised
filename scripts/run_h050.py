"""H050: SharpeWeightedEnsemble -- standalone runner.

Run: uv run python scripts/run_h050.py

Weight kept strategies by their OOS Sharpe (positive only). Strategies with
higher OOS Sharpe get proportionally more capital. Tests whether quality-
weighting beats equal-weight.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from quant.strategies.base import (
    AbsoluteMomentumStrategy,
    DefensiveEquityHedge,
    GoldFearGauge,
    HalloweenMomentumConfirm,
    JanuaryBarometer,
    MovingAverageCrossover,
    RealizedVsImpliedVol,
    RiskParity,
    SeasonalityTrendCombo,
    SellInMay,
    SellInMayBondConfirm,
    SPYGLDRotation,
    SPYTLTCorrelationRegime,
    SPYTLTDualMomentum,
    TimeSeriesMomentum,
    VolScaledTrend,
)
from quant.strategies.base import MultiAssetStrategy
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi


def run_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)

SEP = "=" * 64

KEPT_STRATEGIES = {
    "H001": MovingAverageCrossover(50, 200),
    "H007": TimeSeriesMomentum(),
    "H008": AbsoluteMomentumStrategy(),
    "H016": VolScaledTrend(),
    "H018": SPYTLTDualMomentum(),
    "H019": SPYGLDRotation(),
    "H021": RiskParity(),
    "H022": DefensiveEquityHedge(),
    "H023": SellInMay(),
    "H025": JanuaryBarometer(),
    "H027": SeasonalityTrendCombo(),
    "H031": RealizedVsImpliedVol(),
    "H034": SPYTLTCorrelationRegime(),
    "H035": GoldFearGauge(),
    "H043": SellInMayBondConfirm(),
    "H046": HalloweenMomentumConfirm(),
}


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})

    prices_all = spy.join(tlt, how="left").join(gld, how="left").join(vix, how="left")
    prices_all["tlt_close"] = prices_all["tlt_close"].ffill()
    prices_all["gld_close"] = prices_all["gld_close"].ffill()
    prices_all["vix_close"] = prices_all["vix_close"].ffill()

    folds = make_folds(spy, train_years=5, test_years=2)
    print(f"H050: SharpeWeightedEnsemble\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")

    # Collect OOS returns
    print("\nRunning component strategies...")
    component_returns = {}
    component_sharpes = {}
    for name, strat in KEPT_STRATEGIES.items():
        wf = run_wf(prices_all, strat, folds)
        component_returns[name] = wf.combined_returns
        component_sharpes[name] = sharpe(wf.combined_returns)
        print(f"  {name}: OOS Sharpe {component_sharpes[name]:.3f}")

    returns_df = pd.DataFrame(component_returns)

    # Sharpe weights (clip at 0 to avoid negative weights)
    sw = pd.Series(component_sharpes).clip(lower=0)
    sw_norm = sw / sw.sum()
    print("\nSharpe weights:")
    for name in KEPT_STRATEGIES:
        print(f"  {name}: Sharpe {component_sharpes[name]:.3f}  weight {sw_norm[name]:.4f}")

    sw_rets = (returns_df * sw_norm).sum(axis=1)
    lo, hi = bootstrap_sharpe_ci(sw_rets, n_boot=5000)
    s = sharpe(sw_rets)
    c = cagr(sw_rets)
    dd = max_drawdown(sw_rets)

    print("\nSharpe-Weighted Ensemble:")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")

    corr = sw_rets.corr(component_returns["H001"])
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (sw_rets.index >= fold.test_start) & (sw_rets.index < fold.test_end)
        fold_ret = sw_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    write_summary(
        hypothesis_id="050",
        name="sharpe-weighted-ensemble",
        strategy_class="SharpeWeightedEnsemble",
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
        notes="Weight kept strategies by OOS Sharpe (clipped at 0). H021 (Sharpe 1.03) gets highest weight.",
        canonical_params={"weighting": "sharpe", "clip_negative": True},
    )


if __name__ == "__main__":
    main()
