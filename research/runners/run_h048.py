"""H048: EqualWeightEnsemble -- standalone runner.

Run: uv run python scripts/run_h048.py

Equal-weight all 16 kept strategies from H001-H047.
Ensemble returns = mean of per-$1 OOS net returns across components.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies.base import (
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
from erised.strategies.base import MultiAssetStrategy
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi


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
    print(f"H048: EqualWeightEnsemble\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")
    print(f"Folds: {len(folds)}  Component strategies: {len(KEPT_STRATEGIES)}")

    # Collect OOS returns for each component
    print("\nRunning component strategies...")
    component_returns = {}
    component_fold_returns = {}
    for name, strat in KEPT_STRATEGIES.items():
        wf = run_wf(prices_all, strat, folds)
        component_returns[name] = wf.combined_returns
        component_fold_returns[name] = [fr.net_returns for fr in wf.fold_results]
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    returns_df = pd.DataFrame(component_returns)

    # Equal weight: mean across all strategies
    ew_rets = returns_df.mean(axis=1)
    lo, hi = bootstrap_sharpe_ci(ew_rets, n_boot=5000)
    s = sharpe(ew_rets)
    c = cagr(ew_rets)
    dd = max_drawdown(ew_rets)

    print(f"\nEqual-Weight Ensemble ({len(KEPT_STRATEGIES)} components):")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")

    # H001 correlation
    corr = ew_rets.corr(component_returns["H001"])
    print(f"\nH001 correlation: {corr:+.4f}")

    # Per-fold Sharpe
    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (ew_rets.index >= fold.test_start) & (ew_rets.index < fold.test_end)
        fold_ret = ew_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    write_summary(
        hypothesis_id="048",
        name="equal-weight-ensemble",
        strategy_class="EqualWeightEnsemble",
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
        notes=f"Equal weight ({1 / len(KEPT_STRATEGIES):.3f} each) across all 16 kept strategies H001-H046.",
        canonical_params={"n_components": len(KEPT_STRATEGIES), "weighting": "equal"},
    )


if __name__ == "__main__":
    main()
