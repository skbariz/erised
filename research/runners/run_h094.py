"""H094: H049PlusBestNew -- standalone runner.

Find the kept strategy from H051-H090 with lowest average pairwise correlation
to H049's 4 components (H021+H046+H019+H035). Add as 5th equal-weight component.
Tests whether a 5-component ensemble beats H049 (4-component).

Run: uv run python scripts/run_h094.py
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
from erised.strategies import (
    CarryMomentumSPYTLT,
    CrisisProtocolRiskParity,
    DefensiveSectorOverlay,
    EqualWeightDefensiveFactor,
    FourAssetRiskParity,
    GoldFearGauge,
    H001H021DirectBlend,
    H043H021RegimeBlend,
    HalloweenMomentumConfirm,
    LowVolTargetPortfolio,
    MomentumCarryBlend,
    MonthlyRebalancingRiskParity,
    MovingAverageCrossover,
    MultiAssetDynamicVolTarget,
    MultiAssetStrategy,
    RatchetDrawdownProtection,
    RiskAdjustedMomentum,
    RiskParity,
    RiskParityMomentumFilter,
    RiskParityYieldCurveOverlay,
    SeasonalRiskParity,
    SPYGLDRotation,
    ThreeAssetRiskParity,
    VixRiskParityEnhanced,
    VrpCalendarHybrid,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64

H049_COMPONENTS = {
    "H021": RiskParity(),
    "H046": HalloweenMomentumConfirm(),
    "H019": SPYGLDRotation(),
    "H035": GoldFearGauge(),
}

NEW_KEEPERS = {
    "H051": ThreeAssetRiskParity(),
    "H060": CarryMomentumSPYTLT(),
    "H063": DefensiveSectorOverlay(),
    "H065": EqualWeightDefensiveFactor(),
    "H066": VixRiskParityEnhanced(),
    "H068": VrpCalendarHybrid(),
    "H070": MultiAssetDynamicVolTarget(),
    "H076": FourAssetRiskParity(),
    "H077": RiskParityMomentumFilter(),
    "H078": SeasonalRiskParity(),
    "H079": CrisisProtocolRiskParity(),
    "H080": RiskParityYieldCurveOverlay(),
    "H082": RiskAdjustedMomentum(),
    "H084": MomentumCarryBlend(),
    "H086": LowVolTargetPortfolio(),
    "H087": RatchetDrawdownProtection(),
    "H088": H001H021DirectBlend(),
    "H089": H043H021RegimeBlend(),
    "H090": MonthlyRebalancingRiskParity(),
}


def run_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    dbc = load_research("DBC")[["adj_close"]].rename(columns={"adj_close": "dbc_close"})
    xlp = load_research("XLP")[["adj_close"]].rename(columns={"adj_close": "xlp_close"})
    xlu = load_research("XLU")[["adj_close"]].rename(columns={"adj_close": "xlu_close"})
    ive = load_research("IVE")[["adj_close"]].rename(columns={"adj_close": "ive_close"})

    prices_all = (
        spy.join(tlt, how="left")
        .join(gld, how="left")
        .join(vix, how="left")
        .join(shy, how="left")
        .join(dbc, how="left")
        .join(xlp, how="left")
        .join(xlu, how="left")
        .join(ive, how="left")
    )
    for col in [
        "tlt_close",
        "gld_close",
        "vix_close",
        "shy_close",
        "dbc_close",
        "xlp_close",
        "xlu_close",
        "ive_close",
    ]:
        prices_all[col] = prices_all[col].ffill()

    folds = make_folds(spy, train_years=5, test_years=2)
    print(f"H094: H049PlusBestNew ({len(H049_COMPONENTS)} original + 1 best new component)\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")

    # Run H049 components
    print("\nRunning H049 components...")
    h049_returns = {}
    for name, strat in H049_COMPONENTS.items():
        wf = run_wf(prices_all, strat, folds)
        h049_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    h049_rets = pd.DataFrame(h049_returns).mean(axis=1)
    print(
        f"\nH049 equal-weight baseline: Sharpe {sharpe(h049_rets):.3f}  MaxDD {max_drawdown(h049_rets):.2%}"
    )

    # Run all new keepers
    print("\nRunning new keeper strategies...")
    new_returns = {}
    for name, strat in NEW_KEEPERS.items():
        wf = run_wf(prices_all, strat, folds)
        new_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    # Find best new component: lowest average correlation to H049's 4 components
    print("\nCorrelation of new strategies to H049 components:")
    corrs_to_h049 = {}
    for new_name, new_ret in new_returns.items():
        avg_corr = 0.0
        for _h049_name, h049_ret in h049_returns.items():
            common = pd.concat([new_ret, h049_ret], axis=1).dropna()
            if len(common) > 50:
                avg_corr += common.iloc[:, 0].corr(common.iloc[:, 1])
        avg_corr /= len(h049_returns)
        corrs_to_h049[new_name] = avg_corr
        print(f"  {new_name}: avg corr to H049 = {avg_corr:+.4f}")

    best_new = min(corrs_to_h049, key=lambda k: corrs_to_h049[k])
    print(f"\nBest new component: {best_new} (avg corr to H049 = {corrs_to_h049[best_new]:+.4f})")

    # Build 5-component ensemble
    all_rets = {**h049_returns, best_new: new_returns[best_new]}
    ens_df = pd.DataFrame(all_rets)
    ens_rets = ens_df.mean(axis=1)

    lo, hi = bootstrap_sharpe_ci(ens_rets, n_boot=5000)
    s = sharpe(ens_rets)
    c = cagr(ens_rets)
    dd = max_drawdown(ens_rets)

    print(f"\nH049 + {best_new} (5-component equal weight):")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")

    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    corr = ens_rets.corr(wf_h001.combined_returns.reindex(ens_rets.index))
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
        hypothesis_id="094",
        name="h049-plus-best-new",
        strategy_class="H049PlusBestNew",
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
        notes=f"H049 + {best_new} (lowest avg corr to H049 components: {corrs_to_h049[best_new]:+.4f}). 5-component equal weight.",
        canonical_params={"h049_components": list(h049_returns.keys()), "best_new": best_new},
    )


if __name__ == "__main__":
    main()
