"""H091: Enhanced Min-Corr Ensemble v2 -- standalone runner.

Apply H049's greedy min-corr selection to all 35 kept strategies from H001-H090.
Tests whether new keepers from H051-H090 improve on H049's 4-component subset.

Run: uv run python scripts/run_h091.py
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
    CarryMomentumSPYTLT,
    CrisisProtocolRiskParity,
    DefensiveEquityHedge,
    DefensiveSectorOverlay,
    EqualWeightDefensiveFactor,
    FourAssetRiskParity,
    GoldFearGauge,
    H001H021DirectBlend,
    H043H021RegimeBlend,
    HalloweenMomentumConfirm,
    JanuaryBarometer,
    LowVolTargetPortfolio,
    MomentumCarryBlend,
    MonthlyRebalancingRiskParity,
    MovingAverageCrossover,
    MultiAssetDynamicVolTarget,
    MultiAssetStrategy,
    RatchetDrawdownProtection,
    RealizedVsImpliedVol,
    RiskAdjustedMomentum,
    RiskParity,
    RiskParityMomentumFilter,
    RiskParityYieldCurveOverlay,
    SeasonalityTrendCombo,
    SeasonalRiskParity,
    SellInMay,
    SellInMayBondConfirm,
    SPYGLDRotation,
    SPYTLTCorrelationRegime,
    SPYTLTDualMomentum,
    ThreeAssetRiskParity,
    TimeSeriesMomentum,
    VixRiskParityEnhanced,
    VolScaledTrend,
    VrpCalendarHybrid,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64

KEPT_STRATEGIES = {
    # H001-H046 (original program)
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
    # H051-H090 new keepers
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


def select_min_corr_subset(returns_df: pd.DataFrame, threshold: float) -> list[str]:
    """Greedy max-subset: all pairwise correlations below threshold.
    Seeds with highest-Sharpe strategy; adds next-best that doesn't exceed threshold
    with any already-selected strategy.
    """
    sharpes = {col: sharpe(returns_df[col].dropna()) for col in returns_df.columns}
    ranked = sorted(sharpes.keys(), key=lambda x: sharpes[x], reverse=True)
    selected = [ranked[0]]
    for candidate in ranked[1:]:
        can_add = True
        for sel in selected:
            common = returns_df[[candidate, sel]].dropna()
            if len(common) < 50:
                continue
            if common[candidate].corr(common[sel]) > threshold:
                can_add = False
                break
        if can_add:
            selected.append(candidate)
    return selected


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
    print(f"H091: Enhanced Min-Corr Ensemble v2 ({len(KEPT_STRATEGIES)} candidates)\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")

    print("\nRunning all component strategies...")
    component_returns = {}
    for name, strat in KEPT_STRATEGIES.items():
        wf = run_wf(prices_all, strat, folds)
        component_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    returns_df = pd.DataFrame(component_returns)

    print("\nMin-corr subset size at various thresholds:")
    for thresh in [0.30, 0.40, 0.50, 0.60, 0.70]:
        subset = select_min_corr_subset(returns_df, threshold=thresh)
        print(f"  threshold={thresh:.2f}: {len(subset)} strategies: {subset}")

    # Canonical: threshold 0.50 (consistent with H049)
    threshold = 0.50
    selected = select_min_corr_subset(returns_df, threshold=threshold)
    print(f"\nCanonical (threshold={threshold}): {selected}")

    mc_rets = returns_df[selected].mean(axis=1)
    lo, hi = bootstrap_sharpe_ci(mc_rets, n_boot=5000)
    s = sharpe(mc_rets)
    c = cagr(mc_rets)
    dd = max_drawdown(mc_rets)

    print(f"\nEnhanced Min-Corr Ensemble ({len(selected)} components):")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")

    corr = mc_rets.corr(component_returns["H001"])
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPairwise correlations of selected subset:")
    sel_df = returns_df[selected]
    corr_mat = sel_df.corr().round(3)
    print(corr_mat.to_string())

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (mc_rets.index >= fold.test_start) & (mc_rets.index < fold.test_end)
        fold_ret = mc_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    write_summary(
        hypothesis_id="091",
        name="enhanced-min-corr-ensemble-v2",
        strategy_class="EnhancedMinCorrEnsembleV2",
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
        notes=f"Greedy min-corr selection (threshold={threshold}) over all 35 kept strategies H001-H090. Selected: {selected}.",
        canonical_params={
            "threshold": threshold,
            "n_candidates": len(KEPT_STRATEGIES),
            "selected": selected,
        },
    )


if __name__ == "__main__":
    main()
