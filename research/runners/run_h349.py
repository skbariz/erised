"""H349: HierarchicalBilevelETFMDR -- 2-tier MDR (H001-H100 50% + H301-H340 50%).

Tier 1 = all 37 kept H001-H100 ETF strategies; MDR weights within tier; 50% total allocation.
Tier 2 = all 28 kept H301-H340 ETF strategies; MDR weights within tier; 50% total allocation.
Final return = 0.5 * tier1 + 0.5 * tier2.

Compare to H347 (flat MDR over all 65: TBD) and H095 (MDR H001-H100 only: 1.133).
Tests whether equal tier allocation (hierarchical) beats flat MDR across same 65 strategies.

Run: uv run python scripts/run_h349.py
Expected runtime: ~45 minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies import (
    AbsoluteMomentumStrategy,
    CarryMomentumSPYTLT,
    ConstrainedMacroOverlay,
    CorrelationRiskParityHybrid,
    CreditRegimeRiskParity,
    CrisisProtocolRiskParity,
    DefaultSpreadSPYTimer,
    DefensiveEquityHedge,
    DefensiveMacroStateMachine,
    DefensiveSectorOverlay,
    DoubleCheckH080,
    DualFrequencyVolTarget,
    EarningsYieldOverlay,
    EqualWeightDefensiveFactor,
    FourAssetRiskParity,
    FundingStressH077,
    FundingStressMonitor,
    FundingStressVolTarget,
    GoldFearGauge,
    H001H021DirectBlend,
    H043H021RegimeBlend,
    HalloweenMomentumConfirm,
    JanuaryBarometer,
    LiquiditySpiralProtection,
    LowVolTargetPortfolio,
    MacroCombinationOverlay,
    MacroStateH089,
    MomentumCarryBlend,
    MonthlyRebalancingRiskParity,
    MovingAverageCrossover,
    MultiAssetDynamicVolTarget,
    MultiAssetStrategy,
    MultiSignalStressAdjusted,
    QuarterlyMacroStateOverlay,
    RatchetDrawdownProtection,
    RatchetVolTarget,
    RealizedVsImpliedVol,
    RegimeClassifier,
    ReversionEntryRiskParity,
    RiskAdjustedMomentum,
    RiskBudgetExpansion,
    RiskParity,
    RiskParityMomentumFilter,
    RiskParityYieldCurveOverlay,
    SeasonalityTrendCombo,
    SeasonalRiskParity,
    SellInMay,
    SellInMayBondConfirm,
    SimpleVRPOverlay,
    SPYGLDRotation,
    SPYTLTCorrelationRegime,
    SPYTLTDualMomentum,
    TermSpreadSPYTimer,
    ThreeAssetRiskParity,
    TimeSeriesMomentum,
    ValuationSpreadMarketState,
    VixRiskParityEnhanced,
    VolAccelerationDetector,
    VolPercentileExposure,
    VolRegimeRiskParity,
    VolScaledTrend,
    VrpCalendarHybrid,
    VRPCorrelationRegime,
    VRPRiskParityScaled,
    VRPTrendConfirm,
    VRPVolTarget,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64

H001_H100_STRATEGIES = {
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
    "H097": CorrelationRiskParityHybrid(),
    "H098": ReversionEntryRiskParity(),
}

H301_H340_STRATEGIES = {
    "H301": SimpleVRPOverlay(),
    "H302": VRPRiskParityScaled(),
    "H303": VRPTrendConfirm(),
    "H304": VRPVolTarget(),
    "H306": FundingStressMonitor(),
    "H307": LiquiditySpiralProtection(),
    "H308": CreditRegimeRiskParity(),
    "H309": MultiSignalStressAdjusted(),
    "H310": FundingStressVolTarget(),
    "H311": TermSpreadSPYTimer(),
    "H312": DefaultSpreadSPYTimer(),
    "H313": EarningsYieldOverlay(),
    "H314": MacroCombinationOverlay(),
    "H315": ConstrainedMacroOverlay(),
    "H316": QuarterlyMacroStateOverlay(),
    "H317": ValuationSpreadMarketState(),
    "H318": RegimeClassifier(),
    "H319": RiskBudgetExpansion(),
    "H320": DefensiveMacroStateMachine(),
    "H326": DualFrequencyVolTarget(),
    "H327": VolAccelerationDetector(),
    "H328": VolPercentileExposure(),
    "H329": VolRegimeRiskParity(),
    "H330": RatchetVolTarget(),
    "H335": MacroStateH089(),
    "H337": FundingStressH077(),
    "H338": DoubleCheckH080(),
    "H339": VRPCorrelationRegime(),
}


def run_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def mdr_weights(returns_df: pd.DataFrame) -> np.ndarray:
    clean = returns_df.dropna(how="all")
    cov = clean.cov().values * 252
    vols = np.sqrt(np.diag(cov))
    n = len(vols)

    def neg_dr(w: np.ndarray) -> float:
        port_var = w @ cov @ w
        if port_var <= 0:
            return 0.0
        return -(np.dot(w, vols) / np.sqrt(port_var))

    result = minimize(
        neg_dr,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x


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
    hyg = load_research("HYG")[["adj_close"]].rename(columns={"adj_close": "hyg_close"})
    lqd = load_research("LQD")[["adj_close"]].rename(columns={"adj_close": "lqd_close"})
    ivw = load_research("IVW")[["adj_close"]].rename(columns={"adj_close": "ivw_close"})

    prices_aa = (
        spy.join(tlt, how="left")
        .join(gld, how="left")
        .join(vix, how="left")
        .join(shy, how="left")
        .join(dbc, how="left")
        .join(xlp, how="left")
        .join(xlu, how="left")
        .join(ive, how="left")
        .join(hyg, how="left")
        .join(lqd, how="left")
        .join(ivw, how="left")
    )
    for col in [
        "tlt_close", "gld_close", "vix_close", "shy_close", "dbc_close",
        "xlp_close", "xlu_close", "ive_close", "hyg_close", "lqd_close", "ivw_close",
    ]:
        prices_aa[col] = prices_aa[col].ffill()

    folds = make_folds(spy, train_years=5, test_years=2)
    print(f"H349: HierarchicalBilevelETFMDR\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")
    print(f"  Tier 1: {len(H001_H100_STRATEGIES)} H001-H100 strategies (50%)")
    print(f"  Tier 2: {len(H301_H340_STRATEGIES)} H301-H340 strategies (50%)")

    print("\nRunning Tier 1 (H001-H100)...")
    tier1_returns: dict[str, pd.Series] = {}
    for name, strat in H001_H100_STRATEGIES.items():
        wf = run_wf(prices_aa, strat, folds)
        tier1_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    print("\nRunning Tier 2 (H301-H340)...")
    tier2_returns: dict[str, pd.Series] = {}
    for name, strat in H301_H340_STRATEGIES.items():
        wf = run_wf(prices_aa, strat, folds)
        tier2_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    # MDR within each tier
    tier1_df = pd.DataFrame(tier1_returns)
    tier2_df = pd.DataFrame(tier2_returns)

    print("\nSolving MDR within Tier 1...")
    w1 = mdr_weights(tier1_df)
    print("\nSolving MDR within Tier 2...")
    w2 = mdr_weights(tier2_df)

    tier1_combined = (tier1_df * w1).sum(axis=1)
    tier2_combined = (tier2_df * w2).sum(axis=1)

    common = tier1_combined.index.intersection(tier2_combined.index)
    combined_rets = 0.5 * tier1_combined.loc[common] + 0.5 * tier2_combined.loc[common]

    lo, hi = bootstrap_sharpe_ci(combined_rets, n_boot=5000)
    s = sharpe(combined_rets)
    c = cagr(combined_rets)
    dd = max_drawdown(combined_rets)

    print(f"\n{'=' * 64}")
    print("H349: HierarchicalBilevelETFMDR")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")
    print("\n  Compare H095 (MDR H001-H100 only): Sharpe 1.133  MaxDD -7.16%")

    print("\nTier 1 top weights:")
    t1_names = list(tier1_returns.keys())
    for i in np.argsort(w1)[::-1][:8]:
        print(f"  {t1_names[i]}: {w1[i]:.3f}")

    print("\nTier 2 top weights:")
    t2_names = list(tier2_returns.keys())
    for i in np.argsort(w2)[::-1][:8]:
        print(f"  {t2_names[i]}: {w2[i]:.3f}")

    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    corr = float(combined_rets.corr(wf_h001.combined_returns.reindex(combined_rets.index)))
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        fold_rets = combined_rets.loc[fold.test_start : fold.test_end]
        fs = float(sharpe(fold_rets)) if len(fold_rets) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if lo > 0 and s > 0.51 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="349",
        name="hierarchical-bilevel-etf-mdr",
        strategy_class="HierarchicalBilevelETFMDR",
        family="cross-program ensemble",
        sharpe=s,
        ci_low=lo,
        ci_high=hi,
        cagr=c,
        max_drawdown=dd,
        n_trades=0,
        h001_corr=corr,
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            "2-tier ETF-only MDR: Tier1=37 H001-H100 strategies (50%), "
            "Tier2=28 H301-H340 strategies (50%). "
            "MDR applied within each tier; tiers blended 50/50. "
            f"T1 top: {t1_names[int(np.argmax(w1))]}: {w1.max():.3f}. "
            f"T2 top: {t2_names[int(np.argmax(w2))]}: {w2.max():.3f}."
        ),
        canonical_params={
            "tier1_n": len(H001_H100_STRATEGIES),
            "tier2_n": len(H301_H340_STRATEGIES),
            "tier_weights": [0.5, 0.5],
            "method": "MDR_SLSQP",
        },
    )


if __name__ == "__main__":
    main()
