"""H347: CrossProgramETFMDR -- MDR over 37 H001-H100 + 28 H301-H340 = 65 ETF strategies.

ETF-only cross-program ensemble. Extends H095 (Set 1 MDR, Sharpe 1.133) by adding
all 28 kept H301-H340 macro/VRP strategies. Tests whether Set 4 strategies add
diversification value beyond the H001-H100 pool.

Run: uv run python scripts/run_h347.py
Expected runtime: ~45 minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from quant.strategies.base import (
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
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

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
    all_strategies = {**H001_H100_STRATEGIES, **H301_H340_STRATEGIES}
    print(f"H347: CrossProgramETFMDR ({len(all_strategies)} strategies)\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")
    print(f"  H001-H100: {len(H001_H100_STRATEGIES)} strategies")
    print(f"  H301-H340: {len(H301_H340_STRATEGIES)} strategies")

    print("\nRunning H001-H100 strategies...")
    component_returns: dict[str, pd.Series] = {}
    for name, strat in H001_H100_STRATEGIES.items():
        wf = run_wf(prices_aa, strat, folds)
        component_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    print("\nRunning H301-H340 strategies...")
    for name, strat in H301_H340_STRATEGIES.items():
        wf = run_wf(prices_aa, strat, folds)
        component_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    returns_df = pd.DataFrame(component_returns)
    ew_rets = returns_df.mean(axis=1)
    print(
        f"\nEqual-weight baseline ({len(all_strategies)} strategies):"
        f" Sharpe {sharpe(ew_rets):.3f}  MaxDD {max_drawdown(ew_rets):.2%}"
    )

    print("\nSolving MDR optimisation...")
    w_mdr = mdr_weights(returns_df)
    names = list(component_returns.keys())

    print("\nOptimal MDR weights (> 0.5%):")
    for nm, w in sorted(zip(names, w_mdr, strict=True), key=lambda x: x[1], reverse=True):
        if w > 0.005:
            print(f"  {nm}: {w:.4f}")

    mdr_rets = pd.Series(returns_df.values @ w_mdr, index=returns_df.index)
    lo, hi = bootstrap_sharpe_ci(mdr_rets, n_boot=5000)
    s = sharpe(mdr_rets)
    c = cagr(mdr_rets)
    dd = max_drawdown(mdr_rets)

    print("\nCross-Program ETF MDR (H347):")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")
    print("\n  Compare H095 (37-strat MDR): Sharpe 1.133  MaxDD -7.16%")
    print("  Compare H346 (28-strat Set4 MDR): Sharpe TBD")

    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    corr = float(mdr_rets.corr(wf_h001.combined_returns.reindex(mdr_rets.index)))
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        fold_rets = mdr_rets.loc[fold.test_start : fold.test_end]
        fs = float(sharpe(fold_rets)) if len(fold_rets) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if lo > 0 and s > 0.51 else "kill"
    print(f"Decision: {decision}")

    top_weights = {nm: float(w) for nm, w in zip(names, w_mdr, strict=True) if w > 0.005}
    write_summary(
        hypothesis_id="347",
        name="cross-program-etf-mdr",
        strategy_class="CrossProgramETFMDR",
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
            f"SLSQP MDR over {len(all_strategies)} ETF strategies: "
            f"{len(H001_H100_STRATEGIES)} H001-H100 + {len(H301_H340_STRATEGIES)} H301-H340. "
            f"EW baseline: Sharpe {sharpe(ew_rets):.3f}. "
            f"Top MDR weights: {top_weights}."
        ),
        canonical_params={
            "method": "MDR_SLSQP",
            "n_h001_h100": len(H001_H100_STRATEGIES),
            "n_h301_h340": len(H301_H340_STRATEGIES),
        },
    )


if __name__ == "__main__":
    main()
