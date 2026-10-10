"""H515: ExpandedAAOnlyMDR -- MDR over H001-H100 AA (37) + H301-H340 AA (28) = 65 AA-only strategies.
Tests if Set4 AA strategies improve the AA MDR frontier beyond H095 (1.133). No CS panel needed.
Run: uv run python scripts/run_h515.py
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
from erised.strategies.base import (
    AbsoluteMomentumStrategy, CarryMomentumSPYTLT, ConstrainedMacroOverlay, CorrelationRiskParityHybrid,
    CreditRegimeRiskParity, CrisisProtocolRiskParity, DefaultSpreadSPYTimer, DefensiveEquityHedge,
    DefensiveMacroStateMachine, DefensiveSectorOverlay, DoubleCheckH080, DualFrequencyVolTarget,
    EarningsYieldOverlay, EqualWeightDefensiveFactor, FourAssetRiskParity, FundingStressH077,
    FundingStressMonitor, FundingStressVolTarget, GoldFearGauge, H001H021DirectBlend,
    H043H021RegimeBlend, HalloweenMomentumConfirm, JanuaryBarometer, LiquiditySpiralProtection,
    LowVolTargetPortfolio, MacroCombinationOverlay, MacroStateH089, MomentumCarryBlend,
    MonthlyRebalancingRiskParity, MovingAverageCrossover, MultiAssetDynamicVolTarget, MultiAssetStrategy,
    MultiSignalStressAdjusted, QuarterlyMacroStateOverlay, RatchetDrawdownProtection, RatchetVolTarget,
    RealizedVsImpliedVol, RegimeClassifier, ReversionEntryRiskParity, RiskAdjustedMomentum,
    RiskBudgetExpansion, RiskParity, RiskParityMomentumFilter, RiskParityYieldCurveOverlay,
    SeasonalityTrendCombo, SeasonalRiskParity, SellInMay, SellInMayBondConfirm, SimpleVRPOverlay,
    SPYGLDRotation, SPYTLTCorrelationRegime, SPYTLTDualMomentum, TermSpreadSPYTimer, ThreeAssetRiskParity,
    TimeSeriesMomentum, ValuationSpreadMarketState, VixRiskParityEnhanced, VolAccelerationDetector,
    VolPercentileExposure, VolRegimeRiskParity, VolScaledTrend, VrpCalendarHybrid,
    VRPCorrelationRegime, VRPRiskParityScaled, VRPTrendConfirm, VRPVolTarget,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64

H001_H100 = {
    "H001": MovingAverageCrossover(50, 200), "H007": TimeSeriesMomentum(), "H008": AbsoluteMomentumStrategy(),
    "H016": VolScaledTrend(), "H018": SPYTLTDualMomentum(), "H019": SPYGLDRotation(),
    "H021": RiskParity(), "H022": DefensiveEquityHedge(), "H023": SellInMay(),
    "H025": JanuaryBarometer(), "H027": SeasonalityTrendCombo(), "H031": RealizedVsImpliedVol(),
    "H034": SPYTLTCorrelationRegime(), "H035": GoldFearGauge(), "H043": SellInMayBondConfirm(),
    "H046": HalloweenMomentumConfirm(), "H051": ThreeAssetRiskParity(), "H060": CarryMomentumSPYTLT(),
    "H063": DefensiveSectorOverlay(), "H065": EqualWeightDefensiveFactor(), "H066": VixRiskParityEnhanced(),
    "H068": VrpCalendarHybrid(), "H070": MultiAssetDynamicVolTarget(), "H076": FourAssetRiskParity(),
    "H077": RiskParityMomentumFilter(), "H078": SeasonalRiskParity(), "H079": CrisisProtocolRiskParity(),
    "H080": RiskParityYieldCurveOverlay(), "H082": RiskAdjustedMomentum(), "H084": MomentumCarryBlend(),
    "H086": LowVolTargetPortfolio(), "H087": RatchetDrawdownProtection(), "H088": H001H021DirectBlend(),
    "H089": H043H021RegimeBlend(), "H090": MonthlyRebalancingRiskParity(), "H097": CorrelationRiskParityHybrid(),
    "H098": ReversionEntryRiskParity(),
}
H301_H340 = {
    "H301": SimpleVRPOverlay(), "H302": VRPRiskParityScaled(), "H303": VRPTrendConfirm(),
    "H304": VRPVolTarget(), "H306": FundingStressMonitor(), "H307": LiquiditySpiralProtection(),
    "H308": CreditRegimeRiskParity(), "H309": MultiSignalStressAdjusted(), "H310": FundingStressVolTarget(),
    "H311": TermSpreadSPYTimer(), "H312": DefaultSpreadSPYTimer(), "H313": EarningsYieldOverlay(),
    "H314": MacroCombinationOverlay(), "H315": ConstrainedMacroOverlay(), "H316": QuarterlyMacroStateOverlay(),
    "H317": ValuationSpreadMarketState(), "H318": RegimeClassifier(), "H319": RiskBudgetExpansion(),
    "H320": DefensiveMacroStateMachine(), "H326": DualFrequencyVolTarget(), "H327": VolAccelerationDetector(),
    "H328": VolPercentileExposure(), "H329": VolRegimeRiskParity(), "H330": RatchetVolTarget(),
    "H335": MacroStateH089(), "H337": FundingStressH077(), "H338": DoubleCheckH080(),
    "H339": VRPCorrelationRegime(),
}

def max_diversification_weights(returns_df):
    clean = returns_df.dropna(how="all"); cov = clean.cov().values * 252; vols = np.sqrt(np.diag(cov)); n = len(vols)
    def neg_dr(w):
        pv = w @ cov @ w
        return 0.0 if pv <= 0 else -(np.dot(w, vols) / np.sqrt(pv))
    r = minimize(neg_dr, np.full(n, 1.0/n), method="SLSQP", bounds=[(0,1)]*n,
                 constraints=[{"type":"eq","fun":lambda w: w.sum()-1}], options={"maxiter":1000,"ftol":1e-12})
    return r.x

def run_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy): return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)

def main():
    print(f"H515: ExpandedAAOnlyMDR (H001-H100 37 + H301-H340 28 = 65 AA strategies)\n{SEP}")
    spy = load_research("SPY")
    etfs = {n: load_research(n)[["adj_close"]].rename(columns={"adj_close": f"{n.lower()}_close"})
            for n in ["TLT","GLD","VIX","SHY","HYG","LQD","DBC","XLP","XLU","IVE","IVW"]}
    prices = spy.copy()
    for df in etfs.values(): prices = prices.join(df, how="left")
    for col in [c for c in prices.columns if c != "adj_close"]: prices[col] = prices[col].ffill()
    folds = make_folds(spy, train_years=5, test_years=2)
    print(f"Data: {len(spy)} rows  Folds: {len(folds)}")

    all_strats = {**H001_H100, **H301_H340}
    comp_rets = {}
    for name, strat in all_strats.items():
        wf = run_wf(prices, strat, folds)
        comp_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    ret_df = pd.DataFrame(comp_rets)
    ew = ret_df.mean(axis=1)
    print(f"\nEqual-weight (65 strategies): Sharpe {sharpe(ew):.3f}  MaxDD {max_drawdown(ew):.2%}")
    print("Solving MDR optimisation...")
    w_mdr = max_diversification_weights(ret_df)
    mdr_rets = pd.Series(ret_df.values @ w_mdr, index=ret_df.index)
    top_w = {n: float(w) for n, w in zip(all_strats.keys(), w_mdr) if w > 0.005}
    print(f"Top weights: {top_w}")
    lo, hi = bootstrap_sharpe_ci(mdr_rets, n_boot=5000)
    s = sharpe(mdr_rets); c = cagr(mdr_rets); dd = max_drawdown(mdr_rets)
    print(f"\nMDR: Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd:.2%}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (mdr_rets.index >= fold.test_start) & (mdr_rets.index < fold.test_end)
        fr = mdr_rets[mask]; fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs)); print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    corr = mdr_rets.corr(wh.combined_returns.reindex(mdr_rets.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if s > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    write_summary(
        hypothesis_id="515", name="expanded-aa-only-mdr", strategy_class="ExpandedAAOnlyMDR",
        family="Set7 Cross-Program Synergy", sharpe=s, ci_low=lo, ci_high=hi, cagr=c,
        max_drawdown=dd, n_trades=0, h001_corr=float(corr),
        fold_sharpes=fold_sharpes, decision=decision,
        notes=f"SLSQP MDR over {len(all_strats)} AA-only strategies: 37 from H001-H100 + 28 from H301-H340. No CS panel. Tests if Set4 AA strategies improve over H095 (1.133). Top weights: {top_w}.",
        canonical_params={"n_strategies": len(all_strats), "method": "SLSQP", "top_weights": top_w},
    )

if __name__ == "__main__":
    main()
