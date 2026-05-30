"""H400: UltimateGrandMDR400 -- MDR over H350's 80 components + 20 Set5 CS = 100 total.

Extends H350 (Sharpe 1.525, MaxDD -6.86%) by adding the strongest Set 5 cross-sectional
strategies (H351-H399). Tests whether factor-zoo, ML, governance, ESG, attention-proxy,
and execution-aware signals add MDR benefit beyond H350's existing 80 components.

Components:
  - 37 H001-H100 ETF strategies (same as H350)
  - 28 H301-H340 ETF strategies (same as H350)
  - 9 H150-H199 CS strategies (same as H350)
  - 6 H251-H298 Set 3 CS strategies (same as H350)
  - 20 Set 5 CS strategies (NEW: H351-H399, best per batch)
  Total: 100 strategies

Run: uv run python scripts/run_h400.py
Expected runtime: ~5-6 hours (CS components dominate runtime).
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

from quant.data.fundamentals import load_fundamentals_panel
from quant.data.loader import load_research
from quant.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
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
from quant.strategies.cross_sectional import (
    # H150-H199 CS (same as H350)
    BettingAgainstBeta,
    CostAwareMomentumLongHold,
    InvestmentEfficiencyRatio,
    KnowledgeReturnProxy,
    LongShortReversal,
    LowCoverageMomentumProxy,
    MultiSignalComposite,
    PCAOUMeanReversion,
    PCAResidualReversal,
    ResidualOUMeanReversion,
    ResidualWeeklyReversal,
    SustainabilityFilteredProfitability,
    TailRiskComposite,
    VolScaledMomentum,
    WeeklyReversalSectorNeutral,
    # Set 3 CS (same as H350)
    # Set 5 CS (NEW)
    BuybackQualityGovernance,
    ConsistentMomentumPort,
    ControversyProxyAvoid,
    DecayAdjustedComposite,
    ElasticNetRankStrong,
    ESGProxyComposite,
    ExecutionOptimizedComposite,
    GradientBoostingRank,
    GovernanceQualityComposite,
    GreenProxyQuality,
    LiquidityRegimeCapPort,
    LowAttentionValuePort,
    LowTurnoverGP,
    LowVolDisagreementAvoid,
    MispricingAvoidFilter,
    NonlinearMomentumBins,
    QFactorComposite,
    QInvestmentRank,
    ReturnSpikeRiskFilter,
    SpreadAwarePositionSizer,
    VolatilityAwareComposite,
)
from quant.validation.walk_forward import (
    make_folds,
    walk_forward,
    walk_forward_cross_sectional,
    walk_forward_multi,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
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


def run_asset_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def max_diversification_weights(returns_df: pd.DataFrame) -> np.ndarray:
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
    print(f"H400: UltimateGrandMDR400 (terminal 400-strategy champion)\n{SEP}")

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

    print("Loading S&P 500 stock panel...")
    membership = load_sp500_members()
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)

    common_idx = panel.index.intersection(membership.index)
    panel = panel.loc[common_idx]
    membership = membership.loc[common_idx]

    spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    print(f"  Panel: {panel.shape[0]} days, {panel.shape[1]} tickers")

    print("Loading fundamental data...")
    panel_tickers = panel.columns.tolist()
    net_income = load_fundamentals_panel(panel_tickers, "net_income", panel.index)
    gp = load_fundamentals_panel(panel_tickers, "gross_profit", panel.index)
    ta = load_fundamentals_panel(panel_tickers, "total_assets", panel.index)
    ca = load_fundamentals_panel(panel_tickers, "current_assets", panel.index)
    cl = load_fundamentals_panel(panel_tickers, "current_liabilities", panel.index)
    ocf = load_fundamentals_panel(panel_tickers, "operating_cf", panel.index)
    rev = load_fundamentals_panel(panel_tickers, "revenue", panel.index)
    shares = load_fundamentals_panel(panel_tickers, "shares_outstanding", panel.index)
    be = load_fundamentals_panel(panel_tickers, "book_equity", panel.index)

    folds_aa = make_folds(spy, train_years=5, test_years=2)
    folds_cs = make_folds(panel, train_years=5, test_years=2)
    print(f"\nFolds (AA): {len(folds_aa)}  |  Folds (CS): {len(folds_cs)}")

    # H001-H100 strategies
    print(f"\nRunning {len(H001_H100_STRATEGIES)} H001-H100 strategies...")
    component_returns: dict[str, pd.Series] = {}
    for name, strat in H001_H100_STRATEGIES.items():
        wf = run_asset_wf(prices_aa, strat, folds_aa)
        component_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    # H301-H340 ETF strategies
    print(f"\nRunning {len(H301_H340_STRATEGIES)} H301-H340 ETF strategies...")
    for name, strat in H301_H340_STRATEGIES.items():
        wf = run_asset_wf(prices_aa, strat, folds_aa)
        component_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    # H150-H199 cross-sectional strategies
    cs_factories_h200 = {
        "H157": lambda tp, tm: VolScaledMomentum(),
        "H175": lambda tp, tm: BettingAgainstBeta(spy_rets=spy_rets, sectors=sectors),
        "H185": lambda tp, tm: WeeklyReversalSectorNeutral(sectors=sectors),
        "H186": lambda tp, tm: ResidualWeeklyReversal(spy_rets=spy_rets, sectors=sectors),
        "H189": lambda tp, tm: ResidualOUMeanReversion(spy_rets=spy_rets, sectors=sectors),
        "H191": lambda tp, tm: PCAResidualReversal(train_prices=tp, sectors=sectors),
        "H192": lambda tp, tm: PCAOUMeanReversion(train_prices=tp, sectors=sectors),
        "H194": lambda tp, tm: LongShortReversal(sectors=sectors),
        "H199": lambda tp, tm: MultiSignalComposite(net_income=net_income, sectors=sectors),
    }

    print(f"\nRunning {len(cs_factories_h200)} H150-H199 strategies...")
    for name, factory in cs_factories_h200.items():
        print(f"  {name}...", end="", flush=True)
        wf = walk_forward_cross_sectional(
            panel, membership, factory, folds=folds_cs, spy_returns=spy_rets
        )
        component_returns[name] = wf.combined_returns
        print(f" Sharpe {sharpe(wf.combined_returns):.3f}")

    # H251-H298 Set 3 strategies
    cs_factories_set3 = {
        "H268": lambda tp, tm: KnowledgeReturnProxy(
            gross_profit=gp, total_assets=ta, sectors=sectors
        ),
        "H265": lambda tp, tm: SustainabilityFilteredProfitability(
            gross_profit=gp,
            total_assets=ta,
            current_assets=ca,
            current_liabilities=cl,
            net_income=net_income,
            operating_cf=ocf,
            sectors=sectors,
        ),
        "H254": lambda tp, tm: TailRiskComposite(sectors=sectors, spy_rets=spy_rets),
        "H287": lambda tp, tm: InvestmentEfficiencyRatio(
            revenue=rev,
            total_assets=ta,
            current_assets=ca,
            sectors=sectors,
        ),
        "H293": lambda tp, tm: CostAwareMomentumLongHold(sectors=sectors),
        "H284": lambda tp, tm: LowCoverageMomentumProxy(
            spy_prices=spy["adj_close"], sectors=sectors
        ),
    }

    print(f"\nRunning {len(cs_factories_set3)} Set 3 (H251-H298) strategies...")
    for name, factory in cs_factories_set3.items():
        print(f"  {name}...", end="", flush=True)
        wf = walk_forward_cross_sectional(
            panel, membership, factory, folds=folds_cs, spy_returns=spy_rets
        )
        component_returns[name] = wf.combined_returns
        print(f" Sharpe {sharpe(wf.combined_returns):.3f}")

    # Set 5 CS strategies (NEW for H400)
    # Best 2 representatives per batch for diversity
    cs_factories_set5 = {
        # Batch 1 - q-factor
        "H351": lambda tp, tm: QInvestmentRank(
            total_assets=ta, sectors=sectors,
        ),
        "H355": lambda tp, tm: QFactorComposite(
            total_assets=ta, net_income=net_income, book_equity=be,
            gross_profit=gp, sectors=sectors,
        ),
        # Batch 2 - mispricing
        "H359": lambda tp, tm: MispricingAvoidFilter(
            total_assets=ta, net_income=net_income, operating_cf=ocf,
            shares_outstanding=shares, sectors=sectors,
        ),
        # Batch 3 - nonlinear/ML
        "H364": lambda tp, tm: NonlinearMomentumBins(sectors=sectors),
        "H365": lambda tp, tm: ElasticNetRankStrong(
            gross_profit=gp, total_assets=ta, net_income=net_income,
            operating_cf=ocf, sectors=sectors,
        ),
        # Batch 5 - ML ladder
        "H374": lambda tp, tm: GradientBoostingRank(
            gross_profit=gp, total_assets=ta, net_income=net_income,
            operating_cf=ocf, sectors=sectors,
        ),
        # Batch 7 - governance
        "H384": lambda tp, tm: GovernanceQualityComposite(
            shares_outstanding=shares, total_assets=ta, net_income=net_income,
            operating_cf=ocf, gross_profit=gp, sectors=sectors,
        ),
        "H381": lambda tp, tm: BuybackQualityGovernance(
            shares_outstanding=shares, sectors=sectors,
        ),
        # Batch 8 - ESG proxy
        "H390": lambda tp, tm: ESGProxyComposite(
            gross_profit=gp, total_assets=ta, net_income=net_income,
            operating_cf=ocf, sectors=sectors,
        ),
        "H388": lambda tp, tm: GreenProxyQuality(
            gross_profit=gp, total_assets=ta, sectors=sectors,
        ),
        "H389": lambda tp, tm: ControversyProxyAvoid(
            gross_profit=gp, total_assets=ta, net_income=net_income,
            operating_cf=ocf, sectors=sectors,
        ),
        # Batch 9 - attention/vol proxy
        "H391": lambda tp, tm: LowAttentionValuePort(
            gross_profit=gp, total_assets=ta, sectors=sectors,
        ),
        "H392": lambda tp, tm: ConsistentMomentumPort(sectors=sectors),
        "H393": lambda tp, tm: LowVolDisagreementAvoid(
            gross_profit=gp, total_assets=ta, sectors=sectors,
        ),
        "H395": lambda tp, tm: VolatilityAwareComposite(
            gross_profit=gp, total_assets=ta, net_income=net_income,
            operating_cf=ocf, sectors=sectors,
        ),
        # Batch 10 - execution aware
        "H396": lambda tp, tm: LowTurnoverGP(
            gross_profit=gp, total_assets=ta, sectors=sectors,
        ),
        "H397": lambda tp, tm: SpreadAwarePositionSizer(
            gross_profit=gp, total_assets=ta, net_income=net_income,
            operating_cf=ocf, sectors=sectors,
        ),
        "H398": lambda tp, tm: LiquidityRegimeCapPort(
            gross_profit=gp, total_assets=ta, sectors=sectors,
        ),
        "H399": lambda tp, tm: ExecutionOptimizedComposite(
            gross_profit=gp, total_assets=ta, net_income=net_income,
            operating_cf=ocf, sectors=sectors,
        ),
    }

    print(f"\nRunning {len(cs_factories_set5)} Set 5 (H351-H399) strategies...")
    for name, factory in cs_factories_set5.items():
        print(f"  {name}...", end="", flush=True)
        wf = walk_forward_cross_sectional(
            panel, membership, factory, folds=folds_cs, spy_returns=spy_rets
        )
        component_returns[name] = wf.combined_returns
        print(f" Sharpe {sharpe(wf.combined_returns):.3f}")

    # MDR optimisation
    returns_df = pd.DataFrame(component_returns)
    all_names = list(returns_df.columns)

    ew_rets = returns_df.mean(axis=1)
    print(
        f"\nEqual-weight baseline ({len(all_names)} strategies):"
        f" Sharpe {sharpe(ew_rets):.3f}  MaxDD {max_drawdown(ew_rets):.2%}"
    )

    print("\nSolving MDR optimisation...")
    w_mdr = max_diversification_weights(returns_df)

    print("\nOptimal MDR weights (> 0.5%):")
    for name, w in sorted(zip(all_names, w_mdr, strict=True), key=lambda x: x[1], reverse=True):
        if w > 0.005:
            print(f"  {name}: {w:.4f}")

    mdr_rets = pd.Series(returns_df.values @ w_mdr, index=returns_df.index)
    lo, hi = bootstrap_sharpe_ci(mdr_rets, n_boot=5000)
    s = sharpe(mdr_rets)
    c = cagr(mdr_rets)
    dd = max_drawdown(mdr_rets)

    print("\nUltimate Grand MDR Portfolio (H400):")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")
    print("\n  Compare H350 (80 strategies): Sharpe 1.525  MaxDD -6.86%")
    print("  Compare H300 (52 strategies): Sharpe 1.525  MaxDD -6.86%")

    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = float(mdr_rets.corr(wf_h001.combined_returns.reindex(mdr_rets.index)))
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (mdr_rets.index >= fold.test_start) & (mdr_rets.index < fold.test_end)
        fold_ret = mdr_rets[mask]
        fs = float(sharpe(fold_ret)) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    top_weights = {n: float(w) for n, w in zip(all_names, w_mdr, strict=True) if w > 0.005}
    write_summary(
        hypothesis_id="400",
        name="ultimate-grand-mdr-400",
        strategy_class="UltimateGrandMDR400",
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
            f"SLSQP MDR over {len(all_names)} strategies: "
            f"{len(H001_H100_STRATEGIES)} H001-H100 + {len(H301_H340_STRATEGIES)} H301-H340 ETF "
            f"+ {len(cs_factories_h200)} H150-H199 CS + {len(cs_factories_set3)} Set3 CS "
            f"+ {len(cs_factories_set5)} Set5 CS. "
            f"Extends H350 (80 strategies) with 20 Set 5 CS strategies. "
            f"Top MDR weights: {top_weights}."
        ),
        canonical_params={
            "n_strategies": len(all_names),
            "n_aa_h001": len(H001_H100_STRATEGIES),
            "n_aa_h301": len(H301_H340_STRATEGIES),
            "n_cs_h200": len(cs_factories_h200),
            "n_cs_set3": len(cs_factories_set3),
            "n_cs_set5": len(cs_factories_set5),
            "method": "SLSQP",
            "top_weights": top_weights,
        },
    )


if __name__ == "__main__":
    main()
