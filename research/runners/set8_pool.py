"""Shared pool builder for Set8 Batch3 ensemble runners (H581-H590).

Returns the 55-strategy return matrix (same pool as H570) plus auxiliary data.
Import: from set8_pool import build_pool
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from erised.data.fundamentals import load_fundamentals_panel
from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import sharpe
from erised.strategies.base import (
    AbsoluteMomentumStrategy,
    CarryMomentumSPYTLT,
    CorrelationRiskParityHybrid,
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
    ReversionEntryRiskParity,
    RiskAdjustedMomentum,
    RiskParity,
    RiskParityMomentumFilter,
    RiskParityYieldCurveOverlay,
    SeasonalityTrendCombo,
    SeasonalRiskParity,
    SectorMomentumVolTarget,
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
from erised.strategies.cross_sectional import (
    BettingAgainstBeta,
    CostAwareMomentumLongHold,
    InvestmentEfficiencyRatio,
    KnowledgeReturnProxy,
    LongShortPCAReversal,
    LongShortResidualReversal,
    LongShortReversal,
    LongShortSectorReversal,
    LowCoverageMomentumProxy,
    MultiSignalComposite,
    PCAOUMeanReversion,
    PCAResidualReversal,
    QualityMomentumWeeklyRotation,
    ResidualOUMeanReversion,
    ResidualWeeklyReversal,
    SustainabilityFilteredProfitability,
    TailRiskComposite,
    VolScaledMomentum,
    WeeklyReversalSectorNeutral,
)
from erised.validation.walk_forward import (
    make_folds,
    walk_forward,
    walk_forward_cross_sectional,
    walk_forward_multi,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_DIR = DATA_DIR / "cache"

AA_POOL_BASE = {
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


def _run_aa(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def build_pool(include_h576: bool = False, include_h577: bool = False) -> dict:
    """Build the 55 (or 57) strategy return pool used in Set8 Batch3.

    Returns a dict with keys:
        mat: np.ndarray (T x N) return matrix
        strategy_names: list[str]
        clean: pd.DataFrame (aligned, all-clean rows)
        folds_aa: list of Fold
        spy_close: pd.Series (SPY adj_close)
        vix_series: pd.Series (VIX adj_close)
        prices_aa: pd.DataFrame (multi-asset prices)
        fundamentals: dict of fundamental panels
        panel, membership, spy_rets, sectors, folds_cs
    """
    spy = load_research("SPY")
    etf_names = ["TLT", "GLD", "VIX", "SHY", "DBC", "XLP", "XLU", "IVE"]
    etfs = {
        n: load_research(n)[["adj_close"]].rename(columns={"adj_close": f"{n.lower()}_close"})
        for n in etf_names
    }
    prices_aa = spy.copy()
    for df in etfs.values():
        prices_aa = prices_aa.join(df, how="left")
    for col in [c for c in prices_aa.columns if c != "adj_close"]:
        prices_aa[col] = prices_aa[col].ffill()
    folds_aa = make_folds(spy, train_years=5, test_years=2)

    spy_close = spy["adj_close"]
    vix_series = load_research("VIX")["adj_close"]
    spy_rets = spy["adj_close"].pct_change().fillna(0.0)

    # Disk cache of the (expensive) walk-forward return matrix. Pure memoization:
    # the pool is fully determined by the strategy set + fixed OOS folds, so the
    # cached `clean` matrix is byte-identical to a fresh build. Keyed by include flags.
    cache_key = f"{int(include_h576)}{int(include_h577)}"
    cache_path = CACHE_DIR / f"set8_pool_clean_{cache_key}.parquet"
    if cache_path.exists():
        clean = pd.read_parquet(cache_path)
        mat = clean.fillna(0.0).to_numpy(dtype="float64")
        strategy_names = list(clean.columns)
        print(
            f"[cache] loaded return matrix from {cache_path.name}: "
            f"{mat.shape[1]} strategies, {mat.shape[0]} days"
        )
        return {
            "mat": mat,
            "strategy_names": strategy_names,
            "clean": clean,
            "folds_aa": folds_aa,
            "spy_close": spy_close,
            "vix_series": vix_series,
            "prices_aa": prices_aa,
            "spy_rets": spy_rets,
            "sectors": None,
            "spy": spy,
            "fundamentals": None,
        }

    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    membership = load_sp500_members()
    common = panel.index.intersection(membership.index)
    panel = panel.loc[common]
    membership = membership.loc[common]
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    pt = panel.columns.tolist()
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    ta = load_fundamentals_panel(pt, "total_assets", panel.index)
    ni = load_fundamentals_panel(pt, "net_income", panel.index)
    ca = load_fundamentals_panel(pt, "current_assets", panel.index)
    cl = load_fundamentals_panel(pt, "current_liabilities", panel.index)
    ocf = load_fundamentals_panel(pt, "operating_cf", panel.index)
    rev = load_fundamentals_panel(pt, "revenue", panel.index)
    folds_cs = make_folds(panel, train_years=5, test_years=2)
    print(f"Panel: {panel.shape[0]} days, {panel.shape[1]} tickers")

    aa_pool = dict(AA_POOL_BASE)
    if include_h577:
        aa_pool["H577"] = SectorMomentumVolTarget()

    cs_factories = {
        "H157": lambda tp, tm: VolScaledMomentum(),
        "H175": lambda tp, tm: BettingAgainstBeta(spy_rets=spy_rets, sectors=sectors),
        "H185": lambda tp, tm: WeeklyReversalSectorNeutral(sectors=sectors),
        "H186": lambda tp, tm: ResidualWeeklyReversal(spy_rets=spy_rets, sectors=sectors),
        "H189": lambda tp, tm: ResidualOUMeanReversion(spy_rets=spy_rets, sectors=sectors),
        "H191": lambda tp, tm: PCAResidualReversal(train_prices=tp, sectors=sectors),
        "H192": lambda tp, tm: PCAOUMeanReversion(train_prices=tp, sectors=sectors),
        "H194": lambda tp, tm: LongShortReversal(sectors=sectors),
        "H199": lambda tp, tm: MultiSignalComposite(net_income=ni, sectors=sectors),
        "H254": lambda tp, tm: TailRiskComposite(sectors=sectors, spy_rets=spy_rets),
        "H265": lambda tp, tm: SustainabilityFilteredProfitability(
            gross_profit=gp,
            total_assets=ta,
            current_assets=ca,
            current_liabilities=cl,
            net_income=ni,
            operating_cf=ocf,
            sectors=sectors,
        ),
        "H268": lambda tp, tm: KnowledgeReturnProxy(
            gross_profit=gp, total_assets=ta, sectors=sectors
        ),
        "H284": lambda tp, tm: LowCoverageMomentumProxy(
            spy_prices=spy["adj_close"], sectors=sectors
        ),
        "H287": lambda tp, tm: InvestmentEfficiencyRatio(
            revenue=rev, total_assets=ta, current_assets=ca, sectors=sectors
        ),
        "H293": lambda tp, tm: CostAwareMomentumLongHold(sectors=sectors),
        "H471": lambda tp, tm: LongShortSectorReversal(sectors=sectors),
        "H472": lambda tp, tm: LongShortResidualReversal(spy_rets=spy_rets, sectors=sectors),
        "H473": lambda tp, tm: LongShortPCAReversal(train_prices=tp, sectors=sectors),
    }
    if include_h576:
        cs_factories["H576"] = lambda tp, tm: QualityMomentumWeeklyRotation(
            gross_profit=gp, total_assets=ta, sectors=sectors
        )

    print(f"\nRunning {len(aa_pool)} AA strategies...")
    all_rets = {}
    for name, strat in aa_pool.items():
        wf = _run_aa(prices_aa, strat, folds_aa)
        all_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    print(f"\nRunning {len(cs_factories)} CS strategies...")
    for name, factory in cs_factories.items():
        wf = walk_forward_cross_sectional(
            panel, membership, factory, folds=folds_cs, spy_returns=spy_rets
        )
        all_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    strategy_names = list(all_rets.keys())
    ret_df = pd.DataFrame(all_rets)
    clean = ret_df.dropna(how="all")
    mat = clean.fillna(0.0).to_numpy(dtype="float64")
    print(f"\nPool: {mat.shape[1]} strategies, {mat.shape[0]} days")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(cache_path)
    print(f"[cache] saved return matrix to {cache_path.name}")

    return {
        "mat": mat,
        "strategy_names": strategy_names,
        "clean": clean,
        "folds_aa": folds_aa,
        "spy_close": spy_close,
        "vix_series": vix_series,
        "prices_aa": prices_aa,
        "spy_rets": spy_rets,
        "sectors": sectors,
        "spy": spy,
        "fundamentals": {
            "gp": gp,
            "ta": ta,
            "ni": ni,
            "ca": ca,
            "cl": cl,
            "ocf": ocf,
            "rev": rev,
        },
    }
