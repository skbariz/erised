"""H547: RatchetExpandedH300 -- H543 expanded pool MDR + H087-style ratchet (5% trigger, 15% limit).
Tests if ratchet on L/S-expanded H300 compresses MaxDD while preserving Sharpe above H518 (1.524).
Run: uv run python scripts/run_h547.py
Expected runtime: ~3-4 hours.
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
from erised.data.fundamentals import load_fundamentals_panel
from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies import (
    AbsoluteMomentumStrategy, CarryMomentumSPYTLT, CorrelationRiskParityHybrid,
    CrisisProtocolRiskParity, DefensiveEquityHedge, DefensiveSectorOverlay,
    EqualWeightDefensiveFactor, FourAssetRiskParity, GoldFearGauge,
    H001H021DirectBlend, H043H021RegimeBlend, HalloweenMomentumConfirm,
    JanuaryBarometer, LowVolTargetPortfolio, MomentumCarryBlend,
    MonthlyRebalancingRiskParity, MovingAverageCrossover, MultiAssetDynamicVolTarget,
    MultiAssetStrategy, RatchetDrawdownProtection, RealizedVsImpliedVol,
    ReversionEntryRiskParity, RiskAdjustedMomentum, RiskParity, RiskParityMomentumFilter,
    RiskParityYieldCurveOverlay, SeasonalityTrendCombo, SeasonalRiskParity, SellInMay,
    SellInMayBondConfirm, SPYGLDRotation, SPYTLTCorrelationRegime, SPYTLTDualMomentum,
    ThreeAssetRiskParity, TimeSeriesMomentum, VixRiskParityEnhanced, VolScaledTrend, VrpCalendarHybrid,
)
from erised.strategies.cross_sectional import (
    BettingAgainstBeta, CostAwareMomentumLongHold, InvestmentEfficiencyRatio,
    KnowledgeReturnProxy, LongShortPCAReversal, LongShortResidualReversal,
    LongShortReversal, LongShortSectorReversal, LowCoverageMomentumProxy,
    MultiSignalComposite, PCAOUMeanReversion, PCAResidualReversal,
    ResidualOUMeanReversion, ResidualWeeklyReversal, SustainabilityFilteredProfitability,
    TailRiskComposite, VolScaledMomentum, WeeklyReversalSectorNeutral,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional, walk_forward_multi
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64

AA_POOL = {
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

def mdr_weights(returns_df):
    clean = returns_df.dropna(how="all"); cov = clean.cov().values * 252; vols = np.sqrt(np.diag(cov)); n = len(vols)
    def neg_dr(w):
        pv = w @ cov @ w; return 0.0 if pv <= 0 else -(np.dot(w, vols) / np.sqrt(pv))
    r = minimize(neg_dr, np.full(n, 1.0/n), method="SLSQP", bounds=[(0,1)]*n,
                 constraints=[{"type":"eq","fun":lambda w: w.sum()-1}], options={"maxiter":1000,"ftol":1e-12})
    return r.x

def run_aa(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy): return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)

def main():
    print(f"H547: RatchetExpandedH300 (H543 55-strategy MDR + ratchet dd>5%)\n{SEP}")
    spy = load_research("SPY")
    etfs = {n: load_research(n)[["adj_close"]].rename(columns={"adj_close": f"{n.lower()}_close"})
            for n in ["TLT","GLD","VIX","SHY","DBC","XLP","XLU","IVE"]}
    prices_aa = spy.copy()
    for df in etfs.values(): prices_aa = prices_aa.join(df, how="left")
    for col in [c for c in prices_aa.columns if c != "adj_close"]: prices_aa[col] = prices_aa[col].ffill()
    folds_aa = make_folds(spy, train_years=5, test_years=2)

    tickers = get_all_historical_tickers(); panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; membership = membership.loc[common]
    spy_rets = spy["adj_close"].pct_change().fillna(0.0)
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

    all_rets = {}
    for name, strat in AA_POOL.items():
        wf = run_aa(prices_aa, strat, folds_aa)
        all_rets[name] = wf.combined_returns

    cs_factories = {
        "H157": lambda tp, tm: VolScaledMomentum(), "H175": lambda tp, tm: BettingAgainstBeta(spy_rets=spy_rets, sectors=sectors),
        "H185": lambda tp, tm: WeeklyReversalSectorNeutral(sectors=sectors), "H186": lambda tp, tm: ResidualWeeklyReversal(spy_rets=spy_rets, sectors=sectors),
        "H189": lambda tp, tm: ResidualOUMeanReversion(spy_rets=spy_rets, sectors=sectors), "H191": lambda tp, tm: PCAResidualReversal(train_prices=tp, sectors=sectors),
        "H192": lambda tp, tm: PCAOUMeanReversion(train_prices=tp, sectors=sectors), "H194": lambda tp, tm: LongShortReversal(sectors=sectors),
        "H199": lambda tp, tm: MultiSignalComposite(net_income=ni, sectors=sectors),
        "H254": lambda tp, tm: TailRiskComposite(sectors=sectors, spy_rets=spy_rets),
        "H265": lambda tp, tm: SustainabilityFilteredProfitability(gross_profit=gp, total_assets=ta, current_assets=ca, current_liabilities=cl, net_income=ni, operating_cf=ocf, sectors=sectors),
        "H268": lambda tp, tm: KnowledgeReturnProxy(gross_profit=gp, total_assets=ta, sectors=sectors),
        "H284": lambda tp, tm: LowCoverageMomentumProxy(spy_prices=spy["adj_close"], sectors=sectors),
        "H287": lambda tp, tm: InvestmentEfficiencyRatio(revenue=rev, total_assets=ta, current_assets=ca, sectors=sectors),
        "H293": lambda tp, tm: CostAwareMomentumLongHold(sectors=sectors),
        "H471": lambda tp, tm: LongShortSectorReversal(sectors=sectors), "H472": lambda tp, tm: LongShortResidualReversal(spy_rets=spy_rets, sectors=sectors),
        "H473": lambda tp, tm: LongShortPCAReversal(train_prices=tp, sectors=sectors),
    }
    for name, factory in cs_factories.items():
        wf = walk_forward_cross_sectional(panel, membership, factory, folds=folds_cs, spy_returns=spy_rets)
        all_rets[name] = wf.combined_returns

    ret_df = pd.DataFrame(all_rets)
    print(f"Pool: {len(all_rets)} strategies. Solving MDR...")
    w = mdr_weights(ret_df)
    mdr_base = pd.Series(ret_df.values @ w, index=ret_df.index)
    top_w = {n: float(v) for n, v in zip(all_rets.keys(), w) if v > 0.005}
    print(f"MDR base: Sharpe {sharpe(mdr_base):.3f}  MaxDD {max_drawdown(mdr_base):.2%}")

    # Ratchet overlay: dd>5% triggers scale = (1 - dd/0.15)
    cum = (1.0 + mdr_base.fillna(0.0)).cumprod()
    peak = cum.expanding(min_periods=1).max()
    dd = (cum / peak - 1.0).fillna(0.0)
    dd_shifted = dd.shift(1).fillna(0.0)
    dd_trigger, dd_limit = 0.05, 0.15
    scale_ratchet = (1.0 + dd_shifted / dd_limit).clip(lower=0.0, upper=1.0)
    scale = pd.Series(np.where(dd_shifted < -dd_trigger, scale_ratchet, 1.0), index=dd.index)
    mdr_ratchet = mdr_base * scale

    lo, hi = bootstrap_sharpe_ci(mdr_ratchet, n_boot=5000)
    s = sharpe(mdr_ratchet); c = cagr(mdr_ratchet); dd_final = max_drawdown(mdr_ratchet)
    print(f"\nRatchet MDR: Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd_final:.2%}")
    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (mdr_ratchet.index >= fold.test_start) & (mdr_ratchet.index < fold.test_end)
        fr = mdr_ratchet[mask]; fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs)); print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = mdr_ratchet.corr(wh.combined_returns.reindex(mdr_ratchet.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if s > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    write_summary(hypothesis_id="547", name="ratchet-expanded-h300", strategy_class="RatchetExpandedH300",
        family="Set7 Batch10 Terminal Ensemble", sharpe=s, ci_low=lo, ci_high=hi, cagr=c,
        max_drawdown=dd_final, n_trades=0, h001_corr=float(corr), fold_sharpes=fold_sharpes, decision=decision,
        notes=f"H543 expanded MDR ({len(all_rets)} strategies) with H087-style ratchet: when portfolio dd from peak > 5%, scale by (1-dd/0.15). H300 MaxDD is -6.86% so ratchet triggers rarely. H087 showed ratchet works on H021. MDR base weights: {top_w}.",
        canonical_params={"dd_trigger": dd_trigger, "dd_limit": dd_limit, "method": "SLSQP", "n_strategies": len(all_rets)})

if __name__ == "__main__": main()
