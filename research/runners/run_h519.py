"""H519: HierarchicalAllFoldMDR -- 2-tier MDR: Tier1=H095 AA pool (50%), Tier2=all-fold CS (50%).
Tier2 restricted to all-fold CS only (no EDGAR gap). Tests if clean Tier2 breaks H325 (1.520).
Run: uv run python scripts/run_h519.py
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
from erised.strategies.base import (
    AbsoluteMomentumStrategy, CarryMomentumSPYTLT, CorrelationRiskParityHybrid,
    CrisisProtocolRiskParity, DefensiveEquityHedge, DefensiveSectorOverlay, EqualWeightDefensiveFactor,
    FourAssetRiskParity, GoldFearGauge, H001H021DirectBlend, H043H021RegimeBlend,
    HalloweenMomentumConfirm, JanuaryBarometer, LowVolTargetPortfolio, MomentumCarryBlend,
    MonthlyRebalancingRiskParity, MovingAverageCrossover, MultiAssetDynamicVolTarget, MultiAssetStrategy,
    RatchetDrawdownProtection, RealizedVsImpliedVol, ReversionEntryRiskParity, RiskAdjustedMomentum,
    RiskParity, RiskParityMomentumFilter, RiskParityYieldCurveOverlay, SeasonalityTrendCombo,
    SeasonalRiskParity, SellInMay, SellInMayBondConfirm, SPYGLDRotation, SPYTLTCorrelationRegime,
    SPYTLTDualMomentum, ThreeAssetRiskParity, TimeSeriesMomentum, VixRiskParityEnhanced,
    VolScaledTrend, VrpCalendarHybrid,
)
from erised.strategies.cross_sectional import (
    CreditQualityMomentumCombo, FundamentalDivergenceAvoid, LongShortPCAReversal,
    PatienceFilteredMomentum, SectorRelativeMarginProxy, SemiAnnualMomentumQualityCombo,
    StableMarginMomentum, VolAccelerationCSSwitch,
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
    "H086": LowVolTargetPortfolio(), "H087": RatchetDrawdownProtection(), "H089": H043H021RegimeBlend(),
    "H090": MonthlyRebalancingRiskParity(), "H097": CorrelationRiskParityHybrid(),
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
    print(f"H519: HierarchicalAllFoldMDR (Tier1=H095 AA MDR, Tier2=all-fold CS MDR, 50/50)\n{SEP}")
    spy = load_research("SPY")
    etfs = {n: load_research(n)[["adj_close"]].rename(columns={"adj_close": f"{n.lower()}_close"})
            for n in ["TLT","GLD","VIX","SHY","DBC","XLP","XLU","IVE"]}
    prices_aa = spy.copy()
    for df in etfs.values(): prices_aa = prices_aa.join(df, how="left")
    for col in [c for c in prices_aa.columns if c != "adj_close"]: prices_aa[col] = prices_aa[col].ffill()
    folds_aa = make_folds(spy, train_years=5, test_years=2)

    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; membership = membership.loc[common]
    spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    pt = panel.columns.tolist()
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    ta = load_fundamentals_panel(pt, "total_assets", panel.index)
    ni = load_fundamentals_panel(pt, "net_income", panel.index)
    rev = load_fundamentals_panel(pt, "revenue", panel.index)
    ltd = load_fundamentals_panel(pt, "long_term_debt", panel.index)
    ocf = load_fundamentals_panel(pt, "operating_cf", panel.index)
    folds_cs = make_folds(panel, train_years=5, test_years=2)
    print(f"Panel: {panel.shape[0]} days, {panel.shape[1]} tickers")

    # Tier 1: H095 AA pool
    print("\nTier 1: running H001-H100 AA pool...")
    tier1_rets = {}
    for name, strat in AA_POOL.items():
        wf = run_aa(prices_aa, strat, folds_aa)
        tier1_rets[name] = wf.combined_returns
    t1_df = pd.DataFrame(tier1_rets)
    w1 = mdr_weights(t1_df)
    tier1_port = pd.Series(t1_df.values @ w1, index=t1_df.index)
    top1 = {n: float(w) for n, w in zip(tier1_rets, w1) if w > 0.005}
    print(f"Tier1 MDR: Sharpe {sharpe(tier1_port):.3f}  MaxDD {max_drawdown(tier1_port):.2%}")
    print(f"Top weights: {top1}")

    # Tier 2: all-fold CS stars only
    print("\nTier 2: running all-fold CS stars...")
    cs_factories = {
        "H412": lambda tp, tm: FundamentalDivergenceAvoid(net_income=ni, total_assets=ta, sectors=sectors),
        "H415": lambda tp, tm: StableMarginMomentum(gross_profit=gp, revenue=rev, sectors=sectors),
        "H429": lambda tp, tm: SectorRelativeMarginProxy(gross_profit=gp, revenue=rev, sectors=sectors),
        "H447": lambda tp, tm: CreditQualityMomentumCombo(gross_profit=gp, total_assets=ta, long_term_debt=ltd, operating_cf=ocf, sectors=sectors),
        "H473": lambda tp, tm: LongShortPCAReversal(train_prices=tp, sectors=sectors),
        "H499": lambda tp, tm: VolAccelerationCSSwitch(sectors=sectors, gross_profit=gp, total_assets=ta, spy_returns=spy_rets),
        "H505": lambda tp, tm: PatienceFilteredMomentum(sectors=sectors),
        "H508": lambda tp, tm: SemiAnnualMomentumQualityCombo(gross_profit=gp, total_assets=ta, sectors=sectors),
    }
    tier2_rets = {}
    for name, factory in cs_factories.items():
        wf = walk_forward_cross_sectional(panel, membership, factory, folds=folds_cs, spy_returns=spy_rets)
        tier2_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")
    t2_df = pd.DataFrame(tier2_rets)
    w2 = mdr_weights(t2_df)
    tier2_port = pd.Series(t2_df.values @ w2, index=t2_df.index)
    top2 = {n: float(w) for n, w in zip(tier2_rets, w2) if w > 0.005}
    print(f"Tier2 MDR: Sharpe {sharpe(tier2_port):.3f}  MaxDD {max_drawdown(tier2_port):.2%}")
    print(f"Top weights: {top2}")

    # 50/50 hierarchical blend
    common_idx = tier1_port.index.intersection(tier2_port.index)
    blended = 0.5 * tier1_port.reindex(common_idx) + 0.5 * tier2_port.reindex(common_idx).fillna(0.0)
    lo, hi = bootstrap_sharpe_ci(blended, n_boot=5000)
    s = sharpe(blended); c = cagr(blended); dd = max_drawdown(blended)
    print(f"\nHierarchical 50/50: Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd:.2%}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (blended.index >= fold.test_start) & (blended.index < fold.test_end)
        fr = blended[mask]; fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs)); print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = blended.corr(wh.combined_returns.reindex(blended.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if s > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    write_summary(
        hypothesis_id="519", name="hierarchical-all-fold-mdr", strategy_class="HierarchicalAllFoldMDR",
        family="Set7 Cross-Program Synergy", sharpe=s, ci_low=lo, ci_high=hi, cagr=c,
        max_drawdown=dd, n_trades=0, h001_corr=float(corr),
        fold_sharpes=fold_sharpes, decision=decision,
        notes=f"2-tier hierarchical MDR: Tier1 (50%) = MDR over H001-H100 AA pool ({len(AA_POOL)} strats). Tier2 (50%) = MDR over all-fold CS only (H412+H415+H429+H447+H473+H499+H505+H508). Tier1 top: {top1}. Tier2 top: {top2}.",
        canonical_params={"tier1_top": top1, "tier2_top": top2, "tier1_weight": 0.5, "tier2_weight": 0.5},
    )

if __name__ == "__main__":
    main()
