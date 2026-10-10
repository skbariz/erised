"""H516: CoreDiversifierTripleMDR -- MDR over focused 20-strategy pool (best H300 + Set7 stars).
Top 10 H300 AA contributors + 10 CS all-fold stars. Tests small-pool MDR vs large-pool H300.
Run: uv run python scripts/run_h516.py
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
    CarryMomentumSPYTLT, CorrelationRiskParityHybrid, EqualWeightDefensiveFactor,
    FourAssetRiskParity, GoldFearGauge, LowVolTargetPortfolio, MovingAverageCrossover,
    MultiAssetStrategy, RatchetDrawdownProtection, RiskAdjustedMomentum, SeasonalRiskParity,
    SellInMayBondConfirm, VixRiskParityEnhanced, VolScaledTrend, VrpCalendarHybrid,
)
from erised.strategies.cross_sectional import (
    CreditQualityMomentumCombo, FundamentalDivergenceAvoid, LongShortPCAReversal,
    LongShortReversal, PatienceFilteredMomentum, SectorRelativeMarginProxy,
    SemiAnnualMomentumQualityCombo, StableMarginMomentum, SustainabilityFilteredProfitability,
    VolAccelerationCSSwitch,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional, walk_forward_multi
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64

def max_diversification_weights(returns_df):
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
    print(f"H516: CoreDiversifierTripleMDR (20-strategy focused pool)\n{SEP}")
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
    ca = load_fundamentals_panel(pt, "current_assets", panel.index)
    cl = load_fundamentals_panel(pt, "current_liabilities", panel.index)
    folds_cs = make_folds(panel, train_years=5, test_years=2)
    print(f"Panel: {panel.shape[0]} days, {panel.shape[1]} tickers")

    # Top 10 H300 AA contributors (by H300 MDR weight)
    aa_pool = {
        "H016": VolScaledTrend(), "H043": SellInMayBondConfirm(), "H068": VrpCalendarHybrid(),
        "H076": FourAssetRiskParity(), "H065": EqualWeightDefensiveFactor(), "H078": SeasonalRiskParity(),
        "H035": GoldFearGauge(), "H086": LowVolTargetPortfolio(), "H087": RatchetDrawdownProtection(),
        "H097": CorrelationRiskParityHybrid(),
    }
    all_rets = {}
    for name, strat in aa_pool.items():
        wf = run_aa(prices_aa, strat, folds_aa)
        all_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    # 10 CS all-fold stars (H194 + Set6 stars + Set7 near-zero H001 stars)
    cs_factories = {
        "H194": lambda tp, tm: LongShortReversal(sectors=sectors),
        "H265": lambda tp, tm: SustainabilityFilteredProfitability(gross_profit=gp, total_assets=ta, current_assets=ca, current_liabilities=cl, net_income=ni, operating_cf=ocf, sectors=sectors),
        "H412": lambda tp, tm: FundamentalDivergenceAvoid(net_income=ni, total_assets=ta, sectors=sectors),
        "H415": lambda tp, tm: StableMarginMomentum(gross_profit=gp, revenue=rev, sectors=sectors),
        "H429": lambda tp, tm: SectorRelativeMarginProxy(gross_profit=gp, revenue=rev, sectors=sectors),
        "H447": lambda tp, tm: CreditQualityMomentumCombo(gross_profit=gp, total_assets=ta, long_term_debt=ltd, operating_cf=ocf, sectors=sectors),
        "H473": lambda tp, tm: LongShortPCAReversal(train_prices=tp, sectors=sectors),
        "H499": lambda tp, tm: VolAccelerationCSSwitch(sectors=sectors, gross_profit=gp, total_assets=ta, spy_returns=spy_rets),
        "H505": lambda tp, tm: PatienceFilteredMomentum(sectors=sectors),
        "H508": lambda tp, tm: SemiAnnualMomentumQualityCombo(gross_profit=gp, total_assets=ta, sectors=sectors),
    }
    for name, factory in cs_factories.items():
        wf = walk_forward_cross_sectional(panel, membership, factory, folds=folds_cs, spy_returns=spy_rets)
        all_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    ret_df = pd.DataFrame(all_rets)
    print(f"\nPool: {len(all_rets)} strategies")
    ew = ret_df.mean(axis=1)
    print(f"Equal-weight: Sharpe {sharpe(ew):.3f}  MaxDD {max_drawdown(ew):.2%}")
    print("Solving MDR optimisation...")
    w_mdr = max_diversification_weights(ret_df)
    mdr_rets = pd.Series(ret_df.values @ w_mdr, index=ret_df.index)
    top_w = {n: float(w) for n, w in zip(all_rets.keys(), w_mdr) if w > 0.005}
    print(f"Top weights: {top_w}")
    lo, hi = bootstrap_sharpe_ci(mdr_rets, n_boot=5000)
    s = sharpe(mdr_rets); c = cagr(mdr_rets); dd = max_drawdown(mdr_rets)
    print(f"\nMDR: Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd:.2%}")
    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (mdr_rets.index >= fold.test_start) & (mdr_rets.index < fold.test_end)
        fr = mdr_rets[mask]; fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs)); print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = mdr_rets.corr(wh.combined_returns.reindex(mdr_rets.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if s > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    write_summary(
        hypothesis_id="516", name="core-diversifier-triple-mdr", strategy_class="CoreDiversifierTripleMDR",
        family="Set7 Cross-Program Synergy", sharpe=s, ci_low=lo, ci_high=hi, cagr=c,
        max_drawdown=dd, n_trades=0, h001_corr=float(corr),
        fold_sharpes=fold_sharpes, decision=decision,
        notes=f"SLSQP MDR over {len(all_rets)}-strategy focused pool: top 10 H300 AA contributors + 10 best CS all-fold stars. Tests small-pool MDR vs H300's 52-strategy pool. Top weights: {top_w}.",
        canonical_params={"n_strategies": len(all_rets), "method": "SLSQP", "top_weights": top_w},
    )

if __name__ == "__main__":
    main()
