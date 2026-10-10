"""H518: RatchetH300Port -- H300 MDR weights + H087-style ratchet drawdown protection.
Re-runs H300's 17 components. When portfolio DD > 5%, scale by (1 - dd/0.15).
Run: uv run python scripts/run_h518.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from write_summary import write_summary
from erised.data.fundamentals import load_fundamentals_panel
from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies.base import (
    CarryMomentumSPYTLT, CorrelationRiskParityHybrid, EqualWeightDefensiveFactor,
    FourAssetRiskParity, GoldFearGauge, LowVolTargetPortfolio, MovingAverageCrossover,
    MultiAssetStrategy, RiskAdjustedMomentum, SeasonalRiskParity, SellInMayBondConfirm,
    VixRiskParityEnhanced, VolScaledTrend, VrpCalendarHybrid,
)
from erised.strategies.cross_sectional import (
    BettingAgainstBeta, InvestmentEfficiencyRatio, KnowledgeReturnProxy,
    LongShortReversal, ResidualOUMeanReversion, SustainabilityFilteredProfitability,
    VolScaledMomentum,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional, walk_forward_multi
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64

H300_WEIGHTS = {
    "H016": 0.1099, "H035": 0.0623, "H043": 0.1007, "H060": 0.0283, "H065": 0.0701,
    "H068": 0.0949, "H076": 0.0804, "H078": 0.0305, "H082": 0.0084, "H097": 0.0262,
    "H157": 0.0526, "H175": 0.0075, "H189": 0.0086, "H194": 0.2279,
    "H268": 0.0097, "H265": 0.0727, "H287": 0.0088,
}

def run_aa(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy): return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)

def main():
    print(f"H518: RatchetH300Port (H300 MDR weights + ratchet drawdown protection)\n{SEP}")
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
    ca = load_fundamentals_panel(pt, "current_assets", panel.index)
    cl = load_fundamentals_panel(pt, "current_liabilities", panel.index)
    ocf = load_fundamentals_panel(pt, "operating_cf", panel.index)
    folds_cs = make_folds(panel, train_years=5, test_years=2)
    print(f"Panel: {panel.shape[0]} days, {panel.shape[1]} tickers")

    aa_strats = {
        "H016": VolScaledTrend(), "H035": GoldFearGauge(), "H043": SellInMayBondConfirm(),
        "H060": CarryMomentumSPYTLT(), "H065": EqualWeightDefensiveFactor(), "H068": VrpCalendarHybrid(),
        "H076": FourAssetRiskParity(), "H078": SeasonalRiskParity(), "H082": RiskAdjustedMomentum(),
        "H097": CorrelationRiskParityHybrid(),
    }
    comp_rets = {}
    for name, strat in aa_strats.items():
        wf = run_aa(prices_aa, strat, folds_aa)
        comp_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    cs_factories = {
        "H157": lambda tp, tm: VolScaledMomentum(vol_target=0.12),
        "H175": lambda tp, tm: BettingAgainstBeta(spy_rets=spy_rets, sectors=sectors),
        "H189": lambda tp, tm: ResidualOUMeanReversion(spy_rets=spy_rets, sectors=sectors),
        "H194": lambda tp, tm: LongShortReversal(sectors=sectors),
        "H265": lambda tp, tm: SustainabilityFilteredProfitability(gross_profit=gp, total_assets=ta, current_assets=ca, current_liabilities=cl, net_income=ni, operating_cf=ocf, sectors=sectors),
        "H268": lambda tp, tm: KnowledgeReturnProxy(gross_profit=gp, total_assets=ta, sectors=sectors),
        "H287": lambda tp, tm: InvestmentEfficiencyRatio(revenue=rev, total_assets=ta, current_assets=ca, sectors=sectors),
    }
    for name, factory in cs_factories.items():
        wf = walk_forward_cross_sectional(panel, membership, factory, folds=folds_cs, spy_returns=spy_rets)
        comp_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    ret_df = pd.DataFrame(comp_rets)
    weights_vec = np.array([H300_WEIGHTS.get(n, 0.0) for n in comp_rets.keys()])
    weights_vec /= weights_vec.sum()
    h300_rets = pd.Series(ret_df.values @ weights_vec, index=ret_df.index)
    print(f"\nH300 reconstructed: Sharpe {sharpe(h300_rets):.3f}  MaxDD {max_drawdown(h300_rets):.2%}")

    # Apply H087-style ratchet: when drawdown > dd_trigger, scale by (1 - dd/dd_limit)
    cum = (1.0 + h300_rets.fillna(0.0)).cumprod()
    peak = cum.expanding().max()
    dd = (cum / peak - 1.0).fillna(0.0)
    dd_trigger, dd_limit = 0.05, 0.15
    scale = (1.0 + dd.shift(1).fillna(0.0) / dd_limit).clip(lower=0.0, upper=1.0)
    # Ratchet only activates when past drawdown > trigger
    below_trigger = dd.shift(1).fillna(0.0) < -dd_trigger
    scale = scale.where(below_trigger, 1.0)
    ratcheted = h300_rets * scale
    lo, hi = bootstrap_sharpe_ci(ratcheted, n_boot=5000)
    s = sharpe(ratcheted); c = cagr(ratcheted); dd_final = max_drawdown(ratcheted)
    print(f"Ratcheted: Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd_final:.2%}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (ratcheted.index >= fold.test_start) & (ratcheted.index < fold.test_end)
        fr = ratcheted[mask]; fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs)); print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = ratcheted.corr(wh.combined_returns.reindex(ratcheted.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if s > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    write_summary(
        hypothesis_id="518", name="ratchet-h300", strategy_class="RatchetH300Port",
        family="Set7 Cross-Program Synergy", sharpe=s, ci_low=lo, ci_high=hi, cagr=c,
        max_drawdown=dd_final, n_trades=0, h001_corr=float(corr),
        fold_sharpes=fold_sharpes, decision=decision,
        notes="H300 MDR portfolio (17 non-zero-weight components) with H087-style ratchet drawdown protection. When portfolio drawdown from peak exceeds 5%, scale by (1 - dd/0.15). Ratchet resets at new high-water mark.",
        canonical_params={"dd_trigger": 0.05, "dd_limit": 0.15, "h300_weights": H300_WEIGHTS},
    )

if __name__ == "__main__":
    main()
