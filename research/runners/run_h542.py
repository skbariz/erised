"""H542: Set7AllFoldMDR -- SLSQP MDR over all Set7 strategies with all 12 folds active.
25 CS strategies from H458-H473 (Batches 1-3) + H505/H506/H508/H530/H533/H535/H537/H538/H540.
Run: uv run python scripts/run_h542.py
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
from erised.strategies import MovingAverageCrossover
from erised.strategies.cross_sectional import (
    AssetTurnoverMomentumPort, BetaStabilityPort, DownsideCapturePort,
    FundamentalReversalWeekly, LongShortPCAReversal, LongShortResidualReversal,
    LongShortSectorReversal, LowBetaQualityMomentum, MinReturnCrashRiskPort,
    MinReturnMomentumCombo, MinTurnoverMomentumBand, MomentumWithPortfolioRatchet,
    MultiPeriodMomentumBlend, NetMarginMomentumPort, PatienceFilteredMomentum,
    ReturnAsymmetryPort, ReturnAutocorrelationPort, ReturnConsistencyMomentumCombo,
    ReturnConsistencyPort, ReversionQualityCombo, ROELevelMomentumPort,
    SemiAnnualMomentumQualityCombo, UpsideSharpePort, VolOfVolStabilityPort,
    VolScaledMomentumQuality,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64

def mdr_weights(returns_df):
    clean = returns_df.dropna(how="all"); cov = clean.cov().values * 252; vols = np.sqrt(np.diag(cov)); n = len(vols)
    def neg_dr(w):
        pv = w @ cov @ w; return 0.0 if pv <= 0 else -(np.dot(w, vols) / np.sqrt(pv))
    r = minimize(neg_dr, np.full(n, 1.0/n), method="SLSQP", bounds=[(0,1)]*n,
                 constraints=[{"type":"eq","fun":lambda w: w.sum()-1}], options={"maxiter":1000,"ftol":1e-12})
    return r.x

def main():
    print(f"H542: Set7AllFoldMDR (25 all-fold Set7 CS strategies)\n{SEP}")
    tickers = get_all_historical_tickers(); panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; membership = membership.loc[common]
    spy = load_research("SPY"); spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    pt = panel.columns.tolist()
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    ta = load_fundamentals_panel(pt, "total_assets", panel.index)
    ni = load_fundamentals_panel(pt, "net_income", panel.index)
    rev = load_fundamentals_panel(pt, "revenue", panel.index)
    be = load_fundamentals_panel(pt, "book_equity", panel.index)
    ocf = load_fundamentals_panel(pt, "operating_cf", panel.index)
    folds = make_folds(panel, train_years=5, test_years=2)
    print(f"Panel: {panel.shape[0]} days, {panel.shape[1]} tickers  Folds: {len(folds)}")

    cs_factories = {
        # Batch 1: all-fold DuPont+momentum templates
        "H458": lambda tp, tm: NetMarginMomentumPort(net_income=ni, revenue=rev, sectors=sectors),
        "H459": lambda tp, tm: AssetTurnoverMomentumPort(revenue=rev, total_assets=ta, sectors=sectors),
        "H460": lambda tp, tm: ROELevelMomentumPort(net_income=ni, book_equity=be, sectors=sectors),
        # Batch 2: return distribution signals (all price-only, all 12 folds)
        "H461": lambda tp, tm: MinReturnCrashRiskPort(sectors=sectors),
        "H462": lambda tp, tm: ReturnAsymmetryPort(sectors=sectors),
        "H463": lambda tp, tm: ReturnConsistencyPort(sectors=sectors),
        "H464": lambda tp, tm: DownsideCapturePort(sectors=sectors, spy_rets=spy_rets),
        "H465": lambda tp, tm: UpsideSharpePort(sectors=sectors),
        "H466": lambda tp, tm: VolOfVolStabilityPort(sectors=sectors),
        "H467": lambda tp, tm: BetaStabilityPort(sectors=sectors, spy_rets=spy_rets),
        "H468": lambda tp, tm: ReturnAutocorrelationPort(sectors=sectors),
        "H469": lambda tp, tm: MinReturnMomentumCombo(sectors=sectors),
        "H470": lambda tp, tm: ReturnConsistencyMomentumCombo(sectors=sectors),
        # Batch 3: L/S keepers
        "H471": lambda tp, tm: LongShortSectorReversal(sectors=sectors),
        "H472": lambda tp, tm: LongShortResidualReversal(spy_rets=spy_rets, sectors=sectors),
        "H473": lambda tp, tm: LongShortPCAReversal(train_prices=tp, sectors=sectors),
        # Batch 6: execution-aware
        "H505": lambda tp, tm: PatienceFilteredMomentum(sectors=sectors),
        "H506": lambda tp, tm: MinTurnoverMomentumBand(sectors=sectors),
        "H508": lambda tp, tm: SemiAnnualMomentumQualityCombo(gross_profit=gp, total_assets=ta, sectors=sectors),
        # Batch 8: multi-period momentum
        "H530": lambda tp, tm: MultiPeriodMomentumBlend(sectors=sectors),
        # Batch 9: novel architecture
        "H533": lambda tp, tm: VolScaledMomentumQuality(gross_profit=gp, total_assets=ta, sectors=sectors),
        "H535": lambda tp, tm: LowBetaQualityMomentum(gross_profit=gp, total_assets=ta, spy_returns=spy_rets, sectors=sectors),
        "H537": lambda tp, tm: MomentumWithPortfolioRatchet(sectors=sectors),
        "H538": lambda tp, tm: ReversionQualityCombo(gross_profit=gp, total_assets=ta, sectors=sectors),
        "H540": lambda tp, tm: FundamentalReversalWeekly(net_income=ni, operating_cf=ocf, total_assets=ta, sectors=sectors),
    }
    all_rets = {}
    for name, factory in cs_factories.items():
        wf = walk_forward_cross_sectional(panel, membership, factory, folds=folds, spy_returns=spy_rets)
        all_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    ret_df = pd.DataFrame(all_rets)
    print(f"\nPool: {len(all_rets)} strategies")
    ew = ret_df.mean(axis=1)
    print(f"Equal-weight: Sharpe {sharpe(ew):.3f}  MaxDD {max_drawdown(ew):.2%}")
    print("Solving MDR...")
    w = mdr_weights(ret_df)
    mdr = pd.Series(ret_df.values @ w, index=ret_df.index)
    top_w = {n: float(v) for n, v in zip(all_rets.keys(), w) if v > 0.005}
    print(f"Top weights: {top_w}")
    lo, hi = bootstrap_sharpe_ci(mdr, n_boot=5000)
    s = sharpe(mdr); c = cagr(mdr); dd = max_drawdown(mdr)
    print(f"\nMDR: Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd:.2%}")
    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    folds_aa = make_folds(spy, train_years=5, test_years=2)
    for i, fold in enumerate(folds_aa):
        mask = (mdr.index >= fold.test_start) & (mdr.index < fold.test_end)
        fr = mdr[mask]; fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs)); print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = mdr.corr(wh.combined_returns.reindex(mdr.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if s > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    write_summary(hypothesis_id="542", name="set7-all-fold-mdr", strategy_class="Set7AllFoldMDR",
        family="Set7 Batch10 Terminal Ensemble", sharpe=s, ci_low=lo, ci_high=hi, cagr=c,
        max_drawdown=dd, n_trades=0, h001_corr=float(corr), fold_sharpes=fold_sharpes, decision=decision,
        notes=f"SLSQP MDR over {len(all_rets)} Set7 all-fold CS strategies (H458-H473 + H505/H506/H508/H530/H533/H535/H537/H538/H540). All 12 folds active. MDR ceiling test with clean no-EDGAR-gap pool. Top weights: {top_w}.",
        canonical_params={"n_strategies": len(all_rets), "method": "SLSQP", "top_weights": top_w})

if __name__ == "__main__": main()
