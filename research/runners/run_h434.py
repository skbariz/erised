"""H434: FinancialFragilityAvoid -- Altman Z-score proxy from Compustat fundamentals.
Z_proxy = 1.2*WC/TA + 1.4*NI/TA_mean + 3.3*GP/TA + 0.6*BE/LTD + 1.0*Rev/TA.
Kisgen (2006). Sector-neutral top 20%, quarterly rebalance.
Run: uv run python scripts/run_h434.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from write_summary import write_summary
from erised.data.fundamentals import load_fundamentals_panel
from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import MovingAverageCrossover
from erised.strategies.cross_sectional import FinancialFragilityAvoid
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64

def main():
    print(f"H434: FinancialFragilityAvoid (Altman Z proxy)\n{SEP}")
    membership = load_sp500_members(); tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; membership = membership.loc[common]
    spy = load_research("SPY"); spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    folds = make_folds(panel, train_years=5, test_years=2); pt = panel.columns.tolist()
    ca = load_fundamentals_panel(pt, "current_assets", panel.index)
    cl = load_fundamentals_panel(pt, "current_liabilities", panel.index)
    ta = load_fundamentals_panel(pt, "total_assets", panel.index)
    ni = load_fundamentals_panel(pt, "net_income", panel.index)
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    be = load_fundamentals_panel(pt, "book_equity", panel.index)
    ltd = load_fundamentals_panel(pt, "long_term_debt", panel.index)
    rev = load_fundamentals_panel(pt, "revenue", panel.index)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers  Folds: {len(folds)}")
    wf = walk_forward_cross_sectional(
        panel, membership,
        lambda tp, tm: FinancialFragilityAvoid(current_assets=ca, current_liabilities=cl, total_assets=ta, net_income=ni, gross_profit=gp, book_equity=be, long_term_debt=ltd, revenue=rev, sectors=sectors),
        folds=folds, spy_returns=spy_rets,
    )
    rep = wf.combined_report; lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical: Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")
    print("\nPer-fold Sharpe:")
    for i, f in enumerate(wf.fold_results):
        print(f"  Fold {i:>2}: {sharpe(f.net_returns):+.3f}  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})")
    fs = make_folds(spy, train_years=5, test_years=2)
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=fs)
    corr = wf.combined_returns.corr(wh.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    write_summary(
        hypothesis_id="434", name="financial-fragility-avoid", strategy_class="FinancialFragilityAvoid",
        family="credit risk", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="Altman Z proxy: 1.2*(CA-CL)/TA + 1.4*NI/TA_252d_mean + 3.3*GP/TA + 0.6*BE/LTD + 1.0*Rev/TA. Kisgen (2006) rating threshold proxy.",
        canonical_params={"within_sector_quantile": 0.20, "rebalance_period": 63, "ni_window": 252},
    )

if __name__ == "__main__":
    main()
