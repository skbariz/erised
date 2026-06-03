"""H527: RevenueMarginConsistencyPort -- fraction of 8Q where rev_growth>0 AND gp_margin stable. Quarterly.
Joint persistence: sustained revenue growth AND margin stability simultaneously.
Run: uv run python scripts/run_h527.py
"""
from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from write_summary import write_summary
from quant.data.fundamentals import load_fundamentals_panel
from quant.data.loader import load_research
from quant.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover
from quant.strategies.cross_sectional import RevenueMarginConsistencyPort
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64

def main():
    print(f"H527: RevenueMarginConsistencyPort (joint rev_growth>0 & margin stable, 8Q, quarterly)\n{SEP}")
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; membership = membership.loc[common]
    spy = load_research("SPY"); spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    pt = panel.columns.tolist()
    rev = load_fundamentals_panel(pt, "revenue", panel.index)
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    folds = make_folds(panel, train_years=5, test_years=2)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers  Folds: {len(folds)}")
    wf = walk_forward_cross_sectional(
        panel, membership,
        lambda tp, tm: RevenueMarginConsistencyPort(revenue=rev, gross_profit=gp, sectors=sectors),
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
    write_summary(hypothesis_id="527", name="revenue-margin-consistency", strategy_class="RevenueMarginConsistencyPort",
        family="Set7 Novel Fundamentals", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="Fraction of trailing 8 quarters where: (1) quarterly revenue growth > 0 AND (2) GP/Revenue quarterly change > -1%. Joint persistence of growth AND margin stability. Novel joint condition not tested in H411-H415. Sector-neutral top 20%, quarterly.",
        canonical_params={"lookback": 504, "quarterly_step": 63, "within_sector_quantile": 0.20, "rebalance_period": 63})

if __name__ == "__main__":
    main()
