"""H390: ESGProxyComposite -- batch-8 ESG proxy ensemble.
Run: uv run python scripts/run_h390.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from write_summary import write_summary
from quant.data.fundamentals import load_fundamentals_panel
from quant.data.loader import load_research
from quant.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover
from quant.strategies.cross_sectional import ESGProxyComposite
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64

def main():
    print(f"H390: ESGProxyComposite (batch-8 ESG ensemble)\n{SEP}")
    membership = load_sp500_members(); tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; membership = membership.loc[common]
    spy = load_research("SPY"); spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    folds = make_folds(panel, train_years=5, test_years=2); pt = panel.columns.tolist()
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    ta = load_fundamentals_panel(pt, "total_assets", panel.index)
    ltd = load_fundamentals_panel(pt, "long_term_debt", panel.index)
    sh = load_fundamentals_panel(pt, "shares_outstanding", panel.index)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers  Folds: {len(folds)}")
    wf = walk_forward_cross_sectional(panel, membership, lambda tp, tm: ESGProxyComposite(gross_profit=gp, total_assets=ta, long_term_debt=ltd, shares_outstanding=sh, sectors=sectors), folds=folds, spy_returns=spy_rets)
    rep = wf.combined_report; lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical: Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")
    print("\nPer-fold Sharpe:")
    for i, f in enumerate(wf.fold_results):
        print(f"  Fold {i:>2}: {sharpe(f.net_returns):+.3f}  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})")
    fs = make_folds(spy, train_years=5, test_years=2)
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=fs)
    corr = wf.combined_returns.corr(wh.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    write_summary(hypothesis_id="390", name="esg-proxy-composite", strategy_class="ESGProxyComposite", family="esg-proxy", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr, max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr), fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision, notes="Normalized blend: ex-Staples GP + ex-Energy GP + controversy-avoided GP + GP/TA anchor.", canonical_params={"clip_val": 2.0, "within_sector_quantile": 0.20, "rebalance_period": 63})

if __name__ == "__main__":
    main()
