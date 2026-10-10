"""H536: EarningsQualityGatedMomentum -- 6m momentum with OCF>0, NI>0, OCF/NI>0.5 gate. Monthly.
Cash-backed earnings gate prevents momentum crashes in low-quality names.
Run: uv run python scripts/run_h536.py
"""
from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from write_summary import write_summary
from erised.data.fundamentals import load_fundamentals_panel
from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies import MovingAverageCrossover
from erised.strategies.cross_sectional import EarningsQualityGatedMomentum
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64

def main():
    print(f"H536: EarningsQualityGatedMomentum (OCF>0 & NI>0 & OCF/NI>0.5 gate, 6m mom, monthly)\n{SEP}")
    tickers = get_all_historical_tickers(); panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; membership = membership.loc[common]
    spy = load_research("SPY"); spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    pt = panel.columns.tolist()
    ocf = load_fundamentals_panel(pt, "operating_cf", panel.index)
    ni = load_fundamentals_panel(pt, "net_income", panel.index)
    folds = make_folds(panel, train_years=5, test_years=2)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers  Folds: {len(folds)}")
    wf = walk_forward_cross_sectional(panel, membership,
        lambda tp, tm: EarningsQualityGatedMomentum(operating_cf=ocf, net_income=ni, sectors=sectors),
        folds=folds, spy_returns=spy_rets)
    rep = wf.combined_report; lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical: Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")
    print("\nPer-fold Sharpe:")
    for i, f in enumerate(wf.fold_results): print(f"  Fold {i:>2}: {sharpe(f.net_returns):+.3f}  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})")
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=make_folds(spy, train_years=5, test_years=2))
    corr = wf.combined_returns.corr(wh.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    write_summary(hypothesis_id="536", name="earnings-quality-gated-momentum", strategy_class="EarningsQualityGatedMomentum",
        family="Set7 Sector-Relative", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="6m momentum filtered to cash-backed earnings universe: OCF>0 AND NI>0 AND OCF/NI>0.5. Sector-neutral top 20% within gate-passing stocks. Tests if quality gate prevents momentum crashes. Monthly rebalance.",
        canonical_params={"mom_lookback": 126, "within_sector_quantile": 0.20, "rebalance_period": 21})

if __name__ == "__main__": main()
