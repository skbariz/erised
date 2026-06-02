"""H475: LongShortPCAOUReversal -- dollar-neutral L/S PCA OU (H192 L/S form).
H192 long-only Sharpe 1.175 -- best long-only short-term signal in program.
Primary MDR ceiling crack attempt. Weekly rebalance.
Run: uv run python scripts/run_h475.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from write_summary import write_summary
from quant.data.loader import load_research
from quant.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover
from quant.strategies.cross_sectional import LongShortPCAOUReversal
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64

def main():
    print(f"H475: LongShortPCAOUReversal (dollar-neutral L/S PCA OU -- MDR ceiling crack)\n{SEP}")
    membership = load_sp500_members(); tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; membership = membership.loc[common]
    spy = load_research("SPY"); spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    folds = make_folds(panel, train_years=5, test_years=2)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers  Folds: {len(folds)}")
    wf = walk_forward_cross_sectional(
        panel, membership,
        lambda tp, tm: LongShortPCAOUReversal(train_prices=tp, sectors=sectors),
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
        hypothesis_id="475", name="long-short-pca-ou-reversal", strategy_class="LongShortPCAOUReversal",
        family="Set7 L/S Market-Neutral", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="Dollar-neutral L/S PCA OU: K=10 PCA + OU z-score |z|>1.5. H192 long-only = 1.175 (program best short-term). L/S amplifies purest signal. Primary MDR ceiling crack attempt. Weekly rebalance.",
        canonical_params={"n_components": 10, "ou_window": 252, "entry_z": 1.5, "within_sector_quantile": 0.20, "rebalance_period": 5},
    )

if __name__ == "__main__":
    main()
