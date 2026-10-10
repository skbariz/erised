"""H487: MomentumBreadthAAPort -- H021 with % of stocks above 200d MA.
Classic breadth indicator as rule-based AA. >60% → SPY × 1.20; <40% → × 0.80.
Run: uv run python scripts/run_h487.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from write_summary import write_summary
from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import MomentumBreadthAAPort, MovingAverageCrossover
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64

def main():
    print(f"H487: MomentumBreadthAAPort (% stocks above 200d MA overlay on H021)\n{SEP}")
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]
    ma_200 = panel.rolling(200, min_periods=100).mean()
    above_200 = (panel > ma_200).astype(float)
    breadth_signal = above_200.mean(axis=1)

    spy = load_research("SPY"); tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices = spy.join(tlt, how="left"); prices["tlt_close"] = prices["tlt_close"].ffill()
    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Data: {len(prices)} rows  Folds: {len(folds)}")
    wf = walk_forward_multi(prices, lambda _: MomentumBreadthAAPort(breadth_signal=breadth_signal), folds=folds)
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
        hypothesis_id="487", name="momentum-breadth-aa", strategy_class="MomentumBreadthAAPort",
        family="Set7 Hybrid AA+CS", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="H021 + % stocks above 200d MA breadth. Classic Zweig-style market breadth as rule-based AA. >60% → SPY × 1.20; <40% → × 0.80. Pure price signal, full 12-fold coverage.",
        canonical_params={"vol_lookback": 60, "bull_thresh": 0.60, "bear_thresh": 0.40, "bull_scale": 1.20, "bear_scale": 0.80},
    )

if __name__ == "__main__":
    main()
