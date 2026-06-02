"""H484: QualityBreadthAAPort -- H021 with fraction of stocks improving quality.
>60% stocks with GP/TA above own 252d median → SPY × 1.20; <40% → × 0.80.
Run: uv run python scripts/run_h484.py
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
from quant.strategies.base import QualityBreadthAAPort, MovingAverageCrossover
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64

def main():
    print(f"H484: QualityBreadthAAPort (quality improvement breadth overlay on H021)\n{SEP}")
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; pt = panel.columns.tolist()
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    ta = load_fundamentals_panel(pt, "total_assets", panel.index)
    ta_safe = ta.where(ta > 0)
    gp_ta = (gp / ta_safe).clip(0.0, 5.0)
    gp_ta_median = gp_ta.rolling(252, min_periods=63).median()
    above_median = (gp_ta > gp_ta_median).astype(float)
    quality_breadth = above_median.mean(axis=1)

    spy = load_research("SPY"); tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices = spy.join(tlt, how="left"); prices["tlt_close"] = prices["tlt_close"].ffill()
    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Data: {len(prices)} rows  Folds: {len(folds)}")
    wf = walk_forward_multi(prices, lambda _: QualityBreadthAAPort(quality_breadth=quality_breadth), folds=folds)
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
        hypothesis_id="484", name="quality-breadth-aa", strategy_class="QualityBreadthAAPort",
        family="Set7 Hybrid AA+CS", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="H021 + quality breadth. Fraction of S&P 500 stocks with GP/TA above own 252d median. >60% → SPY × 1.20; <40% → × 0.80. Breadth measures whether improvement is broad-based.",
        canonical_params={"vol_lookback": 60, "bull_thresh": 0.60, "bear_thresh": 0.40, "bull_scale": 1.20, "bear_scale": 0.80},
    )

if __name__ == "__main__":
    main()
