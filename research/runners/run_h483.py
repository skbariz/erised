"""H483: CSReturnDispersionAAPort -- H021 with cross-sectional return dispersion signal.
Low dispersion → passive/ETF regime → SPY × 0.75. Pure price, full 12-fold.
Run: uv run python scripts/run_h483.py
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
from erised.strategies.base import CSReturnDispersionAAPort, MovingAverageCrossover
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64

def main():
    print(f"H483: CSReturnDispersionAAPort (CS return dispersion regime on H021)\n{SEP}")
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]
    daily_rets = panel.pct_change()
    ret_5d = panel.pct_change(5)
    cs_daily_disp = ret_5d.std(axis=1)
    cs_dispersion = cs_daily_disp.rolling(21, min_periods=10).mean()

    spy = load_research("SPY"); tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices = spy.join(tlt, how="left"); prices["tlt_close"] = prices["tlt_close"].ffill()
    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Data: {len(prices)} rows  Folds: {len(folds)}")
    wf = walk_forward_multi(prices, lambda _: CSReturnDispersionAAPort(cs_dispersion=cs_dispersion), folds=folds)
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
        hypothesis_id="483", name="cs-return-dispersion-aa", strategy_class="CSReturnDispersionAAPort",
        family="Set7 Hybrid AA+CS", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="H021 + CS return dispersion regime. Rolling 21d avg of daily CS std of 5d returns. Low dispersion → passive/ETF regime → SPY × 0.75. Israeli et al. (2017) ETF synchronization signal.",
        canonical_params={"vol_lookback": 60, "signal_window": 252, "low_disp_scale": 0.75},
    )

if __name__ == "__main__":
    main()
