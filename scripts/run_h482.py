"""H482: MarketAccrualTimingPort -- H021 with market-wide accruals quality signal.
High aggregate accruals → SPY × 0.80; low accruals (cash-backed) → × 1.20.
Run: uv run python scripts/run_h482.py
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
from quant.strategies.base import MarketAccrualTimingPort, MovingAverageCrossover
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64

def main():
    print(f"H482: MarketAccrualTimingPort (market-wide accruals overlay on H021)\n{SEP}")
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; pt = panel.columns.tolist()
    ni = load_fundamentals_panel(pt, "net_income", panel.index)
    ocf = load_fundamentals_panel(pt, "operating_cf", panel.index)
    ta = load_fundamentals_panel(pt, "total_assets", panel.index)
    ta_safe = ta.where(ta > 0)
    accruals = ((ni - ocf) / ta_safe).clip(-1.0, 1.0)
    market_accruals = accruals.mean(axis=1).rolling(252, min_periods=63).mean()

    spy = load_research("SPY"); tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices = spy.join(tlt, how="left"); prices["tlt_close"] = prices["tlt_close"].ffill()
    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Data: {len(prices)} rows  Folds: {len(folds)}")
    wf = walk_forward_multi(prices, lambda _: MarketAccrualTimingPort(market_accruals=market_accruals), folds=folds)
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
        hypothesis_id="482", name="market-accrual-timing", strategy_class="MarketAccrualTimingPort",
        family="Set7 Hybrid AA+CS", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="H021 + market-wide accruals timing. Rolling 252d mean (NI-OCF)/TA across S&P 500. High accruals → SPY × 0.80; low → × 1.20. Most robust market-level fundamental predictor.",
        canonical_params={"vol_lookback": 60, "signal_window": 252, "bull_scale": 1.20, "bear_scale": 0.80},
    )

if __name__ == "__main__":
    main()
