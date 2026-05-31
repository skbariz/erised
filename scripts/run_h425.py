"""H425: MacroRegimeSectorBlend -- 3-signal composite macro regime → sector blended allocation.

Signals: VIX<25, TLT>63dMA, HYG improving. 3 bullish→80/20; 2→55/25/10/10; 0-1→30/40/15/15.

Run: uv run python scripts/run_h425.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from write_summary import write_summary
from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover, MacroRegimeSectorBlend
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi
SEP = "=" * 64

def main():
    print(f"H425: MacroRegimeSectorBlend (3-signal regime sector allocation)\n{SEP}")
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    hyg = load_research("HYG")[["adj_close"]].rename(columns={"adj_close": "hyg_close"})
    xlp = load_research("XLP")[["adj_close"]].rename(columns={"adj_close": "xlp_close"})
    xlu = load_research("XLU")[["adj_close"]].rename(columns={"adj_close": "xlu_close"})
    prices = spy.join(tlt, how="left").join(vix, how="left").join(hyg, how="left").join(xlp, how="left").join(xlu, how="left")
    for col in ["tlt_close", "vix_close", "hyg_close", "xlp_close", "xlu_close"]:
        prices[col] = prices[col].ffill()
    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Data: {prices.shape[0]} days  Folds: {len(folds)}")
    wf = walk_forward_multi(prices, lambda _: MacroRegimeSectorBlend(), folds=folds)
    rep = wf.combined_report; lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical: Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")
    print("\nPer-fold Sharpe:")
    for i, f in enumerate(wf.fold_results):
        print(f"  Fold {i:>2}: {sharpe(f.net_returns):+.3f}  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})")
    folds_s = make_folds(spy, train_years=5, test_years=2)
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_s)
    corr = wf.combined_returns.corr(wh.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    write_summary(
        hypothesis_id="425", name="macro-regime-sector-blend", strategy_class="MacroRegimeSectorBlend",
        family="macro sector proxy", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="3 signals: VIX<25, TLT>63dMA, HYG improving. n_bull=3: 80/20; n=2: 55/25/10/10; n<=1: 30/40/15/15 (SPY/TLT/XLP/XLU).",
        canonical_params={"vix_threshold": 25.0, "tlt_ma": 63, "credit_lookback": 63},
    )

if __name__ == "__main__":
    main()
