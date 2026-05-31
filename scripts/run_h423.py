"""H423: CreditCycleSectorFilter -- HYG vs LQD credit spread gates sector tilt into XLP+XLU.

When HYG 63d return < LQD 63d return (credit widening): reduce SPY 20%, add XLP+XLU.

Run: uv run python scripts/run_h423.py
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
from quant.strategies.base import MovingAverageCrossover, CreditCycleSectorFilter
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi
SEP = "=" * 64

def main():
    print(f"H423: CreditCycleSectorFilter (HYG/LQD spread -> sector tilt)\n{SEP}")
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    hyg = load_research("HYG")[["adj_close"]].rename(columns={"adj_close": "hyg_close"})
    lqd = load_research("LQD")[["adj_close"]].rename(columns={"adj_close": "lqd_close"})
    xlp = load_research("XLP")[["adj_close"]].rename(columns={"adj_close": "xlp_close"})
    xlu = load_research("XLU")[["adj_close"]].rename(columns={"adj_close": "xlu_close"})
    prices = spy.join(tlt, how="left").join(hyg, how="left").join(lqd, how="left").join(xlp, how="left").join(xlu, how="left")
    for col in ["tlt_close", "hyg_close", "lqd_close", "xlp_close", "xlu_close"]:
        prices[col] = prices[col].ffill()
    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Data: {prices.shape[0]} days  Folds: {len(folds)}")
    wf = walk_forward_multi(prices, lambda _: CreditCycleSectorFilter(), folds=folds)
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
        hypothesis_id="423", name="credit-cycle-sector-filter", strategy_class="CreditCycleSectorFilter",
        family="macro sector proxy", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="HYG 63d ret < LQD 63d ret â†’ reduce SPY 20%, add 10% XLP + 10% XLU. Credit cycle = policy risk regime signal.",
        canonical_params={"vol_lookback": 60, "credit_lookback": 63, "tilt_pct": 0.20},
    )

if __name__ == "__main__":
    main()

