"""H421: RegulatedSectorTiming -- H021 + TLT trend gate â†’ XLP/XLU defensive tilt.

When TLT > 200d MA (rates falling): reduce SPY by 15%, split into XLP + XLU.
Cohen, Diether & Malloy (2013): policy events benefit regulated defensive sectors.

Run: uv run python scripts/run_h421.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from write_summary import write_summary
from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies import MovingAverageCrossover, RegulatedSectorTiming
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi
SEP = "=" * 64

def main():
    print(f"H421: RegulatedSectorTiming (TLT trend -> XLP+XLU tilt)\n{SEP}")
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    xlp = load_research("XLP")[["adj_close"]].rename(columns={"adj_close": "xlp_close"})
    xlu = load_research("XLU")[["adj_close"]].rename(columns={"adj_close": "xlu_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    prices = spy.join(tlt, how="left").join(xlp, how="left").join(xlu, how="left").join(vix, how="left")
    for col in ["tlt_close", "xlp_close", "xlu_close", "vix_close"]:
        prices[col] = prices[col].ffill()
    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Data: {prices.shape[0]} days  Folds: {len(folds)}")
    wf = walk_forward_multi(prices, lambda _: RegulatedSectorTiming(), folds=folds)
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
        hypothesis_id="421", name="regulated-sector-timing", strategy_class="RegulatedSectorTiming",
        family="macro sector proxy", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="H021 + TLT>200dMA gate reduces SPY 15%, adds 7.5% XLP + 7.5% XLU. Rate-falling regime benefits regulated defensives (Cohen et al. 2013 policy proxy).",
        canonical_params={"vol_lookback": 60, "tlt_ma": 200, "tilt_pct": 0.15},
    )

if __name__ == "__main__":
    main()

