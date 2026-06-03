"""H492: VIXPercentileScalerPort -- H021 with linear SPY scaling from VIX 252d percentile rank.
VIX at min pct → SPY × 1.4 (equity tilt); VIX at max pct → SPY × 0.6 (defensive). Continuous.
Run: uv run python scripts/run_h492.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_summary import write_summary
from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import VIXPercentileScalerPort, MovingAverageCrossover
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64

def main():
    print(f"H492: VIXPercentileScalerPort (VIX 252d pct rank → SPY scale [0.6, 1.4])\n{SEP}")
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    import pandas as pd
    vix = pd.read_parquet(DATA_DIR / "VIX_research.parquet")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    prices = spy.join(tlt, how="left").join(vix, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()
    prices["vix_close"] = prices["vix_close"].ffill()
    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Data: {len(prices)} rows  Folds: {len(folds)}")
    wf = walk_forward_multi(prices, lambda _: VIXPercentileScalerPort(), folds=folds)
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
        hypothesis_id="492", name="vix-percentile-scaler", strategy_class="VIXPercentileScalerPort",
        family="Set7 Continuous Macro AA", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="H021 + VIX percentile scaler. VIX 252d trailing percentile rank → linear SPY weight in [0.6, 1.4]. Low VIX pct → equity boost; high pct → defensive. Continuous analog of H066 binary VIX>35 spike override.",
        canonical_params={"vol_lookback": 60, "pct_window": 252, "scale_lo": 0.60, "scale_hi": 1.40},
    )

if __name__ == "__main__":
    main()
