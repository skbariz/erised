"""H495: GLDMomentumOverlay -- SPY/TLT/GLD 3-asset parity with GLD 6m momentum tilt.
GLD up -> GLD weight × 1.2; GLD down -> GLD weight × 0.5, floor at 10%. Renormalize.
Run: uv run python scripts/run_h495.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_summary import write_summary
from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import GLDMomentumOverlay, MovingAverageCrossover
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi
SEP = "=" * 64

def main():
    print(f"H495: GLDMomentumOverlay (SPY/TLT/GLD parity × GLD 6m momentum tilt)\n{SEP}")
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    prices = spy.join(tlt, how="left").join(gld, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()
    prices["gld_close"] = prices["gld_close"].ffill()
    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Data: {len(prices)} rows  Folds: {len(folds)}")
    wf = walk_forward_multi(prices, lambda _: GLDMomentumOverlay(), folds=folds)
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
        hypothesis_id="495", name="gld-momentum-overlay", strategy_class="GLDMomentumOverlay",
        family="Set7 Continuous Macro AA", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="SPY/TLT/GLD inverse-vol parity base + GLD 6m momentum tilt. GLD trending up -> GLD weight × 1.2; down -> × 0.5, floor at 10% of portfolio. Renormalize. Tests whether GLD trend improves H076 3-asset parity.",
        canonical_params={"vol_lookback": 60, "mom_window": 126, "mom_up_mult": 1.2, "mom_dn_mult": 0.5, "gld_floor": 0.10},
    )

if __name__ == "__main__":
    main()
