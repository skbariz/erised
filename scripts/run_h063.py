"""H063: DefensiveSectorOverlay -- standalone walk-forward runner.

Run: uv run python scripts/run_h063.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import DefensiveSectorOverlay, MovingAverageCrossover
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    xlp = load_research("XLP")[["adj_close"]].rename(columns={"adj_close": "xlp_close"})
    prices = spy.join(vix, how="left").join(xlp, how="left")
    prices["vix_close"] = prices["vix_close"].ffill()
    prices["xlp_close"] = prices["xlp_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H063: DefensiveSectorOverlay (canonical spy_ma=200, vix_thresh=25)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: DefensiveSectorOverlay(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (spy_ma=200, vix_thresh=25):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity -- vix_thresh (spy_ma=200):")
    for vt in [20, 25, 30, 35]:
        w = walk_forward_multi(
            prices,
            lambda _, v=vt: DefensiveSectorOverlay(spy_ma=200, vix_thresh=v),
            folds=folds,
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  vix_thresh={vt}  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nSensitivity -- spy_ma (vix_thresh=25):")
    for sm in [50, 100, 200, 252]:
        w = walk_forward_multi(
            prices,
            lambda _, s=sm: DefensiveSectorOverlay(spy_ma=s, vix_thresh=25),
            folds=folds,
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  spy_ma={sm}d  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nPer-fold Sharpe:")
    for i, fold_result in enumerate(wf.fold_results):
        print(
            f"  Fold {i:>2}: {sharpe(fold_result.net_returns):+.3f}"
            f"  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})"
        )

    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}")
    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="063",
        name="defensive-sector-overlay",
        strategy_class="DefensiveSectorOverlay",
        family="equity factor",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="SPY uptrend+VIX calm→100% SPY; uptrend+VIX stressed→50/50; downtrend→100% XLP.",
        canonical_params={"spy_ma": 200, "vix_thresh": 25},
    )


if __name__ == "__main__":
    main()
