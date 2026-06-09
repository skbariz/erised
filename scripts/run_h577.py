"""H577: SectorMomentumVolTarget -- sector ETF momentum with vol-targeting.

Ranks equity ETFs (XLP, XLU, IVE, IVW, QQQ, IWM, VNQ) by 6m momentum;
holds top-3 risk-parity weighted. Applies vol-targeting (target_vol=12%) to
total exposure. Pure momentum signal, no regime conditioning.

Run: uv run python scripts/run_h577.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover, SectorMomentumVolTarget
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64

ETF_NAMES = ["TLT", "XLP", "XLU", "IVE", "IVW", "QQQ", "IWM", "VNQ"]


def main():
    spy = load_research("SPY")
    prices = spy.copy()
    for name in ETF_NAMES:
        etf = load_research(name)[["adj_close"]].rename(
            columns={"adj_close": f"{name.lower()}_close"}
        )
        prices = prices.join(etf, how="left")
    for col in [c for c in prices.columns if c != "adj_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H577: SectorMomentumVolTarget\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")

    wf = walk_forward_multi(prices, lambda _: SectorMomentumVolTarget(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (top_n=3, target_vol=12%): Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")

    print("\nSensitivity -- top_n:")
    for tn in [2, 3, 4]:
        w = walk_forward_multi(prices, lambda _, t=tn: SectorMomentumVolTarget(top_n=t), folds=folds)
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(f"  top_n={tn}  Sharpe {w.combined_report.sharpe:.3f}  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}")

    print("\nPer-fold Sharpe:")
    for i, fold_result in enumerate(wf.fold_results):
        print(f"  Fold {i:>2}: {sharpe(fold_result.net_returns):+.3f}  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})")

    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="577",
        name="sector-momentum-vol-target",
        strategy_class="SectorMomentumVolTarget",
        family="Set8 Batch2 AA",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="7-ETF sector momentum top-3, vol-targeted at 12%. Pure momentum, no regime gate.",
        canonical_params={"mom_window": 126, "vol_lookback": 60, "top_n": 3, "target_vol": 0.12},
    )


if __name__ == "__main__":
    main()
