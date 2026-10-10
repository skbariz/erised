"""H096: PCARiskRegime -- standalone walk-forward runner.

Rolling 252-day PCA on SPY/TLT/GLD returns; sign of PC1 score
(21-day cumulative return projected onto PC1) determines regime:
positive -> SPY, negative -> TLT.

Run: uv run python scripts/run_h096.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies import MovingAverageCrossover, PCARiskRegime
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})

    prices = spy.join(tlt, how="left").join(gld, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()
    prices["gld_close"] = prices["gld_close"].ffill()

    folds = make_folds(spy, train_years=5, test_years=2)
    print(f"H096: PCARiskRegime\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    strat = PCARiskRegime(pca_lookback=252, signal_days=21)
    wf = walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    s = sharpe(wf.combined_returns)
    c = cagr(wf.combined_returns)
    dd = max_drawdown(wf.combined_returns)

    print("\nCanonical (pca_lookback=252, signal_days=21):")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")

    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    corr = wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    print(f"  H001 corr: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (wf.combined_returns.index >= fold.test_start) & (
            wf.combined_returns.index < fold.test_end
        )
        fold_ret = wf.combined_returns[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    print("\nSensitivity -- pca_lookback:")
    for lb in [126, 252, 504]:
        st = PCARiskRegime(pca_lookback=lb, signal_days=21)
        wf_s = walk_forward_multi(prices, lambda _, s=st: s, folds=folds)
        print(
            f"  pca_lookback={lb}: Sharpe {sharpe(wf_s.combined_returns):.3f}  MaxDD {max_drawdown(wf_s.combined_returns):.2%}"
        )

    print("\nSensitivity -- signal_days:")
    for sd in [10, 21, 42]:
        st = PCARiskRegime(pca_lookback=252, signal_days=sd)
        wf_s = walk_forward_multi(prices, lambda _, s=st: s, folds=folds)
        print(
            f"  signal_days={sd}: Sharpe {sharpe(wf_s.combined_returns):.3f}  MaxDD {max_drawdown(wf_s.combined_returns):.2%}"
        )

    n_trades = wf.combined_report.n_trades if hasattr(wf, "combined_report") else 0
    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    write_summary(
        hypothesis_id="096",
        name="pca-risk-regime",
        strategy_class="PCARiskRegime",
        family="regime",
        sharpe=s,
        ci_low=lo,
        ci_high=hi,
        cagr=c,
        max_drawdown=dd,
        n_trades=n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(fs) for fs in fold_sharpes],
        decision=decision,
        notes="Rolling PCA on SPY/TLT/GLD daily returns; PC1 score = dot(21d-cum-ret, PC1). "
        "Score>0 → SPY, score<=0 → TLT. Binary regime switcher.",
        canonical_params={"pca_lookback": 252, "signal_days": 21},
    )


if __name__ == "__main__":
    main()
