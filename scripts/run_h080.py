"""H080: RiskParityYieldCurveOverlay -- standalone walk-forward runner.

Run: uv run python scripts/run_h080.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover, RiskParity, RiskParityYieldCurveOverlay
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    prices = spy.join(tlt, how="left").join(shy, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()
    prices["shy_close"] = prices["shy_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(
        f"H080: RiskParityYieldCurveOverlay (canonical vol_lookback=60, curve_days=63, spy_scale=0.50)\n{SEP}"
    )
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: RiskParityYieldCurveOverlay(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (vol_lookback=60, curve_days=63, spy_scale=0.50):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(60), folds=folds)
    rep_h021 = wf_h021.combined_report
    print(f"\nH021 baseline: Sharpe {rep_h021.sharpe:.3f}  MaxDD {rep_h021.max_drawdown:.2%}")

    print("\nSensitivity -- spy_scale:")
    for ss in [0.25, 0.50, 0.75, 1.00]:
        w = walk_forward_multi(
            prices,
            lambda _, s=ss: RiskParityYieldCurveOverlay(spy_scale=s),
            folds=folds,
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  spy_scale={ss:.2f}  Sharpe {w.combined_report.sharpe:.3f}"
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
        hypothesis_id="080",
        name="risk-parity-yield-curve-overlay",
        strategy_class="RiskParityYieldCurveOverlay",
        family="multi-asset",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="H021 parity with yield curve overlay: SHY>TLT (inverted) cuts SPY weight by spy_scale=50%.",
        canonical_params={"vol_lookback": 60, "curve_days": 63, "spy_scale": 0.50},
    )


if __name__ == "__main__":
    main()
