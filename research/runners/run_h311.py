"""H311: TermSpreadSPYTimer -- H021 with TLT vs SHY term spread signal.

Run: uv run python scripts/run_h311.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import MovingAverageCrossover, RiskParity, TermSpreadSPYTimer
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    prices = spy.join(tlt, how="left").join(shy, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()
    prices["shy_close"] = prices["shy_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H311: TermSpreadSPYTimer (TLT vs SHY 252d term spread on H021)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: TermSpreadSPYTimer(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (spread_lookback=252, spy_scale=0.50):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(60), folds=folds)
    print(f"\nH021 baseline: Sharpe {wf_h021.combined_report.sharpe:.3f}  MaxDD {wf_h021.combined_report.max_drawdown:.2%}")

    print("\nSensitivity -- spread_lookback:")
    for lb in [126, 252, 504]:
        w = walk_forward_multi(prices, lambda _, l=lb: TermSpreadSPYTimer(spread_lookback=l), folds=folds)
        lo_s, hi_s = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(f"  spread_lookback={lb}  Sharpe {w.combined_report.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}")

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
        hypothesis_id="311",
        name="term-spread-spy-timer",
        strategy_class="TermSpreadSPYTimer",
        family="macro timing",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes=(
            "H021 with TLT vs SHY 252d return as term spread proxy. "
            "Inverted curve (SHY > TLT 252d) → SPY weight × 0.50; freed weight to TLT. "
            "Normal curve → H021 unchanged. Goyal-Welch (2008): term spread predicts equity premium."
        ),
        canonical_params={"spread_lookback": 252, "spy_scale": 0.50},
    )


if __name__ == "__main__":
    main()
