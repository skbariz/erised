"""H331: MacroConfirmedSeasonal -- H078 seasonal + macro gate in Nov-Apr.

Run: uv run python scripts/run_h331.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MacroConfirmedSeasonal, MovingAverageCrossover, RiskParity
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    prices = spy.join(tlt, how="left").join(shy, how="left").join(vix, how="left")
    for col in ["tlt_close", "shy_close", "vix_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H331: MacroConfirmedSeasonal (H078 + VRP/term-spread macro gate)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: MacroConfirmedSeasonal(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (fallback_spy=0.60, fallback_tlt=0.40, term_lookback=252):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(), folds=folds)
    rep21 = wf_h021.combined_report
    print(f"\nH021 baseline: Sharpe {rep21.sharpe:.3f}  MaxDD {rep21.max_drawdown:.2%}")

    print("\nSensitivity -- fallback_spy:")
    for fs in [0.50, 0.60, 0.70]:
        wf_s = walk_forward_multi(
            prices,
            lambda _, v=fs: MacroConfirmedSeasonal(fallback_spy=v, fallback_tlt=1.0 - v),
            folds=folds,
        )
        rep_s = wf_s.combined_report
        lo_s, hi_s = bootstrap_sharpe_ci(wf_s.combined_returns, n_boot=1000)
        print(
            f"  fallback_spy={fs:.2f}  Sharpe {rep_s.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {rep_s.max_drawdown:.2%}"
        )

    print("\nSensitivity -- term_lookback:")
    for tl in [126, 252]:
        wf_s = walk_forward_multi(
            prices,
            lambda _, v=tl: MacroConfirmedSeasonal(term_lookback=v),
            folds=folds,
        )
        rep_s = wf_s.combined_report
        lo_s, hi_s = bootstrap_sharpe_ci(wf_s.combined_returns, n_boot=1000)
        print(
            f"  term_lookback={tl}  Sharpe {rep_s.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {rep_s.max_drawdown:.2%}"
        )

    print("\nPer-fold Sharpe:")
    for i, fold_result in enumerate(wf.fold_results):
        print(
            f"  Fold {i:>2}: {sharpe(fold_result.net_returns):+.3f}  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})"
        )

    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}")
    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="331",
        name="macro-confirmed-seasonal",
        strategy_class="MacroConfirmedSeasonal",
        family="calendar",
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
            "H078 seasonal framework + macro gate in Nov-Apr. "
            "Gate: VRP>0 OR TLT 252d > SHY 252d -> H021 weights; "
            "else fallback 60% SPY / 40% TLT. May-Oct: 100% TLT."
        ),
        canonical_params={
            "vol_lookback": 60,
            "term_lookback": 252,
            "fallback_spy": 0.60,
            "fallback_tlt": 0.40,
        },
    )


if __name__ == "__main__":
    main()
