"""H317: ValuationSpreadMarketState -- IVW vs IVE 252d return spread as risk-on/off signal.

Run: uv run python scripts/run_h317.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover, RiskParity, ValuationSpreadMarketState
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    ivw = load_research("IVW")[["adj_close"]].rename(columns={"adj_close": "ivw_close"})
    ive = load_research("IVE")[["adj_close"]].rename(columns={"adj_close": "ive_close"})
    prices = (
        spy.join(tlt, how="left")
        .join(ivw, how="left")
        .join(ive, how="left")
    )
    for col in ["tlt_close", "ivw_close", "ive_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H317: ValuationSpreadMarketState (IVW vs IVE growth/value spread)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: ValuationSpreadMarketState(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (growth_scale=1.10, value_scale=0.90, ivw_lookback=252):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(), folds=folds)
    rep21 = wf_h021.combined_report
    print(f"\nH021 baseline: Sharpe {rep21.sharpe:.3f}  MaxDD {rep21.max_drawdown:.2%}")

    print("\nSensitivity -- growth_scale:")
    for gs in [1.05, 1.10, 1.15]:
        wf_s = walk_forward_multi(
            prices, lambda _, v=gs: ValuationSpreadMarketState(growth_scale=v), folds=folds
        )
        rep_s = wf_s.combined_report
        lo_s, hi_s = bootstrap_sharpe_ci(wf_s.combined_returns, n_boot=1000)
        print(
            f"  growth_scale={gs}  Sharpe {rep_s.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {rep_s.max_drawdown:.2%}"
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
        hypothesis_id="317",
        name="valuation-spread-market-state",
        strategy_class="ValuationSpreadMarketState",
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
            "H021 scaled by IVW vs IVE 252d return spread. Growth-on (IVW > IVE) -> 1.10x; "
            "value-on (IVE > IVW) -> 0.90x. Pre-2000: pure H021. "
            "IVW/IVE folds 0-1 produce zero OOS (acceptable)."
        ),
        canonical_params={"growth_scale": 1.10, "value_scale": 0.90, "ivw_lookback": 252},
    )


if __name__ == "__main__":
    main()
