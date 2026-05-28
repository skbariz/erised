"""H318: RegimeClassifier -- 4-state macro regime (Growth/Recovery/Caution/Stress).

Run: uv run python scripts/run_h318.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover, RegimeClassifier, RiskParity
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    ivw = load_research("IVW")[["adj_close"]].rename(columns={"adj_close": "ivw_close"})
    ive = load_research("IVE")[["adj_close"]].rename(columns={"adj_close": "ive_close"})
    prices = (
        spy.join(tlt, how="left")
        .join(vix, how="left")
        .join(shy, how="left")
        .join(ivw, how="left")
        .join(ive, how="left")
    )
    for col in ["tlt_close", "vix_close", "shy_close", "ivw_close", "ive_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H318: RegimeClassifier (4-state SPY momentum + VIX regime)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: RegimeClassifier(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (vix_low=20, vix_high=30):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(), folds=folds)
    rep21 = wf_h021.combined_report
    print(f"\nH021 baseline: Sharpe {rep21.sharpe:.3f}  MaxDD {rep21.max_drawdown:.2%}")

    print("\nSensitivity -- vix_high:")
    for vh in [25, 30, 35]:
        wf_s = walk_forward_multi(
            prices, lambda _, v=vh: RegimeClassifier(vix_high=v), folds=folds
        )
        rep_s = wf_s.combined_report
        lo_s, hi_s = bootstrap_sharpe_ci(wf_s.combined_returns, n_boot=1000)
        print(
            f"  vix_high={vh}  Sharpe {rep_s.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {rep_s.max_drawdown:.2%}"
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
        hypothesis_id="318",
        name="regime-classifier",
        strategy_class="RegimeClassifier",
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
            "H021 with 4-state macro regime: Growth (SPY>0 AND VIX<20) 1.15x, "
            "Recovery (SPY>0 AND VIX 20-30) 1.00x, Stress (SPY<0 OR VIX>30) 0.70x, "
            "Caution (else) 0.90x. All signals lagged 1 bar. No lookahead."
        ),
        canonical_params={"vix_low": 20, "vix_high": 30},
    )


if __name__ == "__main__":
    main()
