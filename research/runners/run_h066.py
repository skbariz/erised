"""H066: VixRiskParityEnhanced -- standalone walk-forward runner.

Run: uv run python scripts/run_h066.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies import MovingAverageCrossover, RiskParity, VixRiskParityEnhanced
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    prices = spy.join(tlt, how="left").join(vix, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()
    prices["vix_close"] = prices["vix_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H066: VixRiskParityEnhanced (canonical vix_spike=35, vix_resume=30)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: VixRiskParityEnhanced(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (vix_spike=35, vix_resume=30, crisis_tlt=0.80):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    # H021 baseline for comparison
    folds_spy_tlt = make_folds(prices, train_years=5, test_years=2)
    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(60), folds=folds_spy_tlt)
    rep_h021 = wf_h021.combined_report
    print(f"\nH021 baseline: Sharpe {rep_h021.sharpe:.3f}  MaxDD {rep_h021.max_drawdown:.2%}")

    print("\nSensitivity -- vix_spike (vix_resume=30, crisis_tlt=0.80):")
    for vs in [25, 30, 35, 40]:
        w = walk_forward_multi(
            prices,
            lambda _, v=vs: VixRiskParityEnhanced(vix_spike=v, vix_resume=v - 5),
            folds=folds,
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  vix_spike={vs}  Sharpe {w.combined_report.sharpe:.3f}"
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
        hypothesis_id="066",
        name="vix-risk-parity-enhanced",
        strategy_class="VixRiskParityEnhanced",
        family="vol regime",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="H021 RiskParity + VIX spike override: VIX>35 → 80/20 TLT/SPY; hysteresis resumes at VIX<30.",
        canonical_params={"vix_spike": 35, "vix_resume": 30, "crisis_tlt": 0.80},
    )


if __name__ == "__main__":
    main()
