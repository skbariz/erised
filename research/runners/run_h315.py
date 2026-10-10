"""H315: ConstrainedMacroOverlay -- H314 with Campbell-Thompson constraints.

Run: uv run python scripts/run_h315.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies import ConstrainedMacroOverlay, MacroCombinationOverlay, MovingAverageCrossover
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    hyg = load_research("HYG")[["adj_close"]].rename(columns={"adj_close": "hyg_close"})
    lqd = load_research("LQD")[["adj_close"]].rename(columns={"adj_close": "lqd_close"})
    prices = (
        spy.join(tlt, how="left")
        .join(vix, how="left")
        .join(shy, how="left")
        .join(hyg, how="left")
        .join(lqd, how="left")
    )
    for col in ["tlt_close", "vix_close", "shy_close", "hyg_close", "lqd_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H315: ConstrainedMacroOverlay (H314 + Campbell-Thompson constraints)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: ConstrainedMacroOverlay(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (spy_floor=0.50, spy_cap=1.30):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    wf_h314 = walk_forward_multi(prices, lambda _: MacroCombinationOverlay(), folds=folds)
    print(f"\nH314 baseline: Sharpe {wf_h314.combined_report.sharpe:.3f}  MaxDD {wf_h314.combined_report.max_drawdown:.2%}")

    print("\nSensitivity -- spy_floor:")
    for sf in [0.40, 0.50, 0.60]:
        w = walk_forward_multi(prices, lambda _, s=sf: ConstrainedMacroOverlay(spy_floor=s), folds=folds)
        lo_s, hi_s = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(f"  spy_floor={sf}  Sharpe {w.combined_report.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}")

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
        hypothesis_id="315",
        name="constrained-macro-overlay",
        strategy_class="ConstrainedMacroOverlay",
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
            "H314 macro combination with Campbell-Thompson constraints. "
            "SPY floor = 50% of H021 SPY weight (never fully out). "
            "SPY cap = 130% of H021 (no excess leverage without all-bullish signal). "
            "Leverage (scale > 1) only when all 3 signals bullish (total ≥ 3). "
            "Parameter-free constraints based on economic logic."
        ),
        canonical_params={"spy_floor": 0.50, "spy_cap": 1.30},
    )


if __name__ == "__main__":
    main()
