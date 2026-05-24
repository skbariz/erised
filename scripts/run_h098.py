"""H098: ReversionEntryRiskParity -- standalone walk-forward runner.

H021 base weights with a heavy-TLT reversion entry: when SPY drawdown
exceeds 10%, shift to (0.20 SPY, 0.80 TLT) for up to 20 days or until
SPY recovers 5% from trough.

Run: uv run python scripts/run_h098.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from quant.strategies.base import MovingAverageCrossover, ReversionEntryRiskParity
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})

    prices = spy.join(tlt, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()

    folds = make_folds(spy, train_years=5, test_years=2)
    print(f"H098: ReversionEntryRiskParity\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    strat = ReversionEntryRiskParity(
        vol_lookback=60, dd_trigger=0.10, crisis_tlt=0.80, crisis_days=20, recovery=0.05
    )
    wf = walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    s = sharpe(wf.combined_returns)
    c = cagr(wf.combined_returns)
    dd = max_drawdown(wf.combined_returns)

    print("\nCanonical (dd_trigger=0.10, crisis_tlt=0.80, crisis_days=20, recovery=0.05):")
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

    print("\nSensitivity -- dd_trigger:")
    for trig in [0.05, 0.10, 0.15]:
        st = ReversionEntryRiskParity(
            vol_lookback=60, dd_trigger=trig, crisis_tlt=0.80, crisis_days=20, recovery=0.05
        )
        wf_s = walk_forward_multi(prices, lambda _, s=st: s, folds=folds)
        print(
            f"  dd_trigger={trig}: Sharpe {sharpe(wf_s.combined_returns):.3f}  MaxDD {max_drawdown(wf_s.combined_returns):.2%}"
        )

    print("\nSensitivity -- crisis_tlt:")
    for ct in [0.70, 0.80, 0.90]:
        st = ReversionEntryRiskParity(
            vol_lookback=60, dd_trigger=0.10, crisis_tlt=ct, crisis_days=20, recovery=0.05
        )
        wf_s = walk_forward_multi(prices, lambda _, s=st: s, folds=folds)
        print(
            f"  crisis_tlt={ct}: Sharpe {sharpe(wf_s.combined_returns):.3f}  MaxDD {max_drawdown(wf_s.combined_returns):.2%}"
        )

    n_trades = wf.combined_report.n_trades if hasattr(wf, "combined_report") else 0
    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    write_summary(
        hypothesis_id="098",
        name="reversion-entry-risk-parity",
        strategy_class="ReversionEntryRiskParity",
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
        notes="H021 base; crisis entry (0.20 SPY / 0.80 TLT) when SPY drawdown > 10% from peak. "
        "Exits on 5% trough-recovery or 20 days max.",
        canonical_params={
            "vol_lookback": 60,
            "dd_trigger": 0.10,
            "crisis_tlt": 0.80,
            "crisis_days": 20,
            "recovery": 0.05,
        },
    )


if __name__ == "__main__":
    main()
