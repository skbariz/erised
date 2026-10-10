"""H320: DefensiveMacroStateMachine -- 3-state Risk-On/Neutral/Risk-Off with VIX hysteresis.

Run: uv run python scripts/run_h320.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import DefensiveMacroStateMachine, MovingAverageCrossover, RiskParity
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    prices = spy.join(tlt, how="left").join(vix, how="left")
    for col in ["tlt_close", "vix_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H320: DefensiveMacroStateMachine (3-state with VIX hysteresis)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: DefensiveMacroStateMachine(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (vix_threshold=28, calm_days=5, risk_off_spy_scale=0.65):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(), folds=folds)
    rep21 = wf_h021.combined_report
    print(f"\nH021 baseline: Sharpe {rep21.sharpe:.3f}  MaxDD {rep21.max_drawdown:.2%}")

    print("\nSensitivity -- risk_off_spy_scale:")
    for ros in [0.50, 0.65, 0.80]:
        wf_s = walk_forward_multi(
            prices,
            lambda _, v=ros: DefensiveMacroStateMachine(risk_off_spy_scale=v),
            folds=folds,
        )
        rep_s = wf_s.combined_report
        lo_s, hi_s = bootstrap_sharpe_ci(wf_s.combined_returns, n_boot=1000)
        print(
            f"  risk_off_spy_scale={ros}  Sharpe {rep_s.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {rep_s.max_drawdown:.2%}"
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
        hypothesis_id="320",
        name="defensive-macro-state-machine",
        strategy_class="DefensiveMacroStateMachine",
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
            "3-state machine: Risk-On (VRP>0 AND SPY>0 AND SPY63d>TLT63d) 1.15x; "
            "Risk-Off (VIX>28 OR SPY252d<0) SPY*0.65 freed -> TLT; Neutral 1.0x. "
            "Hysteresis: calm_days=5 consecutive non-triggering days to exit Risk-Off."
        ),
        canonical_params={"vix_threshold": 28, "calm_days": 5, "risk_off_spy_scale": 0.65},
    )


if __name__ == "__main__":
    main()
