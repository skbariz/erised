"""H333: MultiCalendarEnsemble -- equal-weight blend of 5 calendar strategies.

H023 (SellInMay) + H025 (JanuaryBarometer) + H043 (SellInMayBondConfirm)
+ H046 (HalloweenMomentumConfirm) + H078 (SeasonalRiskParity).

Run: uv run python scripts/run_h333.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies import (
    HalloweenMomentumConfirm,
    JanuaryBarometer,
    MovingAverageCrossover,
    MultiAssetStrategy,
    SeasonalRiskParity,
    SellInMay,
    SellInMayBondConfirm,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def run_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    prices = spy.join(tlt, how="left").join(vix, how="left")
    for col in ["tlt_close", "vix_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(spy, train_years=5, test_years=2)
    print(f"H333: MultiCalendarEnsemble (equal-weight 5 calendar strategies)\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")

    components = {
        "H023": SellInMay(),
        "H025": JanuaryBarometer(),
        "H043": SellInMayBondConfirm(),
        "H046": HalloweenMomentumConfirm(),
        "H078": SeasonalRiskParity(),
    }

    print("\nRunning calendar components...")
    comp_returns = {}
    for name, strat in components.items():
        wf = run_wf(prices, strat, folds)
        comp_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    ens_df = pd.DataFrame(comp_returns)
    ens_rets = ens_df.mean(axis=1)

    lo, hi = bootstrap_sharpe_ci(ens_rets, n_boot=5000)
    s = sharpe(ens_rets)
    c = cagr(ens_rets)
    dd = max_drawdown(ens_rets)

    print("\nCanonical (equal-weight 5 components):")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (ens_rets.index >= fold.test_start) & (ens_rets.index < fold.test_end)
        fold_ret = ens_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(float(fs))
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    corr = ens_rets.corr(wf_h001.combined_returns.reindex(ens_rets.index))
    print(f"\nH001 correlation: {corr:+.4f}")
    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="333",
        name="multi-calendar-ensemble",
        strategy_class="MultiCalendarEnsemble",
        family="ensemble",
        sharpe=s,
        ci_low=lo,
        ci_high=hi,
        cagr=c,
        max_drawdown=dd,
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            "Equal-weight blend of 5 calendar strategies: "
            "H023 SellInMay + H025 JanuaryBarometer + H043 SellInMayBondConfirm "
            "+ H046 HalloweenMomentumConfirm + H078 SeasonalRiskParity. "
            "Runner-only strategy (no new class); blends combined_returns."
        ),
        canonical_params={"components": list(components.keys()), "weights": "equal"},
    )


if __name__ == "__main__":
    main()
