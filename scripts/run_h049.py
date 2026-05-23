"""H049: MinCorrEnsemble -- standalone runner.

Run: uv run python scripts/run_h049.py

Select the maximum subset of kept strategies where all pairwise OOS return
correlations are below the threshold. Equal-weight the selected subset.
Threshold 0.50 (empirical: all kept strategies have H001 corr > 0.30, so
0.30 would yield a degenerate 1-strategy subset).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from quant.strategies.base import (
    AbsoluteMomentumStrategy,
    DefensiveEquityHedge,
    GoldFearGauge,
    HalloweenMomentumConfirm,
    JanuaryBarometer,
    MovingAverageCrossover,
    RealizedVsImpliedVol,
    RiskParity,
    SeasonalityTrendCombo,
    SellInMay,
    SellInMayBondConfirm,
    SPYGLDRotation,
    SPYTLTCorrelationRegime,
    SPYTLTDualMomentum,
    TimeSeriesMomentum,
    VolScaledTrend,
)
from quant.strategies.base import MultiAssetStrategy
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi


def run_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)

SEP = "=" * 64

KEPT_STRATEGIES = {
    "H001": MovingAverageCrossover(50, 200),
    "H007": TimeSeriesMomentum(),
    "H008": AbsoluteMomentumStrategy(),
    "H016": VolScaledTrend(),
    "H018": SPYTLTDualMomentum(),
    "H019": SPYGLDRotation(),
    "H021": RiskParity(),
    "H022": DefensiveEquityHedge(),
    "H023": SellInMay(),
    "H025": JanuaryBarometer(),
    "H027": SeasonalityTrendCombo(),
    "H031": RealizedVsImpliedVol(),
    "H034": SPYTLTCorrelationRegime(),
    "H035": GoldFearGauge(),
    "H043": SellInMayBondConfirm(),
    "H046": HalloweenMomentumConfirm(),
}


def select_min_corr_subset(returns_df: pd.DataFrame, threshold: float) -> list[str]:
    """Greedy max-subset: all pairwise correlations below threshold.
    Seeds with highest-Sharpe strategy; adds next-best that doesn't exceed threshold
    with any already-selected strategy.
    """
    sharpes = {col: sharpe(returns_df[col].dropna()) for col in returns_df.columns}
    ranked = sorted(sharpes.keys(), key=lambda x: sharpes[x], reverse=True)

    selected = [ranked[0]]
    for candidate in ranked[1:]:
        can_add = True
        for sel in selected:
            common = returns_df[[candidate, sel]].dropna()
            if len(common) < 50:
                continue
            c = common[candidate].corr(common[sel])
            if c > threshold:
                can_add = False
                break
        if can_add:
            selected.append(candidate)
    return selected


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})

    prices_all = spy.join(tlt, how="left").join(gld, how="left").join(vix, how="left")
    prices_all["tlt_close"] = prices_all["tlt_close"].ffill()
    prices_all["gld_close"] = prices_all["gld_close"].ffill()
    prices_all["vix_close"] = prices_all["vix_close"].ffill()

    folds = make_folds(spy, train_years=5, test_years=2)
    print(f"H049: MinCorrEnsemble\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")

    # Collect OOS returns
    print("\nRunning component strategies...")
    component_returns = {}
    for name, strat in KEPT_STRATEGIES.items():
        wf = run_wf(prices_all, strat, folds)
        component_returns[name] = wf.combined_returns

    returns_df = pd.DataFrame(component_returns)

    # Show full pairwise correlation table
    corr_mat = returns_df.corr()
    print("\nPairwise correlations (full 16x16 upper triangle):")
    names = list(KEPT_STRATEGIES.keys())
    header = "      " + "  ".join(f"{n:>5}" for n in names)
    print(header)
    for i, n1 in enumerate(names):
        row = f"  {n1}: " + "  ".join(
            f"{corr_mat.loc[n1, n2]:+.3f}" if j <= i else "     " for j, n2 in enumerate(names)
        )
        print(row)

    # Find min-corr subsets at different thresholds
    print("\nMin-corr subset size at various thresholds:")
    for thresh in [0.30, 0.40, 0.50, 0.60, 0.70]:
        subset = select_min_corr_subset(returns_df, threshold=thresh)
        print(f"  threshold={thresh:.2f}: {len(subset)} strategies: {subset}")

    # Canonical: use 0.50 threshold
    threshold = 0.50
    selected = select_min_corr_subset(returns_df, threshold=threshold)
    print(f"\nCanonical (threshold={threshold}): {selected}")

    mc_rets = returns_df[selected].mean(axis=1)
    lo, hi = bootstrap_sharpe_ci(mc_rets, n_boot=5000)
    s = sharpe(mc_rets)
    c = cagr(mc_rets)
    dd = max_drawdown(mc_rets)

    print(f"\nMin-Corr Ensemble ({len(selected)} components):")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")

    corr = mc_rets.corr(component_returns["H001"])
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (mc_rets.index >= fold.test_start) & (mc_rets.index < fold.test_end)
        fold_ret = mc_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    write_summary(
        hypothesis_id="049",
        name="min-corr-ensemble",
        strategy_class="MinCorrEnsemble",
        family="ensemble",
        sharpe=s,
        ci_low=lo,
        ci_high=hi,
        cagr=c,
        max_drawdown=dd,
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=[float(fs) for fs in fold_sharpes],
        decision=decision,
        notes=f"Greedy max-subset with all pairwise OOS corr < {threshold}. Selected: {selected}.",
        canonical_params={"threshold": threshold, "selected": selected},
    )


if __name__ == "__main__":
    main()
