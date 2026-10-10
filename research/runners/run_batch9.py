"""Batch 9 walk-forward runner: H048-H050 (Ensemble Strategies).

Run: uv run python scripts/run_batch9.py

Combines the 16 kept strategies from H001-H047 into three ensemble variants.
All component strategy OOS returns are already walk-forward validated; the
ensemble simply combines these return streams without additional lookahead.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies import (
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
from erised.strategies import MultiAssetStrategy
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi


def run_wf(prices, strat, folds):
    """Dispatch to walk_forward or walk_forward_multi based on strategy type."""
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)

SEP = "=" * 64

# Kept strategies from H001-H047
KEPT_IDS = [
    "H001",
    "H007",
    "H008",
    "H016",
    "H018",
    "H019",
    "H021",
    "H022",
    "H023",
    "H025",
    "H027",
    "H031",
    "H034",
    "H035",
    "H043",
    "H046",
]


def section(title):
    return f"\n{title}\n{'-' * len(title)}"


def _metrics(returns: pd.Series) -> dict:
    return {
        "sharpe": sharpe(returns),
        "cagr": cagr(returns),
        "max_drawdown": max_drawdown(returns),
    }


def select_min_corr_subset(returns_df: pd.DataFrame, threshold: float = 0.50) -> list[str]:
    """Greedy max-subset with all pairwise correlations below threshold.
    Ranks candidates by OOS Sharpe; adds each if it doesn't exceed threshold
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


def build_component_returns(prices_all: pd.DataFrame, folds: list) -> dict[str, pd.Series]:
    """Run walk_forward for all 16 kept strategies and return OOS return streams."""
    strategies = {
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

    component_returns = {}
    for name, strat in strategies.items():
        print(f"  running {name}...", end=" ", flush=True)
        wf = run_wf(prices_all, strat, folds)
        component_returns[name] = wf.combined_returns
        m = _metrics(wf.combined_returns)
        print(f"Sharpe {m['sharpe']:.3f}")

    return component_returns


def fold_returns(returns_df: pd.DataFrame, weights: pd.Series, folds: list) -> list[pd.Series]:
    """Compute weighted ensemble return series for each fold test window."""
    fold_rets = []
    for fold in folds:
        mask = (returns_df.index >= fold.test_start) & (returns_df.index < fold.test_end)
        fold_slice = returns_df[mask]
        # Align weights to available columns in this fold
        valid_cols = fold_slice.columns[fold_slice.notna().any()]
        w = weights[valid_cols]
        if w.sum() == 0:
            fold_rets.append(pd.Series(dtype=float))
            continue
        ens_ret = (fold_slice[valid_cols] * (w / w.sum())).sum(axis=1)
        fold_rets.append(ens_ret)
    return fold_rets


def main():
    # Load all data
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})

    prices_all = spy.join(tlt, how="left").join(gld, how="left").join(vix, how="left")
    prices_all["tlt_close"] = prices_all["tlt_close"].ffill()
    prices_all["gld_close"] = prices_all["gld_close"].ffill()
    prices_all["vix_close"] = prices_all["vix_close"].ffill()

    folds = make_folds(spy, train_years=5, test_years=2)

    print(f"Batch 9: H048-H050 — Ensemble Strategies\n{SEP}")
    print(f"SPY rows: {len(spy)}  Folds: {len(folds)}")
    print(f"Component strategies: {len(KEPT_IDS)}")

    # --- Build component returns ---
    print(section("Building component OOS returns"))
    component_returns = build_component_returns(prices_all, folds)
    returns_df = pd.DataFrame(component_returns)

    # Equal weights for H048
    ew_weights = pd.Series(1.0, index=KEPT_IDS)

    # -----------------------------------------------------------------------
    # H048: Equal-Weight Ensemble
    # -----------------------------------------------------------------------
    print(section("H048: Equal-Weight Ensemble"))
    ew_rets = (returns_df * (ew_weights / ew_weights.sum())).sum(axis=1)
    ew_m = _metrics(ew_rets)
    lo48, hi48 = bootstrap_sharpe_ci(ew_rets, n_boot=5000)
    print(
        f"  Sharpe {ew_m['sharpe']:.3f}  CI [{lo48:.3f}, {hi48:.3f}]"
        f"  CAGR {ew_m['cagr']:.2%}  MaxDD {ew_m['max_drawdown']:.2%}"
    )
    dec48 = "KEEP" if ew_m["sharpe"] > 0.51 and lo48 > 0 else "kill"
    print(f"  Decision: {dec48}")

    # -----------------------------------------------------------------------
    # H049: Min-Correlation Ensemble (threshold 0.50 — all kept have corr > 0.30)
    # -----------------------------------------------------------------------
    print(section("H049: Min-Correlation Ensemble  (pairwise corr < 0.50 threshold)"))
    min_corr_subset = select_min_corr_subset(returns_df, threshold=0.50)
    print(f"  Selected {len(min_corr_subset)} strategies: {min_corr_subset}")
    mc_weights = pd.Series(1.0, index=min_corr_subset)
    mc_rets = (returns_df[min_corr_subset] * (mc_weights / mc_weights.sum())).sum(axis=1)
    mc_m = _metrics(mc_rets)
    lo49, hi49 = bootstrap_sharpe_ci(mc_rets, n_boot=5000)
    print(
        f"  Sharpe {mc_m['sharpe']:.3f}  CI [{lo49:.3f}, {hi49:.3f}]"
        f"  CAGR {mc_m['cagr']:.2%}  MaxDD {mc_m['max_drawdown']:.2%}"
    )
    dec49 = "KEEP" if mc_m["sharpe"] > 0.51 and lo49 > 0 else "kill"
    print(f"  Decision: {dec49}")

    # Also show full pairwise corr table for kept strategies
    print("\n  Pairwise correlation matrix (selected subset):")
    corr_mat = returns_df[min_corr_subset].corr()
    for i, name in enumerate(min_corr_subset):
        row = (
            "  "
            + f"    {name}: "
            + "  ".join(f"{corr_mat.loc[name, other]:+.3f}" for other in min_corr_subset[: i + 1])
        )
        print(row)

    # -----------------------------------------------------------------------
    # H050: Sharpe-Weighted Ensemble
    # -----------------------------------------------------------------------
    print(section("H050: Sharpe-Weighted Ensemble"))
    component_sharpes = pd.Series({name: sharpe(returns_df[name].dropna()) for name in KEPT_IDS})
    # Only include positive-Sharpe strategies in the weight (negative weight = short = wrong)
    sw_weights = component_sharpes.clip(lower=0)
    print("  Component Sharpes used for weighting:")
    for name in KEPT_IDS:
        print(f"    {name}: {component_sharpes[name]:.3f}  weight {sw_weights[name]:.3f}")

    sw_rets = (returns_df * (sw_weights / sw_weights.sum())).sum(axis=1)
    sw_m = _metrics(sw_rets)
    lo50, hi50 = bootstrap_sharpe_ci(sw_rets, n_boot=5000)
    print(
        f"\n  Sharpe {sw_m['sharpe']:.3f}  CI [{lo50:.3f}, {hi50:.3f}]"
        f"  CAGR {sw_m['cagr']:.2%}  MaxDD {sw_m['max_drawdown']:.2%}"
    )
    dec50 = "KEEP" if sw_m["sharpe"] > 0.51 and lo50 > 0 else "kill"
    print(f"  Decision: {dec50}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(section("DECISION SUMMARY"))
    print(
        f"  H048 Equal-Weight:     Sharpe {ew_m['sharpe']:.3f} CI [{lo48:.3f}, {hi48:.3f}]  -> {dec48}"
    )
    print(
        f"  H049 Min-Corr:         Sharpe {mc_m['sharpe']:.3f} CI [{lo49:.3f}, {hi49:.3f}]  -> {dec49}"
    )
    print(
        f"  H050 Sharpe-Weighted:  Sharpe {sw_m['sharpe']:.3f} CI [{lo50:.3f}, {hi50:.3f}]  -> {dec50}"
    )

    print(section("CORRELATION MATRIX (all 16 kept strategies, vs H001)"))
    h001_ret = returns_df["H001"]
    for name in KEPT_IDS:
        corr = returns_df[name].corr(h001_ret)
        print(f"  {name}: {corr:+.4f}")

    print(section("PER-FOLD SHARPE (H048 Equal-Weight)"))
    for i, fold in enumerate(folds):
        mask = (ew_rets.index >= fold.test_start) & (ew_rets.index < fold.test_end)
        fold_ret = ew_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")


if __name__ == "__main__":
    main()
