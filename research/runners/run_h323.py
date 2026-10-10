"""H323: CappedEqualWeightEnsemble -- Top-10 H001-H100 strategies by OOS Sharpe, equal-weighted.

Takes the 10 best-performing H001-H100 kept strategies and equal-weights at 10% each.
Tests whether top-N selection beats MDR optimization (H095 Sharpe 1.133).
No optimizer needed -- simple rank-and-equal-weight.

Run: uv run python scripts/run_h323.py
Expected runtime: ~25 minutes.
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
    AbsoluteMomentumStrategy,
    CarryMomentumSPYTLT,
    CrisisProtocolRiskParity,
    DefensiveEquityHedge,
    DefensiveSectorOverlay,
    EqualWeightDefensiveFactor,
    FourAssetRiskParity,
    GoldFearGauge,
    H001H021DirectBlend,
    H043H021RegimeBlend,
    HalloweenMomentumConfirm,
    JanuaryBarometer,
    LowVolTargetPortfolio,
    MomentumCarryBlend,
    MonthlyRebalancingRiskParity,
    MovingAverageCrossover,
    MultiAssetDynamicVolTarget,
    MultiAssetStrategy,
    RatchetDrawdownProtection,
    RealizedVsImpliedVol,
    RiskAdjustedMomentum,
    RiskParity,
    RiskParityMomentumFilter,
    RiskParityYieldCurveOverlay,
    SeasonalityTrendCombo,
    SeasonalRiskParity,
    SellInMay,
    SellInMayBondConfirm,
    SPYGLDRotation,
    SPYTLTCorrelationRegime,
    SPYTLTDualMomentum,
    ThreeAssetRiskParity,
    TimeSeriesMomentum,
    VixRiskParityEnhanced,
    VolScaledTrend,
    VrpCalendarHybrid,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64
TOP_N = 10  # canonical: top 10 by OOS Sharpe

# Same 35-strategy universe as H095
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
    "H051": ThreeAssetRiskParity(),
    "H060": CarryMomentumSPYTLT(),
    "H063": DefensiveSectorOverlay(),
    "H065": EqualWeightDefensiveFactor(),
    "H066": VixRiskParityEnhanced(),
    "H068": VrpCalendarHybrid(),
    "H070": MultiAssetDynamicVolTarget(),
    "H076": FourAssetRiskParity(),
    "H077": RiskParityMomentumFilter(),
    "H078": SeasonalRiskParity(),
    "H079": CrisisProtocolRiskParity(),
    "H080": RiskParityYieldCurveOverlay(),
    "H082": RiskAdjustedMomentum(),
    "H084": MomentumCarryBlend(),
    "H086": LowVolTargetPortfolio(),
    "H087": RatchetDrawdownProtection(),
    "H088": H001H021DirectBlend(),
    "H089": H043H021RegimeBlend(),
    "H090": MonthlyRebalancingRiskParity(),
}


def run_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    dbc = load_research("DBC")[["adj_close"]].rename(columns={"adj_close": "dbc_close"})
    xlp = load_research("XLP")[["adj_close"]].rename(columns={"adj_close": "xlp_close"})
    xlu = load_research("XLU")[["adj_close"]].rename(columns={"adj_close": "xlu_close"})
    ive = load_research("IVE")[["adj_close"]].rename(columns={"adj_close": "ive_close"})

    prices_all = (
        spy.join(tlt, how="left")
        .join(gld, how="left")
        .join(vix, how="left")
        .join(shy, how="left")
        .join(dbc, how="left")
        .join(xlp, how="left")
        .join(xlu, how="left")
        .join(ive, how="left")
    )
    for col in [
        "tlt_close",
        "gld_close",
        "vix_close",
        "shy_close",
        "dbc_close",
        "xlp_close",
        "xlu_close",
        "ive_close",
    ]:
        prices_all[col] = prices_all[col].ffill()

    folds = make_folds(spy, train_years=5, test_years=2)
    print(f"H323: CappedEqualWeightEnsemble (top {TOP_N} by OOS Sharpe)\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")

    print(f"\nRunning all {len(KEPT_STRATEGIES)} component strategies...")
    component_returns: dict[str, pd.Series] = {}
    for name, strat in KEPT_STRATEGIES.items():
        wf = run_wf(prices_all, strat, folds)
        component_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    returns_df = pd.DataFrame(component_returns)
    names = list(KEPT_STRATEGIES.keys())

    # Rank by OOS Sharpe
    strat_sharpes = {n: sharpe(component_returns[n].dropna()) for n in names}
    ranked = sorted(strat_sharpes, key=lambda k: strat_sharpes[k], reverse=True)

    print("\nAll strategies ranked by OOS Sharpe:")
    for i, n in enumerate(ranked, 1):
        marker = " <-- selected" if i <= TOP_N else ""
        print(f"  {i:>2}. {n}: {strat_sharpes[n]:.3f}{marker}")

    top_names = ranked[:TOP_N]
    top_df = returns_df[top_names]

    print(f"\nTop {TOP_N} selected: {top_names}")

    # Equal-weight baseline (all 35)
    ew_all_rets = returns_df.mean(axis=1)
    print(
        f"\nEqual-weight baseline (all {len(KEPT_STRATEGIES)}): "
        f"Sharpe {sharpe(ew_all_rets):.3f}  MaxDD {max_drawdown(ew_all_rets):.2%}"
    )

    # Capped equal-weight (top N)
    cew_rets = top_df.mean(axis=1)

    lo, hi = bootstrap_sharpe_ci(cew_rets, n_boot=5000)
    s = sharpe(cew_rets)
    c = cagr(cew_rets)
    dd = max_drawdown(cew_rets)

    print(f"\nCapped Equal-Weight Portfolio (H323, top {TOP_N}):")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")
    print("\n  Compare H095 (MDR all 35):  Sharpe 1.133  MaxDD -7.16%")
    print("  Compare H048 (EW all 35):   Sharpe 0.800  MaxDD -23.35%")

    # Sensitivity: top N by OOS Sharpe
    print("\nSensitivity -- top N by OOS Sharpe:")
    for top_n in [5, 10, 15, 20, 25, 35]:
        top_sub = ranked[:top_n]
        sub_rets = returns_df[top_sub].mean(axis=1)
        lo_v, hi_v = bootstrap_sharpe_ci(sub_rets, n_boot=1000)
        print(
            f"  top_{top_n:>2}  Sharpe {sharpe(sub_rets):.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {max_drawdown(sub_rets):.2%}"
        )

    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    corr = cew_rets.corr(wf_h001.combined_returns.reindex(cew_rets.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (cew_rets.index >= fold.test_start) & (cew_rets.index < fold.test_end)
        fold_ret = cew_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    top_weights = {n: round(1.0 / TOP_N, 4) for n in top_names}
    write_summary(
        hypothesis_id="323",
        name="capped-equal-weight-ensemble",
        strategy_class="CappedEqualWeightEnsemble",
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
        notes=(
            f"Top {TOP_N} H001-H100 kept strategies by OOS Sharpe, equal-weighted at "
            f"{1.0/TOP_N:.0%} each. "
            f"Selection: {top_names}. "
            f"Compare H095 (MDR over 35): Sharpe 1.133, MaxDD -7.16%. "
            f"Compare H048 (EW 35): Sharpe 0.800, MaxDD -23.35%."
        ),
        canonical_params={
            "top_n": TOP_N,
            "selection": "oos_sharpe_rank",
            "weighting": "equal",
            "selected_strategies": top_names,
            "top_weights": top_weights,
        },
    )


if __name__ == "__main__":
    main()
