"""H099: Batch19AugmentedMinCorr -- greedy min-corr ensemble over all keepers plus H096-H098.

Tests whether the three new Batch 19 individual strategies (PCARiskRegime,
CorrelationRiskParityHybrid, ReversionEntryRiskParity) add value to the
existing 35-strategy candidate pool. Uses the same greedy min-corr selection
as H091 but with an expanded universe.

Run: uv run python scripts/run_h099.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies.base import (
    AbsoluteMomentumStrategy,
    CarryMomentumSPYTLT,
    CorrelationRiskParityHybrid,
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
    PCARiskRegime,
    RatchetDrawdownProtection,
    RealizedVsImpliedVol,
    ReversionEntryRiskParity,
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

ALL_CANDIDATES = {
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
    # Batch 19 new candidates
    "H096": PCARiskRegime(pca_lookback=252, signal_days=21),
    "H097": CorrelationRiskParityHybrid(vol_lookback=60, corr_lookback=60, crisis_tlt=0.90),
    "H098": ReversionEntryRiskParity(
        vol_lookback=60, dd_trigger=0.10, crisis_tlt=0.80, crisis_days=20, recovery=0.05
    ),
}


def run_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def greedy_min_corr(returns_df: pd.DataFrame, threshold: float = 0.50) -> list[str]:
    """Greedy: seed on highest-Sharpe, add next best with all pairwise corr < threshold."""
    strat_sharpes = {c: sharpe(returns_df[c].dropna()) for c in returns_df.columns}
    seed = max(strat_sharpes, key=lambda k: strat_sharpes[k])
    selected = [seed]
    remaining = [c for c in returns_df.columns if c != seed]
    while remaining:
        best_candidate = None
        best_sharpe = -np.inf
        for cand in remaining:
            ok = all(returns_df[cand].corr(returns_df[s]) < threshold for s in selected)
            if ok and strat_sharpes[cand] > best_sharpe:
                best_sharpe = strat_sharpes[cand]
                best_candidate = cand
        if best_candidate is None:
            break
        selected.append(best_candidate)
        remaining.remove(best_candidate)
    return selected


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
    n_cand = len(ALL_CANDIDATES)
    print(f"H099: Batch19AugmentedMinCorr ({n_cand} candidates)\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")

    print("\nRunning all component strategies...")
    component_returns = {}
    for name, strat in ALL_CANDIDATES.items():
        wf = run_wf(prices_all, strat, folds)
        component_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    returns_df = pd.DataFrame(component_returns)

    ew_rets = returns_df.mean(axis=1)
    print(
        f"\nEqual-weight ({n_cand} strategies): Sharpe {sharpe(ew_rets):.3f}  MaxDD {max_drawdown(ew_rets):.2%}"
    )

    for threshold in [0.40, 0.50, 0.60]:
        selected = greedy_min_corr(returns_df, threshold=threshold)
        sub = returns_df[selected].mean(axis=1)
        lo_v, hi_v = bootstrap_sharpe_ci(sub, n_boot=1000)
        print(
            f"\nMin-corr subset (threshold={threshold}): {selected}"
            f"\n  Sharpe {sharpe(sub):.3f}  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {max_drawdown(sub):.2%}"
        )

    # Canonical: threshold=0.50
    selected = greedy_min_corr(returns_df, threshold=0.50)
    ensemble_rets = returns_df[selected].mean(axis=1)
    lo, hi = bootstrap_sharpe_ci(ensemble_rets, n_boot=5000)
    s = sharpe(ensemble_rets)
    c = cagr(ensemble_rets)
    dd = max_drawdown(ensemble_rets)

    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    corr = ensemble_rets.corr(wf_h001.combined_returns.reindex(ensemble_rets.index))

    print(f"\nCanonical (threshold=0.50): {selected}")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}  H001 corr: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (ensemble_rets.index >= fold.test_start) & (ensemble_rets.index < fold.test_end)
        fold_ret = ensemble_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    new_entrants = [n for n in selected if n in ("H096", "H097", "H098")]
    write_summary(
        hypothesis_id="099",
        name="batch19-augmented-min-corr",
        strategy_class="Batch19AugmentedMinCorr",
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
        notes=f"Greedy min-corr (threshold=0.50) over {n_cand} candidates (35 H001-H090 keepers + H096/H097/H098). "
        f"Selected: {selected}. New Batch 19 entrants in selection: {new_entrants}.",
        canonical_params={"threshold": 0.50, "n_candidates": n_cand, "selected": selected},
    )


if __name__ == "__main__":
    main()
