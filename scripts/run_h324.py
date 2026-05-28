"""H324: InverseVolWeightedEnsemble -- H001-H100 strategies weighted by 1/sigma_i.

Risk-parity at the strategy layer: weight each kept H001-H100 strategy proportional to
the inverse of its annualized OOS return volatility. No optimizer needed.
Same 35-strategy universe as H095.

Run: uv run python scripts/run_h324.py
Expected runtime: ~25 minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from quant.strategies.base import (
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
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64

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


def inverse_vol_weights(returns_df: pd.DataFrame) -> tuple[np.ndarray, list[float]]:
    """Compute inverse-vol weights: w_i = (1/sigma_i) / sum(1/sigma_j).

    Returns (weights, annualised_vols).
    """
    clean = returns_df.dropna(how="all")
    vols = clean.std() * np.sqrt(252)  # annualised
    inv_vols = 1.0 / vols
    weights = (inv_vols / inv_vols.sum()).values
    return weights, vols.tolist()


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
    print(f"H324: InverseVolWeightedEnsemble ({len(KEPT_STRATEGIES)} strategies)\n{SEP}")
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

    # Equal-weight baseline
    ew_rets = returns_df.mean(axis=1)
    print(
        f"\nEqual-weight baseline ({len(KEPT_STRATEGIES)} strategies): "
        f"Sharpe {sharpe(ew_rets):.3f}  MaxDD {max_drawdown(ew_rets):.2%}"
    )

    # Inverse-vol weights
    w_iv, vols = inverse_vol_weights(returns_df)

    print("\nInverse-vol weights (sorted by weight, > 0.5%):")
    vol_by_name = dict(zip(names, vols, strict=True))
    for name, w in sorted(zip(names, w_iv, strict=True), key=lambda x: x[1], reverse=True):
        if w > 0.005:
            print(f"  {name}: {w:.4f}  (ann_vol={vol_by_name[name]:.4f})")

    iv_rets = pd.Series(returns_df.values @ w_iv, index=returns_df.index)

    lo, hi = bootstrap_sharpe_ci(iv_rets, n_boot=5000)
    s = sharpe(iv_rets)
    c = cagr(iv_rets)
    dd = max_drawdown(iv_rets)

    print("\nInverse-Vol Portfolio (H324):")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")
    print("\n  Compare H095 (MDR): Sharpe 1.133  MaxDD -7.16%")
    print("  Compare H048 (EW):  Sharpe 0.800  MaxDD -23.35%")
    print("  Compare H050 (Sharpe-weighted): Sharpe 0.820  MaxDD -22.94%")

    # Sensitivity: shrink vols toward mean (like Ledoit-Wolf for vols)
    print("\nSensitivity -- vol shrinkage (blend toward equal weight):")
    for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
        # alpha=0 -> pure inverse-vol, alpha=1 -> equal-weight
        w_shrunk = (1 - alpha) * w_iv + alpha * np.full(len(w_iv), 1.0 / len(w_iv))
        shrunk_rets = pd.Series(returns_df.values @ w_shrunk, index=returns_df.index)
        lo_v, hi_v = bootstrap_sharpe_ci(shrunk_rets, n_boot=1000)
        print(
            f"  alpha={alpha:.2f}  Sharpe {sharpe(shrunk_rets):.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {max_drawdown(shrunk_rets):.2%}"
        )

    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    corr = iv_rets.corr(wf_h001.combined_returns.reindex(iv_rets.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (iv_rets.index >= fold.test_start) & (iv_rets.index < fold.test_end)
        fold_ret = iv_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    top_weights = {n: float(w) for n, w in zip(names, w_iv, strict=True) if w > 0.005}
    write_summary(
        hypothesis_id="324",
        name="inverse-vol-weighted-ensemble",
        strategy_class="InverseVolWeightedEnsemble",
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
            f"Inverse-vol weighting (w_i = 1/sigma_i, normalised) over "
            f"{len(KEPT_STRATEGIES)} H001-H100 kept strategies. "
            f"No optimizer. Compare H095 (MDR): Sharpe 1.133, MaxDD -7.16%. "
            f"Top weights: {top_weights}."
        ),
        canonical_params={
            "n_strategies": len(KEPT_STRATEGIES),
            "method": "inverse_vol",
            "top_weights": top_weights,
        },
    )


if __name__ == "__main__":
    main()
