"""H322: NoShortConstrainedMinVar -- Minimum variance portfolio over H001-H100 strategies.

Minimize portfolio variance (w^T Sigma w) subject to: sum(w)=1, w>=0, w<=0.25.
Uses sample covariance. Same 35-strategy universe as H095.

Run: uv run python scripts/run_h322.py
Expected runtime: ~25 minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies.base import (
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
MAX_WEIGHT = 0.25  # no single strategy > 25%

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


def min_var_weights(returns_df: pd.DataFrame, max_w: float = MAX_WEIGHT) -> np.ndarray:
    """SLSQP: minimize w^T Cov w.  sum(w)=1, 0 <= w_i <= max_w."""
    clean = returns_df.dropna(how="all")
    cov = clean.cov().values * 252
    n = cov.shape[0]

    def port_var(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    result = minimize(
        port_var,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, max_w)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    return result.x


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
    print(
        f"H322: NoShortConstrainedMinVar ({len(KEPT_STRATEGIES)} strategies, "
        f"max_w={MAX_WEIGHT:.0%})\n{SEP}"
    )
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")

    print("\nRunning all component strategies...")
    component_returns = {}
    for name, strat in KEPT_STRATEGIES.items():
        wf = run_wf(prices_all, strat, folds)
        component_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    returns_df = pd.DataFrame(component_returns)

    # Equal-weight baseline
    ew_rets = returns_df.mean(axis=1)
    print(
        f"\nEqual-weight baseline ({len(KEPT_STRATEGIES)} strategies): "
        f"Sharpe {sharpe(ew_rets):.3f}  MaxDD {max_drawdown(ew_rets):.2%}"
    )

    # Min-var optimisation
    print(f"\nSolving min-var optimisation (max_w={MAX_WEIGHT:.0%})...")
    w_mv = min_var_weights(returns_df)
    names = list(KEPT_STRATEGIES.keys())

    print("\nOptimal min-var weights (> 0.5%):")
    for name, w in sorted(zip(names, w_mv, strict=True), key=lambda x: x[1], reverse=True):
        if w > 0.005:
            print(f"  {name}: {w:.4f}")

    mv_rets = pd.Series(returns_df.values @ w_mv, index=returns_df.index)

    lo, hi = bootstrap_sharpe_ci(mv_rets, n_boot=5000)
    s = sharpe(mv_rets)
    c = cagr(mv_rets)
    dd = max_drawdown(mv_rets)

    print("\nMin-Var Portfolio (H322):")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")
    print("\n  Compare H095 (MDR): Sharpe 1.133  MaxDD -7.16%")
    print("  Compare H048 (EW):  Sharpe 0.800  MaxDD -23.35%")

    # Sensitivity: different max-weight caps
    print("\nSensitivity -- max weight cap:")
    for cap in [0.15, 0.20, 0.25, 0.30, 1.0]:
        w_cap = min_var_weights(returns_df, max_w=cap)
        cap_rets = pd.Series(returns_df.values @ w_cap, index=returns_df.index)
        lo_v, hi_v = bootstrap_sharpe_ci(cap_rets, n_boot=1000)
        n_active = sum(1 for w in w_cap if w > 0.005)
        print(
            f"  cap={cap:.0%}  Sharpe {sharpe(cap_rets):.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {max_drawdown(cap_rets):.2%}"
            f"  n_active={n_active}"
        )

    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    corr = mv_rets.corr(wf_h001.combined_returns.reindex(mv_rets.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (mv_rets.index >= fold.test_start) & (mv_rets.index < fold.test_end)
        fold_ret = mv_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    top_weights = {n: float(w) for n, w in zip(names, w_mv, strict=True) if w > 0.005}
    write_summary(
        hypothesis_id="322",
        name="no-short-constrained-min-var",
        strategy_class="NoShortConstrainedMinVar",
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
            f"SLSQP min-variance over {len(KEPT_STRATEGIES)} H001-H100 kept strategies. "
            f"Constraints: sum(w)=1, w_i>=0, w_i<={MAX_WEIGHT:.0%}. "
            f"Compare H095 (MDR): Sharpe 1.133, MaxDD -7.16%. "
            f"Top weights: {top_weights}."
        ),
        canonical_params={
            "n_strategies": len(KEPT_STRATEGIES),
            "method": "SLSQP",
            "objective": "min_variance",
            "max_weight": MAX_WEIGHT,
            "top_weights": top_weights,
        },
    )


if __name__ == "__main__":
    main()
