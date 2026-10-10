"""H321: LedoitWolfShrinkageMDR -- MDR with Ledoit-Wolf shrinkage covariance.

Identical to H095 MaxDiversificationPortfolio but replaces the sample covariance
matrix with the Ledoit-Wolf analytical shrinkage estimator (sklearn). Tests whether
covariance estimation quality is the binding constraint for H095.

Run: uv run python scripts/run_h321.py
Expected runtime: ~25 minutes (same component strategies as H095).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
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


def lw_mdr_weights(returns_df: pd.DataFrame) -> np.ndarray:
    """SLSQP MDR with Ledoit-Wolf shrinkage covariance.

    DR = sum(w_i * sigma_i) / sqrt(w^T Cov_lw w)
    Constraints: sum(w) = 1, w >= 0.
    """
    clean = returns_df.dropna(how="all").values

    # Ledoit-Wolf shrinkage on daily returns, then annualise
    lw = LedoitWolf().fit(clean)
    cov = lw.covariance_ * 252
    vols = np.sqrt(np.diag(cov))
    n = len(vols)

    def neg_dr(w: np.ndarray) -> float:
        port_var = w @ cov @ w
        if port_var <= 0:
            return 0.0
        return -(np.dot(w, vols) / np.sqrt(port_var))

    result = minimize(
        neg_dr,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
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
    print(f"H321: LedoitWolfShrinkageMDR ({len(KEPT_STRATEGIES)} strategies)\n{SEP}")
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

    # Ledoit-Wolf MDR
    print("\nSolving Ledoit-Wolf MDR optimisation...")
    w_lw = lw_mdr_weights(returns_df)
    names = list(KEPT_STRATEGIES.keys())

    print("\nOptimal LW-MDR weights (> 0.5%):")
    for name, w in sorted(zip(names, w_lw, strict=True), key=lambda x: x[1], reverse=True):
        if w > 0.005:
            print(f"  {name}: {w:.4f}")

    lw_rets = pd.Series(returns_df.values @ w_lw, index=returns_df.index)

    lo, hi = bootstrap_sharpe_ci(lw_rets, n_boot=5000)
    s = sharpe(lw_rets)
    c = cagr(lw_rets)
    dd = max_drawdown(lw_rets)

    print("\nLedoit-Wolf MDR Portfolio (H321):")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")
    print("\n  Compare H095 (sample cov MDR): Sharpe 1.133  MaxDD -7.16%")

    # H095 sample-covariance MDR for comparison
    clean = returns_df.dropna(how="all")
    cov_sample = clean.cov().values * 252
    vols_sample = np.sqrt(np.diag(cov_sample))
    n = len(vols_sample)

    def neg_dr_sample(w):
        port_var = w @ cov_sample @ w
        if port_var <= 0:
            return 0.0
        return -(np.dot(w, vols_sample) / np.sqrt(port_var))

    res_sample = minimize(
        neg_dr_sample,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w_sample = res_sample.x
    sample_rets = pd.Series(returns_df.values @ w_sample, index=returns_df.index)
    print(
        f"  Sample-cov MDR (recomputed):   Sharpe {sharpe(sample_rets):.3f}  "
        f"MaxDD {max_drawdown(sample_rets):.2%}"
    )

    # Weight comparison
    print("\nWeight differences (LW vs sample, top movers by abs diff):")
    diffs = [(names[i], float(w_lw[i] - w_sample[i])) for i in range(n)]
    diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, diff in diffs[:10]:
        print(f"  {name}: {diff:+.4f}  (LW={w_lw[names.index(name)]:.4f})")

    # H001 correlation
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    corr = lw_rets.corr(wf_h001.combined_returns.reindex(lw_rets.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (lw_rets.index >= fold.test_start) & (lw_rets.index < fold.test_end)
        fold_ret = lw_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    top_weights = {n: float(w) for n, w in zip(names, w_lw, strict=True) if w > 0.005}
    write_summary(
        hypothesis_id="321",
        name="ledoit-wolf-shrinkage-mdr",
        strategy_class="LedoitWolfShrinkageMDR",
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
            f"SLSQP MDR with Ledoit-Wolf shrinkage covariance over {len(KEPT_STRATEGIES)} "
            f"H001-H100 kept strategies. "
            f"Same universe as H095 (sample cov MDR, Sharpe 1.133, MaxDD -7.16%). "
            f"Top weights: {top_weights}."
        ),
        canonical_params={
            "n_strategies": len(KEPT_STRATEGIES),
            "method": "SLSQP",
            "covariance": "LedoitWolf",
            "top_weights": top_weights,
        },
    )


if __name__ == "__main__":
    main()
