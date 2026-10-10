"""H200: CrossProgramMDRPortfolio -- terminal MDR ensemble combining H001-H100 and H150-H199.

Combines 37 kept asset-allocation strategies (H001-H100) with 9 curated cross-sectional equity
strategies (H150-H199) via Max-Diversification Ratio optimization. The two program families
are inherently decorrelated: risk-parity allocates across SPY/TLT/GLD; cross-sectional
trades relative positions within the S&P 500 equity universe.

Run: uv run python scripts/run_h200.py
Expected runtime: ~2 hours (panel data loading + 46-strategy walk-forward).
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

from erised.data.fundamentals import load_fundamentals_panel
from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies import (
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
from erised.strategies.cross_sectional import (
    BettingAgainstBeta,
    LongShortReversal,
    MultiSignalComposite,
    PCAOUMeanReversion,
    PCAResidualReversal,
    ResidualOUMeanReversion,
    ResidualWeeklyReversal,
    VolScaledMomentum,
    WeeklyReversalSectorNeutral,
)
from erised.validation.walk_forward import (
    make_folds,
    walk_forward,
    walk_forward_cross_sectional,
    walk_forward_multi,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64

# ---------------------------------------------------------------------------
# H001-H100 kept strategies (H095 pool + H097 + H098)
# ---------------------------------------------------------------------------

H001_H100_STRATEGIES = {
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
    "H097": CorrelationRiskParityHybrid(),
    "H098": ReversionEntryRiskParity(),
}


def run_asset_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def max_diversification_weights(returns_df: pd.DataFrame) -> np.ndarray:
    """SLSQP: maximise DR = sum(w*sigma) / sqrt(w^T Cov w). sum(w)=1, w>=0."""
    clean = returns_df.dropna(how="all")
    cov = clean.cov().values * 252
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
    print(f"H200: CrossProgramMDRPortfolio\n{SEP}")

    # ------------------------------------------------------------------
    # Load asset-allocation data (H001-H100)
    # ------------------------------------------------------------------
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    dbc = load_research("DBC")[["adj_close"]].rename(columns={"adj_close": "dbc_close"})
    xlp = load_research("XLP")[["adj_close"]].rename(columns={"adj_close": "xlp_close"})
    xlu = load_research("XLU")[["adj_close"]].rename(columns={"adj_close": "xlu_close"})
    ive = load_research("IVE")[["adj_close"]].rename(columns={"adj_close": "ive_close"})

    prices_aa = (
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
        prices_aa[col] = prices_aa[col].ffill()

    # ------------------------------------------------------------------
    # Load cross-sectional data (H150-H199)
    # ------------------------------------------------------------------
    print("Loading S&P 500 stock panel...")
    membership = load_sp500_members()
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)

    common_idx = panel.index.intersection(membership.index)
    panel = panel.loc[common_idx]
    membership = membership.loc[common_idx]

    spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    print(f"  Panel: {panel.shape[0]} days, {panel.shape[1]} tickers")

    print("Loading fundamental data (net_income for H199)...")
    try:
        net_income = load_fundamentals_panel(panel.columns.tolist(), "net_income", panel.index)
    except Exception:
        net_income = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)

    # ------------------------------------------------------------------
    # Folds (aligned: both use SPY-length calendar)
    # ------------------------------------------------------------------
    folds_aa = make_folds(spy, train_years=5, test_years=2)
    folds_cs = make_folds(panel, train_years=5, test_years=2)
    print(f"\nFolds (AA): {len(folds_aa)}  |  Folds (CS): {len(folds_cs)}")

    # ------------------------------------------------------------------
    # Run H001-H100 components
    # ------------------------------------------------------------------
    print(f"\nRunning {len(H001_H100_STRATEGIES)} H001-H100 strategies...")
    component_returns = {}
    for name, strat in H001_H100_STRATEGIES.items():
        wf = run_asset_wf(prices_aa, strat, folds_aa)
        component_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    # ------------------------------------------------------------------
    # Run H150-H199 cross-sectional components
    # ------------------------------------------------------------------
    cs_factories = {
        "H157": lambda tp, tm: VolScaledMomentum(),
        "H175": lambda tp, tm: BettingAgainstBeta(spy_rets=spy_rets, sectors=sectors),
        "H185": lambda tp, tm: WeeklyReversalSectorNeutral(sectors=sectors),
        "H186": lambda tp, tm: ResidualWeeklyReversal(spy_rets=spy_rets, sectors=sectors),
        "H189": lambda tp, tm: ResidualOUMeanReversion(spy_rets=spy_rets, sectors=sectors),
        "H191": lambda tp, tm: PCAResidualReversal(train_prices=tp, sectors=sectors),
        "H192": lambda tp, tm: PCAOUMeanReversion(train_prices=tp, sectors=sectors),
        "H194": lambda tp, tm: LongShortReversal(sectors=sectors),
        "H199": lambda tp, tm: MultiSignalComposite(net_income=net_income, sectors=sectors),
    }

    print(f"\nRunning {len(cs_factories)} H150-H199 cross-sectional strategies...")
    for name, factory in cs_factories.items():
        print(f"  {name}...", end="", flush=True)
        wf = walk_forward_cross_sectional(
            panel, membership, factory, folds=folds_cs, spy_returns=spy_rets
        )
        component_returns[name] = wf.combined_returns
        print(f" Sharpe {sharpe(wf.combined_returns):.3f}")

    # ------------------------------------------------------------------
    # Combine returns and align on common index
    # ------------------------------------------------------------------
    returns_df = pd.DataFrame(component_returns)
    all_names = list(returns_df.columns)

    # Equal-weight baseline
    ew_rets = returns_df.mean(axis=1)
    print(
        f"\nEqual-weight baseline ({len(all_names)} strategies):"
        f" Sharpe {sharpe(ew_rets):.3f}  MaxDD {max_drawdown(ew_rets):.2%}"
    )

    # ------------------------------------------------------------------
    # Max-diversification optimisation
    # ------------------------------------------------------------------
    print("\nSolving MDR optimisation...")
    w_mdr = max_diversification_weights(returns_df)

    print("\nOptimal MDR weights (> 0.5%):")
    for name, w in sorted(zip(all_names, w_mdr, strict=True), key=lambda x: x[1], reverse=True):
        if w > 0.005:
            print(f"  {name}: {w:.4f}")

    mdr_rets = pd.Series(returns_df.values @ w_mdr, index=returns_df.index)

    lo, hi = bootstrap_sharpe_ci(mdr_rets, n_boot=5000)
    s = sharpe(mdr_rets)
    c = cagr(mdr_rets)
    dd = max_drawdown(mdr_rets)

    print("\nCross-Program MDR Portfolio:")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")

    # ------------------------------------------------------------------
    # Sensitivity: vary H150-H199 component count
    # ------------------------------------------------------------------
    strat_sharpes = {n: sharpe(component_returns[n].dropna()) for n in all_names}
    print("\nSensitivity -- H150+ component count:")
    cs_names_ranked = sorted(
        [n for n in all_names if n.startswith("H1")],
        key=lambda k: strat_sharpes[k],
        reverse=True,
    )
    for n_cs in [3, 6, 9]:
        top_cs = cs_names_ranked[:n_cs]
        sub_names = [n for n in all_names if not n.startswith("H1")] + top_cs
        sub_df = returns_df[sub_names]
        w_sub = max_diversification_weights(sub_df)
        sub_rets = pd.Series(sub_df.values @ w_sub, index=sub_df.index)
        lo_v, hi_v = bootstrap_sharpe_ci(sub_rets, n_boot=1000)
        print(
            f"  top_{n_cs}_cs  Sharpe {sharpe(sub_rets):.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {max_drawdown(sub_rets):.2%}"
        )

    # ------------------------------------------------------------------
    # Per-fold Sharpe
    # ------------------------------------------------------------------
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = mdr_rets.corr(wf_h001.combined_returns.reindex(mdr_rets.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (mdr_rets.index >= fold.test_start) & (mdr_rets.index < fold.test_end)
        fold_ret = mdr_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    top_weights = {n: float(w) for n, w in zip(all_names, w_mdr, strict=True) if w > 0.005}
    write_summary(
        hypothesis_id="200",
        name="cross-program-mdr-portfolio",
        strategy_class="CrossProgramMDRPortfolio",
        family="cross-program ensemble",
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
            f"SLSQP MDR over {len(all_names)} strategies: "
            f"{len(H001_H100_STRATEGIES)} from H001-H100 + {len(cs_factories)} from H150-H199. "
            f"Top weights: {top_weights}."
        ),
        canonical_params={
            "n_strategies": len(all_names),
            "n_aa_strategies": len(H001_H100_STRATEGIES),
            "n_cs_strategies": len(cs_factories),
            "method": "SLSQP",
            "top_weights": top_weights,
        },
    )


if __name__ == "__main__":
    main()
