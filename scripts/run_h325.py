"""H325: HierarchicalRiskParityEnsemble -- 2-tier MDR over H001-H100 (50%) + H150-H199 (50%).

Tier 1 = all kept H001-H100 macro/ETF strategies; MDR weights within tier; 50% total allocation.
Tier 2 = all kept H150-H199 cross-sectional strategies; MDR weights within tier; 50% total allocation.
Final return = 0.5 * tier1 + 0.5 * tier2.

Compare to H300 (flat MDR over all 52 strategies; Sharpe 1.525) and H095 (MDR over H001-H100; Sharpe 1.133).

Run: uv run python scripts/run_h325.py
Expected runtime: ~3 hours (same cross-sectional components as H300).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from write_summary import write_summary

from quant.data.fundamentals import load_fundamentals_panel
from quant.data.loader import load_research
from quant.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from quant.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from quant.strategies.base import (
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
from quant.strategies.cross_sectional import (
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
from quant.validation.walk_forward import (
    make_folds,
    walk_forward,
    walk_forward_cross_sectional,
    walk_forward_multi,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64

# Same 37-strategy H001-H100 pool as H300
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


def mdr_weights(returns_df: pd.DataFrame) -> np.ndarray:
    """SLSQP MDR: maximise DR = sum(w*sigma) / sqrt(w^T Cov w). sum(w)=1, w>=0."""
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
    print(f"H325: HierarchicalRiskParityEnsemble (Tier1=H001-H100 50%, Tier2=H150-H199 50%)\n{SEP}")

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

    print("Loading S&P 500 stock panel...")
    membership = load_sp500_members()
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)

    common_idx = panel.index.intersection(membership.index)
    panel = panel.loc[common_idx]
    membership = membership.loc[common_idx]

    spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]

    panel_tickers = panel.columns.tolist()
    net_income = load_fundamentals_panel(panel_tickers, "net_income", panel.index)

    folds_aa = make_folds(spy, train_years=5, test_years=2)
    folds_cs = make_folds(panel, train_years=5, test_years=2)

    # --- Tier 1: H001-H100 ---
    print(f"\nRunning {len(H001_H100_STRATEGIES)} Tier-1 (H001-H100) strategies...")
    tier1_returns: dict[str, pd.Series] = {}
    for name, strat in H001_H100_STRATEGIES.items():
        wf = run_asset_wf(prices_aa, strat, folds_aa)
        tier1_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    # --- Tier 2: H150-H199 ---
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

    print(f"\nRunning {len(cs_factories)} Tier-2 (H150-H199) strategies...")
    tier2_returns: dict[str, pd.Series] = {}
    for name, factory in cs_factories.items():
        print(f"  {name}...", end="", flush=True)
        wf = walk_forward_cross_sectional(
            panel, membership, factory, folds=folds_cs, spy_returns=spy_rets
        )
        tier2_returns[name] = wf.combined_returns
        print(f" Sharpe {sharpe(wf.combined_returns):.3f}")

    # --- MDR within each tier ---
    tier1_df = pd.DataFrame(tier1_returns)
    tier2_df = pd.DataFrame(tier2_returns)

    w1 = mdr_weights(tier1_df)
    w2 = mdr_weights(tier2_df)

    tier1_combined = (tier1_df * w1).sum(axis=1)
    tier2_combined = (tier2_df * w2).sum(axis=1)

    # Align index
    common = tier1_combined.index.intersection(tier2_combined.index)
    combined_rets = 0.5 * tier1_combined.loc[common] + 0.5 * tier2_combined.loc[common]

    sh = sharpe(combined_rets)
    ci = bootstrap_sharpe_ci(combined_rets, n_boot=5000)
    cagr_val = cagr(combined_rets)
    mdd = max_drawdown(combined_rets)

    print(f"\n{'=' * 64}")
    print("H325: HierarchicalRiskParityEnsemble")
    print(f"  Sharpe {sh:.3f}  CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  CAGR {cagr_val:.2%}  MaxDD {mdd:.2%}")

    print("\nTier 1 top weights:")
    t1_names = list(tier1_returns.keys())
    for i in np.argsort(w1)[::-1][:8]:
        print(f"  {t1_names[i]}: {w1[i]:.3f}")

    print("\nTier 2 top weights:")
    t2_names = list(tier2_returns.keys())
    for i in np.argsort(w2)[::-1][:5]:
        print(f"  {t2_names[i]}: {w2[i]:.3f}")

    # H001 correlation
    wf_h001 = walk_forward(prices_aa, lambda _: MovingAverageCrossover(), folds=folds_aa)
    h001_rets = wf_h001.combined_returns.reindex(combined_rets.index).fillna(0)
    corr = float(np.corrcoef(combined_rets.values, h001_rets.values)[0, 1])
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if ci[0] > 0 and sh > 0.51 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="325",
        name="hierarchical-risk-parity-ensemble",
        strategy_class="HierarchicalRiskParityEnsemble",
        family="ensemble",
        sharpe=sh,
        ci_low=ci[0],
        ci_high=ci[1],
        cagr=cagr_val,
        max_drawdown=mdd,
        n_trades=int(len(combined_rets)),
        h001_corr=corr,
        fold_sharpes=[],
        decision=decision,
        notes=(
            "2-tier MDR ensemble: Tier1=37 H001-H100 ETF strategies (50%), "
            "Tier2=9 H150-H199 cross-sectional strategies (50%). "
            "MDR weights applied within each tier; tiers blended 50/50."
        ),
    )


if __name__ == "__main__":
    main()
