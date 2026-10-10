"""Compute Sortino ratios for all H001-H100 direct (non-ensemble) kept strategies.

Runs each strategy's full walk-forward to collect daily OOS returns,
then computes Sortino = annualised_mean / downside_std (target = 0).
Also computes Calmar ratio.

Saves empirical Sortino/Sharpe ratio for calibrating H150-H199 approximations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from erised.data.loader import load_research
from erised.metrics.performance import sharpe as calc_sharpe
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
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

TRADING_DAYS = 252
SEP = "=" * 64


def sortino_ratio(returns: pd.Series, target: float = 0.0) -> float:
    r = returns.dropna()
    annual_mean = r.mean() * TRADING_DAYS
    downside = r[r < target]
    if len(downside) == 0:
        return float("inf")
    ds = np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS)
    return annual_mean / ds if ds > 0 else 0.0


def calmar_ratio(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    cagr_val = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
    cum = (1 + r).cumprod()
    dd = (cum / cum.expanding().max() - 1).min()
    return cagr_val / abs(dd) if dd < 0 else 0.0


def ann_cagr(returns: pd.Series) -> float:
    r = returns.dropna()
    return (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1


def max_dd(returns: pd.Series) -> float:
    r = returns.dropna()
    cum = (1 + r).cumprod()
    return (cum / cum.expanding().max() - 1).min()


def run_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


# All direct (non-ensemble) H001-H100 kept strategies
DIRECT_STRATEGIES = {
    "H001": ("moving-average-crossover", MovingAverageCrossover(50, 200)),
    "H007": ("time-series-momentum", TimeSeriesMomentum()),
    "H008": ("absolute-momentum", AbsoluteMomentumStrategy()),
    "H016": ("vol-scaled-trend", VolScaledTrend()),
    "H018": ("spy-tlt-dual-momentum", SPYTLTDualMomentum()),
    "H019": ("spy-gld-rotation", SPYGLDRotation()),
    "H021": ("risk-parity", RiskParity()),
    "H022": ("defensive-equity-hedge", DefensiveEquityHedge()),
    "H023": ("sell-in-may", SellInMay()),
    "H025": ("january-barometer", JanuaryBarometer()),
    "H027": ("seasonality-trend-combo", SeasonalityTrendCombo()),
    "H031": ("realized-vs-implied-vol", RealizedVsImpliedVol()),
    "H034": ("spy-tlt-corr-regime", SPYTLTCorrelationRegime()),
    "H035": ("gold-fear-gauge", GoldFearGauge()),
    "H043": ("sell-in-may-bond-confirm", SellInMayBondConfirm()),
    "H046": ("halloween-momentum-confirm", HalloweenMomentumConfirm()),
    "H051": ("three-asset-risk-parity", ThreeAssetRiskParity()),
    "H060": ("carry-momentum-spy-tlt", CarryMomentumSPYTLT()),
    "H063": ("defensive-sector-overlay", DefensiveSectorOverlay()),
    "H065": ("equal-weight-defensive-factor", EqualWeightDefensiveFactor()),
    "H066": ("vix-risk-parity-enhanced", VixRiskParityEnhanced()),
    "H068": ("vrp-calendar-hybrid", VrpCalendarHybrid()),
    "H070": ("multi-asset-dynamic-vol-target", MultiAssetDynamicVolTarget()),
    "H076": ("four-asset-risk-parity", FourAssetRiskParity()),
    "H077": ("risk-parity-momentum-filter", RiskParityMomentumFilter()),
    "H078": ("seasonal-risk-parity", SeasonalRiskParity()),
    "H079": ("crisis-protocol-risk-parity", CrisisProtocolRiskParity()),
    "H080": ("risk-parity-yield-curve-overlay", RiskParityYieldCurveOverlay()),
    "H082": ("risk-adjusted-momentum", RiskAdjustedMomentum()),
    "H084": ("momentum-carry-blend", MomentumCarryBlend()),
    "H086": ("low-vol-target-portfolio", LowVolTargetPortfolio()),
    "H087": ("ratchet-drawdown-protection", RatchetDrawdownProtection()),
    "H088": ("h001-h021-direct-blend", H001H021DirectBlend()),
    "H089": ("h043-h021-regime-blend", H043H021RegimeBlend()),
    "H090": ("monthly-rebalancing-risk-parity", MonthlyRebalancingRiskParity()),
    "H097": ("correlation-risk-parity-hybrid", CorrelationRiskParityHybrid()),
    "H098": ("reversion-entry-risk-parity", ReversionEntryRiskParity()),
}


def main():
    spy = load_research("SPY")
    prices_aa = (
        spy.join(
            load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"}),
            how="left",
        )
        .join(
            load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"}),
            how="left",
        )
        .join(
            load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"}),
            how="left",
        )
        .join(
            load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"}),
            how="left",
        )
        .join(
            load_research("DBC")[["adj_close"]].rename(columns={"adj_close": "dbc_close"}),
            how="left",
        )
        .join(
            load_research("XLP")[["adj_close"]].rename(columns={"adj_close": "xlp_close"}),
            how="left",
        )
        .join(
            load_research("XLU")[["adj_close"]].rename(columns={"adj_close": "xlu_close"}),
            how="left",
        )
        .join(
            load_research("IVE")[["adj_close"]].rename(columns={"adj_close": "ive_close"}),
            how="left",
        )
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

    folds = make_folds(spy, train_years=5, test_years=2)

    print(f"\nComputing Sortino for {len(DIRECT_STRATEGIES)} H001-H100 direct strategies")
    print(SEP)

    results = {}
    for hid, (name, strat) in DIRECT_STRATEGIES.items():
        wf = run_wf(prices_aa, strat, folds)
        r = wf.combined_returns
        sh = calc_sharpe(r)
        so = sortino_ratio(r)
        ca = calmar_ratio(r)
        cg = ann_cagr(r)
        dd = max_dd(r)
        ratio = so / sh if sh > 0 and so < 20 else None
        results[hid] = {
            "hid": hid,
            "name": name,
            "sharpe": round(sh, 4),
            "sortino": round(so, 4),
            "calmar": round(ca, 4),
            "cagr": round(cg, 4),
            "maxdd": round(dd, 4),
            "sortino_sharpe_ratio": round(ratio, 4) if ratio else None,
            "estimated": False,
        }
        r_str = f"{ratio:.3f}" if ratio else "inf"
        print(f"  {hid}  Sharpe {sh:.3f}  Sortino {so:.3f}  Calmar {ca:.3f}  S/Sh {r_str}")

    valid = [
        v["sortino_sharpe_ratio"] for v in results.values() if v["sortino_sharpe_ratio"] is not None
    ]
    median_ratio = float(np.median(valid))
    mean_ratio = float(np.mean(valid))
    p25 = float(np.percentile(valid, 25))
    p75 = float(np.percentile(valid, 75))

    print(f"\nEmpirical Sortino/Sharpe ratios across {len(valid)} strategies:")
    print(f"  Median {median_ratio:.3f}  Mean {mean_ratio:.3f}  P25 {p25:.3f}  P75 {p75:.3f}")

    out = (Path(__file__).resolve().parents[1] / "results/sortino_h001_h100.json")
    out.parent.mkdir(exist_ok=True)
    payload = {
        "strategies": results,
        "n": len(valid),
        "median_ratio": median_ratio,
        "mean_ratio": mean_ratio,
        "p25_ratio": p25,
        "p75_ratio": p75,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
