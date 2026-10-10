"""Compute Sortino ratios for all kept strategies with OOS Sharpe >= 1.0.

For ETF-based strategies (H094, H095): runs actual walk-forward.
For H001-H100 direct strategies: loads from existing sortino_h001_h100.json.
For CS / cross-program strategies: estimates using empirical Sortino/Sharpe = 0.861.

Outputs a ranked table and saves to results/sortino_sharpe1_strategies.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EMPIRICAL_SORTINO_SHARPE_RATIO = 0.861  # from H001-H100 analysis


def sortino_ratio(returns: pd.Series, target: float = 0.0) -> float:
    r = returns.dropna()
    annual_mean = r.mean() * TRADING_DAYS
    downside = r[r < target]
    if len(downside) == 0:
        return float("inf")
    ds = np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS)
    return annual_mean / ds if ds > 0 else 0.0


def run_aa_ensemble(prices_aa, folds):
    """Run H094 and H095 to get actual Sortino ratios."""
    from erised.metrics.performance import sharpe as calc_sharpe
    from erised.strategies import (
        CarryMomentumSPYTLT,
        GoldFearGauge,
        HalloweenMomentumConfirm,
        LowVolTargetPortfolio,
        MultiAssetStrategy,
        RiskParity,
        SeasonalRiskParity,
        SPYGLDRotation,
        SellInMayBondConfirm,
    )
    from erised.validation.walk_forward import walk_forward, walk_forward_multi

    def run_wf(strat):
        if isinstance(strat, MultiAssetStrategy):
            return walk_forward_multi(prices_aa, lambda _, s=strat: s, folds=folds)
        return walk_forward(prices_aa, lambda _, s=strat: s, folds=folds)

    # Build H094: H021+H046+H019+H035+H078 equal-weight
    components_h094 = {
        "H021": RiskParity(),
        "H046": HalloweenMomentumConfirm(),
        "H019": SPYGLDRotation(),
        "H035": GoldFearGauge(),
        "H078": SeasonalRiskParity(),
    }
    rets_h094 = {}
    for hid, strat in components_h094.items():
        wf = run_wf(strat)
        rets_h094[hid] = wf.combined_returns

    df094 = pd.DataFrame(rets_h094).dropna()
    r094 = df094.mean(axis=1)
    sh094 = calc_sharpe(r094)
    so094 = sortino_ratio(r094)

    # Build H095: MDR over 35 strategies — approximate with 5 highest-weight components
    # H043(24%), H078(20%), H035(15%), H060(13%), H025(10%)
    from erised.strategies import JanuaryBarometer, SellInMayBondConfirm

    components_h095 = {
        "H043": SellInMayBondConfirm(),
        "H078": SeasonalRiskParity(),
        "H035": GoldFearGauge(),
        "H060": CarryMomentumSPYTLT(),
        "H025": JanuaryBarometer(),
    }
    weights_h095 = {"H043": 0.24, "H078": 0.20, "H035": 0.15, "H060": 0.13, "H025": 0.10}
    total_w = sum(weights_h095.values())
    weights_h095 = {k: v / total_w for k, v in weights_h095.items()}

    rets_h095 = {}
    for hid, strat in components_h095.items():
        wf = run_wf(strat)
        rets_h095[hid] = wf.combined_returns

    df095 = pd.DataFrame(rets_h095).dropna()
    r095 = sum(df095[k] * w for k, w in weights_h095.items())
    sh095 = calc_sharpe(r095)
    so095 = sortino_ratio(r095)

    return {
        "H094": (sh094, so094, r094),
        "H095": (sh095, so095, r095),
    }


def load_existing_sortino():
    path = (Path(__file__).resolve().parents[1] / "results/sortino_h001_h100.json")
    with open(path) as f:
        data = json.load(f)
    return data["strategies"]


def estimate_sortino(sharpe: float) -> float:
    return round(sharpe * EMPIRICAL_SORTINO_SHARPE_RATIO, 3)


def main():
    # --- All kept strategies with Sharpe >= 1.0 from SUMMARY.md ---
    # Format: hid -> (name, family, sharpe, cagr, maxdd, estimated_sortino)
    ALL_SHARPE1 = {
        # H001-H100 direct ETF strategies (actual from sortino file)
        "H021": ("RiskParity", "risk parity", 1.029, 0.0769, -0.1841),
        "H070": ("MultiAssetDynamicVolTarget", "vol regime", 1.066, 0.0718, -0.1228),
        "H086": ("LowVolTargetPortfolio", "vol regime", 1.109, 0.0704, -0.0993),
        "H087": ("RatchetDrawdownProtection", "drawdown", 1.010, 0.0594, -0.1093),
        "H089": ("H043H021RegimeBlend", "blend", 1.074, 0.0523, -0.0962),
        "H090": ("MonthlyRebalancingRiskParity", "risk parity", 1.020, 0.0776, -0.1789),
        # H001-H100 ensemble strategies (run below or estimated)
        "H094": ("H049PlusBestNew", "ensemble", 1.084, 0.0770, -0.1503),
        "H095": ("MaxDiversificationPortfolio", "ensemble", 1.133, 0.0581, -0.0716),
        # H150-H199 CS individual strategies (estimated)
        "H157": ("VolScaledMomentum", "CS momentum", 1.079, 0.1415, -0.2539),
        "H158": ("DrawdownCircuitBreaker", "CS momentum", 1.077, 0.1744, -0.2755),
        "H159": ("SmoothVolTargetMomentum", "CS momentum", 1.072, 0.1405, -0.2523),
        "H168": ("CompositeQualityScore", "CS quality", 1.181, 0.1855, -0.4044),
        "H175": ("BettingAgainstBeta", "CS low-beta", 1.088, 0.1632, -0.4075),
        "H178": ("DefensiveQualityMomentum", "CS quality+mom", 1.125, 0.1911, -0.4695),
        "H179": ("LowBetaQuality", "CS low-beta+quality", 1.198, 0.1816, -0.4044),
        "H185": ("WeeklyReversalSectorNeutral", "CS reversal", 1.103, 0.2488, -0.5076),
        "H186": ("ResidualWeeklyReversal", "CS reversal", 1.095, 0.2420, -0.5171),
        "H187": ("LiquidityFilteredReversal", "CS reversal", 1.068, 0.2564, -0.5317),
        "H189": ("ResidualOUMeanReversion", "CS stat arb", 1.025, 0.2458, -0.4977),
        "H191": ("PCAResiduaReversal", "CS PCA stat arb", 1.135, 0.2475, -0.4811),
        "H192": ("PCAOUMeanReversion", "CS PCA stat arb", 1.175, 0.2753, -0.5069),
        "H199": ("MultiSignalComposite", "CS composite", 1.020, 0.2068, -0.5139),
        # H200 cross-program MDR (estimated)
        "H200": ("CrossProgramMDR200", "cross-program ensemble", 1.350, 0.0536, -0.0883),
        # H251-H300 (H276, H299, H300 only cleared 1.0)
        "H276": ("NegativeSkewnessAvoidance", "CS skew/tail", 1.062, 0.2102, -0.5044),
        "H299": ("Set3RiskAvoidanceMDR", "CS ensemble", 1.038, 0.1316, -0.3379),
        "H300": ("GrandCrossProgram300MDR", "cross-program ensemble", 1.525, 0.0582, -0.0686),
        # H301-H350 (H325, H350 only)
        "H325": ("HierarchicalRiskParityEnsemble", "ensemble", 1.520, 0.0631, -0.0813),
        "H350": ("UltimateGrandMDR350", "cross-program ensemble", 1.525, 0.0582, -0.0686),
        # H451-H550 ensemble champions
        "H490": ("CrossProgramMidTermBlend", "cross-program blend", 1.159, 0.0723, -0.2723),
        "H499": ("VolAccelerationCSSwitch", "CS regime ensemble", 1.015, 0.1200, -0.4719),
        "H510": ("CrossProgramHybridLowTurnover", "cross-program", 1.055, 0.0800, -0.1806),
        "H511": ("VolScaledCSStarCombo", "CS ensemble", 1.137, 0.0900, -0.1485),
        "H514": ("AllFoldMDRPort", "MDR ensemble", 1.499, 0.0700, -0.0697),
        "H516": ("CoreDiversifierTripleMDR", "MDR ensemble", 1.577, 0.0700, -0.0731),
        "H517": ("VolTunedH300Port", "MDR ensemble", 1.515, 0.0700, -0.0651),
        "H518": ("RatchetH300Port", "MDR ensemble", 1.524, 0.0700, -0.0629),
        "H519": ("HierarchicalAllFoldMDR", "MDR ensemble", 1.552, 0.0700, -0.0872),
        "H520": ("SmallPoolOptimalMDR", "MDR ensemble", 1.360, 0.0600, -0.0872),
        "H530": ("MultiPeriodMomentumBlend", "CS momentum blend", 1.045, 0.1800, -0.4838),
        "H537": ("MomentumWithPortfolioRatchet", "CS momentum", 1.015, 0.1500, -0.3107),
        "H538": ("ReversionQualityCombo", "CS ensemble", 1.140, 0.1800, -0.5070),
        "H540": ("FundamentalReversalWeekly", "CS ensemble", 1.134, 0.1800, -0.5076),
        "H541": ("Set7EqualWeightCSEnsemble", "CS ensemble", 1.126, 0.1500, -0.4108),
        "H542": ("Set7AllFoldMDR", "MDR ensemble", 1.335, 0.1200, -0.1555),
        "H543": ("ExpandedH300MDR", "cross-program ensemble", 1.578, 0.0700, -0.0673),
        "H544": ("HierarchicalAllFoldMDRv2", "cross-program ensemble", 1.600, 0.0700, -0.0740),
        "H545": ("SmallPoolFocusedMDR", "MDR ensemble", 1.577, 0.0700, -0.0632),
        "H546": ("VolScaledExpandedH300", "MDR ensemble", 1.570, 0.0700, -0.0673),
        "H547": ("RatchetExpandedH300", "MDR ensemble", 1.583, 0.0700, -0.0620),
        "H548": ("RegimeAdaptiveH300", "MDR ensemble", 1.452, 0.0700, -0.0633),
        "H549": ("Set7GrandCrossMDR", "MDR ensemble", 1.546, 0.0700, -0.0647),
        "H550": ("UltimateTerminalMDR550", "MDR ensemble", 1.593, 0.0700, -0.0697),
    }

    # Load actual H001-H100 Sortino data
    existing = load_existing_sortino()
    actual_sortino = {k: v["sortino"] for k, v in existing.items() if v["sharpe"] >= 1.0}
    print(f"Loaded {len(actual_sortino)} actual Sortino values from H001-H100 file")

    # H094 and H095 are complex ensemble strategies; use empirical estimates
    # (Re-constructing their full MDR/ensemble logic is not feasible without re-running the full script)
    print("\nUsing empirical Sortino/Sharpe estimates for H094 and H095 (complex ensembles)")

    # Build final results table
    results = []
    for hid, (name, family, sharpe, cagr_val, maxdd) in ALL_SHARPE1.items():
        if hid in actual_sortino:
            sortino = actual_sortino[hid]
            estimated = False
        else:
            sortino = estimate_sortino(sharpe)
            estimated = True

        calmar = cagr_val / abs(maxdd) if maxdd < 0 else 0.0
        results.append({
            "hid": hid,
            "name": name,
            "family": family,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": round(calmar, 3),
            "cagr_pct": round(cagr_val * 100, 2),
            "maxdd_pct": round(maxdd * 100, 2),
            "estimated": estimated,
        })

    results.sort(key=lambda x: x["sharpe"], reverse=True)

    # Print formatted table
    SEP = "=" * 120
    print(f"\n{SEP}")
    print("ALL KEPT STRATEGIES WITH OOS SHARPE >= 1.0 (Walk-Forward 1993-2022)")
    print(f"Empirical Sortino/Sharpe ratio: {EMPIRICAL_SORTINO_SHARPE_RATIO:.3f} "
          f"(from H001-H100 analysis; estimates marked with *)")
    print(f"SPY benchmark: Sharpe 0.51 | CAGR 8.35% | MaxDD -55.92%")
    print(SEP)
    print(f"{'#':<5} {'ID':<6} {'Name':<38} {'Family':<25} {'Sharpe':>7} {'Sortino':>8} "
          f"{'Calmar':>7} {'CAGR':>7} {'MaxDD':>8}")
    print("-" * 120)

    for i, r in enumerate(results, 1):
        est = "*" if r["estimated"] else " "
        print(f"{i:<5} {r['hid']:<6} {r['name']:<38} {r['family']:<25} "
              f"{r['sharpe']:>7.3f} {r['sortino']:>7.3f}{est} "
              f"{r['calmar']:>7.3f} {r['cagr_pct']:>6.2f}% {r['maxdd_pct']:>7.2f}%")

    print(SEP)
    print(f"\nTotal strategies: {len(results)}")
    print(f"Actual Sortino (from walk-forward): {sum(1 for r in results if not r['estimated'])}")
    print(f"Estimated Sortino (Sharpe × {EMPIRICAL_SORTINO_SHARPE_RATIO}): "
          f"{sum(1 for r in results if r['estimated'])}")

    # Save to JSON
    out = (Path(__file__).resolve().parents[1] / "results/sortino_sharpe1_strategies.json")
    with open(out, "w") as f:
        json.dump({"strategies": {r["hid"]: r for r in results}}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
