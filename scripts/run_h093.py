"""H093: EnsembleAntiMomentumRebalance -- standalone runner.

H049's 4 components with monthly anti-momentum rebalancing.
Weight proportional to (max_trailing_Sharpe - component_trailing_Sharpe).
Upweights underperformers; downweights outperformers. Rebalances monthly.
First 24 months use equal weights (insufficient history).

Run: uv run python scripts/run_h093.py
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
    GoldFearGauge,
    HalloweenMomentumConfirm,
    MovingAverageCrossover,
    MultiAssetStrategy,
    RiskParity,
    SPYGLDRotation,
)
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64

H049_COMPONENTS = {
    "H021": RiskParity(),
    "H046": HalloweenMomentumConfirm(),
    "H019": SPYGLDRotation(),
    "H035": GoldFearGauge(),
}

COMPONENTS = list(H049_COMPONENTS.keys())


def run_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def compute_anti_momentum_returns(returns_df: pd.DataFrame, lookback_months: int = 24) -> pd.Series:
    """Monthly anti-momentum weighting. Upweights components with lower trailing Sharpe."""
    lookback_days = lookback_months * 21

    combined = pd.Series(0.0, index=returns_df.index)
    idx = returns_df.index

    # Identify first trading day of each month
    seen: set = set()
    month_starts = []
    for d in idx:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            month_starts.append(d)

    current_weights = np.full(len(COMPONENTS), 1.0 / len(COMPONENTS))

    for m_idx, month_start in enumerate(month_starts):
        # Determine next month start (or end of series)
        next_start = (
            month_starts[m_idx + 1]
            if m_idx + 1 < len(month_starts)
            else idx[-1] + pd.Timedelta(days=1)
        )

        # Compute trailing-window Sharpe for each component (using data before this month)
        pos = idx.get_loc(month_start)
        if pos >= lookback_days:
            window_start = idx[pos - lookback_days]
            trailing = returns_df.loc[window_start : idx[pos - 1]]
            trail_sharpes = np.array([sharpe(trailing[c].dropna()) for c in COMPONENTS])
            max_sh = trail_sharpes.max()
            raw_w = np.maximum(0.0, max_sh - trail_sharpes)
            if raw_w.sum() < 1e-10:
                raw_w = np.ones(len(COMPONENTS))
            current_weights = raw_w / raw_w.sum()

        # Apply weights to this month's daily returns
        mask = (idx >= month_start) & (idx < next_start)
        month_rets = returns_df[COMPONENTS].loc[mask].values
        combined.loc[idx[mask]] = (month_rets * current_weights).sum(axis=1)

    return combined


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})

    prices_all = (
        spy.join(tlt, how="left").join(gld, how="left").join(vix, how="left").join(shy, how="left")
    )
    for col in ["tlt_close", "gld_close", "vix_close", "shy_close"]:
        prices_all[col] = prices_all[col].ffill()

    folds = make_folds(spy, train_years=5, test_years=2)
    print(f"H093: EnsembleAntiMomentumRebalance (H049, trailing 24m Sharpe anti-momentum)\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")

    print("\nRunning H049 component strategies...")
    component_returns = {}
    for name, strat in H049_COMPONENTS.items():
        wf = run_wf(prices_all, strat, folds)
        component_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    returns_df = pd.DataFrame(component_returns)

    h049_rets = returns_df.mean(axis=1)
    print(
        f"\nH049 equal-weight baseline: Sharpe {sharpe(h049_rets):.3f}  MaxDD {max_drawdown(h049_rets):.2%}"
    )

    print("\nCanonical (lookback=24m):")
    ens_rets = compute_anti_momentum_returns(returns_df, lookback_months=24)
    lo, hi = bootstrap_sharpe_ci(ens_rets, n_boot=5000)
    s = sharpe(ens_rets)
    c = cagr(ens_rets)
    dd = max_drawdown(ens_rets)
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")

    print("\nSensitivity -- lookback:")
    for lb in [12, 24, 36, 48]:
        er = compute_anti_momentum_returns(returns_df, lookback_months=lb)
        lo_v, hi_v = bootstrap_sharpe_ci(er, n_boot=1000)
        print(
            f"  lookback={lb}m  Sharpe {sharpe(er):.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {max_drawdown(er):.2%}"
        )

    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    corr = ens_rets.corr(wf_h001.combined_returns.reindex(ens_rets.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (ens_rets.index >= fold.test_start) & (ens_rets.index < fold.test_end)
        fold_ret = ens_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    write_summary(
        hypothesis_id="093",
        name="ensemble-anti-momentum-rebalance",
        strategy_class="EnsembleAntiMomentumRebalance",
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
        notes="H049 components. Monthly anti-momentum: weights proportional to (max_trailing_Sharpe - component_trailing_Sharpe). 24m lookback.",
        canonical_params={"components": COMPONENTS, "lookback_months": 24},
    )


if __name__ == "__main__":
    main()
