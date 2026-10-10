"""H100: HoldoutEvaluation -- final sealed holdout assessment.

Evaluates the flagship H049 ensemble (H021 + H046 + H019 + H035) on the
sealed 2023+ holdout data. This is the true out-of-sample read on the
research program. Only run once — results cannot be used to guide further
research.

Run: uv run python scripts/run_h100.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from write_summary import write_summary

from erised.data.loader import load_holdout, load_research
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies.base import (
    GoldFearGauge,
    HalloweenMomentumConfirm,
    MovingAverageCrossover,
    MultiAssetStrategy,
    RiskParity,
    SPYGLDRotation,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64

H049_COMPONENTS = {
    "H021": RiskParity(),
    "H046": HalloweenMomentumConfirm(),
    "H019": SPYGLDRotation(),
    "H035": GoldFearGauge(),
}
H049_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25])


def apply_strategy_on_full_data(prices_research, prices_holdout, strat):
    """Apply a strategy to combined research+holdout, return holdout-period returns."""
    combined = pd.concat([prices_research, prices_holdout])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    if isinstance(strat, MultiAssetStrategy):
        weights = strat.generate_weights(combined)
        # Compute portfolio return from weights (shifted by 1 to avoid lookahead)
        port_rets = pd.Series(0.0, index=combined.index)
        if "SPY" in weights.columns:
            port_rets += weights["SPY"].shift(1).fillna(0) * combined["adj_close"].pct_change()
        if "TLT" in weights.columns and "tlt_close" in combined.columns:
            port_rets += weights["TLT"].shift(1).fillna(0) * combined["tlt_close"].pct_change()
        if "GLD" in weights.columns and "gld_close" in combined.columns:
            port_rets += weights["GLD"].shift(1).fillna(0) * combined["gld_close"].pct_change()
    else:
        positions = strat.generate_positions(combined)
        port_rets = combined["adj_close"].pct_change() * positions.shift(1).fillna(0)
    holdout_start = prices_holdout.index[0]
    return port_rets[port_rets.index >= holdout_start]


def main():
    print(f"H100: HoldoutEvaluation (H049 flagship ensemble)\n{SEP}")  # noqa: F541
    print(
        "WARNING: This is final evaluation. Results must not be used to guide further research.\n"
    )

    # Load research data for each component's price history context
    spy_r = load_research("SPY")
    tlt_r = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld_r = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    shy_r = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})

    prices_research = (
        spy_r.join(tlt_r, how="left").join(gld_r, how="left").join(shy_r, how="left")
    )
    for col in ["tlt_close", "gld_close", "shy_close"]:
        prices_research[col] = prices_research[col].ffill()

    # Load holdout data
    spy_h = load_holdout("SPY", _i_understand_this_is_final_evaluation=True)
    tlt_h = load_holdout("TLT", _i_understand_this_is_final_evaluation=True)[["adj_close"]].rename(
        columns={"adj_close": "tlt_close"}
    )
    gld_h = load_holdout("GLD", _i_understand_this_is_final_evaluation=True)[["adj_close"]].rename(
        columns={"adj_close": "gld_close"}
    )
    shy_h = load_holdout("SHY", _i_understand_this_is_final_evaluation=True)[["adj_close"]].rename(
        columns={"adj_close": "shy_close"}
    )

    prices_holdout = spy_h.join(tlt_h, how="left").join(gld_h, how="left").join(shy_h, how="left")
    for col in ["tlt_close", "gld_close", "shy_close"]:
        prices_holdout[col] = prices_holdout[col].ffill()

    holdout_start = prices_holdout.index[0]
    holdout_end = prices_holdout.index[-1]
    print(
        f"Research: {prices_research.index.min().date()} -- {prices_research.index.max().date()} ({len(prices_research)} rows)"
    )
    print(f"Holdout:  {holdout_start.date()} -- {holdout_end.date()} ({len(prices_holdout)} rows)")

    # SPY holdout benchmark
    spy_holdout_rets = prices_holdout["adj_close"].pct_change().dropna()
    spy_s = sharpe(spy_holdout_rets)
    spy_c = cagr(spy_holdout_rets)
    spy_dd = max_drawdown(spy_holdout_rets)
    print(f"\nSPY holdout: Sharpe {spy_s:.3f}  CAGR {spy_c:.2%}  MaxDD {spy_dd:.2%}")

    # OOS walk-forward on research as reference (should match H049 reported stats)
    folds = make_folds(spy_r, train_years=5, test_years=2)

    def run_wf(prices, strat, folds):
        if isinstance(strat, MultiAssetStrategy):
            return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
        return walk_forward(prices, lambda _, s=strat: s, folds=folds)

    research_comp_rets = {}
    for name, strat in H049_COMPONENTS.items():
        wf = run_wf(prices_research, strat, folds)
        research_comp_rets[name] = wf.combined_returns

    research_df = pd.DataFrame(research_comp_rets)
    research_ens = research_df.mean(axis=1)
    print(
        f"\nH049 OOS research (verification): Sharpe {sharpe(research_ens):.3f}  MaxDD {max_drawdown(research_ens):.2%}"
    )

    # Holdout evaluation: apply each component to combined data, extract holdout period
    print("\nHoldout returns per component:")
    holdout_comp_rets = {}
    for name, strat in H049_COMPONENTS.items():
        ret_h = apply_strategy_on_full_data(prices_research, prices_holdout, strat)
        holdout_comp_rets[name] = ret_h
        s_h = sharpe(ret_h.dropna())
        c_h = cagr(ret_h.dropna())
        dd_h = max_drawdown(ret_h.dropna())
        print(f"  {name}: Sharpe {s_h:.3f}  CAGR {c_h:.2%}  MaxDD {dd_h:.2%}")

    holdout_df = pd.DataFrame(holdout_comp_rets).dropna(how="all")
    ensemble_h = holdout_df.mean(axis=1).dropna()

    lo, hi = bootstrap_sharpe_ci(ensemble_h, n_boot=5000)
    s = sharpe(ensemble_h)
    c = cagr(ensemble_h)
    dd = max_drawdown(ensemble_h)

    # H001 holdout correlation
    h001_holdout = apply_strategy_on_full_data(
        prices_research, prices_holdout, MovingAverageCrossover(50, 200)
    )
    corr = ensemble_h.corr(h001_holdout.reindex(ensemble_h.index))

    print("\nH049 Ensemble Holdout Result:")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}  H001 corr: {corr:+.4f}")

    vs_spy = s - spy_s
    print(f"\nVs SPY holdout: dSharpe {vs_spy:+.3f}  dMaxDD {dd - spy_dd:+.2%}")

    decision = "KEEP" if s > spy_s and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    write_summary(
        hypothesis_id="100",
        name="holdout-evaluation",
        strategy_class="HoldoutEvaluation",
        family="ensemble",
        sharpe=s,
        ci_low=lo,
        ci_high=hi,
        cagr=c,
        max_drawdown=dd,
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=[],
        decision=decision,
        notes=f"Final sealed holdout ({holdout_start.date()} to {holdout_end.date()}) evaluation of H049 "
        f"(H021+H046+H019+H035, equal-weight). SPY holdout: Sharpe {spy_s:.3f}, CAGR {spy_c:.2%}, MaxDD {spy_dd:.2%}. "
        f"H049 vs SPY: dSharpe {vs_spy:+.3f}.",
        canonical_params={
            "components": list(H049_COMPONENTS.keys()),
            "weights": list(H049_WEIGHTS),
        },
    )


if __name__ == "__main__":
    main()
