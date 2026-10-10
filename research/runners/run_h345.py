"""H345: GrandStateAllocator -- five-state macro regime blend (H021 / H086 / H078).

States driven by VRP + VIX + term spread:
  Risk-On  (VRP>0 AND VIX<20):              60% H021 + 20% H086 + 20% H078
  Growth   (VRP>0 AND VIX 20-28):           50% H021 + 30% H086 + 20% H078
  Neutral  (VIX 20-28 AND VRP neutral):     40% H021 + 30% H086 + 30% H078
  Caution  (VRP<0 AND VIX 20-28):           30% H021 + 30% H086 + 40% H078
  Crisis   (VIX>28 OR term spread inverted): 20% H021 + 30% H086 + 50% H078

Run: uv run python scripts/run_h345.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies import (
    LowVolTargetPortfolio,
    MovingAverageCrossover,
    RiskParity,
    SeasonalRiskParity,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64

# State weights: [w_h021, w_h086, w_h078]
STATE_WEIGHTS = {
    "risk_on": (0.60, 0.20, 0.20),
    "growth": (0.50, 0.30, 0.20),
    "neutral": (0.40, 0.30, 0.30),
    "caution": (0.30, 0.30, 0.40),
    "crisis": (0.20, 0.30, 0.50),
}


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    prices = spy.join(tlt, how="left").join(shy, how="left").join(vix, how="left")
    for col in ["tlt_close", "shy_close", "vix_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H345: GrandStateAllocator (H021/H086/H078 via VRP+VIX+term spread, 5 states)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    # --- run component strategies ---
    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(), folds=folds)
    wf_h086 = walk_forward_multi(prices, lambda _: LowVolTargetPortfolio(), folds=folds)
    wf_h078 = walk_forward_multi(prices, lambda _: SeasonalRiskParity(), folds=folds)
    ret_h021 = wf_h021.combined_returns
    ret_h086 = wf_h086.combined_returns
    ret_h078 = wf_h078.combined_returns

    print(
        f"\nH021 standalone: Sharpe {wf_h021.combined_report.sharpe:.3f}  MaxDD {wf_h021.combined_report.max_drawdown:.2%}"
    )
    print(
        f"H086 standalone: Sharpe {wf_h086.combined_report.sharpe:.3f}  MaxDD {wf_h086.combined_report.max_drawdown:.2%}"
    )
    print(
        f"H078 standalone: Sharpe {wf_h078.combined_report.sharpe:.3f}  MaxDD {wf_h078.combined_report.max_drawdown:.2%}"
    )

    # --- regime signals (all lagged 1 day) ---
    spy_r = spy["adj_close"].pct_change()
    vix_var = (prices["vix_close"] / 100) ** 2
    realized_var = spy_r.rolling(21).var() * 252
    vrp = (vix_var - realized_var).shift(1)  # lagged

    vix_lag = prices["vix_close"].shift(1)
    tlt_mom = prices["tlt_close"].pct_change(252).shift(1)
    shy_mom = prices["shy_close"].pct_change(252).shift(1)
    term_spread_inverted = (tlt_mom < shy_mom).astype(float)

    vrp_positive = vrp > 0
    vix_low = vix_lag < 20
    vix_medium = (vix_lag >= 20) & (vix_lag <= 28)
    vix_high = vix_lag > 28

    # five-state priority (first match wins): Crisis, Risk-On, Growth, Caution, Neutral
    crisis = vix_high | term_spread_inverted.astype(bool)
    risk_on = (~crisis) & vrp_positive & vix_low
    growth = (~crisis) & (~risk_on) & vrp_positive & vix_medium
    caution = (~crisis) & (~risk_on) & (~growth) & (~vrp_positive) & vix_medium
    neutral = ~(crisis | risk_on | growth | caution)

    idx = ret_h021.index

    w_h021_ser = pd.Series(STATE_WEIGHTS["neutral"][0], index=prices.index)
    w_h021_ser = w_h021_ser.where(~crisis, STATE_WEIGHTS["crisis"][0])
    w_h021_ser = w_h021_ser.where(~risk_on, STATE_WEIGHTS["risk_on"][0])
    w_h021_ser = w_h021_ser.where(~growth, STATE_WEIGHTS["growth"][0])
    w_h021_ser = w_h021_ser.where(~caution, STATE_WEIGHTS["caution"][0])

    w_h086_ser = pd.Series(STATE_WEIGHTS["neutral"][1], index=prices.index)
    w_h086_ser = w_h086_ser.where(~crisis, STATE_WEIGHTS["crisis"][1])
    w_h086_ser = w_h086_ser.where(~risk_on, STATE_WEIGHTS["risk_on"][1])
    w_h086_ser = w_h086_ser.where(~growth, STATE_WEIGHTS["growth"][1])
    w_h086_ser = w_h086_ser.where(~caution, STATE_WEIGHTS["caution"][1])

    w_h078_ser = pd.Series(STATE_WEIGHTS["neutral"][2], index=prices.index)
    w_h078_ser = w_h078_ser.where(~crisis, STATE_WEIGHTS["crisis"][2])
    w_h078_ser = w_h078_ser.where(~risk_on, STATE_WEIGHTS["risk_on"][2])
    w_h078_ser = w_h078_ser.where(~growth, STATE_WEIGHTS["growth"][2])
    w_h078_ser = w_h078_ser.where(~caution, STATE_WEIGHTS["caution"][2])

    w1 = w_h021_ser.reindex(idx).fillna(STATE_WEIGHTS["neutral"][0])
    w2 = w_h086_ser.reindex(idx).fillna(STATE_WEIGHTS["neutral"][1])
    w3 = w_h078_ser.reindex(idx).fillna(STATE_WEIGHTS["neutral"][2])

    blended_rets = (
        w1 * ret_h021
        + w2 * ret_h086.reindex(idx).fillna(0.0)
        + w3 * ret_h078.reindex(idx).fillna(0.0)
    )

    lo, hi = bootstrap_sharpe_ci(blended_rets, n_boot=5000)
    bl_sharpe = float(sharpe(blended_rets))
    bl_cagr = float((1 + blended_rets).prod() ** (252 / len(blended_rets)) - 1)
    cum = (1 + blended_rets).cumprod()
    bl_maxdd = float((cum / cum.cummax() - 1).min())
    bl_ntrades = int(
        wf_h021.combined_report.n_trades
        + wf_h086.combined_report.n_trades
        + wf_h078.combined_report.n_trades
    )

    print("\nBlended result:")
    print(f"  Sharpe {bl_sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {bl_cagr:.2%}  MaxDD {bl_maxdd:.2%}  N_trades {bl_ntrades}")

    # --- state distribution (OOS) ---
    def pct(s: pd.Series) -> str:
        return f"{s.reindex(idx).fillna(False).astype(bool).sum() / len(idx):.1%}"

    print("\nState distribution (OOS):")
    print(f"  Risk-On: {pct(risk_on)}  Growth: {pct(growth)}  Neutral: {pct(neutral)}")
    print(f"  Caution: {pct(caution)}  Crisis: {pct(crisis)}")

    # --- per-fold Sharpe ---
    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        fold_rets = blended_rets.loc[fold.test_start : fold.test_end]
        fs = float(sharpe(fold_rets)) if len(fold_rets) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    # --- H001 correlation ---
    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = float(blended_rets.corr(wf_h001.combined_returns.reindex(blended_rets.index)))
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if lo > 0 and bl_sharpe > 0.51 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="345",
        name="grand-state-allocator",
        strategy_class="GrandStateAllocator",
        family="sleeve allocation",
        sharpe=bl_sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=bl_cagr,
        max_drawdown=bl_maxdd,
        n_trades=bl_ntrades,
        h001_corr=corr,
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            "Runner-level ensemble: H021 + H086 + H078 blended via five-state macro regime. "
            "States: Risk-On (VRP>0 AND VIX<20), Growth (VRP>0 AND VIX 20-28), "
            "Neutral (VIX 20-28), Caution (VRP<0 AND VIX 20-28), "
            "Crisis (VIX>28 OR term spread inverted). "
            "All signals lagged 1 day. Term spread = TLT_252d vs SHY_252d momentum."
        ),
        canonical_params={
            "vix_low_threshold": 20,
            "vix_high_threshold": 28,
            "vrp_realized_window": 21,
            "term_spread_window": 252,
            "state_weights": STATE_WEIGHTS,
        },
    )


if __name__ == "__main__":
    main()
