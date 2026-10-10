"""H593: VIXThreeStateLeverage -- 3-state VIX caps.

VIX<15 -> cap=3.0x; 15<=VIX<25 -> cap=2.0x; VIX>=25 -> cap=1.0x. tv=6.5%, lb=20d.
Tests whether finer VIX granularity beats H581's binary VIX<20 (2.119).

Run: uv run python scripts/run_h593.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from set8_lev import (
    TRADING_DAYS,
    apply_leverage,
    fold_sharpe_list,
    h001_correlation,
    mdr_warm_start,
    portfolio_metrics,
    slsqp,
    top_weights,
)
from set8_pool import build_pool
from write_summary import write_summary

from erised.metrics.performance import bootstrap_sharpe_ci

TARGET_VOL = 0.065
LOOKBACK = 20
LOW_VIX, HIGH_VIX = 15.0, 25.0
CAP_STRONG, CAP_NEUTRAL, CAP_BEAR = 3.0, 2.0, 1.0


def main():
    print("H593: VIXThreeStateLeverage\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]

    vixr = vix.reindex(idx).ffill().bfill()
    cap = pd.Series(
        np.select([vixr < LOW_VIX, vixr < HIGH_VIX], [CAP_STRONG, CAP_NEUTRAL], default=CAP_BEAR),
        index=idx,
    )
    fr = {
        "strong": float((vixr < LOW_VIX).mean()),
        "neutral": float(((vixr >= LOW_VIX) & (vixr < HIGH_VIX)).mean()),
        "bear": float((vixr >= HIGH_VIX).mean()),
    }
    print(f"State fractions: {fr}")
    w0 = mdr_warm_start(mat, n)

    def neg(w):
        r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    w = slsqp(neg, w0, n)
    r = apply_leverage(mat @ w, idx, TARGET_VOL, cap, LOOKBACK)
    sh, cal, cg, dd, _ = portfolio_metrics(r)
    print(f"Sh={sh:.3f} Cal={cal:.3f} CAGR={cg:.2%} DD={dd:.2%}  (H581 binary 2.119)")
    best_port = pd.Series(r, index=idx)
    lo, hi = bootstrap_sharpe_ci(best_port, n_boot=5000)
    fs = fold_sharpe_list(best_port, folds)
    corr = h001_correlation(best_port, spy, folds)
    tw = top_weights(names, w)
    print(f"CI [{lo:.3f}, {hi:.3f}]  Folds {min(fs):.2f}..{max(fs):.2f}  H001 corr {corr:+.4f}")
    decision = "KEEP" if sh > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="593",
        name="vix-three-state-leverage",
        strategy_class="VIXThreeStateLeverage",
        family="Set8 Batch4 VIX-Regime Synthesis",
        sharpe=sh,
        ci_low=lo,
        ci_high=hi,
        cagr=cg,
        max_drawdown=dd,
        n_trades=0,
        h001_corr=corr,
        fold_sharpes=fs,
        decision=decision,
        notes=(
            f"3-state VIX: <{LOW_VIX} cap={CAP_STRONG}x, {LOW_VIX}-{HIGH_VIX} cap={CAP_NEUTRAL}x, "
            f">={HIGH_VIX} cap={CAP_BEAR}x. tv={TARGET_VOL:.1%}/lb={LOOKBACK}d. State fractions {fr}. "
            f"H581 ref (binary VIX<20): 2.119. Delta vs H581: {sh - 2.119:+.3f}."
        ),
        canonical_params={
            "objective": "lev-sharpe-vix-3state",
            "low_vix": LOW_VIX,
            "high_vix": HIGH_VIX,
            "cap_strong": CAP_STRONG,
            "cap_neutral": CAP_NEUTRAL,
            "cap_bear": CAP_BEAR,
            "leverage_target_vol": TARGET_VOL,
            "leverage_lookback": LOOKBACK,
            "state_fractions": fr,
            "top_weights": tw,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
