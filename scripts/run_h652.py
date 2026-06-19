"""H652: DeployableSparse652 -- deployable Sharpe champion constrained to a sparse top-K weight vector.

Same lag-aware, interior-only construction as H650, but after the full Sharpe optimization the weight
vector is truncated to its K largest sleeves and renormalized (K in {3,5,8,12}). Finalizes on the smallest
K that holds within 0.05 Sharpe of the full blend -- an operational-cost (fewer sleeves) lever on the
deployable frontier.

Run: uv run python scripts/run_h652.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from set8_lev import (
    TRADING_DAYS,
    apply_leverage,
    finalize_run,
    mdr_warm_start,
    portfolio_metrics,
    slsqp,
    two_state_cap,
    vix_momentum_mask,
)
from set8_pool import build_pool

LOOKBACK = 20
K_GRID = [3, 5, 8, 12]


def lev_sharpe_neg(mat, idx, tv, cap):
    def neg(w):
        r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
        s = r.std() * np.sqrt(TRADING_DAYS)
        return -(r.mean() * TRADING_DAYS / s) if s > 0 else 0.0

    return neg


def sparsify(w, k):
    """Keep the k largest weights, zero the rest, renormalize to sum 1."""
    keep = np.argsort(w)[::-1][:k]
    out = np.zeros_like(w)
    out[keep] = w[keep]
    s = out.sum()
    return out / s if s > 0 else out


def lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def main():
    print("H652: DeployableSparse652 -- deployable sparse top-K strategy\n" + "=" * 64)
    pool = build_pool()
    mat, names, clean = pool["mat"], pool["strategy_names"], pool["clean"]
    folds, vix, spy = pool["folds_aa"], pool["vix_series"], pool["spy"]
    idx, n = clean.index, mat.shape[1]
    w0 = mdr_warm_start(mat, n)

    print("\nInterior-only candidates (full Sharpe-optimal weights, lagged signal):")
    cands = {}
    for thr in [15.0, 16.0, 17.0]:
        for asym in (False, True):
            tv, cap = lagged_caps(vix, idx, thr, asym)
            w = slsqp(lev_sharpe_neg(mat, idx, tv, cap), w0, n)
            sh = portfolio_metrics(apply_leverage(mat @ w, idx, tv, cap, LOOKBACK))[0]
            key = f"t{thr:.0f}w20{'-asym' if asym else ''}"
            cands[key] = (sh, w, tv, cap, thr, asym)
            print(f"  {key}: Sh={sh:.3f}")

    best_key = max(cands, key=lambda k: cands[k][0])
    sh_full, w_full, tv, cap, thr, asym = cands[best_key]
    print(
        f"\nFull-blend champion config: {best_key} (Sharpe {sh_full:.3f}, {int((w_full > 0.005).sum())} sleeves)"
    )

    print("\nTop-K sparsification:")
    ksweep = {}
    for k in K_GRID:
        wk = sparsify(w_full, k)
        shk = portfolio_metrics(apply_leverage(mat @ wk, idx, tv, cap, LOOKBACK))[0]
        ksweep[str(k)] = round(float(shk), 4)
        print(f"  K={k}: Sh={shk:.3f} (drop {sh_full - shk:+.3f})")

    chosen_k = next((k for k in K_GRID if sh_full - ksweep[str(k)] <= 0.05), max(K_GRID))
    w = sparsify(w_full, chosen_k)
    print(f"\nChosen K (smallest within 0.05 of full): {chosen_k}")
    r = apply_leverage(mat @ w, idx, tv, cap, LOOKBACK)
    sh, cal, cg, dd, so = portfolio_metrics(r)
    print(f"Sharpe {sh:.3f}; vs H650 full-blend (1.980): {sh - 1.980:+.3f}")

    finalize_run(
        "652",
        "deployable-sparse",
        "DeployableSparse652",
        "Set13 Deployable",
        r,
        idx,
        w,
        names,
        folds,
        spy,
        notes=(
            f"Deployable sparse top-K: full Sharpe-optimal weights on the 1-day-LAGGED signal "
            f"(config {best_key}, Sharpe {sh_full:.3f}) truncated to K={chosen_k} sleeves. "
            f"K-sweep {ksweep}. Sharpe {sh:.3f}, Calmar {cal:.3f}, Sortino {so:.3f}. "
            f"vs H650 full-blend (1.980): {sh - 1.980:+.3f}. Operational-cost (fewer-sleeve) variant."
        ),
        extra={
            "objective": "deployable-sparse-topk-lagged-fit-interior",
            "champion_config": best_key,
            "chosen_k": chosen_k,
            "k_sweep": ksweep,
            "full_blend_sharpe": round(float(sh_full), 4),
            "vix_threshold": thr,
            "window": 20,
            "asymmetric": asym,
            "regime_lag_days": 1,
            "leverage_lookback": LOOKBACK,
        },
        label="DeployableSparse652",
    )


if __name__ == "__main__":
    main()
