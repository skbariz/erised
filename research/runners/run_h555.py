"""H555: Leveraged evaluation at 1.0x, 1.25x, 1.5x, 2.0x notional.

Pure calculation — no walk-forward needed. Applies leverage to existing
OOS return series using annual borrowing cost of 50 bps on the leveraged fraction.
Evaluates all 11 Target-bar survivors + H552 across leverage levels.

Run: uv run python scripts/run_h555.py
Expected runtime: <30 seconds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

TRADING_DAYS = 252
BORROW_COST_ANNUAL = 0.005  # 50 bps per year on leveraged fraction
RESULTS_DIR = (Path(__file__).resolve().parents[1] / "results")

TARGET_BAR_STRATEGIES = {
    "H544": "hierarchical-all-fold-mdr-v2",
    "H550": "ultimate-terminal-mdr-550",
    "H547": "ratchet-expanded-h300",
    "H543": "expanded-h300-mdr",
    "H545": "small-pool-focused-mdr",
    "H546": "vol-scaled-expanded-h300",
    "H549": "set7-grand-cross-mdr",
    "H518": "ratchet-h300-port",
    "H300": "grand-cross-program-300-mdr",
    "H350": "ultimate-grand-mdr-350",
    "H517": "vol-tuned-h300-port",
    "H552": "maxdd-constrained-mdr",
}

LEVERAGE_LEVELS = [1.0, 1.25, 1.5, 2.0]


def apply_leverage(cagr: float, maxdd: float, sharpe: float, leverage: float) -> dict:
    """
    Approximate leveraged metrics.

    Under constant leverage L:
    - CAGR_lev ≈ L * CAGR - (L-1) * borrow_cost
    - MaxDD_lev ≈ L * MaxDD
    - Sharpe_lev ≈ Sharpe (both mean and vol scale by L, net of borrow)
      Actually Sharpe_lev = (L*CAGR - (L-1)*borrow - rf) / (L*vol)
      ≈ Sharpe - (L-1)*borrow_cost / (L*vol_approx)
      For simplicity: Sharpe_net ≈ Sharpe * (1 - (L-1)*borrow/(CAGR*L))
    - Calmar_lev = CAGR_lev / |MaxDD_lev|
    """
    borrow = (leverage - 1) * BORROW_COST_ANNUAL
    cagr_lev = leverage * cagr - borrow
    maxdd_lev = leverage * maxdd  # maxdd is negative, scales linearly
    # Sharpe: both excess return and vol scale by L; net of borrow reduces return
    # Approx: Sharpe_lev ≈ Sharpe - borrow/port_vol where port_vol ≈ CAGR/Sharpe
    vol_approx = abs(cagr) / sharpe if sharpe > 0 else 0.01
    sharpe_lev = (leverage * cagr - borrow) / (leverage * vol_approx) if vol_approx > 0 else 0
    calmar_lev = cagr_lev / abs(maxdd_lev) if maxdd_lev < 0 else 0
    return {
        "leverage": leverage,
        "cagr_pct": round(cagr_lev * 100, 2),
        "maxdd_pct": round(maxdd_lev * 100, 2),
        "sharpe": round(sharpe_lev, 3),
        "calmar": round(calmar_lev, 3),
        "meets_cagr_bar": cagr_lev >= 0.08,
        "meets_maxdd_bar": maxdd_lev >= -0.15,
        "meets_all_bars": cagr_lev >= 0.08 and maxdd_lev >= -0.15 and sharpe_lev >= 1.4,
    }


def load_strategy_metrics(hid: str, folder: str) -> dict | None:
    candidates = sorted(RESULTS_DIR.glob(f"{int(hid[1:])}-{folder}*"))
    if not candidates:
        candidates = sorted(RESULTS_DIR.glob(f"{hid.lstrip('H').lstrip('0')}-*"))
    if not candidates:
        return None
    path = candidates[0] / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    print("H555: Leveraged Evaluation — Target Bar Survivors")
    print("=" * 80)
    print(f"Borrow cost: {BORROW_COST_ANNUAL*100:.1f}% p.a. on leveraged fraction")
    print(f"Leverage levels tested: {LEVERAGE_LEVELS}")
    print()

    results = {}
    for hid, folder in TARGET_BAR_STRATEGIES.items():
        d = load_strategy_metrics(hid, folder)
        if d is None:
            print(f"  {hid}: summary.json not found, skipping")
            continue
        sh = d["oos_sharpe"]
        cg = d["cagr"]
        mdd = d["max_drawdown"]
        calmar_1x = cg / abs(mdd) if mdd < 0 else 0
        lev_results = {lev: apply_leverage(cg, mdd, sh, lev) for lev in LEVERAGE_LEVELS}
        results[hid] = {
            "name": d["name"],
            "base": {"sharpe": sh, "cagr_pct": round(cg*100,2), "maxdd_pct": round(mdd*100,2), "calmar": round(calmar_1x,3)},
            "leveraged": lev_results,
        }

    # Print table: for each strategy, show metrics at each leverage level
    SEP = "-" * 110
    HEADER = f"{'ID':<6} {'Lev':>5} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7} {'CAGR>=8':>8} {'DD<=-15':>8} {'All bars':>9}"
    print(HEADER)
    print(SEP)

    for hid, res in results.items():
        b = res["base"]
        print(f"{hid:<6} {'1.0x':>5} {b['sharpe']:>7.3f} {b['cagr_pct']:>7.2f}% {b['maxdd_pct']:>7.2f}% {b['calmar']:>7.3f}   {'--':>7}   {'--':>7}   {'--':>8}")
        for lev, lr in res["leveraged"].items():
            if lev == 1.0:
                continue
            flag = "*" if lr["meets_all_bars"] else " "
            all_str = ("YES" + flag) if lr["meets_all_bars"] else "no "
            print(f"{'':6} {lev:.2f}x {lr['sharpe']:>7.3f} {lr['cagr_pct']:>7.2f}% {lr['maxdd_pct']:>7.2f}% {lr['calmar']:>7.3f}   {'YES' if lr['meets_cagr_bar'] else 'no':>7}   {'YES' if lr['meets_maxdd_bar'] else 'no':>7}   {all_str:>8}")
        print()

    # Summary: which leverage levels allow each strategy to meet institutional bars?
    print("=" * 80)
    print("INSTITUTIONAL BAR PASSAGE (CAGR >= 8%, MaxDD <= -15%, Sharpe >= 1.4)")
    print("=" * 80)
    for hid, res in results.items():
        passing_levs = [lev for lev, lr in res["leveraged"].items() if lr["meets_all_bars"]]
        if passing_levs:
            best_lev = passing_levs[0]
            lr = res["leveraged"][best_lev]
            print(f"  {hid} ({res['name'][:35]}): passes at {best_lev:.2f}x — "
                  f"Sharpe {lr['sharpe']:.3f}, CAGR {lr['cagr_pct']:.2f}%, MaxDD {lr['maxdd_pct']:.2f}%")
        else:
            # Find minimum leverage for CAGR ≥ 8%
            for lev, lr in sorted(res["leveraged"].items()):
                if lr["meets_cagr_bar"]:
                    print(f"  {hid}: CAGR bar at {lev:.2f}x but MaxDD {lr['maxdd_pct']:.2f}% fails DD bar")
                    break
            else:
                print(f"  {hid}: does not reach CAGR ≥ 8% at any tested leverage level")

    # Save results
    out = (Path(__file__).resolve().parents[1] / "results/h555-leverage-evaluation.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
