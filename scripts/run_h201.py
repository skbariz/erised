"""H201: SectorETFMomentum -- ETF-level sector rotation via 6m momentum.

Run: uv run python scripts/run_h201.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover, SectorETFMomentum
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64

_EQUITY_ETFS = ["XLP", "XLU", "IVE", "IVW", "QQQ", "IWM", "VNQ"]


def build_prices() -> object:
    spy = load_research("SPY")
    prices = spy.copy()
    for etf in _EQUITY_ETFS:
        col = load_research(etf)[["adj_close"]].rename(columns={"adj_close": f"{etf.lower()}_close"})
        prices = prices.join(col, how="left")
    for etf in _EQUITY_ETFS:
        prices[f"{etf.lower()}_close"] = prices[f"{etf.lower()}_close"].ffill()
    return prices


def main() -> None:
    print(f"H201: SectorETFMomentum (top-3 equity ETFs by 6m return, risk-parity)\n{SEP}")

    prices = build_prices()
    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}  |  ETFs loaded: {_EQUITY_ETFS}")

    wf = walk_forward_multi(prices, lambda _: SectorETFMomentum(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)

    print("\nCanonical (6m window, top-3, 60d vol):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity -- top_n:")
    for n in [1, 2, 3, 4, 5]:
        r = walk_forward_multi(prices, lambda _, _n=n: SectorETFMomentum(top_n=_n), folds=folds)
        lo_s, hi_s = bootstrap_sharpe_ci(r.combined_returns, n_boot=1000)
        print(
            f"  top_n={n}  Sharpe {r.combined_report.sharpe:.3f}"
            f"  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {r.combined_report.max_drawdown:.2%}"
        )

    print("\nSensitivity -- mom_window (days):")
    for mw in [63, 126, 189, 252]:
        r = walk_forward_multi(prices, lambda _, _m=mw: SectorETFMomentum(mom_window=_m), folds=folds)
        lo_s, hi_s = bootstrap_sharpe_ci(r.combined_returns, n_boot=1000)
        print(
            f"  mom={mw}d  Sharpe {r.combined_report.sharpe:.3f}"
            f"  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {r.combined_report.max_drawdown:.2%}"
        )

    print("\nPer-fold Sharpe:")
    for i, fold_result in enumerate(wf.fold_results):
        print(
            f"  Fold {i:>2}: {sharpe(fold_result.net_returns):+.3f}"
            f"  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})"
        )

    spy = load_research("SPY")
    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="201",
        name="sector-etf-momentum",
        strategy_class="SectorETFMomentum",
        family="sector/industry momentum",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes=(
            "Moskowitz-Grinblatt (1999) at ETF level. "
            "Rank XLP/XLU/IVE/IVW/QQQ/IWM/VNQ by 126d return; long top-3 risk-parity weighted. "
            "SPY fallback before ETF inception dates."
        ),
        canonical_params={"mom_window": 126, "vol_lookback": 60, "top_n": 3},
    )


if __name__ == "__main__":
    main()
