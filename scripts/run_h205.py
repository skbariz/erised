"""H205: SectorMomentumRiskParity -- top equity ETF by 6m momentum paired with TLT.

Run: uv run python scripts/run_h205.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover, RiskParity, SectorMomentumRiskParity
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64

_EQUITY_ETFS = ["XLP", "XLU", "IVE", "IVW", "QQQ", "IWM", "VNQ"]


def build_prices() -> object:
    spy = load_research("SPY")
    prices = spy.copy()
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices = prices.join(tlt, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()
    for etf in _EQUITY_ETFS:
        col = load_research(etf)[["adj_close"]].rename(columns={"adj_close": f"{etf.lower()}_close"})
        prices = prices.join(col, how="left")
    for etf in _EQUITY_ETFS:
        prices[f"{etf.lower()}_close"] = prices[f"{etf.lower()}_close"].ffill()
    return prices


def main() -> None:
    print(f"H205: SectorMomentumRiskParity (top ETF + TLT, risk-parity)\n{SEP}")

    prices = build_prices()
    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: SectorMomentumRiskParity(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)

    # H021 baseline for comparison
    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(), folds=folds)
    rep021 = wf_h021.combined_report

    print("\nCanonical (6m mom, 60d vol):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")
    print(f"\nH021 baseline: Sharpe {rep021.sharpe:.3f}  MaxDD {rep021.max_drawdown:.2%}")

    print("\nSensitivity -- mom_window:")
    for mw in [63, 126, 189, 252]:
        r = walk_forward_multi(prices, lambda _, _m=mw: SectorMomentumRiskParity(mom_window=_m), folds=folds)
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
        hypothesis_id="205",
        name="sector-momentum-risk-parity",
        strategy_class="SectorMomentumRiskParity",
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
            "H021 variant: top equity ETF (XLP/XLU/IVE/IVW/QQQ/IWM/VNQ by 126d return) "
            "paired with TLT via inverse-vol risk parity. "
            "Concentrates equity leg in sector with strongest momentum."
        ),
        canonical_params={"mom_window": 126, "vol_lookback": 60},
    )


if __name__ == "__main__":
    main()
