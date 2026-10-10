"""Compare stock/bond risk parity with and without a portfolio volatility target.

    uv run python examples/03_multi_asset_risk_parity.py

Multi-asset strategies return a weights DataFrame (one column per asset) instead
of a single position. Asset prices are passed as extra columns: ``adj_close`` is
SPY and ``tlt_close`` is TLT.
"""

from __future__ import annotations

from erised.data.synthetic import synthetic_bars
from erised.metrics.performance import bootstrap_sharpe_ci
from erised.strategies import LowVolTargetPortfolio, RiskParity
from erised.validation.walk_forward import make_folds, walk_forward_multi


def main() -> None:
    spy = synthetic_bars(start="2002-07-30", seed=42)
    tlt = synthetic_bars(
        start="2002-07-30", initial_price=90.0, daily_drift=0.0001, daily_vol=0.007, seed=43
    )
    prices = spy.join(tlt["adj_close"].rename("tlt_close"))

    folds = make_folds(prices, train_years=5, test_years=2)
    candidates = {
        "Risk parity": lambda _train: RiskParity(vol_lookback=60),
        "Risk parity + 8% vol target": lambda _train: LowVolTargetPortfolio(vol_target=0.08),
    }
    for label, factory in candidates.items():
        wf = walk_forward_multi(prices, factory, folds=folds)
        lo, hi = bootstrap_sharpe_ci(wf.combined_returns)
        rep = wf.combined_report
        print(
            f"{label:<30} Sharpe {rep.sharpe:5.2f}  CI [{lo:5.2f}, {hi:5.2f}]  "
            f"CAGR {rep.cagr:6.2%}  MaxDD {rep.max_drawdown:7.2%}"
        )


if __name__ == "__main__":
    main()
