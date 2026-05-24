"""Walk-forward validation.

Rolling-window OOS evaluation. The data is divided into a sequence of
(train, test) folds where each test segment is strictly after its train
segment. The strategy's parameters may be fit on each train segment;
the test segment is then scored using those parameters. Aggregate
metrics across all test segments give the OOS performance estimate.

For parameter-free strategies (e.g. fixed-config MA crossover), the
train segment is unused and walk-forward simply concatenates test
segments. The framework still exposes the train range so future
parameterized strategies fit naturally.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from ..backtest.engine import BacktestResult, run_backtest, run_multiasset_backtest
from ..backtest.panel_engine import run_panel_backtest
from ..metrics.performance import PerformanceReport, report
from ..strategies.base import MultiAssetStrategy, Strategy


@dataclass
class Fold:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __str__(self) -> str:
        return (
            f"Fold {self.fold_idx}: train [{self.train_start.date()} -> {self.train_end.date()}], "
            f"test [{self.test_start.date()} -> {self.test_end.date()}]"
        )


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    fold_results: list[BacktestResult]
    combined_returns: pd.Series
    combined_positions: pd.Series
    combined_report: PerformanceReport


def make_folds(
    prices: pd.DataFrame,
    train_years: float = 5.0,
    test_years: float = 1.0,
    step_years: float | None = None,
    min_train_years: float = 5.0,
) -> list[Fold]:
    """Build a sequence of walk-forward folds.

    train_years : length of each training window.
    test_years  : length of each test window.
    step_years  : how far to advance between folds. Defaults to test_years
                  (non-overlapping test segments).
    min_train_years : minimum train length; folds before this are dropped.
    """
    if step_years is None:
        step_years = test_years

    idx = prices.index
    start, end = idx.min(), idx.max()

    # Use Timedelta (not DateOffset) so arithmetic and comparisons are consistent.
    train_offset = pd.Timedelta(days=int(train_years * 365.25))
    test_offset = pd.Timedelta(days=int(test_years * 365.25))
    step_offset = pd.Timedelta(days=int(step_years * 365.25))
    min_train_offset = pd.Timedelta(days=int(min_train_years * 365.25))

    folds = []
    fold_idx = 0
    test_start = start + train_offset
    while True:
        test_end = test_start + test_offset
        if test_end > end:
            break
        train_start = test_start - train_offset
        if train_start < start:
            train_start = start
        if (test_start - train_start) < min_train_offset:
            test_start = test_start + step_offset
            continue
        folds.append(
            Fold(
                fold_idx=fold_idx,
                train_start=pd.Timestamp(train_start),
                train_end=pd.Timestamp(test_start),  # exclusive
                test_start=pd.Timestamp(test_start),
                test_end=pd.Timestamp(test_end),  # exclusive
            )
        )
        fold_idx += 1
        test_start = test_start + step_offset
    return folds


def walk_forward(
    prices: pd.DataFrame,
    strategy_factory: Callable[[pd.DataFrame], Strategy],
    folds: list[Fold] | None = None,
    train_years: float = 5.0,
    test_years: float = 1.0,
    commission_bps: float = 0.5,
    slippage_bps: float = 1.0,
) -> WalkForwardResult:
    """Run walk-forward backtests.

    strategy_factory(train_df) -> Strategy : a callable that produces a strategy
        fitted on (or simply parameterized for) the train window. For
        parameter-free strategies, it can ignore train_df.
    """
    if folds is None:
        folds = make_folds(prices, train_years=train_years, test_years=test_years)
    if len(folds) == 0:
        raise ValueError("No folds produced. Check data range and fold sizes.")

    fold_results: list[BacktestResult] = []
    combined_returns_pieces = []
    combined_positions_pieces = []

    for fold in folds:
        train_df = prices.loc[fold.train_start : fold.train_end - pd.Timedelta(days=1)]
        # The strategy needs lookback history to compute features at the test
        # window's start. Pass the full series up through test_end and slice the
        # result. This is NOT lookahead: it's just giving the strategy its
        # warm-up data. Strategies must still only use past data per timestamp.
        warmup_and_test_df = prices.loc[: fold.test_end - pd.Timedelta(days=1)]
        strategy = strategy_factory(train_df)
        all_positions = strategy.generate_positions(warmup_and_test_df)
        # Restrict the backtest to the test window.
        test_df = prices.loc[fold.test_start : fold.test_end - pd.Timedelta(days=1)]
        test_positions = all_positions.reindex(test_df.index).fillna(0.0)
        bt = run_backtest(
            test_df,
            test_positions,
            name=f"{strategy.name}_fold{fold.fold_idx}",
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )
        fold_results.append(bt)
        combined_returns_pieces.append(bt.net_returns)
        combined_positions_pieces.append(bt.positions)

    combined_returns = pd.concat(combined_returns_pieces).sort_index()
    combined_positions = pd.concat(combined_positions_pieces).sort_index()
    combined_report = report(combined_returns, positions=combined_positions)

    return WalkForwardResult(
        folds=folds,
        fold_results=fold_results,
        combined_returns=combined_returns,
        combined_positions=combined_positions,
        combined_report=combined_report,
    )


def walk_forward_multi(
    prices: pd.DataFrame,
    strategy_factory: Callable[[pd.DataFrame], MultiAssetStrategy],
    folds: list[Fold] | None = None,
    train_years: float = 5.0,
    test_years: float = 1.0,
    commission_bps: float = 0.5,
    slippage_bps: float = 1.0,
) -> WalkForwardResult:
    """Walk-forward validation for MultiAssetStrategy.

    Mirrors walk_forward() but calls strategy.generate_weights() and
    run_multiasset_backtest() instead of generate_positions() / run_backtest().
    """
    if folds is None:
        folds = make_folds(prices, train_years=train_years, test_years=test_years)
    if len(folds) == 0:
        raise ValueError("No folds produced. Check data range and fold sizes.")

    fold_results: list[BacktestResult] = []
    combined_returns_pieces = []
    combined_positions_pieces = []

    for fold in folds:
        train_df = prices.loc[fold.train_start : fold.train_end - pd.Timedelta(days=1)]
        warmup_and_test_df = prices.loc[: fold.test_end - pd.Timedelta(days=1)]
        strategy = strategy_factory(train_df)
        all_weights = strategy.generate_weights(warmup_and_test_df)

        test_df = prices.loc[fold.test_start : fold.test_end - pd.Timedelta(days=1)]
        test_weights = all_weights.reindex(test_df.index).fillna(0.0)

        bt = run_multiasset_backtest(
            test_df,
            test_weights,
            name=f"{strategy.name}_fold{fold.fold_idx}",
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )
        fold_results.append(bt)
        combined_returns_pieces.append(bt.net_returns)
        combined_positions_pieces.append(bt.positions)

    combined_returns = pd.concat(combined_returns_pieces).sort_index()
    combined_positions = pd.concat(combined_positions_pieces).sort_index()
    combined_report = report(combined_returns, positions=combined_positions)

    return WalkForwardResult(
        folds=folds,
        fold_results=fold_results,
        combined_returns=combined_returns,
        combined_positions=combined_positions,
        combined_report=combined_report,
    )


def walk_forward_cross_sectional(
    prices_panel: pd.DataFrame,
    membership: pd.DataFrame,
    strategy_factory: Callable,
    folds: list[Fold] | None = None,
    train_years: float = 5.0,
    test_years: float = 2.0,
    commission_bps: float = 0.5,
    slippage_bps: float = 2.0,
    spy_returns: pd.Series | None = None,
) -> WalkForwardResult:
    """Walk-forward validation for CrossSectionalStrategy.

    strategy_factory(train_prices, train_membership) -> strategy instance.
    strategy.generate_weights(prices_panel, membership) -> weights DataFrame.

    prices_panel : adj_close panel, rows=dates, cols=tickers.
    membership   : bool panel, same shape, True where stock is in S&P 500.
    spy_returns  : optional SPY daily returns for benchmark equity curve.
    """
    if folds is None:
        folds = make_folds(prices_panel, train_years=train_years, test_years=test_years)
    if len(folds) == 0:
        raise ValueError("No folds produced. Check data range and fold sizes.")

    fold_results: list[BacktestResult] = []
    combined_returns_pieces: list[pd.Series] = []
    combined_positions_pieces: list[pd.Series] = []

    for fold in folds:
        train_prices = prices_panel.loc[fold.train_start : fold.train_end - pd.Timedelta(days=1)]
        train_membership = membership.reindex(train_prices.index).fillna(False)

        warmup_prices = prices_panel.loc[: fold.test_end - pd.Timedelta(days=1)]
        warmup_membership = membership.reindex(warmup_prices.index).fillna(False)

        strategy = strategy_factory(train_prices, train_membership)
        all_weights = strategy.generate_weights(warmup_prices, warmup_membership)

        test_prices = prices_panel.loc[fold.test_start : fold.test_end - pd.Timedelta(days=1)]
        test_weights = all_weights.reindex(index=test_prices.index).fillna(0.0)

        mem_aligned = membership.reindex(
            index=test_prices.index, columns=test_weights.columns
        ).fillna(False)
        test_weights = test_weights * mem_aligned

        fold_spy = spy_returns.reindex(test_prices.index) if spy_returns is not None else None
        bt = run_panel_backtest(
            test_prices,
            test_weights,
            name=f"{strategy.name}_fold{fold.fold_idx}",
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            spy_returns=fold_spy,
        )
        fold_results.append(bt)
        combined_returns_pieces.append(bt.net_returns)
        combined_positions_pieces.append(bt.positions)

    combined_returns_cs = pd.concat(combined_returns_pieces).sort_index()
    combined_positions_cs = pd.concat(combined_positions_pieces).sort_index()
    combined_report_cs = report(combined_returns_cs, positions=combined_positions_cs)

    return WalkForwardResult(
        folds=folds,
        fold_results=fold_results,
        combined_returns=combined_returns_cs,
        combined_positions=combined_positions_cs,
        combined_report=combined_report_cs,
    )
