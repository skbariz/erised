"""Performance metrics.

All functions take a pandas Series of returns (not prices) indexed by date.
Daily frequency assumed; annualization factor is configurable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerformanceReport:
    n_periods: int
    total_return: float
    cagr: float
    annualized_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    avg_win: float
    avg_loss: float
    turnover: float  # mean abs position change per period
    n_trades: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    def __str__(self) -> str:
        return (
            f"Periods:       {self.n_periods}\n"
            f"Total return:  {self.total_return:>8.2%}\n"
            f"CAGR:          {self.cagr:>8.2%}\n"
            f"Ann vol:       {self.annualized_vol:>8.2%}\n"
            f"Sharpe:        {self.sharpe:>8.2f}\n"
            f"Sortino:       {self.sortino:>8.2f}\n"
            f"Max DD:        {self.max_drawdown:>8.2%}\n"
            f"Calmar:        {self.calmar:>8.2f}\n"
            f"Hit rate:      {self.hit_rate:>8.2%}\n"
            f"Avg win:       {self.avg_win:>8.4%}\n"
            f"Avg loss:      {self.avg_loss:>8.4%}\n"
            f"Turnover/day:  {self.turnover:>8.2%}\n"
            f"Trades:        {self.n_trades}"
        )


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if len(returns) == 0:
        return 0.0
    total = float((1 + returns).prod())
    years = len(returns) / periods_per_year
    if total <= 0 or years <= 0:
        return float("nan")
    return total ** (1 / years) - 1


def annualized_vol(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    if len(returns) < 2:
        return 0.0
    rf_per_period = rf / periods_per_year
    excess = returns - rf_per_period
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    if len(returns) < 2:
        return 0.0
    rf_per_period = rf / periods_per_year
    excess = returns - rf_per_period
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    dd_std = downside.std(ddof=1)
    if dd_std == 0:
        return 0.0
    return float(excess.mean() / dd_std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Returns the most negative drawdown (e.g. -0.30 for a 30% drawdown)."""
    if len(returns) == 0:
        return 0.0
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def calmar(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("nan")
    return cagr(returns, periods_per_year) / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).mean())


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Stationary bootstrap confidence interval for the Sharpe ratio.

    Uses i.i.d. resampling. For more rigor on autocorrelated returns,
    swap in a stationary block bootstrap later.
    """
    rng = np.random.default_rng(seed)
    arr = returns.to_numpy()
    n = len(arr)
    if n < 30:
        return float("nan"), float("nan")
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        sd = sample.std(ddof=1)
        sharpes[i] = (sample.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    lo, hi = np.quantile(sharpes, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def turnover(positions: pd.Series) -> float:
    """Mean absolute change in position per period."""
    if len(positions) < 2:
        return 0.0
    return float(positions.diff().abs().mean())


def report(
    returns: pd.Series,
    positions: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> PerformanceReport:
    """Compute a full report. `positions` is optional; if omitted, turnover=NaN."""
    if positions is None:
        positions = pd.Series(index=returns.index, dtype=float)
        tov = float("nan")
        n_trades = int(0)
    else:
        tov = turnover(positions)
        # Count trades as position changes that cross zero or change sign
        diff = positions.diff().fillna(positions.iloc[0])
        n_trades = int((diff.abs() > 1e-9).sum())

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    total = float((1 + returns).prod() - 1)
    return PerformanceReport(
        n_periods=len(returns),
        total_return=total,
        cagr=cagr(returns, periods_per_year),
        annualized_vol=annualized_vol(returns, periods_per_year),
        sharpe=sharpe(returns, periods_per_year=periods_per_year),
        sortino=sortino(returns, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(returns),
        calmar=calmar(returns, periods_per_year),
        hit_rate=hit_rate(returns),
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        turnover=tov,
        n_trades=n_trades,
    )
