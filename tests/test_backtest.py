"""Tests for the backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import assert_no_lookahead, assert_no_lookahead_multi, run_backtest
from quant.strategies.base import (
    AbsoluteMomentumStrategy,
    ADXFilteredMACrossover,
    BuyAndHold,
    ConsecutiveDownDays,
    DefensiveEquityHedge,
    DualMALongShort,
    FastMACrossover,
    JanuaryBarometer,
    LongShortMACrossover,
    LongShortTSMomentum,
    MACDCrossover,
    MeanReversionStrategy,
    MovingAverageCrossover,
    QuarterEndDrift,
    RealizedVsImpliedVol,
    RiskParity,
    RSIReversionStrategy,
    SeasonalityTrendCombo,
    SellInMay,
    ShortHoldMeanReversion,
    SPYGLDRotation,
    SPYTLTDualMomentum,
    ThreeAssetMomentum,
    TimeSeriesMomentum,
    TrendVIXGate,
    TurnOfMonth,
    VIXFilteredReversionStrategy,
    VIXMeanReversion,
    VIXRegimeFilter,
    VIXSpikeEntry,
    VIXTrendSignal,
    VolatilityRegime,
    VolScaledTrend,
)


def _make_prices(returns: list[float], start: str = "2020-01-02") -> pd.DataFrame:
    """Build a tiny prices DataFrame from a list of daily returns."""
    idx = pd.bdate_range(start=start, periods=len(returns) + 1)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "adj_close": prices,
            "volume": [1_000_000] * len(prices),
        },
        index=idx,
    )
    return df


def _make_prices_with_vix(spy_rets, vix_levels, start="2020-01-02"):
    """Build prices DataFrame with a vix_close column for H006 tests."""
    df = _make_prices(spy_rets, start=start)
    assert len(vix_levels) == len(df), "vix_levels must match price rows"
    df["vix_close"] = vix_levels
    return df


# ---------------------------------------------------------------------------
# H003 – ShortHoldMeanReversion
# ---------------------------------------------------------------------------


def test_buy_and_hold_matches_benchmark():
    """A buy-and-hold strategy should equal the benchmark (minus initial trade cost)."""
    prices = _make_prices([0.01, -0.02, 0.03, 0.005, -0.01])
    bh = BuyAndHold()
    positions = bh.generate_positions(prices)
    result = run_backtest(prices, positions, commission_bps=0, slippage_bps=0)
    # With zero costs, BH net returns == bench returns
    np.testing.assert_allclose(result.net_returns.values, result.bench_returns.values, atol=1e-12)


def test_no_lookahead_position_shifted():
    """Position at time t must be applied to period (t, t+1].

    If we set position=+1 on day 0, we should NOT earn the return on day 0;
    we should earn the return on day 1.
    """
    prices = _make_prices([0.05])  # 5% return on day 1
    # Set position to +1 starting day 0
    positions = pd.Series([1.0, 1.0], index=prices.index)
    result = run_backtest(prices, positions, commission_bps=0, slippage_bps=0)
    # Day 0 return should be 0 (we hadn't traded yet)
    # Day 1 return should be 5%
    assert result.net_returns.iloc[0] == 0.0
    assert result.net_returns.iloc[1] == pytest.approx(0.05)


def test_costs_charged_on_position_change():
    """A position flip from 0 to 1 should incur cost = (commission + slippage) * 1.0."""
    prices = _make_prices([0.0, 0.0, 0.0])
    positions = pd.Series([0.0, 1.0, 1.0, 1.0], index=prices.index)
    result = run_backtest(prices, positions, commission_bps=1.0, slippage_bps=1.0)
    # Cost rate = 2 bps = 0.0002 per unit traded
    # First nonzero position appears day 1 -> charge on day 2 (shifted)
    assert result.costs.iloc[0] == 0.0
    assert result.costs.iloc[1] == 0.0
    assert result.costs.iloc[2] == pytest.approx(0.0002)
    assert result.costs.iloc[3] == 0.0


def test_short_position_inverts_returns():
    """Going short with +5% market move should produce -5% gross return."""
    prices = _make_prices([0.05])
    positions = pd.Series([-1.0, -1.0], index=prices.index)
    result = run_backtest(prices, positions, commission_bps=0, slippage_bps=0)
    assert result.net_returns.iloc[1] == pytest.approx(-0.05)


def test_position_magnitude_validated():
    """Positions > 1 or < -1 should raise."""
    prices = _make_prices([0.0])
    positions = pd.Series([2.0, 2.0], index=prices.index)
    with pytest.raises(ValueError, match="position magnitude"):
        run_backtest(prices, positions)


# ───────────────────────────────────────────────────────────────────────────
# H001 – MovingAverageCrossover
# ───────────────────────────────────────────────────────────────────────────


def test_ma_crossover_no_lookahead():
    """The MA crossover strategy must be lookahead-free."""
    # Build a longer series so MAs can warm up
    rng = np.random.default_rng(0)
    n = 1000
    rets = rng.normal(0.0005, 0.01, n).tolist()
    prices = _make_prices(rets)
    strat = MovingAverageCrossover(fast=20, slow=50)
    assert_no_lookahead(strat.generate_positions, prices, n_checks=10)


def test_buy_and_hold_compounds_correctly():
    """A 10% then -5% return: equity = 1.10 * 0.95 = 1.045"""
    prices = _make_prices([0.10, -0.05])
    positions = pd.Series([1.0, 1.0, 1.0], index=prices.index)
    result = run_backtest(prices, positions, commission_bps=0, slippage_bps=0)
    # Day 0: no position yet (shifted) -> return 0, equity 1
    # Day 1: applied position 1, return 10%, equity 1.10
    # Day 2: applied position 1, return -5%, equity 1.045
    assert result.equity.iloc[0] == pytest.approx(1.0)
    assert result.equity.iloc[1] == pytest.approx(1.10)
    assert result.equity.iloc[2] == pytest.approx(1.045)


# ───────────────────────────────────────────────────────────────────────────
# H002 – MeanReversionStrategy
# ───────────────────────────────────────────────────────────────────────────


def test_mean_reversion_no_lookahead():
    """MeanReversionStrategy must be lookahead-free across all parameter combos."""
    rng = np.random.default_rng(42)
    n = 1000
    rets = rng.normal(0.0005, 0.012, n).tolist()
    prices = _make_prices(rets)
    for lookback, threshold, hold_days in [(20, 2.0, 5), (10, 1.5, 3), (30, 2.5, 7)]:
        strat = MeanReversionStrategy(lookback=lookback, threshold=threshold, hold_days=hold_days)
        assert_no_lookahead(strat.generate_positions, prices, n_checks=10)


def test_mean_reversion_fires_on_down_move_and_holds():
    """Entry fires on a large down day and the position is held for exactly hold_days bars.

    Construction: 25 alternating ±0.01 returns establish a stable distribution
    (mean=0, std≈0.01). A −10% return on bar 26 produces z≈−9.75, well past the
    2σ threshold. No subsequent signal fires, so the hold should clear exactly
    after hold_days bars.
    """
    lookback = 20
    hold_days = 5
    n_warmup = 25
    # Alternating ±1%: mean=0, std≈1%, so −10% gives z≈−9.75
    rets = [0.01 if i % 2 == 0 else -0.01 for i in range(n_warmup)]
    rets.append(-0.10)  # signal bar
    rets.extend([0.001] * (hold_days + 2))  # neutral tail
    prices = _make_prices(rets)
    strat = MeanReversionStrategy(lookback=lookback, threshold=2.0, hold_days=hold_days)
    pos = strat.generate_positions(prices)

    # iloc[0] is the initial price row (no return); signal bar is iloc[n_warmup + 1]
    signal_iloc = n_warmup + 1
    for i in range(hold_days):
        assert pos.iloc[signal_iloc + i] == pytest.approx(1.0), f"Expected long at hold offset {i}"
    assert pos.iloc[signal_iloc + hold_days] == pytest.approx(0.0), (
        "Expected flat after hold expires"
    )


def test_mean_reversion_flat_before_warmup():
    """Position must be 0 until rolling stats have enough history."""
    lookback = 20
    rets = [0.005] * 40
    prices = _make_prices(rets)
    strat = MeanReversionStrategy(lookback=lookback, threshold=2.0, hold_days=5)
    pos = strat.generate_positions(prices)
    # rolling_std.shift(1) first non-NaN at iloc[lookback + 1]; everything before is flat.
    assert (pos.iloc[: lookback + 1] == 0.0).all(), "Expected all-flat during warm-up"


# ───────────────────────────────────────────────────────────────────────────
# H003 – ShortHoldMeanReversion
# ───────────────────────────────────────────────────────────────────────────


def test_short_hold_mean_reversion_no_lookahead():
    rng = np.random.default_rng(1)
    prices = _make_prices(rng.normal(0.0005, 0.012, 1000).tolist())
    assert_no_lookahead(ShortHoldMeanReversion().generate_positions, prices, n_checks=10)


def test_short_hold_mean_reversion_hold_expires_at_2():
    """Signal fires and hold expires after exactly 2 bars."""
    n_warmup = 25
    rets = [0.01 if i % 2 == 0 else -0.01 for i in range(n_warmup)]
    rets.append(-0.10)
    rets.extend([0.001] * 5)
    prices = _make_prices(rets)
    pos = ShortHoldMeanReversion().generate_positions(prices)
    signal_iloc = n_warmup + 1
    assert pos.iloc[signal_iloc] == pytest.approx(1.0)
    assert pos.iloc[signal_iloc + 1] == pytest.approx(1.0)
    assert pos.iloc[signal_iloc + 2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# H004 – RSIReversionStrategy
# ---------------------------------------------------------------------------


# ───────────────────────────────────────────────────────────────────────────
# H004 – RSIReversionStrategy
# ───────────────────────────────────────────────────────────────────────────


def test_rsi_reversion_no_lookahead():
    rng = np.random.default_rng(2)
    prices = _make_prices(rng.normal(0.0005, 0.012, 1000).tolist())
    assert_no_lookahead(RSIReversionStrategy().generate_positions, prices, n_checks=10)


def test_rsi_reversion_flat_before_warmup():
    """No position before RSI has enough history."""
    prices = _make_prices([0.005] * 50)
    pos = RSIReversionStrategy(period=14).generate_positions(prices)
    assert (pos.iloc[:14] == 0.0).all()


def test_rsi_reversion_exits_above_exit_rsi():
    """When RSI exceeds exit_rsi while in a hold, position becomes 0."""
    rng = np.random.default_rng(99)
    # Build a series long enough for EWM warmup
    prices = _make_prices(rng.normal(0.0005, 0.012, 500).tolist())
    strat = RSIReversionStrategy(period=14, entry_rsi=30.0, exit_rsi=50.0, hold_days=5)
    pos = strat.generate_positions(prices)
    # positions are valid (no NaN, within [-1, 1])
    assert pos.notna().all()
    assert (pos.abs() <= 1.0).all()


# ---------------------------------------------------------------------------
# H005 – ConsecutiveDownDays
# ---------------------------------------------------------------------------


# ───────────────────────────────────────────────────────────────────────────
# H005 – ConsecutiveDownDays
# ───────────────────────────────────────────────────────────────────────────


def test_consecutive_down_days_no_lookahead():
    rng = np.random.default_rng(3)
    prices = _make_prices(rng.normal(0.0005, 0.012, 1000).tolist())
    assert_no_lookahead(ConsecutiveDownDays().generate_positions, prices, n_checks=10)


def test_consecutive_down_days_fires_on_streak():
    """Entry fires after exactly n_down consecutive down days."""
    n_down, hold_days = 3, 3
    # 10 up days, then 3 down days, then 10 up days
    rets = [0.01] * 10 + [-0.01] * n_down + [0.01] * 10
    prices = _make_prices(rets)
    strat = ConsecutiveDownDays(n_down=n_down, hold_days=hold_days)
    pos = strat.generate_positions(prices)
    # Signal fires at iloc[10 + n_down] (0-indexed, with initial price row at iloc[0])
    signal_iloc = 1 + 10 + n_down - 1  # last of the 3 down days
    assert pos.iloc[signal_iloc] == pytest.approx(1.0), f"Expected entry at iloc {signal_iloc}"
    assert pos.iloc[signal_iloc + hold_days] == pytest.approx(0.0)


def test_consecutive_down_days_no_signal_after_up_break():
    """2 down, 1 up, 2 down should NOT fire a 3-day streak signal."""
    rets = [0.01] * 5 + [-0.01, -0.01, 0.01, -0.01, -0.01] + [0.01] * 5
    prices = _make_prices(rets)
    strat = ConsecutiveDownDays(n_down=3, hold_days=3)
    pos = strat.generate_positions(prices)
    assert (pos == 0.0).all()


# ---------------------------------------------------------------------------
# H006 – VIXFilteredReversionStrategy
# ---------------------------------------------------------------------------


# ───────────────────────────────────────────────────────────────────────────
# H006 – VIXFilteredReversionStrategy
# ───────────────────────────────────────────────────────────────────────────


def test_vix_filtered_reversion_no_lookahead():
    rng = np.random.default_rng(4)
    n = 1000
    spy_rets = rng.normal(0.0005, 0.012, n).tolist()
    vix = [20.0] * (n + 1)
    prices = _make_prices_with_vix(spy_rets, vix)
    assert_no_lookahead(VIXFilteredReversionStrategy().generate_positions, prices, n_checks=10)


def test_vix_filtered_reversion_suppressed_in_low_vix():
    """No entries when VIX is below the threshold even on large down days."""
    n_warmup = 25
    rets = [0.01 if i % 2 == 0 else -0.01 for i in range(n_warmup)]
    rets.append(-0.10)
    rets.extend([0.001] * 5)
    vix = [10.0] * (len(rets) + 1)  # VIX = 10, below threshold of 20
    prices = _make_prices_with_vix(rets, vix)
    strat = VIXFilteredReversionStrategy(vix_threshold=20.0)
    pos = strat.generate_positions(prices)
    assert (pos == 0.0).all(), "Expected no entries when VIX below threshold"


def test_vix_filtered_reversion_requires_vix_column():
    prices = _make_prices([0.01] * 10)
    with pytest.raises(ValueError, match="vix_close"):
        VIXFilteredReversionStrategy().generate_positions(prices)


# ---------------------------------------------------------------------------
# H007 – TimeSeriesMomentum
# ---------------------------------------------------------------------------


# ───────────────────────────────────────────────────────────────────────────
# H007 – TimeSeriesMomentum
# ───────────────────────────────────────────────────────────────────────────


def test_ts_momentum_no_lookahead():
    rng = np.random.default_rng(5)
    prices = _make_prices(rng.normal(0.0005, 0.012, 1000).tolist())
    assert_no_lookahead(TimeSeriesMomentum().generate_positions, prices, n_checks=10)


def test_ts_momentum_flat_before_warmup():
    """No position before long_days + skip_days of history."""
    strat = TimeSeriesMomentum(long_months=12, skip_months=1)
    prices = _make_prices([0.001] * (strat.long_days + 5))
    pos = strat.generate_positions(prices)
    assert (pos.iloc[: strat.long_days] == 0.0).all()


# ---------------------------------------------------------------------------
# H008 – AbsoluteMomentumStrategy
# ---------------------------------------------------------------------------


# ───────────────────────────────────────────────────────────────────────────
# H008 – AbsoluteMomentumStrategy
# ───────────────────────────────────────────────────────────────────────────


def test_absolute_momentum_no_lookahead():
    rng = np.random.default_rng(6)
    prices = _make_prices(rng.normal(0.0005, 0.012, 1000).tolist())
    assert_no_lookahead(AbsoluteMomentumStrategy().generate_positions, prices, n_checks=10)


def test_absolute_momentum_long_in_uptrend():
    """Sustained uptrend → positive 12-month return → long."""
    strat = AbsoluteMomentumStrategy(lookback_months=12)
    prices = _make_prices([0.001] * (strat.lookback_days + 10))
    pos = strat.generate_positions(prices)
    # After warmup, positive drift → momentum positive → long
    assert pos.iloc[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# H009 – MACDCrossover
# ---------------------------------------------------------------------------


# ───────────────────────────────────────────────────────────────────────────
# H009 – MACDCrossover
# ───────────────────────────────────────────────────────────────────────────


def test_macd_no_lookahead():
    rng = np.random.default_rng(7)
    prices = _make_prices(rng.normal(0.0005, 0.012, 1000).tolist())
    assert_no_lookahead(MACDCrossover().generate_positions, prices, n_checks=10)


def test_macd_flat_during_warmup():
    strat = MACDCrossover(fast=12, slow=26, signal=9)
    prices = _make_prices([0.001] * (strat.slow + strat.signal + 5))
    pos = strat.generate_positions(prices)
    warmup = strat.slow + strat.signal
    assert (pos.iloc[:warmup] == 0.0).all()


# ---------------------------------------------------------------------------
# H010 – FastMACrossover
# ---------------------------------------------------------------------------


# ───────────────────────────────────────────────────────────────────────────
# H010 – FastMACrossover
# ───────────────────────────────────────────────────────────────────────────


def test_fast_ma_no_lookahead():
    rng = np.random.default_rng(8)
    prices = _make_prices(rng.normal(0.0005, 0.012, 1000).tolist())
    assert_no_lookahead(FastMACrossover().generate_positions, prices, n_checks=10)


def test_fast_ma_flat_before_slow_ma_warm():
    strat = FastMACrossover(fast=10, slow=50)
    prices = _make_prices([0.001] * 55)
    pos = strat.generate_positions(prices)
    # slow MA(50) first valid at iloc[49]; everything before that must be flat
    assert (pos.iloc[: strat.slow - 1] == 0.0).all()


# ---------------------------------------------------------------------------
# H011 – ADXFilteredMACrossover
# ---------------------------------------------------------------------------


# ───────────────────────────────────────────────────────────────────────────
# H011 – ADXFilteredMACrossover
# ───────────────────────────────────────────────────────────────────────────


def test_adx_filtered_ma_no_lookahead():
    rng = np.random.default_rng(9)
    prices = _make_prices(rng.normal(0.0005, 0.012, 1000).tolist())
    # ADX needs OHLC — _make_prices already provides high/low/close
    assert_no_lookahead(ADXFilteredMACrossover().generate_positions, prices, n_checks=10)


def test_adx_filtered_ma_suppresses_weak_trend():
    """With a very high ADX threshold (100), no MA signal should pass through."""
    rng = np.random.default_rng(10)
    prices = _make_prices(rng.normal(0.0, 0.005, 1000).tolist())  # low-vol, flat
    strat = ADXFilteredMACrossover(adx_threshold=100.0)  # impossible threshold
    pos = strat.generate_positions(prices)
    assert (pos == 0.0).all()


# ---------------------------------------------------------------------------
# H012 – VolatilityRegime
# ---------------------------------------------------------------------------


# ───────────────────────────────────────────────────────────────────────────
# H012 – VolatilityRegime
# ───────────────────────────────────────────────────────────────────────────


def test_vol_regime_no_lookahead():
    rng = np.random.default_rng(11)
    prices = _make_prices(rng.normal(0.0005, 0.012, 1500).tolist())
    assert_no_lookahead(VolatilityRegime().generate_positions, prices, n_checks=10)


def test_vol_regime_flat_before_warmup():
    strat = VolatilityRegime(vol_lookback=20, regime_lookback=252)
    min_bars = strat.vol_lookback + strat.regime_lookback // 2 + 5
    prices = _make_prices([0.005] * min_bars)
    pos = strat.generate_positions(prices)
    assert (pos.iloc[: strat.vol_lookback + 1] == 0.0).all()


# ───────────────────────────────────────────────────────────────────────────
# H013 – LongShortMACrossover
# ───────────────────────────────────────────────────────────────────────────


def test_ls_ma_crossover_no_lookahead():
    rng = np.random.default_rng(12)
    prices = _make_prices(rng.normal(0.0005, 0.012, 1000).tolist())
    assert_no_lookahead(LongShortMACrossover().generate_positions, prices, n_checks=10)


def test_ls_ma_crossover_short_in_downtrend():
    """After warmup in a sustained downtrend, position must be -1 (not 0 as in H001)."""
    rets = [-0.003] * 300
    prices = _make_prices(rets)
    h001 = MovingAverageCrossover(50, 200).generate_positions(prices)
    h013 = LongShortMACrossover(50, 200).generate_positions(prices)
    # H001 goes flat; H013 goes short
    assert (h001.iloc[200:] == 0.0).all()
    assert (h013.iloc[200:] == -1.0).all()


def test_ls_ma_crossover_positions_binary():
    """All positions must be in {-1, 0, +1}."""
    rng = np.random.default_rng(13)
    prices = _make_prices(rng.normal(0.0005, 0.012, 500).tolist())
    pos = LongShortMACrossover().generate_positions(prices)
    assert pos.isin([-1.0, 0.0, 1.0]).all()


# ───────────────────────────────────────────────────────────────────────────
# H014 – LongShortTSMomentum
# ───────────────────────────────────────────────────────────────────────────


def test_ls_ts_momentum_no_lookahead():
    rng = np.random.default_rng(14)
    prices = _make_prices(rng.normal(0.0005, 0.012, 1000).tolist())
    assert_no_lookahead(LongShortTSMomentum().generate_positions, prices, n_checks=10)


def test_ls_ts_momentum_short_in_downtrend():
    """Sustained downtrend → negative 12m return → short position."""
    strat = LongShortTSMomentum(long_months=12, skip_months=1)
    rets = [-0.001] * (strat.long_days + 30)
    prices = _make_prices(rets)
    pos = strat.generate_positions(prices)
    assert pos.iloc[-1] == pytest.approx(-1.0)


def test_ls_ts_momentum_positions_binary():
    """All non-NaN positions must be in {-1, 0, +1}."""
    rng = np.random.default_rng(15)
    prices = _make_prices(rng.normal(0.0005, 0.012, 500).tolist())
    pos = LongShortTSMomentum().generate_positions(prices)
    assert pos.isin([-1.0, 0.0, 1.0]).all()


# ───────────────────────────────────────────────────────────────────────────
# H015 – DualMALongShort
# ───────────────────────────────────────────────────────────────────────────


def test_dual_ma_ls_no_lookahead():
    rng = np.random.default_rng(16)
    prices = _make_prices(rng.normal(0.0005, 0.012, 1000).tolist())
    assert_no_lookahead(DualMALongShort().generate_positions, prices, n_checks=10)


def test_dual_ma_ls_warms_up_faster_than_h013():
    """DualMALongShort (slow=100) produces non-zero positions before H013 (slow=200) warms up."""
    rets = [0.001] * 210
    prices = _make_prices(rets)
    h013 = LongShortMACrossover().generate_positions(prices)
    h015 = DualMALongShort().generate_positions(prices)
    # H013 (slow=200) flat through index 198 (slow-1 exclusive); H015 active from index 99
    assert (h013.iloc[:199] == 0.0).all()
    assert (h015.iloc[100:] != 0.0).any()


def test_dual_ma_ls_positions_binary():
    rng = np.random.default_rng(17)
    prices = _make_prices(rng.normal(0.0005, 0.012, 500).tolist())
    pos = DualMALongShort().generate_positions(prices)
    assert pos.isin([-1.0, 0.0, 1.0]).all()


# ───────────────────────────────────────────────────────────────────────────
# H016 – VolScaledTrend
# ───────────────────────────────────────────────────────────────────────────


def test_vol_scaled_trend_no_lookahead():
    rng = np.random.default_rng(18)
    prices = _make_prices(rng.normal(0.0005, 0.012, 1000).tolist())
    assert_no_lookahead(VolScaledTrend().generate_positions, prices, n_checks=10)


def test_vol_scaled_trend_positions_in_range():
    """All positions must be in [-1, 1]."""
    rng = np.random.default_rng(19)
    prices = _make_prices(rng.normal(0.0005, 0.02, 1000).tolist())
    pos = VolScaledTrend().generate_positions(prices)
    assert (pos.abs() <= 1.0 + 1e-9).all()


def test_vol_scaled_trend_scales_down_in_high_vol():
    """In high-vol conditions, position magnitude is < 1 (not binary full notional)."""
    # ±5% / ±4% alternating → realized vol well above 10% target
    rets = [0.05 if i % 2 == 0 else -0.04 for i in range(500)]
    prices = _make_prices(rets)
    strat = VolScaledTrend(target_vol=0.10)
    pos = strat.generate_positions(prices)
    non_zero = pos[pos != 0.0]
    if len(non_zero) > 0:
        assert (non_zero.abs() < 1.0).all()


# ───────────────────────────────────────────────────────────────────────────
# H017 – TrendVIXGate
# ───────────────────────────────────────────────────────────────────────────


def test_trend_vix_gate_no_lookahead():
    rng = np.random.default_rng(20)
    n = 1000
    spy_rets = rng.normal(0.0005, 0.012, n).tolist()
    vix = [20.0] * (n + 1)
    prices = _make_prices_with_vix(spy_rets, vix)
    assert_no_lookahead(TrendVIXGate().generate_positions, prices, n_checks=10)


def test_trend_vix_gate_flat_in_crash():
    """When VIX > 40, position must be 0 regardless of MA signal."""
    rets = [0.001] * 300  # uptrend — MA signal would be long
    vix = [50.0] * 301  # crash-level VIX throughout
    prices = _make_prices_with_vix(rets, vix)
    pos = TrendVIXGate(vix_extreme=40.0).generate_positions(prices)
    assert (pos == 0.0).all()


def test_trend_vix_gate_active_in_normal_vix():
    """Below the VIX gate, strategy behaves like a standard long/short MA."""
    rets = [0.001] * 300
    vix = [15.0] * 301  # normal VIX, gate inactive
    prices = _make_prices_with_vix(rets, vix)
    pos = TrendVIXGate(vix_extreme=40.0).generate_positions(prices)
    # After warmup in uptrend, position should be +1
    assert pos.iloc[-1] == pytest.approx(1.0)


def test_trend_vix_gate_requires_vix_column():
    prices = _make_prices([0.01] * 10)
    with pytest.raises(ValueError, match="vix_close"):
        TrendVIXGate().generate_positions(prices)


# ---------------------------------------------------------------------------
# H018: SPYTLTDualMomentum helpers
# ---------------------------------------------------------------------------


def _make_multiasset_prices(spy_rets, tlt_rets=None, gld_rets=None, start="2010-01-04"):
    """Build a prices DataFrame with SPY + optional TLT/GLD columns."""
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)

    spy_px = [100.0]
    for r in spy_rets:
        spy_px.append(spy_px[-1] * (1 + r))

    df = pd.DataFrame(
        {
            "open": spy_px,
            "high": [p * 1.01 for p in spy_px],
            "low": [p * 0.99 for p in spy_px],
            "close": spy_px,
            "adj_close": spy_px,
            "volume": [1_000_000] * len(spy_px),
        },
        index=idx,
    )

    if tlt_rets is not None:
        tlt_px = [100.0]
        for r in tlt_rets:
            tlt_px.append(tlt_px[-1] * (1 + r))
        df["tlt_close"] = tlt_px

    if gld_rets is not None:
        gld_px = [100.0]
        for r in gld_rets:
            gld_px.append(gld_px[-1] * (1 + r))
        df["gld_close"] = gld_px

    return df


# ---------------------------------------------------------------------------
# H018 tests
# ---------------------------------------------------------------------------


def test_spy_tlt_dual_momentum_no_lookahead():
    n = 700
    spy_rets = [0.001] * n
    tlt_rets = [-0.0005] * n
    prices = _make_multiasset_prices(spy_rets, tlt_rets=tlt_rets)
    s = SPYTLTDualMomentum()
    assert_no_lookahead_multi(s.generate_weights, prices)


def test_spy_tlt_dual_momentum_spy_positive_holds_spy():
    """When SPY momentum positive, should be 100% SPY, 0% TLT."""
    spy_rets = [0.001] * 300  # rising SPY
    tlt_rets = [-0.001] * 300  # falling TLT
    prices = _make_multiasset_prices(spy_rets, tlt_rets=tlt_rets)
    w = SPYTLTDualMomentum().generate_weights(prices)
    assert w["SPY"].iloc[-1] == pytest.approx(1.0)
    assert w["TLT"].iloc[-1] == pytest.approx(0.0)


def test_spy_tlt_dual_momentum_cash_when_both_negative():
    """When both SPY and TLT momentum negative, both weights must be 0."""
    spy_rets = [-0.001] * 300  # falling SPY
    tlt_rets = [-0.001] * 300  # falling TLT
    prices = _make_multiasset_prices(spy_rets, tlt_rets=tlt_rets)
    w = SPYTLTDualMomentum().generate_weights(prices)
    assert w["SPY"].iloc[-1] == pytest.approx(0.0)
    assert w["TLT"].iloc[-1] == pytest.approx(0.0)


def test_spy_tlt_dual_momentum_tlt_when_spy_negative():
    """SPY falling, TLT rising → should allocate to TLT."""
    spy_rets = [-0.001] * 300
    tlt_rets = [0.001] * 300  # TLT rising
    prices = _make_multiasset_prices(spy_rets, tlt_rets=tlt_rets)
    w = SPYTLTDualMomentum().generate_weights(prices)
    assert w["TLT"].iloc[-1] == pytest.approx(1.0)
    assert w["SPY"].iloc[-1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# H019 tests
# ---------------------------------------------------------------------------


def test_spy_gld_rotation_no_lookahead():
    n = 700
    spy_rets = [0.001] * n
    gld_rets = [0.0005] * n
    prices = _make_multiasset_prices(spy_rets, gld_rets=gld_rets)
    s = SPYGLDRotation()
    assert_no_lookahead_multi(s.generate_weights, prices)


def test_spy_gld_rotation_spy_when_positive():
    spy_rets = [0.001] * 300
    gld_rets = [0.0005] * 300
    prices = _make_multiasset_prices(spy_rets, gld_rets=gld_rets)
    w = SPYGLDRotation().generate_weights(prices)
    assert w["SPY"].iloc[-1] == pytest.approx(1.0)
    assert w["GLD"].iloc[-1] == pytest.approx(0.0)


def test_spy_gld_rotation_gld_when_spy_negative():
    spy_rets = [-0.001] * 300
    gld_rets = [0.0005] * 300
    prices = _make_multiasset_prices(spy_rets, gld_rets=gld_rets)
    w = SPYGLDRotation().generate_weights(prices)
    assert w["GLD"].iloc[-1] == pytest.approx(1.0)
    assert w["SPY"].iloc[-1] == pytest.approx(0.0)


def test_spy_gld_rotation_always_fully_invested():
    """Sum of weights must always be 0 (pre-warmup) or 1 (post-warmup)."""
    spy_rets = [-0.001] * 300
    gld_rets = [0.001] * 300
    prices = _make_multiasset_prices(spy_rets, gld_rets=gld_rets)
    w = SPYGLDRotation().generate_weights(prices)
    row_sums = (w["SPY"] + w["GLD"]).iloc[270:]  # after warmup
    assert np.allclose(row_sums.values, 1.0)


# ---------------------------------------------------------------------------
# H020 tests
# ---------------------------------------------------------------------------


def test_three_asset_momentum_no_lookahead():
    n = 700
    spy_rets = [0.001] * n
    tlt_rets = [0.0005] * n
    gld_rets = [0.0008] * n
    prices = _make_multiasset_prices(spy_rets, tlt_rets=tlt_rets, gld_rets=gld_rets)
    s = ThreeAssetMomentum()
    assert_no_lookahead_multi(s.generate_weights, prices)


def test_three_asset_momentum_picks_best():
    """Asset with highest 12m return should get weight 1.0."""
    spy_rets = [0.002] * 300  # best
    tlt_rets = [0.001] * 300
    gld_rets = [0.0005] * 300
    prices = _make_multiasset_prices(spy_rets, tlt_rets=tlt_rets, gld_rets=gld_rets)
    w = ThreeAssetMomentum().generate_weights(prices)
    assert w["SPY"].iloc[-1] == pytest.approx(1.0)
    assert w["TLT"].iloc[-1] == pytest.approx(0.0)
    assert w["GLD"].iloc[-1] == pytest.approx(0.0)


def test_three_asset_momentum_cash_when_all_negative():
    spy_rets = [-0.001] * 300
    tlt_rets = [-0.001] * 300
    gld_rets = [-0.001] * 300
    prices = _make_multiasset_prices(spy_rets, tlt_rets=tlt_rets, gld_rets=gld_rets)
    w = ThreeAssetMomentum().generate_weights(prices)
    assert w.iloc[-1].sum() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# H021 tests
# ---------------------------------------------------------------------------


def test_risk_parity_no_lookahead():
    n = 700
    spy_rets = [0.001 * (1 + 0.1 * (i % 10)) for i in range(n)]
    tlt_rets = [0.0005 * (1 + 0.1 * ((i + 3) % 10)) for i in range(n)]
    prices = _make_multiasset_prices(spy_rets, tlt_rets=tlt_rets)
    s = RiskParity()
    assert_no_lookahead_multi(s.generate_weights, prices)


def test_risk_parity_weights_sum_to_one():
    """After warmup, SPY + TLT weights must sum to 1.0."""
    spy_rets = [0.001] * 300
    tlt_rets = [0.0005] * 300
    prices = _make_multiasset_prices(spy_rets, tlt_rets=tlt_rets)
    w = RiskParity(vol_lookback=60).generate_weights(prices)
    # After warmup + 1-bar shift (61 + 1 = 62 bars), weights should sum to 1
    valid = (w["SPY"] + w["TLT"]) > 0
    row_sums = (w["SPY"] + w["TLT"])[valid]
    assert np.allclose(row_sums.values, 1.0)


def test_risk_parity_higher_vol_gets_lower_weight():
    """Asset with higher realized vol should receive lower weight."""
    import numpy as np

    rng = np.random.default_rng(42)
    # SPY: high vol (2%), TLT: low vol (0.5%)
    spy_rets = (rng.normal(0, 0.02, 300)).tolist()
    tlt_rets = (rng.normal(0, 0.005, 300)).tolist()
    prices = _make_multiasset_prices(spy_rets, tlt_rets=tlt_rets)
    w = RiskParity(vol_lookback=60).generate_weights(prices)
    # After warmup, TLT should have higher weight than SPY
    assert w["TLT"].iloc[-1] > w["SPY"].iloc[-1]


# ---------------------------------------------------------------------------
# H022 tests
# ---------------------------------------------------------------------------


def test_defensive_equity_hedge_no_lookahead():
    n = 700
    spy_rets = [0.001] * n
    tlt_rets = [-0.0005] * n
    prices = _make_multiasset_prices(spy_rets, tlt_rets=tlt_rets)
    s = DefensiveEquityHedge()
    assert_no_lookahead_multi(s.generate_weights, prices)


def test_defensive_equity_hedge_full_spy_in_uptrend():
    """Uptrend → 100% SPY, 0% TLT."""
    spy_rets = [0.001] * 300
    tlt_rets = [0.0] * 300
    prices = _make_multiasset_prices(spy_rets, tlt_rets=tlt_rets)
    w = DefensiveEquityHedge().generate_weights(prices)
    assert w["SPY"].iloc[-1] == pytest.approx(1.0)
    assert w["TLT"].iloc[-1] == pytest.approx(0.0)


def test_defensive_equity_hedge_splits_in_downtrend():
    """Downtrend → 50% SPY, 50% TLT."""
    spy_rets = [-0.001] * 300
    tlt_rets = [0.0] * 300
    prices = _make_multiasset_prices(spy_rets, tlt_rets=tlt_rets)
    w = DefensiveEquityHedge().generate_weights(prices)
    assert w["SPY"].iloc[-1] == pytest.approx(0.5)
    assert w["TLT"].iloc[-1] == pytest.approx(0.5)


def test_defensive_equity_hedge_always_fully_invested():
    """Weights must always sum to 0 (pre-warmup) or 1 (post-warmup)."""
    spy_rets = [-0.001] * 300
    tlt_rets = [0.001] * 300
    prices = _make_multiasset_prices(spy_rets, tlt_rets=tlt_rets)
    w = DefensiveEquityHedge().generate_weights(prices)
    valid = (w["SPY"] + w["TLT"]) > 0
    row_sums = (w["SPY"] + w["TLT"])[valid]
    assert np.allclose(row_sums.values, 1.0)


# ---------------------------------------------------------------------------
# Helpers for calendar strategy tests
# ---------------------------------------------------------------------------


def _make_prices_dated(n_rets: int, start: str = "1995-01-03") -> pd.DataFrame:
    """Build prices anchored at a known calendar date for calendar-effect tests."""
    return _make_prices([0.0005] * n_rets, start=start)


# ---------------------------------------------------------------------------
# H023 — SellInMay
# ---------------------------------------------------------------------------


def test_sell_in_may_no_lookahead():
    prices = _make_prices([0.001] * 700, start="1995-01-03")
    assert_no_lookahead(SellInMay().generate_positions, prices)


def test_sell_in_may_long_in_november():
    """November is a favorable month — position must be 1.0."""
    prices = _make_prices_dated(500, start="1995-01-03")
    pos = SellInMay().generate_positions(prices)
    nov_dates = pos.index[pos.index.month == 11]
    assert len(nov_dates) > 0
    assert (pos[nov_dates] == 1.0).all()


def test_sell_in_may_flat_in_june():
    """June is an unfavorable month — position must be 0.0."""
    prices = _make_prices_dated(500, start="1995-01-03")
    pos = SellInMay().generate_positions(prices)
    jun_dates = pos.index[pos.index.month == 6]
    assert len(jun_dates) > 0
    assert (pos[jun_dates] == 0.0).all()


def test_sell_in_may_transitions_at_boundary():
    """Position flips at May 1 and November 1."""
    prices = _make_prices_dated(500, start="1995-01-03")
    pos = SellInMay().generate_positions(prices)
    apr_dates = pos.index[pos.index.month == 4]
    may_dates = pos.index[pos.index.month == 5]
    oct_dates = pos.index[pos.index.month == 10]
    nov_dates = pos.index[pos.index.month == 11]
    assert (pos[apr_dates] == 1.0).all()
    assert (pos[may_dates] == 0.0).all()
    assert (pos[oct_dates] == 0.0).all()
    assert (pos[nov_dates] == 1.0).all()


# ---------------------------------------------------------------------------
# H024 — TurnOfMonth
# ---------------------------------------------------------------------------


def test_turn_of_month_no_lookahead():
    """Ranks must be stable when future price rows are added."""
    prices = _make_prices([0.001] * 700, start="1995-01-03")
    assert_no_lookahead(TurnOfMonth().generate_positions, prices)


def test_turn_of_month_first_day_long():
    """First trading day of a month must be in the window."""
    # Start at a month boundary: 1995-01-03 is the first business day of January
    prices = _make_prices_dated(50, start="1995-01-03")
    pos = TurnOfMonth(n_days=3).generate_positions(prices)
    assert pos.iloc[0] == pytest.approx(1.0)


def test_turn_of_month_middle_of_month_flat():
    """Middle trading days of a month must be flat."""
    # Jan 1995 has ~21 trading days; the 10th trading day is in the middle
    prices = _make_prices_dated(30, start="1995-01-03")
    pos = TurnOfMonth(n_days=3).generate_positions(prices)
    # Day at index 9 (10th trading day, 0-indexed): not first 3, not last 3 of Jan
    assert pos.iloc[9] == pytest.approx(0.0)


def test_turn_of_month_last_day_long():
    """Last trading day of a month must be in the window."""
    # Build enough data that we can check the last day of January 1995
    prices = _make_prices_dated(25, start="1995-01-03")
    pos = TurnOfMonth(n_days=3).generate_positions(prices)
    jan_dates = pos.index[pos.index.month == 1]
    assert pos[jan_dates[-1]] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# H025 — JanuaryBarometer
# ---------------------------------------------------------------------------


def test_january_barometer_no_lookahead():
    prices = _make_prices([0.001] * 700, start="1995-01-03")
    assert_no_lookahead(JanuaryBarometer().generate_positions, prices)


def test_january_barometer_long_after_positive_january():
    """Rising January → position = 1.0 in February."""
    # Start Jan 1995, rising prices throughout
    prices = _make_prices([0.003] * 400, start="1995-01-03")
    pos = JanuaryBarometer().generate_positions(prices)
    feb_dates = pos.index[(pos.index.year == 1995) & (pos.index.month == 2)]
    assert len(feb_dates) > 0
    assert (pos[feb_dates] == 1.0).all()


def test_january_barometer_flat_after_negative_january():
    """Falling January → position = 0.0 in February."""
    # Jan 1995 falls, then flat for the rest
    n_total = 400
    # Jan 1995 has ~21 trading days, make them fall
    jan_rets = [-0.005] * 21
    rest_rets = [0.001] * (n_total - 21)
    prices = _make_prices(jan_rets + rest_rets, start="1995-01-03")
    pos = JanuaryBarometer().generate_positions(prices)
    feb_dates = pos.index[(pos.index.year == 1995) & (pos.index.month == 2)]
    assert len(feb_dates) > 0
    assert (pos[feb_dates] == 0.0).all()


def test_january_barometer_flat_in_january():
    """Position must be 0.0 in January itself (signal not yet set)."""
    prices = _make_prices([0.001] * 300, start="1995-01-03")
    pos = JanuaryBarometer().generate_positions(prices)
    jan_dates = pos.index[pos.index.month == 1]
    assert (pos[jan_dates] == 0.0).all()


# ---------------------------------------------------------------------------
# H026 — QuarterEndDrift
# ---------------------------------------------------------------------------


def test_quarter_end_drift_no_lookahead():
    """Ranks must be stable when future price rows are added."""
    prices = _make_prices([0.001] * 700, start="1995-01-03")
    assert_no_lookahead(QuarterEndDrift().generate_positions, prices)


def test_quarter_end_drift_last_day_of_quarter_long():
    """The last trading day of a quarter must be in the drift window."""
    # Mar 31 1995 is a Friday — last business day of Q1
    prices = _make_prices_dated(80, start="1995-01-03")
    pos = QuarterEndDrift(n_days=5).generate_positions(prices)
    # Last day of Q1 1995
    q1_dates = pos.index[(pos.index.year == 1995) & (pos.index.month == 3)]
    assert pos[q1_dates[-1]] == pytest.approx(1.0)


def test_quarter_end_drift_early_in_quarter_flat():
    """Early trading days of a quarter (not near end) must be flat."""
    prices = _make_prices_dated(80, start="1995-01-03")
    pos = QuarterEndDrift(n_days=5).generate_positions(prices)
    # 10th trading day of Q1 (mid-January) must be flat
    assert pos.iloc[9] == pytest.approx(0.0)


def test_quarter_end_drift_only_active_near_quarter_end():
    """No more than n_days per quarter should be in the window."""
    prices = _make_prices_dated(300, start="1995-01-03")
    pos = QuarterEndDrift(n_days=5).generate_positions(prices)
    # Count active days per quarter
    from quant.strategies.base import _calendar_quarter_rank_end

    rank_end = _calendar_quarter_rank_end(prices.index)
    # Active days = rank_end < 5; should be exactly 5 per quarter for full quarters
    quarters = prices.index.to_period("Q")
    for q in quarters.unique():
        q_active = pos[quarters == q].sum()
        assert q_active <= 5


# ---------------------------------------------------------------------------
# H027 — SeasonalityTrendCombo
# ---------------------------------------------------------------------------


def test_seasonality_trend_combo_no_lookahead():
    prices = _make_prices([0.001] * 700, start="1995-01-03")
    assert_no_lookahead(SeasonalityTrendCombo().generate_positions, prices)


def test_seasonality_trend_combo_long_in_uptrend_favorable():
    """Uptrend in November (favorable, post-warmup) → long."""
    # 800 bars from 1995-01-03 ≈ 3.2 years; MA(200) warms up ~Oct 1995
    prices = _make_prices([0.001] * 800, start="1995-01-03")
    pos = SeasonalityTrendCombo().generate_positions(prices)
    # Check any November after 1996 (well past 200-bar warmup)
    nov_dates = pos.index[(pos.index.month == 11) & (pos.index.year >= 1996)]
    assert len(nov_dates) > 0
    assert np.allclose(pos[nov_dates].values, 1.0)


def test_seasonality_trend_combo_flat_in_uptrend_unfavorable():
    """Uptrend in July (unfavorable month) → flat despite trend signal."""
    prices = _make_prices([0.001] * 800, start="1995-01-03")
    pos = SeasonalityTrendCombo().generate_positions(prices)
    jul_dates = pos.index[(pos.index.month == 7) & (pos.index.year >= 1996)]
    assert len(jul_dates) > 0
    assert np.allclose(pos[jul_dates].values, 0.0)


def test_seasonality_trend_combo_flat_in_downtrend_favorable():
    """Downtrend in December (favorable) → flat because trend is off."""
    prices = _make_prices([-0.001] * 800, start="1995-01-03")
    pos = SeasonalityTrendCombo().generate_positions(prices)
    dec_dates = pos.index[(pos.index.month == 12) & (pos.index.year >= 1996)]
    assert len(dec_dates) > 0
    assert np.allclose(pos[dec_dates].values, 0.0)


# ---------------------------------------------------------------------------
# H028 — VIXSpikeEntry
# ---------------------------------------------------------------------------


def _make_vix_prices(spy_rets, vix_levels, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    px = [100.0]
    for r in spy_rets:
        px.append(px[-1] * (1 + r))
    df = pd.DataFrame(
        {
            "adj_close": px,
            "open": px,
            "high": px,
            "low": px,
            "close": px,
            "volume": [1_000_000] * (n + 1),
        },
        index=idx,
    )
    assert len(vix_levels) == n + 1
    df["vix_close"] = vix_levels
    return df


def test_vix_spike_entry_no_lookahead():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0005, 0.01, 700).tolist()
    vix = (20 + rng.normal(0, 5, 701)).clip(10, 80).tolist()
    prices = _make_vix_prices(spy, vix)
    assert_no_lookahead(VIXSpikeEntry().generate_positions, prices)


def test_vix_spike_entry_active_after_spike():
    """Position should be 1 in the days immediately following a VIX spike."""
    spy = [0.001] * 100
    vix = [15.0] * 50 + [35.0] + [15.0] * 50  # spike at index 50
    prices = _make_vix_prices(spy, vix)
    pos = VIXSpikeEntry(vix_threshold=30.0, hold_days=20).generate_positions(prices)
    # After spike (shifted by 1), should be active for 20 bars
    assert pos.iloc[52] == 1.0


def test_vix_spike_entry_flat_before_spike():
    """Position should be 0 when VIX has never spiked."""
    spy = [0.001] * 100
    vix = [15.0] * 101
    prices = _make_vix_prices(spy, vix)
    pos = VIXSpikeEntry(vix_threshold=30.0).generate_positions(prices)
    assert (pos == 0.0).all()


def test_vix_spike_entry_binary():
    rng = np.random.default_rng(1)
    spy = rng.normal(0, 0.01, 200).tolist()
    vix = (20 + rng.normal(0, 8, 201)).clip(10, 80).tolist()
    prices = _make_vix_prices(spy, vix)
    pos = VIXSpikeEntry().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# H029 — VIXRegimeFilter
# ---------------------------------------------------------------------------


def test_vix_regime_filter_no_lookahead():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0005, 0.01, 700).tolist()
    vix = (18 + rng.normal(0, 4, 701)).clip(10, 50).tolist()
    prices = _make_vix_prices(spy, vix)
    assert_no_lookahead(VIXRegimeFilter().generate_positions, prices)


def test_vix_regime_filter_long_when_low():
    """VIX below threshold → position 1.0."""
    spy = [0.001] * 100
    vix = [15.0] * 101
    prices = _make_vix_prices(spy, vix)
    pos = VIXRegimeFilter(vix_threshold=20.0).generate_positions(prices)
    assert (pos.iloc[1:] == 1.0).all()


def test_vix_regime_filter_flat_when_high():
    """VIX above threshold → position 0.0."""
    spy = [0.001] * 100
    vix = [30.0] * 101
    prices = _make_vix_prices(spy, vix)
    pos = VIXRegimeFilter(vix_threshold=20.0).generate_positions(prices)
    assert (pos == 0.0).all()


def test_vix_regime_filter_binary():
    rng = np.random.default_rng(2)
    spy = rng.normal(0, 0.01, 200).tolist()
    vix = (20 + rng.normal(0, 5, 201)).clip(10, 50).tolist()
    prices = _make_vix_prices(spy, vix)
    pos = VIXRegimeFilter().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# H030 — VIXTrendSignal
# ---------------------------------------------------------------------------


def test_vix_trend_signal_no_lookahead():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0005, 0.01, 700).tolist()
    vix = (18 + rng.normal(0, 4, 701)).clip(10, 60).tolist()
    prices = _make_vix_prices(spy, vix)
    assert_no_lookahead(VIXTrendSignal().generate_positions, prices)


def test_vix_trend_signal_long_when_vix_below_ma():
    """VIX below its MA → declining fear → long (+1)."""
    spy = [0.001] * 300
    # VIX trends down: starts at 25, gradually falls to 10
    vix = [25 - 0.05 * i for i in range(301)]
    prices = _make_vix_prices(spy, vix)
    pos = VIXTrendSignal(ma_period=20).generate_positions(prices)
    # Well after warmup (bar 50+), VIX declining → long
    assert pos.iloc[100] == pytest.approx(1.0)


def test_vix_trend_signal_short_when_vix_above_ma():
    """VIX above its MA → rising fear → short (−1)."""
    spy = [0.001] * 300
    # VIX trends up: starts at 10, gradually rises to 40
    vix = [10 + 0.1 * i for i in range(301)]
    prices = _make_vix_prices(spy, vix)
    pos = VIXTrendSignal(ma_period=20).generate_positions(prices)
    assert pos.iloc[100] == pytest.approx(-1.0)


def test_vix_trend_signal_valid_range():
    """Positions after warmup should be only +1 or -1."""
    rng = np.random.default_rng(3)
    spy = rng.normal(0, 0.01, 300).tolist()
    vix = (20 + rng.normal(0, 5, 301)).clip(10, 60).tolist()
    prices = _make_vix_prices(spy, vix)
    pos = VIXTrendSignal(ma_period=20).generate_positions(prices)
    valid = pos.iloc[21:]
    assert set(valid.values.round(6)).issubset({-1.0, 1.0})


# ---------------------------------------------------------------------------
# H031 — RealizedVsImpliedVol
# ---------------------------------------------------------------------------


def test_realized_vs_implied_no_lookahead():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0005, 0.01, 700).tolist()
    vix = (18 + rng.normal(0, 4, 701)).clip(10, 60).tolist()
    prices = _make_vix_prices(spy, vix)
    assert_no_lookahead(RealizedVsImpliedVol().generate_positions, prices)


def test_realized_vs_implied_long_when_vix_higher():
    """When VIX > realized vol, position should be 1."""
    spy = [0.0] * 200  # zero returns → realized vol ≈ 0
    vix = [20.0] * 201  # VIX >> realized
    prices = _make_vix_prices(spy, vix)
    pos = RealizedVsImpliedVol(realized_window=20).generate_positions(prices)
    assert (pos.iloc[22:] == 1.0).all()


def test_realized_vs_implied_binary():
    rng = np.random.default_rng(4)
    spy = rng.normal(0, 0.01, 300).tolist()
    vix = (20 + rng.normal(0, 4, 301)).clip(10, 60).tolist()
    prices = _make_vix_prices(spy, vix)
    pos = RealizedVsImpliedVol().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# H032 — VIXMeanReversion
# ---------------------------------------------------------------------------


def test_vix_mean_reversion_no_lookahead():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0005, 0.01, 700).tolist()
    vix = (18 + rng.normal(0, 4, 701)).clip(10, 60).tolist()
    prices = _make_vix_prices(spy, vix)
    assert_no_lookahead(VIXMeanReversion().generate_positions, prices)


def test_vix_mean_reversion_active_on_extreme_vix():
    """Extreme VIX spike (> 2σ above mean) should activate position."""
    spy = [0.001] * 200
    # Low VIX for warmup, then one extreme spike
    vix = [15.0] * 160 + [60.0] + [15.0] * 40
    prices = _make_vix_prices(spy, vix)
    pos = VIXMeanReversion(lookback=60, n_sigma=2.0).generate_positions(prices)
    # Day after the spike (shifted by 1), should be active
    assert pos.iloc[161] == 1.0


def test_vix_mean_reversion_flat_when_normal():
    """Normal VIX conditions → position 0."""
    spy = [0.001] * 200
    vix = [18.0] * 201
    prices = _make_vix_prices(spy, vix)
    pos = VIXMeanReversion(lookback=60, n_sigma=2.0).generate_positions(prices)
    assert (pos.iloc[62:] == 0.0).all()


def test_vix_mean_reversion_binary():
    rng = np.random.default_rng(5)
    spy = rng.normal(0, 0.01, 300).tolist()
    vix = (20 + rng.normal(0, 5, 301)).clip(10, 60).tolist()
    prices = _make_vix_prices(spy, vix)
    pos = VIXMeanReversion().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})
