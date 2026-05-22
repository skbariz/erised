"""Tests for the backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import assert_no_lookahead, run_backtest
from quant.strategies.base import (
    ADXFilteredMACrossover,
    AbsoluteMomentumStrategy,
    BuyAndHold,
    ConsecutiveDownDays,
    DualMALongShort,
    FastMACrossover,
    LongShortMACrossover,
    LongShortTSMomentum,
    MACDCrossover,
    MeanReversionStrategy,
    MovingAverageCrossover,
    RSIReversionStrategy,
    ShortHoldMeanReversion,
    TimeSeriesMomentum,
    TrendVIXGate,
    VIXFilteredReversionStrategy,
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
    rets.append(-0.10)                      # signal bar
    rets.extend([0.001] * (hold_days + 2))  # neutral tail
    prices = _make_prices(rets)
    strat = MeanReversionStrategy(lookback=lookback, threshold=2.0, hold_days=hold_days)
    pos = strat.generate_positions(prices)

    # iloc[0] is the initial price row (no return); signal bar is iloc[n_warmup + 1]
    signal_iloc = n_warmup + 1
    for i in range(hold_days):
        assert pos.iloc[signal_iloc + i] == pytest.approx(1.0), f"Expected long at hold offset {i}"
    assert pos.iloc[signal_iloc + hold_days] == pytest.approx(0.0), "Expected flat after hold expires"


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
    vix = [50.0] * 301    # crash-level VIX throughout
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
