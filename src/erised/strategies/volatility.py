"""Volatility-regime, VIX, variance-risk-premium, volatility-targeting, and drawdown-control strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import _h021_base_weights, _h021_weights, _vrp_series
from .base import MultiAssetStrategy, Strategy


class TrendVIXGate(Strategy):
    """H017: MA(50,200) long/short, forced flat when VIX exceeds vix_extreme.

    Hypothesis: in crash conditions (VIX > 40) trend signals are whipsawed by
    panic volatility. Sitting flat avoids the worst drawdown periods while
    keeping the long/short signal active in normal regimes.
    Requires a 'vix_close' column in the prices DataFrame.
    """

    def __init__(self, fast: int = 50, slow: int = 200, vix_extreme: float = 40.0):
        self.fast = fast
        self.slow = slow
        self.vix_extreme = vix_extreme
        self.name = f"trend_vix_gate_{fast}_{slow}_{int(vix_extreme)}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("TrendVIXGate requires 'vix_close' column")

        px = prices["adj_close"]
        fast_ma = px.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = px.rolling(self.slow, min_periods=self.slow).mean()

        pos = pd.Series(np.where(fast_ma > slow_ma, 1.0, -1.0), index=px.index)
        pos = pos.where(slow_ma.notna(), 0.0)

        # Force flat when VIX signals crash territory
        in_crash = prices["vix_close"] > self.vix_extreme
        pos = pos.where(~in_crash, 0.0)

        pos.name = "position"
        return pos


class VolScaledTrend(Strategy):
    """H016: MA trend direction scaled by volatility targeting.

    Position = sign(MA signal) × min(target_vol / 20d_realized_vol, 1.0).
    Sizes up in calm markets and down in turbulent ones, converting H001's
    binary 0/+1 into a continuous signal. Tests whether dynamic position sizing
    improves risk-adjusted returns over constant full-notional exposure.
    """

    def __init__(
        self,
        fast: int = 50,
        slow: int = 200,
        target_vol: float = 0.10,
        vol_lookback: int = 20,
    ):
        self.fast = fast
        self.slow = slow
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.name = f"vol_scaled_trend_{fast}_{slow}_{int(target_vol * 100)}pct"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        r = px.pct_change()

        fast_ma = px.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = px.rolling(self.slow, min_periods=self.slow).mean()

        direction = pd.Series(np.where(fast_ma > slow_ma, 1.0, -1.0), index=px.index)
        direction = direction.where(slow_ma.notna(), 0.0)

        # Shift realized vol by 1 bar so today's close is not used in today's signal
        realized_vol = r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        scale = (self.target_vol / realized_vol).clip(upper=1.0)
        scale = scale.where(realized_vol.notna() & (realized_vol > 0), 0.0)

        pos = (direction * scale).clip(-1.0, 1.0)
        pos = pos.where(slow_ma.notna(), 0.0)
        pos.name = "position"
        return pos


class VIXFilteredReversionStrategy(Strategy):
    """Hypothesis 006: mean reversion only in elevated-VIX regimes.

    Same z-score entry as MeanReversionStrategy but gated by VIX > vix_threshold.
    Hypothesis: reversions succeed in genuine panic; not in low-vol drift.
    Requires a 'vix_close' column in the prices DataFrame.
    """

    def __init__(
        self,
        lookback: int = 20,
        threshold: float = 2.0,
        hold_days: int = 2,
        vix_threshold: float = 20.0,
    ):
        self.lookback = lookback
        self.threshold = threshold
        self.hold_days = hold_days
        self.vix_threshold = vix_threshold
        self.name = f"vix_rev_{lookback}_{threshold}_{hold_days}_{int(vix_threshold)}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VIXFilteredReversionStrategy requires 'vix_close' column")

        r = prices["adj_close"].pct_change()
        rolling_mean = r.rolling(self.lookback, min_periods=self.lookback).mean().shift(1)
        rolling_std = r.rolling(self.lookback, min_periods=self.lookback).std(ddof=1).shift(1)
        z = (r - rolling_mean) / rolling_std

        vix_elevated = prices["vix_close"] > self.vix_threshold

        entry = ((z < -self.threshold) & vix_elevated).astype(float)
        entry = entry.where(rolling_std.notna() & (rolling_std > 0), 0.0)

        pos = entry.rolling(self.hold_days, min_periods=1).max()
        pos.name = "position"
        return pos


class VolatilityRegime(Strategy):
    """Hypothesis 012: long in low realized-volatility regimes.

    Long when 20-day realized vol is below its rolling historical median
    (using only past vol data). Flat in turbulent regimes. Tests the
    low-volatility anomaly: calm markets produce better risk-adjusted returns.
    """

    def __init__(self, vol_lookback: int = 20, regime_lookback: int = 252):
        self.vol_lookback = vol_lookback
        self.regime_lookback = regime_lookback
        self.name = f"vol_regime_{vol_lookback}_{regime_lookback}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        r = prices["adj_close"].pct_change()
        realized_vol = r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std()

        # Rolling median of PAST vols (shift by 1 so today's vol isn't in its own baseline)
        rolling_median = (
            realized_vol.shift(1)
            .rolling(self.regime_lookback, min_periods=self.regime_lookback // 2)
            .median()
        )

        pos = (realized_vol < rolling_median).astype(float)
        pos = pos.where(rolling_median.notna(), 0.0)
        pos.name = "position"
        return pos


class VIXSpikeEntry(Strategy):
    """H028: Contrarian long after VIX spikes above a threshold.

    When VIX crosses above vix_threshold, enter long SPY and hold for
    hold_days trading days. Fear spikes historically mark local bottoms —
    this captures the mean-reversion bounce in equities.

    Requires a 'vix_close' column in the prices DataFrame.
    """

    def __init__(self, vix_threshold: float = 30.0, hold_days: int = 20):
        self.vix_threshold = vix_threshold
        self.hold_days = hold_days
        self.name = f"vix_spike_{int(vix_threshold)}_{hold_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VIXSpikeEntry requires 'vix_close' column")
        vix = prices["vix_close"].shift(1)
        spike = (vix > self.vix_threshold).astype(float)
        # Active during hold_days window after a spike (rolling max of spike flag)
        pos = spike.rolling(self.hold_days, min_periods=1).max()
        pos.name = "position"
        return pos


class VIXRegimeFilter(Strategy):
    """H029: Long SPY only when VIX is below threshold (low-fear regime).

    VIX below vix_threshold signals a calm, risk-on environment historically
    associated with positive equity returns and lower drawdown risk.

    Requires a 'vix_close' column in the prices DataFrame.
    """

    def __init__(self, vix_threshold: float = 20.0):
        self.vix_threshold = vix_threshold
        self.name = f"vix_regime_{int(vix_threshold)}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VIXRegimeFilter requires 'vix_close' column")
        vix = prices["vix_close"].shift(1)
        pos = (vix < self.vix_threshold).astype(float)
        pos.name = "position"
        return pos


class VIXTrendSignal(Strategy):
    """H030: Trade equity direction via VIX trend (rising fear → short).

    When VIX is above its rolling MA (trending up), short SPY: rising fear
    implies falling equity prices. When VIX is below its MA (trending down),
    long SPY. Uses 1-bar lag to avoid lookahead.

    Requires a 'vix_close' column in the prices DataFrame.
    """

    def __init__(self, ma_period: int = 20):
        self.ma_period = ma_period
        self.name = f"vix_trend_{ma_period}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VIXTrendSignal requires 'vix_close' column")
        vix = prices["vix_close"]
        vix_ma = vix.rolling(self.ma_period, min_periods=self.ma_period).mean()
        valid = vix_ma.notna()
        pos = pd.Series(0.0, index=prices.index, name="position")
        pos = pos.where(~(valid & (vix.shift(1) < vix_ma.shift(1))), 1.0)
        pos = pos.where(~(valid & (vix.shift(1) >= vix_ma.shift(1))), -1.0)
        return pos


class RealizedVsImpliedVol(Strategy):
    """H031: Long when implied vol (VIX) exceeds realized vol (vol risk premium).

    The volatility risk premium (VRP = VIX − realized_vol) is historically
    positive on average. Long SPY in VRP-positive regimes (implied > realized),
    flat otherwise. Normalized vol: both in annualized percentage terms.

    Requires a 'vix_close' column in the prices DataFrame.
    """

    def __init__(self, realized_window: int = 20):
        self.realized_window = realized_window
        self.name = f"vrp_{realized_window}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("RealizedVsImpliedVol requires 'vix_close' column")
        rets = prices["adj_close"].pct_change()
        realized_vol = (
            rets.rolling(self.realized_window, min_periods=self.realized_window).std().shift(1)
            * np.sqrt(252)
            * 100
        )
        vix = prices["vix_close"].shift(1)
        valid = realized_vol.notna() & vix.notna()
        pos = pd.Series(0.0, index=prices.index, name="position")
        pos[valid & (vix > realized_vol)] = 1.0
        return pos


class VIXMeanReversion(Strategy):
    """H032: Contrarian long when VIX is at a statistically extreme high.

    Computes a rolling z-score of VIX relative to its own history. A high
    z-score (> n_sigma) signals extreme fear that tends to revert — the
    reversion coincides with equity rallies. Unlike H028 (absolute threshold),
    this normalizes by the current volatility regime.

    Requires a 'vix_close' column in the prices DataFrame.
    """

    def __init__(self, lookback: int = 60, n_sigma: float = 2.0):
        self.lookback = lookback
        self.n_sigma = n_sigma
        self.name = f"vix_mr_{lookback}d_{n_sigma}s"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VIXMeanReversion requires 'vix_close' column")
        vix = prices["vix_close"]
        vix_ma = vix.rolling(self.lookback, min_periods=self.lookback).mean().shift(1)
        vix_std = vix.rolling(self.lookback, min_periods=self.lookback).std().shift(1)
        vix_z = (vix.shift(1) - vix_ma) / vix_std.replace(0, np.nan)
        pos = (vix_z > self.n_sigma).astype(float)
        pos.name = "position"
        return pos


class VolOfVolCalm(Strategy):
    """H042: Long when VIX's own rolling std dev is in the bottom tercile.

    Second-order VIX signal: stable fear (low vol-of-vol) indicates a calm,
    predictable risk environment, historically associated with steady equity
    gains. Spikes in VIX's own volatility often precede drawdowns before the
    VIX level itself rises — this signal turns off earlier than H001 in
    deteriorating regimes. Targeting H001 corr 0.10–0.30.

    vov_window  : rolling window for VIX std dev (default 20 days).
    history_min : min observations to estimate the percentile (default 252).
    percentile  : upper percentile of calm VIX-vol to accept (default 33).
    Requires 'vix_close' column in the prices DataFrame.
    """

    def __init__(
        self,
        vov_window: int = 20,
        history_min: int = 252,
        percentile: float = 33.0,
    ):
        self.vov_window = vov_window
        self.history_min = history_min
        self.percentile = percentile
        self.name = f"vol_of_vol_calm_{vov_window}d_p{int(percentile)}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VolOfVolCalm requires 'vix_close' column")

        vix = prices["vix_close"]
        # Rolling std of VIX (vol-of-vol), shifted to avoid lookahead
        vov = vix.rolling(self.vov_window, min_periods=self.vov_window).std().shift(1)

        # Expanding percentile: is current VoV in the calmest bottom tercile?
        upper_bound = vov.expanding(min_periods=self.history_min).quantile(self.percentile / 100.0)
        valid = vov.notna() & upper_bound.notna()
        pos = (vov <= upper_bound).astype(float)
        pos = pos.where(valid, 0.0)
        pos.name = "position"
        return pos


class DefensiveSectorOverlay(MultiAssetStrategy):
    """H063: SPY/XLP rotation based on SPY trend + VIX stress level.

    SPY above MA + VIX < vix_thresh → 100% SPY.
    SPY above MA + VIX >= vix_thresh → 50% SPY + 50% XLP.
    SPY below MA → 100% XLP.
    Before VIX/XLP data: 100% SPY.
    """

    name = "defensive_sector_overlay"

    def __init__(self, spy_ma: int = 200, vix_thresh: float = 25.0) -> None:
        self.spy_ma = spy_ma
        self.vix_thresh = vix_thresh

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        vix = prices.get("vix_close", pd.Series(np.nan, index=prices.index))
        xlp = prices.get("xlp_close", pd.Series(np.nan, index=prices.index))

        spy_ma = prices["adj_close"].shift(1).rolling(self.spy_ma).mean()
        spy_up = prices["adj_close"].shift(1) > spy_ma

        vix_calm = vix.shift(1) < self.vix_thresh
        has_vix = vix.shift(1).notna()
        has_xlp = xlp.notna()

        w_spy = pd.Series(1.0, index=prices.index)
        w_xlp = pd.Series(0.0, index=prices.index)

        # SPY uptrend + VIX calm → 100% SPY (default, no change)
        # SPY uptrend + VIX stressed → 50/50
        stress = spy_ma.notna() & spy_up & has_vix & ~vix_calm & has_xlp
        w_spy[stress] = 0.5
        w_xlp[stress] = 0.5

        # SPY downtrend → 100% XLP
        bearish = spy_ma.notna() & ~spy_up & has_xlp
        w_spy[bearish] = 0.0
        w_xlp[bearish] = 1.0

        return pd.DataFrame({"SPY": w_spy, "XLP": w_xlp}, index=prices.index)


class VixRiskParityEnhanced(MultiAssetStrategy):
    """H066: H021 RiskParity with VIX spike override.

    Normal: inverse-vol weights for SPY and TLT (identical to H021).
    Crisis: when VIX > vix_spike, override to 80% TLT / 20% SPY.
    Recovery: resume normal risk parity when VIX < vix_resume (hysteresis).
    """

    name = "vix_risk_parity_enhanced"

    def __init__(
        self,
        vol_lookback: int = 60,
        vix_spike: float = 35.0,
        vix_resume: float = 30.0,
        crisis_tlt: float = 0.80,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vix_spike = vix_spike
        self.vix_resume = vix_resume
        self.crisis_tlt = crisis_tlt

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy_rp = (inv_spy / total).fillna(0.0)
        w_tlt_rp = (inv_tlt / total).fillna(0.0)

        vix = prices["vix_close"].shift(1)
        w_spy = w_spy_rp.copy()
        w_tlt = w_tlt_rp.copy()

        # Hysteresis loop: crisis when VIX > spike, resume when VIX < resume
        in_crisis = False
        for i in range(len(prices)):
            v = vix.iloc[i]
            if np.isnan(v):
                continue
            if not in_crisis and v > self.vix_spike:
                in_crisis = True
            elif in_crisis and v < self.vix_resume:
                in_crisis = False
            if in_crisis:
                w_spy.iloc[i] = 1.0 - self.crisis_tlt
                w_tlt.iloc[i] = self.crisis_tlt

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class VixPercentileRegime(Strategy):
    """H067: Expanding-quantile adaptive VIX threshold.

    Long SPY when the current VIX is below its expanding (33rd percentile)
    percentile of all historical VIX readings to date. Adapts to the
    prevailing volatility regime unlike H029's fixed 20 threshold.
    """

    def __init__(self, quantile: float = 0.33) -> None:
        self.quantile = quantile
        self.name = f"vix_percentile_{int(quantile * 100)}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VixPercentileRegime requires 'vix_close' column")
        vix = prices["vix_close"]
        # Expanding quantile of past VIX values; shift 1 to avoid lookahead
        vix_thresh = vix.expanding(min_periods=63).quantile(self.quantile).shift(1)
        pos = pd.Series(0.0, index=prices.index, name="position")
        valid = vix_thresh.notna()
        pos[valid & (vix.shift(1) < vix_thresh)] = 1.0
        return pos


class VrpCalendarHybrid(Strategy):
    """H068: Volatility Risk Premium (H031) gated by Nov-Apr calendar (H043).

    Long SPY only when BOTH the VRP signal fires (VIX > realized vol) AND
    the date falls in the seasonally favorable Nov-Apr window. The calendar
    gate forces the strategy flat May-Oct, reducing H001 correlation.
    """

    def __init__(
        self,
        realized_window: int = 20,
        favorable_months: tuple = (11, 12, 1, 2, 3, 4),
    ) -> None:
        self.realized_window = realized_window
        self.favorable_months = frozenset(favorable_months)
        self.name = f"vrp_calendar_{realized_window}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VrpCalendarHybrid requires 'vix_close' column")
        rets = prices["adj_close"].pct_change()
        realized_vol = (
            rets.rolling(self.realized_window, min_periods=self.realized_window).std().shift(1)
            * np.sqrt(252)
            * 100
        )
        vix = prices["vix_close"].shift(1)
        in_season = pd.Series(
            prices.index.month.isin(self.favorable_months).astype(float), index=prices.index
        )
        vrp_signal = (vix > realized_vol) & realized_vol.notna() & vix.notna()
        pos = pd.Series(0.0, index=prices.index, name="position")
        pos[vrp_signal & (in_season == 1.0)] = 1.0
        return pos


class VixTermStructureProxy(Strategy):
    """H069: VIX term structure proxy via VIX/VIX_63d_MA ratio.

    When VIX is depressed vs. its 63d average (ratio < 0.85), the term
    structure is contango-like (calm) → long SPY.
    When VIX is elevated vs. its average (ratio > 1.15), the term structure
    resembles backwardation (crisis) → flat.
    Between thresholds, hold the prior position (hysteresis).
    """

    def __init__(
        self,
        ma_days: int = 63,
        contango_thresh: float = 0.85,
        backwardation_thresh: float = 1.15,
    ) -> None:
        self.ma_days = ma_days
        self.contango_thresh = contango_thresh
        self.backwardation_thresh = backwardation_thresh
        self.name = (
            f"vix_term_{ma_days}d_{int(contango_thresh * 100)}_{int(backwardation_thresh * 100)}"
        )

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "vix_close" not in prices.columns:
            raise ValueError("VixTermStructureProxy requires 'vix_close' column")
        vix = prices["vix_close"]
        vix_ma = vix.rolling(self.ma_days, min_periods=self.ma_days).mean().shift(1)
        vix_lagged = vix.shift(1)
        ratio = vix_lagged / vix_ma.replace(0, np.nan)

        pos_arr = np.zeros(len(prices))
        current = 0.0
        for i in range(len(prices)):
            r = ratio.iloc[i]
            if np.isnan(r):
                pos_arr[i] = 0.0
                continue
            if r < self.contango_thresh:
                current = 1.0
            elif r > self.backwardation_thresh:
                current = 0.0
            # else: hold prior
            pos_arr[i] = current

        return pd.Series(pos_arr, index=prices.index, name="position")


class MultiAssetDynamicVolTarget(MultiAssetStrategy):
    """H070: H021 risk-parity allocation scaled by portfolio-level vol targeting.

    Two-layer risk management: H021 sets the SPY/TLT mix via inverse-vol
    weights (asset-level risk parity). A second layer then scales the TOTAL
    portfolio exposure down when 20d realized portfolio vol exceeds vol_target.
    Exposure is capped at 1.0 (no leverage).
    """

    name = "multi_asset_dynamic_vol_target"

    def __init__(
        self,
        vol_lookback: int = 60,
        port_vol_lookback: int = 20,
        vol_target: float = 0.10,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.port_vol_lookback = port_vol_lookback
        self.vol_target = vol_target

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        # H021-style inverse-vol weights
        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy = (inv_spy / total).fillna(0.0)
        w_tlt = (inv_tlt / total).fillna(0.0)

        # Portfolio realized returns using prior-day weights (shift 1 to avoid lookahead)
        port_r = w_spy.shift(1) * spy_r + w_tlt.shift(1) * tlt_r
        port_vol = port_r.rolling(
            self.port_vol_lookback, min_periods=self.port_vol_lookback
        ).std().shift(1) * np.sqrt(252)

        # Scale total exposure; cap at 1.0
        scale = (self.vol_target / port_vol).clip(upper=1.0)
        scale = scale.where(port_vol.notna() & (port_vol > 0), 1.0)

        w_spy_scaled = w_spy * scale
        w_tlt_scaled = w_tlt * scale

        return pd.DataFrame({"SPY": w_spy_scaled, "TLT": w_tlt_scaled}, index=prices.index)


class LowVolTargetPortfolio(MultiAssetStrategy):
    """H086: H021 inverse-vol allocation with stricter 8% portfolio vol target.

    Identical mechanism to H070 (MultiAssetDynamicVolTarget) but defaults to
    vol_target=8% instead of 10%. Lower target means earlier de-risking and
    smaller positions in high-vol regimes, trading CAGR for Max DD improvement.
    """

    name = "low_vol_target_portfolio"

    def __init__(
        self,
        vol_lookback: int = 60,
        port_vol_lookback: int = 20,
        vol_target: float = 0.08,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.port_vol_lookback = port_vol_lookback
        self.vol_target = vol_target

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy = (inv_spy / total).fillna(0.0)
        w_tlt = (inv_tlt / total).fillna(0.0)

        port_r = w_spy.shift(1) * spy_r + w_tlt.shift(1) * tlt_r
        port_vol = port_r.rolling(
            self.port_vol_lookback, min_periods=self.port_vol_lookback
        ).std().shift(1) * np.sqrt(252)
        scale = (self.vol_target / port_vol).clip(upper=1.0)
        scale = scale.where(port_vol.notna() & (port_vol > 0), 1.0)

        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class RatchetDrawdownProtection(MultiAssetStrategy):
    """H087: H021 weights scaled proportionally by portfolio drawdown from peak.

    scale = max(0, 1 − portfolio_drawdown / drawdown_limit)
    At 0% drawdown: full H021 allocation (scale=1.0).
    At drawdown_limit: flat (scale=0.0).
    Uses H021 unscaled portfolio returns to track the drawdown (no circularity).
    """

    name = "ratchet_drawdown_protection"

    def __init__(self, vol_lookback: int = 60, drawdown_limit: float = 0.15) -> None:
        self.vol_lookback = vol_lookback
        self.drawdown_limit = drawdown_limit

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy = (inv_spy / total).fillna(0.0)
        w_tlt = (inv_tlt / total).fillna(0.0)

        # Simulate H021 portfolio cumulative return (using lagged weights → no lookahead)
        port_r = w_spy.shift(1) * spy_r + w_tlt.shift(1) * tlt_r
        port_cum = (1.0 + port_r.fillna(0.0)).cumprod()
        rolling_peak = port_cum.expanding(min_periods=1).max()
        dd = (port_cum / rolling_peak - 1.0).fillna(0.0)

        # Ratchet scale based on yesterday's drawdown
        scale = (1.0 + dd.shift(1).fillna(0.0) / self.drawdown_limit).clip(lower=0.0, upper=1.0)

        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class CorrelationRiskParityHybrid(MultiAssetStrategy):
    """H097: H034 correlation regime overlaid on H021 inverse-vol weights.

    Normal: H021 inverse-vol weights.
    Crisis: when rolling 60d SPY-TLT correlation turns positive → override to
            crisis_spy / crisis_tlt split (default 10% / 90%).
    """

    name = "correlation_risk_parity_hybrid"

    def __init__(
        self,
        vol_lookback: int = 60,
        corr_lookback: int = 60,
        crisis_tlt: float = 0.90,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.corr_lookback = corr_lookback
        self.crisis_tlt = crisis_tlt

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback).std()
        tlt_vol = tlt_r.rolling(self.vol_lookback).std()

        inv_spy = spy_vol.replace(0, np.nan).rdiv(1)
        inv_tlt = tlt_vol.replace(0, np.nan).rdiv(1)
        total = inv_spy + inv_tlt

        w_spy_h021 = (inv_spy / total).fillna(0.5)
        w_tlt_h021 = (inv_tlt / total).fillna(0.5)

        # H034 signal: positive rolling correlation = crisis contagion
        corr = spy_r.rolling(self.corr_lookback).corr(tlt_r).shift(1)
        crisis = corr > 0

        tlt_ok = prices["tlt_close"].notna()
        crisis = crisis & tlt_ok

        w_spy = w_spy_h021.where(~crisis, 1.0 - self.crisis_tlt)
        w_tlt = w_tlt_h021.where(~crisis, self.crisis_tlt)

        w_spy = w_spy.where(tlt_ok, 0.0)
        w_tlt = w_tlt.where(tlt_ok, 0.0)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class ReversionEntryRiskParity(MultiAssetStrategy):
    """H098: H021 with contrarian heavy-TLT entry on SPY drawdown events.

    Normal: H021 inverse-vol weights.
    Entry: SPY drawdown from peak exceeds dd_trigger →
           shift to crisis_tlt/crisis_spy for up to crisis_days bars.
    Exit early: SPY recovers recovery% from crisis trough.
    """

    name = "reversion_entry_risk_parity"

    def __init__(
        self,
        vol_lookback: int = 60,
        dd_trigger: float = 0.10,
        crisis_tlt: float = 0.80,
        crisis_days: int = 20,
        recovery: float = 0.05,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.dd_trigger = dd_trigger
        self.crisis_tlt = crisis_tlt
        self.crisis_days = crisis_days
        self.recovery = recovery

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy = prices["adj_close"]
        tlt = prices["tlt_close"]
        spy_r = spy.pct_change()
        tlt_r = tlt.pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback).std()
        tlt_vol = tlt_r.rolling(self.vol_lookback).std()
        inv_spy = spy_vol.replace(0, np.nan).rdiv(1)
        inv_tlt = tlt_vol.replace(0, np.nan).rdiv(1)
        total = inv_spy + inv_tlt
        w_spy_h021 = (inv_spy / total).fillna(0.5)
        w_tlt_h021 = (inv_tlt / total).fillna(0.5)

        spy_peak = spy.expanding(min_periods=1).max()
        spy_dd = (spy / spy_peak - 1.0).fillna(0.0)

        n = len(prices)
        w_spy_out = pd.Series(0.0, index=prices.index)
        w_tlt_out = pd.Series(0.0, index=prices.index)

        in_crisis = False
        days_left = 0
        crisis_trough = np.inf

        for i in range(n):
            tlt_avail = not pd.isna(tlt.iloc[i])
            dd_lag = spy_dd.iloc[i - 1] if i > 0 else 0.0
            spy_lag = spy.iloc[i - 1] if i > 0 else spy.iloc[i]

            if not tlt_avail:
                in_crisis = False
                days_left = 0
                continue

            if not in_crisis:
                if dd_lag < -self.dd_trigger:
                    in_crisis = True
                    days_left = self.crisis_days
                    crisis_trough = spy_lag
            else:
                if spy_lag < crisis_trough:
                    crisis_trough = spy_lag
                if spy_lag > crisis_trough * (1.0 + self.recovery):
                    in_crisis = False
                    days_left = 0
                else:
                    days_left -= 1
                    if days_left <= 0:
                        in_crisis = False

            if in_crisis:
                w_spy_out.iloc[i] = 1.0 - self.crisis_tlt
                w_tlt_out.iloc[i] = self.crisis_tlt
            else:
                w_spy_out.iloc[i] = w_spy_h021.iloc[i]
                w_tlt_out.iloc[i] = w_tlt_h021.iloc[i]

        return pd.DataFrame({"SPY": w_spy_out, "TLT": w_tlt_out}, index=prices.index)


class SimpleVRPOverlay(MultiAssetStrategy):
    """H301: H021 with VRP z-score bounded exposure scaler.

    VRP = (VIX/100)^2 - realized_var_21d * 252  (both annualized).
    Z-score of VRP over vrp_window bars, clipped to [-2, 2].
    scale = 1.0 + vrp_sensitivity * vrp_z ∈ [0.5, 1.5] for vrp_sensitivity=0.25.
    Positive VRP → increase exposure; negative VRP → reduce.
    """

    name = "simple_vrp_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
        vrp_window: int = 252,
        vrp_sensitivity: float = 0.25,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vrp_window = vrp_window
        self.vrp_sensitivity = vrp_sensitivity

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_mean = vrp.rolling(self.vrp_window, min_periods=63).mean()
        vrp_std = vrp.rolling(self.vrp_window, min_periods=63).std().replace(0, np.nan)
        vrp_z = ((vrp - vrp_mean) / vrp_std).clip(-2.0, 2.0).fillna(0.0)

        scale = (1.0 + self.vrp_sensitivity * vrp_z).clip(0.5, 1.5)
        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class VRPRiskParityScaled(MultiAssetStrategy):
    """H302: H021 with 3-tier VRP percentile exposure multiplier.

    VRP > rolling 70th percentile → high_scale (1.25×).
    VRP < rolling 30th percentile → low_scale (0.75×).
    Middle tier → 1.0× (standard H021).
    """

    name = "vrp_risk_parity_scaled"

    def __init__(
        self,
        vol_lookback: int = 60,
        vrp_window: int = 252,
        high_pctile: float = 0.70,
        low_pctile: float = 0.30,
        high_scale: float = 1.25,
        low_scale: float = 0.75,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vrp_window = vrp_window
        self.high_pctile = high_pctile
        self.low_pctile = low_pctile
        self.high_scale = high_scale
        self.low_scale = low_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vrp = _vrp_series(spy_r, prices["vix_close"])
        hi = vrp.rolling(self.vrp_window, min_periods=63).quantile(self.high_pctile)
        lo = vrp.rolling(self.vrp_window, min_periods=63).quantile(self.low_pctile)

        scale_arr = np.where(vrp > hi, self.high_scale, np.where(vrp < lo, self.low_scale, 1.0))
        scale = pd.Series(np.where(hi.isna() | lo.isna(), 1.0, scale_arr), index=prices.index)
        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class VRPTrendConfirm(MultiAssetStrategy):
    """H303: H021 conditional on VRP + SPY 200d MA dual confirmation.

    Both favorable (VRP > 0 AND SPY > MA): full_scale (1.0).
    Exactly one favorable: partial_scale (0.85).
    Neither favorable: min_scale (0.60).
    """

    name = "vrp_trend_confirm"

    def __init__(
        self,
        vol_lookback: int = 60,
        ma_window: int = 200,
        full_scale: float = 1.0,
        partial_scale: float = 0.85,
        min_scale: float = 0.60,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.ma_window = ma_window
        self.full_scale = full_scale
        self.partial_scale = partial_scale
        self.min_scale = min_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_ok = (vrp > 0).fillna(False)

        spy_ma = prices["adj_close"].rolling(self.ma_window, min_periods=self.ma_window // 2).mean()
        trend_ok = (prices["adj_close"].shift(1) > spy_ma.shift(1)).fillna(False)

        n_ok = vrp_ok.astype(float) + trend_ok.astype(float)
        scale_arr = np.where(
            n_ok >= 2, self.full_scale, np.where(n_ok == 1, self.partial_scale, self.min_scale)
        )
        scale = pd.Series(scale_arr, index=prices.index)
        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class VRPVolTarget(MultiAssetStrategy):
    """H304: H086 vol-target with VRP z-score amplifier.

    vol_scale = min(1, vol_target / port_vol)  [H086 mechanism].
    vrp_factor = clip(1 + vrp_amplifier * vrp_z, 0.8, 1.2).
    final_scale = min(max_scale, vol_scale * vrp_factor).
    """

    name = "vrp_vol_target"

    def __init__(
        self,
        vol_lookback: int = 60,
        port_vol_lookback: int = 20,
        vol_target: float = 0.08,
        vrp_window: int = 252,
        vrp_amplifier: float = 0.15,
        max_scale: float = 1.25,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.port_vol_lookback = port_vol_lookback
        self.vol_target = vol_target
        self.vrp_window = vrp_window
        self.vrp_amplifier = vrp_amplifier
        self.max_scale = max_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        port_r = w_spy.shift(1) * spy_r + w_tlt.shift(1) * tlt_r
        port_vol = port_r.rolling(
            self.port_vol_lookback, min_periods=self.port_vol_lookback
        ).std().shift(1) * np.sqrt(252)
        vol_scale = (self.vol_target / port_vol).clip(upper=1.0)
        vol_scale = vol_scale.where(port_vol.notna() & (port_vol > 0), 1.0)

        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_mean = vrp.rolling(self.vrp_window, min_periods=63).mean()
        vrp_std = vrp.rolling(self.vrp_window, min_periods=63).std().replace(0, np.nan)
        vrp_z = ((vrp - vrp_mean) / vrp_std).clip(-2.0, 2.0).fillna(0.0)

        vrp_factor = (1.0 + self.vrp_amplifier * vrp_z).clip(0.8, 1.2)
        scale = (vol_scale * vrp_factor).clip(upper=self.max_scale)
        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class VRPSeasonalHybrid(MultiAssetStrategy):
    """H305: H078 seasonal framework + VRP overlay in Nov-Apr active window.

    May-Oct: 100% TLT (same as H078).
    Nov-Apr + VRP > 0: H021 at favorable_scale (1.0).
    Nov-Apr + VRP ≤ 0: H021 weights * unfavorable_scale (0.75); remainder cash.
    """

    name = "vrp_seasonal_hybrid"

    def __init__(
        self,
        vol_lookback: int = 60,
        unfavorable_scale: float = 0.75,
        winter_months: tuple = (11, 12, 1, 2, 3, 4),
    ) -> None:
        self.vol_lookback = vol_lookback
        self.unfavorable_scale = unfavorable_scale
        self.winter_months = frozenset(winter_months)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy_rp, w_tlt_rp = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_positive = (vrp > 0).fillna(False)

        in_season = pd.Series(prices.index.month.isin(self.winter_months), index=prices.index)
        tlt_avail = prices["tlt_close"].notna()

        # Start with H021 weights
        w_spy = w_spy_rp.copy()
        w_tlt = w_tlt_rp.copy()

        # May-Oct: 100% TLT (if available) or SPY
        summer = ~in_season
        w_spy = w_spy.where(~summer, 0.0)
        w_tlt = w_tlt.where(~summer, tlt_avail.astype(float))

        # Nov-Apr, VRP unfavorable: scale down
        winter_unfav = in_season & ~vrp_positive
        w_spy = w_spy.where(~winter_unfav, w_spy_rp * self.unfavorable_scale)
        w_tlt = w_tlt.where(~winter_unfav, w_tlt_rp * self.unfavorable_scale)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class LiquiditySpiralProtection(MultiAssetStrategy):
    """H307: H021 with triple-confirm liquidity spiral crisis override.

    Crisis: VIX > vix_spike AND HYG 63d < LQD 63d AND SPY 21d vol > 2× 63d vol.
    Crisis mode: crisis_tlt% TLT / (1-crisis_tlt)% SPY.
    Exit: after resume_days consecutive non-crisis bars.
    Pre-HYG era: 2-of-2 (VIX + vol acceleration) trigger.
    """

    name = "liquidity_spiral_protection"

    def __init__(
        self,
        vol_lookback: int = 60,
        vix_spike: float = 35.0,
        crisis_tlt: float = 0.90,
        resume_days: int = 10,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vix_spike = vix_spike
        self.crisis_tlt = crisis_tlt
        self.resume_days = resume_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vix = prices["vix_close"].shift(1).fillna(0.0)
        spy_vol_21 = spy_r.rolling(21, min_periods=10).std().shift(1)
        spy_vol_63 = spy_r.rolling(63, min_periods=30).std().shift(1)
        vol_spike = (spy_vol_21 > spy_vol_63 * 2.0).fillna(False)

        has_credit = "hyg_close" in prices.columns and "lqd_close" in prices.columns
        if has_credit:
            hyg_ret = prices["hyg_close"].pct_change(63).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(63).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            credit_stress = (hyg_ret < lqd_ret).where(hyg_avail, False).fillna(False)
            crisis_flag = (vix > self.vix_spike) & credit_stress & vol_spike
            # Pre-credit era: use 2-of-2 VIX + vol
            crisis_flag = crisis_flag.where(hyg_avail, (vix > self.vix_spike) & vol_spike)
        else:
            crisis_flag = (vix > self.vix_spike) & vol_spike

        w_spy_out = w_spy.copy()
        w_tlt_out = w_tlt.copy()
        in_crisis = False
        calm_count = 0

        for i in range(len(prices)):
            if crisis_flag.iloc[i]:
                in_crisis = True
                calm_count = 0
            elif in_crisis:
                calm_count += 1
                if calm_count >= self.resume_days:
                    in_crisis = False
                    calm_count = 0
            if in_crisis:
                w_spy_out.iloc[i] = 1.0 - self.crisis_tlt
                w_tlt_out.iloc[i] = self.crisis_tlt

        return pd.DataFrame({"SPY": w_spy_out, "TLT": w_tlt_out}, index=prices.index)


class FundingStressVolTarget(MultiAssetStrategy):
    """H310: H086 vol-target compounded with binary funding stress reduction.

    vol_scale = min(1, vol_target / port_vol)  [H086].
    stress_flag = VIX > 30 AND HYG 63d < LQD 63d (fallback: VIX > 30 only).
    final_scale = vol_scale * (stress_reduction if stress_flag else 1.0).
    """

    name = "funding_stress_vol_target"

    def __init__(
        self,
        vol_lookback: int = 60,
        port_vol_lookback: int = 20,
        vol_target: float = 0.08,
        vix_threshold: float = 30.0,
        stress_reduction: float = 0.65,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.port_vol_lookback = port_vol_lookback
        self.vol_target = vol_target
        self.vix_threshold = vix_threshold
        self.stress_reduction = stress_reduction

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        port_r = w_spy.shift(1) * spy_r + w_tlt.shift(1) * tlt_r
        port_vol = port_r.rolling(
            self.port_vol_lookback, min_periods=self.port_vol_lookback
        ).std().shift(1) * np.sqrt(252)
        vol_scale = (self.vol_target / port_vol).clip(upper=1.0)
        vol_scale = vol_scale.where(port_vol.notna() & (port_vol > 0), 1.0)

        vix = prices["vix_close"].shift(1).fillna(0.0)
        vix_stress = vix > self.vix_threshold

        if "hyg_close" in prices.columns and "lqd_close" in prices.columns:
            hyg_ret = prices["hyg_close"].pct_change(63).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(63).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            credit_off = (hyg_ret < lqd_ret) & hyg_avail & hyg_ret.notna()
            stress_flag = (vix_stress & credit_off).where(hyg_avail, vix_stress)
        else:
            stress_flag = vix_stress

        stress_scale = stress_flag.map({True: self.stress_reduction, False: 1.0})
        scale = vol_scale * stress_scale

        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class RegimeClassifier(MultiAssetStrategy):
    """H318: H021 with 4-state macro regime based on SPY momentum and VIX level.

    States (all signals lagged 1 bar):
      Growth   : SPY 252d annualized > 0 AND VIX < vix_low  -> 1.15x
      Recovery : SPY 252d annualized > 0 AND vix_low <= VIX <= vix_high -> 1.00x
      Stress   : SPY 252d annualized < 0 OR VIX > vix_high  -> 0.70x
      Caution  : all else                                    -> 0.90x
    """

    name = "regime_classifier"

    def __init__(
        self,
        vol_lookback: int = 60,
        vix_low: float = 20.0,
        vix_high: float = 30.0,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vix_low = vix_low
        self.vix_high = vix_high

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        spy_ann = prices["adj_close"].pct_change(252).shift(1)
        vix = prices["vix_close"].shift(1)

        spy_up = spy_ann > 0
        vix_low_mask = vix < self.vix_low
        vix_mid_mask = (vix >= self.vix_low) & (vix <= self.vix_high)
        vix_high_mask = vix > self.vix_high

        growth = spy_up & vix_low_mask
        recovery = spy_up & vix_mid_mask
        stress = (~spy_up) | vix_high_mask
        # caution is the residual (spy_up but vix above mid -- already covered by stress if vix_high)
        # Actually: caution = spy_up AND vix in mid range not already recovery, OR spy_down AND low vix
        # Simpler: use np.select priority order
        scale = pd.Series(
            np.select(
                [stress, growth, recovery],
                [0.70, 1.15, 1.00],
                default=0.90,  # Caution
            ),
            index=prices.index,
        )

        # Fill NaN vix/spy periods with neutral
        no_data = spy_ann.isna() | vix.isna()
        scale = scale.where(~no_data, 1.0)

        return pd.DataFrame(
            {"SPY": (w_spy * scale).clip(upper=1.0), "TLT": (w_tlt * scale).clip(upper=1.0)},
            index=prices.index,
        )


class RiskBudgetExpansion(MultiAssetStrategy):
    """H319: H086-style vol-target with macro-state-dependent target level.

    Favorable (VRP > 0 AND SPY 252d > 0) : vol_target = vol_target_favorable (10%).
    Unfavorable (VRP < 0 OR VIX > 60th rolling percentile) : vol_target = vol_target_unfavorable (6%).
    Baseline : vol_target = vol_target_base (8%).

    scale = min(1.0, vol_target / portfolio_vol)
    portfolio_vol = rolling 21d std of H021 daily returns * sqrt(252).
    """

    name = "risk_budget_expansion"

    def __init__(
        self,
        vol_lookback: int = 60,
        vol_target_favorable: float = 0.10,
        vol_target_base: float = 0.08,
        vol_target_unfavorable: float = 0.06,
        vix_pct_window: int = 252,
        vix_pct_threshold: float = 0.60,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vol_target_favorable = vol_target_favorable
        self.vol_target_base = vol_target_base
        self.vol_target_unfavorable = vol_target_unfavorable
        self.vix_pct_window = vix_pct_window
        self.vix_pct_threshold = vix_pct_threshold

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # H021 portfolio daily returns
        port_ret = w_spy.shift(1) * spy_r + w_tlt.shift(1) * tlt_r
        port_vol = port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
        port_vol = port_vol.shift(1).replace(0, np.nan)

        # VRP signal
        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_pos = vrp > 0

        # SPY 252d momentum
        spy_ann = prices["adj_close"].pct_change(252).shift(1)
        spy_up = spy_ann > 0

        # VIX 60th percentile (rolling)
        vix_shifted = prices["vix_close"].shift(1)
        vix_pct60 = vix_shifted.rolling(self.vix_pct_window, min_periods=63).quantile(
            self.vix_pct_threshold
        )
        vix_elevated = vix_shifted > vix_pct60

        favorable = vrp_pos & spy_up
        unfavorable = (~vrp_pos) | vix_elevated

        vol_target = pd.Series(
            np.select(
                [favorable, unfavorable],
                [self.vol_target_favorable, self.vol_target_unfavorable],
                default=self.vol_target_base,
            ),
            index=prices.index,
        )

        scale = (vol_target / port_vol).clip(upper=1.0).fillna(1.0)

        return pd.DataFrame(
            {"SPY": w_spy * scale, "TLT": w_tlt * scale},
            index=prices.index,
        )


class DefensiveMacroStateMachine(MultiAssetStrategy):
    """H320: 3-state Risk-On / Neutral / Risk-Off machine with VIX hysteresis.

    Risk-On  : VRP > 0 AND SPY 252d > 0 AND SPY 63d > TLT 63d -> H021 x risk_on_scale
    Risk-Off : VIX > vix_threshold OR SPY 252d < 0 (hysteresis: calm_days consecutive
               non-triggering days required to exit)
    Neutral  : all else -> H021 x 1.0

    In Risk-Off: SPY weight = H021_SPY x risk_off_spy_scale; freed weight -> TLT.
    """

    name = "defensive_macro_state_machine"

    def __init__(
        self,
        vol_lookback: int = 60,
        vix_threshold: float = 28.0,
        calm_days: int = 5,
        risk_on_scale: float = 1.15,
        risk_off_spy_scale: float = 0.65,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vix_threshold = vix_threshold
        self.calm_days = calm_days
        self.risk_on_scale = risk_on_scale
        self.risk_off_spy_scale = risk_off_spy_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vix = prices["vix_close"].shift(1)
        spy_ann = prices["adj_close"].pct_change(252).shift(1)
        spy_63 = prices["adj_close"].pct_change(63).shift(1)
        tlt_63 = prices["tlt_close"].pct_change(63).shift(1)

        vrp = _vrp_series(spy_r, prices["vix_close"])

        risk_off_raw = (vix > self.vix_threshold) | (spy_ann < 0)
        risk_on_raw = (vrp > 0) & (spy_ann > 0) & (spy_63 > tlt_63)

        # Apply hysteresis to Risk-Off: stay in Risk-Off until calm_days consecutive
        # non-triggering days
        n = len(prices)
        state = np.zeros(n, dtype=np.int8)  # 0=Neutral, 1=Risk-On, -1=Risk-Off
        calm_count = 0
        for i in range(n):
            ro = bool(risk_off_raw.iloc[i]) if not pd.isna(risk_off_raw.iloc[i]) else False
            ron = bool(risk_on_raw.iloc[i]) if not pd.isna(risk_on_raw.iloc[i]) else False
            prev = state[i - 1] if i > 0 else 0

            if prev == -1:
                # Already in Risk-Off -- need calm_days to exit
                if ro:
                    calm_count = 0
                    state[i] = -1
                else:
                    calm_count += 1
                    if calm_count >= self.calm_days:
                        state[i] = 0
                        calm_count = 0
                    else:
                        state[i] = -1
            elif ro:
                state[i] = -1
                calm_count = 0
            elif ron:
                state[i] = 1
                calm_count = 0
            else:
                state[i] = 0
                calm_count = 0

        state_s = pd.Series(state, index=prices.index)

        # Build weights per state
        # Risk-Off: SPY reduced, freed weight to TLT
        freed = w_spy * (1.0 - self.risk_off_spy_scale)
        spy_out = pd.Series(
            np.select(
                [state_s == -1, state_s == 1],
                [w_spy * self.risk_off_spy_scale, (w_spy * self.risk_on_scale).clip(upper=1.0)],
                default=w_spy,
            ),
            index=prices.index,
        )
        tlt_out = pd.Series(
            np.select(
                [state_s == -1, state_s == 1],
                [(w_tlt + freed).clip(upper=1.0), (w_tlt * self.risk_on_scale).clip(upper=1.0)],
                default=w_tlt,
            ),
            index=prices.index,
        )

        return pd.DataFrame({"SPY": spy_out, "TLT": tlt_out}, index=prices.index)


class DualFrequencyVolTarget(MultiAssetStrategy):
    """H326: Vol-target using max(21d, 63d) portfolio vol — more conservative than H086."""

    name = "dual_frequency_vol_target"

    def __init__(
        self,
        vol_target: float = 0.08,
        short_window: int = 21,
        long_window: int = 63,
        vol_lookback: int = 60,
    ) -> None:
        self.vol_target = vol_target
        self.short_window = short_window
        self.long_window = long_window
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        port_r = w_spy * spy_r + w_tlt * tlt_r
        vol_short = port_r.rolling(self.short_window, min_periods=10).std().shift(1) * np.sqrt(252)
        vol_long = port_r.rolling(self.long_window, min_periods=20).std().shift(1) * np.sqrt(252)
        vol_conservative = vol_short.combine(vol_long, np.maximum).replace(0, np.nan)
        scale = (self.vol_target / vol_conservative).clip(upper=1.0).fillna(1.0)

        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class VolAccelerationDetector(MultiAssetStrategy):
    """H327: H086 vol-target with penalty when vol accelerates, bonus when decelerating."""

    name = "vol_acceleration_detector"

    def __init__(
        self,
        vol_target: float = 0.08,
        vol_lookback: int = 60,
        accel_threshold: float = 1.5,
        decel_threshold: float = 0.75,
        accel_penalty: float = 0.75,
        decel_bonus: float = 1.15,
    ) -> None:
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.accel_threshold = accel_threshold
        self.decel_threshold = decel_threshold
        self.accel_penalty = accel_penalty
        self.decel_bonus = decel_bonus

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        port_r = w_spy * spy_r + w_tlt * tlt_r
        vol_21 = port_r.rolling(21, min_periods=10).std().shift(1) * np.sqrt(252)
        vol_63 = port_r.rolling(63, min_periods=20).std().shift(1) * np.sqrt(252)

        # H086 base scale
        vol_scale = (self.vol_target / vol_21.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)

        # Acceleration ratio; guard against zero denominator
        vol_ratio = (vol_21 / vol_63.replace(0, np.nan)).fillna(1.0)
        penalty = pd.Series(
            np.select(
                [vol_ratio > self.accel_threshold, vol_ratio < self.decel_threshold],
                [self.accel_penalty, self.decel_bonus],
                default=1.0,
            ),
            index=prices.index,
        )

        scale = (vol_scale * penalty).clip(upper=1.0)
        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class VolPercentileExposure(MultiAssetStrategy):
    """H328: 3-tier exposure based on 252d rolling percentile of 21d portfolio vol."""

    name = "vol_percentile_exposure"

    def __init__(
        self,
        vol_target: float = 0.08,
        vol_window: int = 21,
        pctile_window: int = 252,
        low_threshold: float = 0.33,
        high_threshold: float = 0.67,
        low_scale: float = 1.20,
        vol_lookback: int = 60,
    ) -> None:
        self.vol_target = vol_target
        self.vol_window = vol_window
        self.pctile_window = pctile_window
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.low_scale = low_scale
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        port_r = w_spy * spy_r + w_tlt * tlt_r
        vol_21 = port_r.rolling(self.vol_window, min_periods=10).std().shift(1) * np.sqrt(252)

        # Rolling percentile rank of current vol
        def _rolling_pctile(s: pd.Series, window: int) -> pd.Series:
            return s.rolling(window, min_periods=window // 2).apply(
                lambda x: float(np.sum(x[:-1] <= x[-1])) / max(len(x) - 1, 1),
                raw=True,
            )

        vol_pctile = _rolling_pctile(vol_21, self.pctile_window)

        vol_target_high = (
            (self.vol_target / vol_21.replace(0, np.nan)).clip(upper=0.60).fillna(0.60)
        )
        scale = pd.Series(
            np.select(
                [vol_pctile < self.low_threshold, vol_pctile > self.high_threshold],
                [self.low_scale, vol_target_high],
                default=1.0,
            ),
            index=prices.index,
        )

        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class VolRegimeRiskParity(MultiAssetStrategy):
    """H329: 3-state vol regime — low vol allows 1.20x; high vol applies tighter 7% target."""

    name = "vol_regime_risk_parity"

    def __init__(
        self,
        vol_window: int = 21,
        regime_window: int = 252,
        low_mult: float = 0.70,
        high_mult: float = 1.30,
        vol_target_high: float = 0.07,
        low_scale: float = 1.20,
        vol_lookback: int = 60,
    ) -> None:
        self.vol_window = vol_window
        self.regime_window = regime_window
        self.low_mult = low_mult
        self.high_mult = high_mult
        self.vol_target_high = vol_target_high
        self.low_scale = low_scale
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        port_r = w_spy * spy_r + w_tlt * tlt_r
        vol_21 = port_r.rolling(self.vol_window, min_periods=10).std().shift(1) * np.sqrt(252)
        vol_mean = vol_21.rolling(self.regime_window, min_periods=self.regime_window // 2).mean()

        is_low = vol_21 < self.low_mult * vol_mean
        is_high = vol_21 > self.high_mult * vol_mean

        high_scale = (self.vol_target_high / vol_21.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
        scale = pd.Series(
            np.select(
                [is_low, is_high],
                [self.low_scale, high_scale],
                default=1.0,
            ),
            index=prices.index,
        )

        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class RatchetVolTarget(MultiAssetStrategy):
    """H330: min(H086 vol-scale, H087 ratchet drawdown-scale) — most conservative wins."""

    name = "ratchet_vol_target"

    def __init__(
        self,
        vol_target: float = 0.08,
        drawdown_sensitivity: float = 3.0,
        vol_lookback: int = 60,
    ) -> None:
        self.vol_target = vol_target
        self.drawdown_sensitivity = drawdown_sensitivity
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        port_r = w_spy * spy_r + w_tlt * tlt_r

        # Vol scale (H086 mechanism)
        port_vol = port_r.rolling(21, min_periods=10).std().shift(1) * np.sqrt(252)
        vol_scale = (self.vol_target / port_vol.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)

        # Ratchet drawdown scale (H087 mechanism)
        cum = (1 + port_r).cumprod()
        peak = cum.cummax()
        dd_depth = ((peak - cum) / peak.replace(0, np.nan)).fillna(0.0)
        dd_scale = (1.0 - dd_depth * self.drawdown_sensitivity).clip(lower=0.10)

        # Take the more conservative of the two
        final_scale = np.minimum(vol_scale, dd_scale)

        return pd.DataFrame(
            {"SPY": w_spy * final_scale, "TLT": w_tlt * final_scale}, index=prices.index
        )


class VRPRefinedH060(MultiAssetStrategy):
    """H336: CarryMomentumSPYTLT (H060) + VRP overlay.

    H060 base: 100% SPY if curve steep (TLT>SHY 63d) AND SPY above 200d MA;
               100% TLT if curve inverted; flat otherwise.
    VRP modifier: if H060 would be 100% SPY and VRP < 0 →
                  spy_weight × vrp_cut (0.70 SPY + 0.30 TLT).
    When VRP ≥ 0 or H060 is already in TLT/flat: no change.

    VRP < 0 means the market under-pays for variance insurance — realized vol
    is running above implied. This is a mild stress signal that justifies
    trimming the pure-equity bet H060 would otherwise make.
    """

    name = "vrp_refined_h060"

    def __init__(
        self,
        mom_days: int = 63,
        spy_ma: int = 200,
        vrp_cut: float = 0.70,
    ) -> None:
        self.mom_days = mom_days
        self.spy_ma = spy_ma
        self.vrp_cut = vrp_cut

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()

        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt = prices.get("tlt_close", pd.Series(np.nan, index=prices.index))

        shy_r = shy.shift(1) / shy.shift(1 + self.mom_days) - 1
        tlt_r = tlt.shift(1) / tlt.shift(1 + self.mom_days) - 1
        spy_ma = prices["adj_close"].shift(1).rolling(self.spy_ma).mean()

        valid = shy_r.notna() & tlt_r.notna()
        inverted = valid & (shy_r > tlt_r)
        spy_trend_up = prices["adj_close"].shift(1) > spy_ma

        # H060 base weights
        w_spy = pd.Series(0.0, index=prices.index)
        w_tlt = pd.Series(0.0, index=prices.index)
        w_spy[valid & ~inverted & spy_trend_up] = 1.0
        w_tlt[inverted] = 1.0

        # VRP overlay: reduce SPY when VRP < 0 and H060 is in 100% SPY
        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_negative = (vrp < 0).fillna(False)
        fully_spy = w_spy >= 1.0
        cut_condition = vrp_negative & fully_spy

        w_spy = w_spy.where(~cut_condition, self.vrp_cut)
        w_tlt = w_tlt.where(~cut_condition, 1.0 - self.vrp_cut)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class VRPCorrelationRegime(MultiAssetStrategy):
    """H339: CorrelationRiskParityHybrid (H097) + VRP boost in normal regime.

    H097 base: if 60d rolling SPY-TLT correlation > 0 → crisis (80% TLT);
               else H021 weights.
    VRP boost: in normal regime (corr < 0) AND VRP > 252d median VRP →
               SPY weight × vrp_boost (1.10), clipped to 1.0.

    When correlation is negative (SPY and TLT moving independently as expected),
    a positive VRP confirms the market is in a compensating regime, making the
    risk-parity position even more attractive. The boost captures this regime.
    """

    name = "vrp_correlation_regime"

    def __init__(
        self,
        vol_lookback: int = 60,
        corr_lookback: int = 60,
        crisis_tlt: float = 0.90,
        vrp_window: int = 252,
        vrp_boost: float = 1.10,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.corr_lookback = corr_lookback
        self.crisis_tlt = crisis_tlt
        self.vrp_window = vrp_window
        self.vrp_boost = vrp_boost

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy_h021, w_tlt_h021 = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # H097 correlation signal
        corr = spy_r.rolling(self.corr_lookback).corr(tlt_r).shift(1)
        crisis = (corr > 0) & prices["tlt_close"].notna()

        tlt_avail = prices["tlt_close"].notna()
        w_spy = w_spy_h021.where(~crisis, 1.0 - self.crisis_tlt)
        w_tlt = w_tlt_h021.where(~crisis, self.crisis_tlt)
        w_spy = w_spy.where(tlt_avail, 0.0)
        w_tlt = w_tlt.where(tlt_avail, 0.0)

        # VRP boost in normal regime
        vrp = _vrp_series(spy_r, prices["vix_close"])
        vrp_median = vrp.rolling(self.vrp_window, min_periods=63).median()
        vrp_above_median = (vrp > vrp_median) & vrp_median.notna()

        normal_regime = ~crisis
        boost_condition = normal_regime & vrp_above_median
        w_spy_boosted = (w_spy * self.vrp_boost).clip(upper=1.0)
        w_spy = w_spy.where(~boost_condition, w_spy_boosted)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class GrandCrisisProtocol(MultiAssetStrategy):
    """H340: Union of four crisis signals with tiered SPY/TLT response.

    Signal 1: H097 correlation crisis (60d SPY-TLT corr > 0).
    Signal 2: H079 crisis (portfolio drawdown > dd_threshold AND VIX > vix_threshold).
    Signal 3: H306 high stress (stress_score > stress_high).
    Signal 4: VRP < vrp_severe (severe negative VRP).

    Tiered response:
      2+ signals active → 20% SPY / 80% TLT
      1  signal active  → 40% SPY / 60% TLT
      0  signals active → H021 inverse-vol weights

    Diversifies across four independent crisis detectors: correlation contagion,
    drawdown depth, macro stress, and VRP. Any single detector can fire without
    triggering maximum defense; the most defensive posture requires corroboration.
    """

    name = "grand_crisis_protocol"

    def __init__(
        self,
        vol_lookback: int = 60,
        corr_lookback: int = 60,
        dd_threshold: float = 0.15,
        vix_threshold: float = 30.0,
        stress_high: float = 0.67,
        vrp_severe: float = -0.005,
        spy_one: float = 0.40,
        tlt_one: float = 0.60,
        spy_two: float = 0.20,
        tlt_two: float = 0.80,
        vix_pctile_window: int = 252,
        credit_lookback: int = 63,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.corr_lookback = corr_lookback
        self.dd_threshold = dd_threshold
        self.vix_threshold = vix_threshold
        self.stress_high = stress_high
        self.vrp_severe = vrp_severe
        self.spy_one = spy_one
        self.tlt_one = tlt_one
        self.spy_two = spy_two
        self.tlt_two = tlt_two
        self.vix_pctile_window = vix_pctile_window
        self.credit_lookback = credit_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0914
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy_h021, w_tlt_h021 = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # Signal 1: H097 correlation crisis
        corr = spy_r.rolling(self.corr_lookback).corr(tlt_r).shift(1)
        s1 = ((corr > 0) & prices["tlt_close"].notna()).fillna(False).astype(int)

        # Signal 2: H079 portfolio drawdown + VIX
        port_r = w_spy_h021.shift(1) * spy_r + w_tlt_h021.shift(1) * tlt_r
        cum = (1 + port_r).cumprod()
        peak = cum.cummax()
        dd_depth = ((peak - cum) / peak.replace(0, np.nan)).fillna(0.0)
        vix = prices["vix_close"].shift(1).fillna(0.0)
        s2 = ((dd_depth > self.dd_threshold) & (vix > self.vix_threshold)).astype(int)

        # Signal 3: H306-style stress composite
        vix_s = prices["vix_close"].shift(1)
        vix_hi = vix_s.rolling(self.vix_pctile_window, min_periods=63).quantile(0.75)
        vix_lo = vix_s.rolling(self.vix_pctile_window, min_periods=63).quantile(0.25)
        vix_range = (vix_hi - vix_lo).replace(0, np.nan)
        vix_pctile = ((vix_s - vix_lo) / vix_range).clip(0.0, 1.0).fillna(0.5)

        spy_vol_21 = spy_r.rolling(21, min_periods=10).std().shift(1)
        spy_vol_63 = spy_r.rolling(63, min_periods=30).std().shift(1)
        vol_accel = (spy_vol_21 > spy_vol_63 * 1.5).fillna(False).astype(float)

        n_sig = 2.0
        credit_c = pd.Series(0.0, index=prices.index)
        if "hyg_close" in prices.columns and "lqd_close" in prices.columns:
            hyg_ret = prices["hyg_close"].pct_change(self.credit_lookback).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(self.credit_lookback).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            credit_ok = hyg_avail & hyg_ret.notna() & lqd_ret.notna()
            credit_c = (hyg_ret < lqd_ret).where(credit_ok, False).astype(float)
            n_sig = 3.0

        stress_score = (vix_pctile + credit_c + vol_accel) / n_sig
        s3 = (stress_score > self.stress_high).astype(int)

        # Signal 4: Severe negative VRP
        vrp = _vrp_series(spy_r, prices["vix_close"])
        s4 = (vrp < self.vrp_severe).fillna(False).astype(int)

        total_signals = s1 + s2 + s3 + s4

        tlt_avail = prices["tlt_close"].notna()
        w_spy = pd.Series(
            np.select(
                [total_signals >= 2, total_signals == 1],
                [self.spy_two, self.spy_one],
                default=w_spy_h021,
            ),
            index=prices.index,
        )
        w_tlt = pd.Series(
            np.select(
                [total_signals >= 2, total_signals == 1],
                [self.tlt_two, self.tlt_one],
                default=w_tlt_h021,
            ),
            index=prices.index,
        )

        # Pre-TLT era: SPY only
        w_spy = w_spy.where(tlt_avail, 1.0)
        w_tlt = w_tlt.where(tlt_avail, 0.0)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class MarketVolBreadthAAPort(MultiAssetStrategy):
    """H485: H021 overlaid with cross-sectional vol-decline breadth signal.

    Signal: fraction of S&P 500 stocks where 21d realized vol < 63d realized vol (pre-computed).
    Breadth > 0.60 → systemic vol normalization → SPY × 1.15.
    Breadth < 0.40 → systemic vol buildup → SPY × 0.85.
    Pure price signal, full 12-fold coverage.
    """

    name = "market_vol_breadth_aa_port"

    def __init__(
        self,
        vol_breadth: pd.Series,
        vol_lookback: int = 60,
        bull_thresh: float = 0.60,
        bear_thresh: float = 0.40,
        bull_scale: float = 1.15,
        bear_scale: float = 0.85,
    ) -> None:
        self.vol_breadth = vol_breadth
        self.vol_lookback = vol_lookback
        self.bull_thresh = bull_thresh
        self.bear_thresh = bear_thresh
        self.bull_scale = bull_scale
        self.bear_scale = bear_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        breadth = self.vol_breadth.reindex(prices.index).ffill().shift(1)
        bull = (breadth > self.bull_thresh) & tlt_avail
        bear = (breadth < self.bear_thresh) & tlt_avail
        spy_scaled = w_spy.copy()
        spy_scaled = (w_spy * self.bull_scale).where(bull, spy_scaled)
        spy_scaled = (w_spy * self.bear_scale).where(bear, spy_scaled)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        spy_final = spy_scaled.where(tlt_avail, 1.0)
        tlt_final = tlt_adj.where(tlt_avail, 0.0)
        return pd.DataFrame({"SPY": spy_final, "TLT": tlt_final}, index=prices.index)


class VIXPercentileScalerPort(MultiAssetStrategy):
    """H492: H021 with linear SPY-weight scaling from VIX trailing percentile rank.

    Signal: VIX 252d trailing percentile rank (0=lowest VIX in past year, 1=highest).
    scale = 1.0 + (0.5 - vix_pct) * (scale_hi - scale_lo)
    → VIX at min (pct=0): scale=scale_hi (equity tilt).
    → VIX at median (pct=0.5): scale=1.0 (neutral).
    → VIX at max (pct=1): scale=scale_lo (defensive).
    Continuous analog of H066/H070's binary VIX threshold.
    """

    name = "vix_percentile_scaler_port"

    def __init__(
        self,
        vol_lookback: int = 60,
        pct_window: int = 252,
        scale_lo: float = 0.60,
        scale_hi: float = 1.40,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.pct_window = pct_window
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()

        vix = prices["vix_close"].shift(1)
        vix_pct = vix.rolling(self.pct_window, min_periods=self.pct_window // 2).rank(pct=True)

        scale = (
            (1.0 + (0.5 - vix_pct) * (self.scale_hi - self.scale_lo))
            .clip(self.scale_lo, self.scale_hi)
            .fillna(1.0)
        )

        spy_scaled = (w_spy * scale).clip(0.0, 1.0)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        return pd.DataFrame(
            {"SPY": spy_scaled.where(tlt_avail, 1.0), "TLT": tlt_adj.where(tlt_avail, 0.0)},
            index=prices.index,
        )


class VIXMeanReversionPort(MultiAssetStrategy):
    """H496: Contrarian VIX overlay on H021 — buy when VIX is historically extreme.

    Signal: VIX / VIX_252d_MA ratio.
    ratio ≥ high_thresh (default 1.5): extreme fear → SPY weight × scale_hi (contrarian buy).
    ratio < low_thresh (default 0.7): extreme calm → SPY normal (scale 1.0).
    Between thresholds: linear interpolation.
    Tests if VIX mean reversion adds value vs H066/H492 (which penalize high VIX).
    """

    name = "vix_mean_reversion_port"

    def __init__(
        self,
        vol_lookback: int = 60,
        vix_ma_window: int = 252,
        low_thresh: float = 0.70,
        high_thresh: float = 1.50,
        scale_lo: float = 1.00,
        scale_hi: float = 1.20,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vix_ma_window = vix_ma_window
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()

        vix = prices["vix_close"].shift(1)
        vix_ma = vix.rolling(self.vix_ma_window, min_periods=self.vix_ma_window // 2).mean()
        ratio = (vix / vix_ma.replace(0, np.nan)).clip(0.0, 3.0).fillna(1.0)

        # Linear ramp: ratio ≤ low_thresh → scale_lo; ratio ≥ high_thresh → scale_hi
        t_lo, t_hi = self.low_thresh, self.high_thresh
        scale = np.where(
            ratio >= t_hi,
            self.scale_hi,
            np.where(
                ratio < t_lo,
                self.scale_lo,
                self.scale_lo + (ratio - t_lo) / (t_hi - t_lo) * (self.scale_hi - self.scale_lo),
            ),
        )
        scale = pd.Series(scale, index=prices.index)

        spy_scaled = (w_spy * scale).clip(0.0, 1.0)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        return pd.DataFrame(
            {"SPY": spy_scaled.where(tlt_avail, 1.0), "TLT": tlt_adj.where(tlt_avail, 0.0)},
            index=prices.index,
        )


class SectorMomentumVolTarget(MultiAssetStrategy):
    """H577: Sector momentum with vol-targeting overlay.

    Ranks equity ETFs (XLP, XLU, IVE, IVW, QQQ, IWM, VNQ) by 6-month
    momentum; holds top-N risk-parity weighted. Applies vol-targeting:
    scales total exposure by min(1, target_vol/realized_vol). Pure momentum
    signal without regime conditioning (contrast with H063/H065).
    """

    name = "sector_momentum_vol_target"

    _ETFS = ["XLP", "XLU", "IVE", "IVW", "QQQ", "IWM", "VNQ"]

    def __init__(
        self,
        mom_window: int = 126,
        vol_lookback: int = 60,
        top_n: int = 3,
        target_vol: float = 0.12,
    ) -> None:
        self.mom_window = mom_window
        self.vol_lookback = vol_lookback
        self.top_n = top_n
        self.target_vol = target_vol

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        col_map = {a: f"{a.lower()}_close" for a in self._ETFS}
        available = {a: col for a, col in col_map.items() if col in prices.columns}

        if not available:
            return pd.DataFrame({"SPY": pd.Series(1.0, index=prices.index)})

        mom_df = pd.DataFrame(
            {a: prices[col].pct_change(self.mom_window) for a, col in available.items()}
        )
        ret_df = pd.DataFrame({a: prices[col].pct_change() for a, col in available.items()})
        vol_df = ret_df.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std()
        inv_vol_df = vol_df.replace(0, np.nan).rdiv(1).fillna(1.0)

        mom_rank = mom_df.rank(axis=1, ascending=False, na_option="keep")
        top_mask = (mom_rank <= self.top_n) & mom_rank.notna()

        raw = inv_vol_df.where(top_mask, 0.0)
        total = raw.sum(axis=1).replace(0, np.nan)
        etf_w = raw.div(total, axis=0).fillna(0.0)

        # Vol-targeting: scale total ETF exposure
        port_r = (etf_w.shift(1) * ret_df).sum(axis=1)
        port_vol = port_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        scale = (self.target_vol / port_vol.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)

        etf_w_scaled = etf_w.multiply(scale, axis=0)

        no_etf = top_mask.sum(axis=1) == 0
        etf_w_scaled["SPY"] = no_etf.astype(float)

        return etf_w_scaled.shift(1).fillna(0.0)
