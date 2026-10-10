"""Macro, credit, yield-curve, funding-stress, and market-breadth regime overlays."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import _h021_base_weights, _h021_weights, _vrp_series
from .base import MultiAssetStrategy, Strategy


class CommodityMomentumOverlay(Strategy):
    """H052: Long SPY when DBC is below its MA (no commodity inflation regime).

    Rising commodity prices signal an inflationary environment that historically
    compresses equity multiples and triggers bond selloffs simultaneously.
    When DBC trends above its MA, the inflation regime is active — reducing
    equity exposure avoids the dual equity+bond drawdown of inflationary bears.

    Long SPY when DBC < ma_period MA (no inflation pressure); flat otherwise.
    When DBC data unavailable, defaults to long SPY.

    Requires: dbc_close column.
    """

    def __init__(self, ma_period: int = 200):
        self.ma_period = ma_period
        self.name = f"commodity_momentum_{ma_period}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "dbc_close" not in prices.columns:
            return pd.Series(1.0, index=prices.index, name="position")
        dbc = prices["dbc_close"]
        dbc_ma = dbc.rolling(self.ma_period, min_periods=self.ma_period).mean().shift(1)
        no_inflation = dbc.shift(1) < dbc_ma
        pos = no_inflation.astype(float)
        pos = pos.where(dbc_ma.notna(), 0.0)
        pos.name = "position"
        return pos


class CreditSpreadRegime(Strategy):
    """H053: Long SPY when HYG outperforms LQD (healthy credit appetite).

    Credit spread widening (HYG underperforming LQD) signals rising corporate
    default risk — a leading indicator of equity stress. When high-yield bonds
    are gaining relative to investment-grade, credit risk appetite is positive
    and equity exposure is favorable.

    Signal: long SPY when HYG momentum_days return > LQD momentum_days return.
    Falls back to flat when HYG data unavailable (pre-2007).

    Requires: hyg_close, lqd_close columns.
    """

    def __init__(self, momentum_days: int = 63):
        self.momentum_days = momentum_days
        self.name = f"credit_spread_regime_{momentum_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "hyg_close" not in prices.columns or "lqd_close" not in prices.columns:
            return pd.Series(0.0, index=prices.index, name="position")
        hyg = prices["hyg_close"]
        lqd = prices["lqd_close"]
        hyg_mom = hyg / hyg.shift(self.momentum_days) - 1
        lqd_mom = lqd / lqd.shift(self.momentum_days) - 1
        valid = hyg_mom.notna() & lqd_mom.notna()
        pos = (hyg_mom >= lqd_mom).astype(float)
        pos = pos.where(valid, 0.0)
        pos.name = "position"
        return pos


class MacroFearComposite(Strategy):
    """H059: Position scaled by 4-signal macro fear composite.

    Fear votes: VIX > vix_thresh, GLD above gld_ma, HYG < LQD momentum,
    yield curve inverted (SHY > TLT on mom_days). Position = max(0, 1 - 2*score)
    where score = fear_count / 4. Two or more fear signals → flat.
    """

    def __init__(
        self,
        vix_thresh: float = 25.0,
        gld_ma: int = 50,
        mom_days: int = 63,
    ) -> None:
        self.vix_thresh = vix_thresh
        self.gld_ma = gld_ma
        self.mom_days = mom_days

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        vix = prices.get("vix_close", pd.Series(np.nan, index=prices.index))
        gld = prices.get("gld_close", pd.Series(np.nan, index=prices.index))
        hyg = prices.get("hyg_close", pd.Series(np.nan, index=prices.index))
        lqd = prices.get("lqd_close", pd.Series(np.nan, index=prices.index))
        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt = prices.get("tlt_close", pd.Series(np.nan, index=prices.index))

        fear = pd.DataFrame(0.0, index=prices.index, columns=["vix", "gld", "credit", "curve"])

        vix_sig = vix.shift(1)
        fear["vix"] = (vix_sig > self.vix_thresh).where(vix_sig.notna(), np.nan).fillna(0.0)

        gld_ma = gld.shift(1).rolling(self.gld_ma).mean()
        fear["gld"] = (gld.shift(1) > gld_ma).where(gld_ma.notna(), np.nan).fillna(0.0)

        hyg_r = hyg.shift(1) / hyg.shift(1 + self.mom_days) - 1
        lqd_r = lqd.shift(1) / lqd.shift(1 + self.mom_days) - 1
        fear["credit"] = (hyg_r < lqd_r).where(hyg_r.notna() & lqd_r.notna(), np.nan).fillna(0.0)

        shy_r = shy.shift(1) / shy.shift(1 + self.mom_days) - 1
        tlt_r = tlt.shift(1) / tlt.shift(1 + self.mom_days) - 1
        fear["curve"] = (shy_r > tlt_r).where(shy_r.notna() & tlt_r.notna(), np.nan).fillna(0.0)

        return (1.0 - 2.0 * fear.mean(axis=1)).clip(lower=0.0)


class SmallCapCyclical(MultiAssetStrategy):
    """H062: Long IWM when credit healthy AND IWM outperforms SPY on momentum.

    Both conditions met → IWM; otherwise → SPY.
    Falls back to SPY when HYG/LQD data unavailable.
    """

    name = "small_cap_cyclical"

    def __init__(self, credit_days: int = 63, mom_days: int = 126) -> None:
        self.credit_days = credit_days
        self.mom_days = mom_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        hyg = prices.get("hyg_close", pd.Series(np.nan, index=prices.index))
        lqd = prices.get("lqd_close", pd.Series(np.nan, index=prices.index))
        iwm = prices.get("iwm_close", pd.Series(np.nan, index=prices.index))

        hyg_r = hyg.shift(1) / hyg.shift(1 + self.credit_days) - 1
        lqd_r = lqd.shift(1) / lqd.shift(1 + self.credit_days) - 1
        spy_mom = prices["adj_close"].shift(1) / prices["adj_close"].shift(1 + self.mom_days) - 1
        iwm_mom = iwm.shift(1) / iwm.shift(1 + self.mom_days) - 1

        credit_ok = (hyg_r >= lqd_r) & hyg_r.notna() & lqd_r.notna()
        size_ok = (iwm_mom >= spy_mom) & iwm_mom.notna()

        w_spy = pd.Series(1.0, index=prices.index)
        w_iwm = pd.Series(0.0, index=prices.index)

        small_cap_regime = credit_ok & size_ok
        w_spy[small_cap_regime] = 0.0
        w_iwm[small_cap_regime] = 1.0

        return pd.DataFrame({"SPY": w_spy, "IWM": w_iwm}, index=prices.index)


class EqualWeightDefensiveFactor(MultiAssetStrategy):
    """H065: Static 25/25/25/25 equal-weight SPY+XLP+XLU+IVE.

    Pure factor quality tilt — no timing. Renormalizes when IVE
    unavailable (pre-2000): equal-weight among available assets.
    """

    name = "equal_weight_defensive_factor"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {
            "SPY": "adj_close",
            "XLP": "xlp_close",
            "XLU": "xlu_close",
            "IVE": "ive_close",
        }
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        weights = pd.DataFrame(0.0, index=prices.index, columns=list(available.keys()))
        for i, idx in enumerate(prices.index):
            active = [
                asset
                for asset, col in available.items()
                if col == "adj_close" or not np.isnan(prices[col].iloc[i])
            ]
            if active:
                w = 1.0 / len(active)
                for asset in active:
                    weights.at[idx, asset] = w
        return weights


class FundingStressMonitor(MultiAssetStrategy):
    """H306: H021 scaled by 3-signal composite funding stress index.

    stress = mean(vix_pctile_252d, credit_stress, vol_accel)
      credit_stress = 1 if HYG 63d return < LQD 63d return, else 0 (0 if unavailable)
      vol_accel = 1 if SPY 21d vol > 1.5× SPY 63d vol, else 0
      vix_pctile = rolling 252d percentile of VIX level ∈ [0, 1]
    scale = max(min_scale, 1 - stress_sensitivity * stress)
    """

    name = "funding_stress_monitor"

    def __init__(
        self,
        vol_lookback: int = 60,
        stress_sensitivity: float = 0.50,
        min_scale: float = 0.40,
        vix_pctile_window: int = 252,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.stress_sensitivity = stress_sensitivity
        self.min_scale = min_scale
        self.vix_pctile_window = vix_pctile_window

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vix = prices["vix_close"].shift(1)
        vix_hi = vix.rolling(self.vix_pctile_window, min_periods=63).quantile(0.75)
        vix_lo = vix.rolling(self.vix_pctile_window, min_periods=63).quantile(0.25)
        vix_range = (vix_hi - vix_lo).replace(0, np.nan)
        vix_pctile = ((vix - vix_lo) / vix_range).clip(0.0, 1.0).fillna(0.5)

        spy_vol_21 = spy_r.rolling(21, min_periods=10).std().shift(1)
        spy_vol_63 = spy_r.rolling(63, min_periods=30).std().shift(1)
        vol_accel = (spy_vol_21 > spy_vol_63 * 1.5).fillna(False).astype(float)

        n_signals = 2.0
        credit_stress = pd.Series(0.0, index=prices.index)
        if "hyg_close" in prices.columns and "lqd_close" in prices.columns:
            hyg_ret = prices["hyg_close"].pct_change(63).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(63).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            credit_ok = hyg_avail & hyg_ret.notna() & lqd_ret.notna()
            credit_stress = (hyg_ret < lqd_ret).where(credit_ok, False).astype(float)
            n_signals = 3.0

        stress = (vix_pctile + credit_stress + vol_accel) / n_signals
        scale = (1.0 - self.stress_sensitivity * stress).clip(lower=self.min_scale)

        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class MultiSignalStressAdjusted(MultiAssetStrategy):
    """H309: H021 with 5-signal stress monitor and tiered exposure.

    Signals: VIX>30, SPY 5% DD from 20d high, SHY>TLT 63d, HYG<LQD 63d, SPY vol accel.
    Exposure tiers: 0=100%, 1=85%, 2=70%, 3=55%, 4+=40%.
    Pre-HYG era (4 signals): 0=100%, 1=87%, 2=73%, 3+=40%.
    """

    name = "multi_signal_stress_adjusted"

    def __init__(
        self,
        vol_lookback: int = 60,
        vix_threshold: float = 30.0,
        dd_threshold: float = 0.05,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vix_threshold = vix_threshold
        self.dd_threshold = dd_threshold

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vix = prices["vix_close"].shift(1).fillna(0.0)
        s1 = (vix > self.vix_threshold).astype(float)

        spy_px = prices["adj_close"].shift(1)
        rolling_hi = spy_px.rolling(20, min_periods=5).max()
        dd = (spy_px / rolling_hi - 1.0).fillna(0.0)
        s2 = (dd < -self.dd_threshold).astype(float)

        has_shy = "shy_close" in prices.columns
        s3 = pd.Series(0.0, index=prices.index)
        if has_shy:
            shy_ret = prices["shy_close"].pct_change(63).shift(1)
            tlt_ret_63 = prices["tlt_close"].pct_change(63).shift(1)
            s3 = (shy_ret > tlt_ret_63).fillna(False).astype(float)

        has_credit = "hyg_close" in prices.columns and "lqd_close" in prices.columns
        s4 = pd.Series(0.0, index=prices.index)
        if has_credit:
            hyg_ret = prices["hyg_close"].pct_change(63).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(63).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            s4 = ((hyg_ret < lqd_ret) & hyg_avail).fillna(False).astype(float)

        spy_vol_21 = spy_r.rolling(21, min_periods=10).std().shift(1)
        spy_vol_63 = spy_r.rolling(63, min_periods=30).std().shift(1)
        s5 = (spy_vol_21 > spy_vol_63 * 1.5).fillna(False).astype(float)

        total = s1 + s2 + s3 + s4 + s5
        tiers_5 = [1.0, 0.85, 0.70, 0.55, 0.40]
        scale_arr = np.select(
            [total == 0, total == 1, total == 2, total == 3],
            tiers_5[:4],
            default=tiers_5[4],
        )
        scale = pd.Series(scale_arr, index=prices.index)
        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class DefaultSpreadSPYTimer(MultiAssetStrategy):
    """H312: H021 with HYG vs LQD default spread signal.

    HYG 63d return > LQD 63d return → credit risk-on → H021 full.
    LQD outperforming → credit risk-off → SPY weight * spy_scale; freed to TLT.
    Pre-HYG era: H021 unmodified.
    """

    name = "default_spread_spy_timer"

    def __init__(
        self,
        vol_lookback: int = 60,
        credit_lookback: int = 63,
        spy_scale: float = 0.60,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.credit_lookback = credit_lookback
        self.spy_scale = spy_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        if "hyg_close" not in prices.columns or "lqd_close" not in prices.columns:
            return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)

        hyg_ret = prices["hyg_close"].pct_change(self.credit_lookback).shift(1)
        lqd_ret = prices["lqd_close"].pct_change(self.credit_lookback).shift(1)
        hyg_avail = prices["hyg_close"].notna()
        credit_off = (hyg_ret < lqd_ret) & hyg_avail & hyg_ret.notna()

        freed = w_spy * (1.0 - self.spy_scale)
        new_spy = w_spy.where(~credit_off, w_spy * self.spy_scale)
        new_tlt = w_tlt.where(~credit_off, (w_tlt + freed).clip(upper=1.0))
        return pd.DataFrame({"SPY": new_spy, "TLT": new_tlt}, index=prices.index)


class EarningsYieldOverlay(MultiAssetStrategy):
    """H313: H021 with SPY short-vs-long-run CAGR differential valuation signal.

    Signal: SPY 252d return minus SPY 1260d annualized return (5yr CAGR proxy).
    When short-run excess is high (market extended): reduce SPY weight by spy_cut.
    When short-run below long-run (market lagging): neutral to H021.
    Uses z-score over expanding window (5yr min) to calibrate thresholds.
    """

    name = "earnings_yield_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
        short_window: int = 252,
        long_window: int = 1260,
        zscore_min: int = 1260,
        spy_cut: float = 0.15,
        zscore_threshold: float = 1.0,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.short_window = short_window
        self.long_window = long_window
        self.zscore_min = zscore_min
        self.spy_cut = spy_cut
        self.zscore_threshold = zscore_threshold

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        spy_px = prices["adj_close"]
        ret_1yr = spy_px.pct_change(self.short_window).shift(1)
        ret_5yr_ann = (
            (spy_px / spy_px.shift(self.long_window)) ** (252.0 / self.long_window) - 1.0
        ).shift(1)
        excess = (ret_1yr - ret_5yr_ann).fillna(0.0)

        ex_mean = excess.expanding(min_periods=self.zscore_min).mean()
        ex_std = excess.expanding(min_periods=self.zscore_min).std().replace(0, np.nan)
        ex_z = ((excess - ex_mean) / ex_std).fillna(0.0)

        extended = ex_z > self.zscore_threshold
        freed = w_spy * self.spy_cut
        new_spy = w_spy.where(~extended, w_spy * (1.0 - self.spy_cut))
        new_tlt = w_tlt.where(~extended, (w_tlt + freed).clip(upper=1.0))
        return pd.DataFrame({"SPY": new_spy, "TLT": new_tlt}, index=prices.index)


class ConstrainedMacroOverlay(MultiAssetStrategy):
    """H315: H314 macro combination with Campbell-Thompson constraints.

    Applies three economic constraints to H314's macro signal:
      Floor: SPY weight ≥ spy_floor * H021 SPY weight (never fully defensive).
      Cap: SPY weight ≤ spy_cap * H021 SPY weight (no excess leverage).
      Sign constraint: leverage (scale > 1) only when all 3 signals bullish or VRP very positive.
    """

    name = "constrained_macro_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
        spy_floor: float = 0.50,
        spy_cap: float = 1.30,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.spy_floor = spy_floor
        self.spy_cap = spy_cap

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        vrp = _vrp_series(spy_r, prices["vix_close"])
        s_vrp = np.where(vrp > 0, 1.0, -1.0)

        tlt_ret = prices["tlt_close"].pct_change(252).shift(1)
        shy_ret = prices["shy_close"].pct_change(252).shift(1)
        s_term = np.where(
            tlt_ret.notna() & shy_ret.notna(),
            np.where(tlt_ret > shy_ret, 1.0, -1.0),
            0.0,
        )

        s_credit = pd.Series(0.0, index=prices.index)
        if "hyg_close" in prices.columns and "lqd_close" in prices.columns:
            hyg_ret = prices["hyg_close"].pct_change(63).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(63).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            s_credit = pd.Series(
                np.where(
                    hyg_avail & hyg_ret.notna(),
                    np.where(hyg_ret > lqd_ret, 1.0, -1.0),
                    0.0,
                ),
                index=prices.index,
            )

        total = (
            pd.Series(s_vrp, index=prices.index) + pd.Series(s_term, index=prices.index) + s_credit
        )

        # Unconstrained H314 scale
        raw_scale = np.select(
            [total >= 2, total == 1, total == 0],
            [1.20, 1.05, 0.90],
            default=0.75,
        )
        raw_scale = pd.Series(raw_scale, index=prices.index)

        # Sign constraint: only leverage if all 3 signals bullish (total ≥ 3)
        constrained_scale = raw_scale.where(total >= 3, raw_scale.clip(upper=1.0))

        # SPY floor and cap
        spy_scaled = (w_spy * constrained_scale).clip(
            lower=w_spy * self.spy_floor,
            upper=w_spy * self.spy_cap,
        )
        tlt_scaled = w_tlt * constrained_scale

        return pd.DataFrame({"SPY": spy_scaled, "TLT": tlt_scaled}, index=prices.index)


class QuarterlyMacroStateOverlay(MultiAssetStrategy):
    """H316: H021 with quarterly-resampled composite macro signal.

    Three signals scored +1 / -1 / 0:
      Term spread : TLT 252d return vs SHY 252d return (always available).
      Equity momentum : SPY 252d return > 0 -> +1, else -1 (always available).
      Growth/value : IVW 252d vs IVE 252d return spread (post-2000 only -> 0 before).

    Composite sum in [-3, +3] resampled to last business day of each quarter,
    then forward-filled daily so macro state updates at most 4x per year.

    Sum >= 2 : H021 x high_scale
    Sum in [0, 1] : H021 x 1.00
    Sum <= -1 : H021 x low_scale
    """

    name = "quarterly_macro_state_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
        high_scale: float = 1.20,
        low_scale: float = 0.80,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.high_scale = high_scale
        self.low_scale = low_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # --- Term spread signal ---
        tlt_ret = prices["tlt_close"].pct_change(252).shift(1)
        shy_ret = prices["shy_close"].pct_change(252).shift(1)
        s_term = pd.Series(
            np.where(
                tlt_ret.notna() & shy_ret.notna(),
                np.where(tlt_ret > shy_ret, 1.0, -1.0),
                0.0,
            ),
            index=prices.index,
        )

        # --- Equity momentum signal ---
        spy_ret252 = prices["adj_close"].pct_change(252).shift(1)
        s_equity = pd.Series(
            np.where(spy_ret252.notna(), np.where(spy_ret252 > 0, 1.0, -1.0), 0.0),
            index=prices.index,
        )

        # --- Growth/value signal (post-2000 IVW/IVE only) ---
        s_gv = pd.Series(0.0, index=prices.index)
        if "ivw_close" in prices.columns and "ive_close" in prices.columns:
            ivw_ret = prices["ivw_close"].pct_change(252).shift(1)
            ive_ret = prices["ive_close"].pct_change(252).shift(1)
            both_avail = prices["ivw_close"].notna() & prices["ive_close"].notna()
            s_gv = pd.Series(
                np.where(
                    both_avail & ivw_ret.notna() & ive_ret.notna(),
                    np.where(ivw_ret > ive_ret, 1.0, -1.0),
                    0.0,
                ),
                index=prices.index,
            )

        composite_daily = s_term + s_equity + s_gv

        # Resample to quarter-end, then forward-fill so signal is slow-moving
        composite_q = composite_daily.resample("QE").last()
        composite_ff = composite_q.reindex(prices.index).ffill()

        scale = pd.Series(1.0, index=prices.index)
        scale = scale.where(composite_ff < 2, self.high_scale)
        scale = scale.where(composite_ff > -1, self.low_scale)
        # middle band [0, 1] keeps scale at 1.0 (already initialised)
        scale = scale.where(~((composite_ff >= 0) & (composite_ff < 2)), 1.0)
        # apply: sum >= 2 -> high_scale, sum <= -1 -> low_scale, else 1.0
        scale = np.select(
            [composite_ff >= 2, composite_ff <= -1],
            [self.high_scale, self.low_scale],
            default=1.0,
        )
        scale = pd.Series(scale, index=prices.index)

        return pd.DataFrame(
            {"SPY": (w_spy * scale).clip(upper=1.0), "TLT": (w_tlt * scale).clip(upper=1.0)},
            index=prices.index,
        )


class ValuationSpreadMarketState(MultiAssetStrategy):
    """H317: H021 scaled by IVW vs IVE 252-day return spread.

    IVW (Growth ETF) outperforming IVE (Value ETF) -> growth/risk-on regime
    -> H021 x growth_scale.
    IVE outperforming IVW -> value/defensive regime -> H021 x value_scale.
    Pre-2000 (no IVW/IVE data): pure H021 (scale = 1.0).
    """

    name = "valuation_spread_market_state"

    def __init__(
        self,
        vol_lookback: int = 60,
        ivw_lookback: int = 252,
        growth_scale: float = 1.10,
        value_scale: float = 0.90,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.ivw_lookback = ivw_lookback
        self.growth_scale = growth_scale
        self.value_scale = value_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        scale = pd.Series(1.0, index=prices.index)

        if "ivw_close" in prices.columns and "ive_close" in prices.columns:
            ivw_ret = prices["ivw_close"].pct_change(self.ivw_lookback).shift(1)
            ive_ret = prices["ive_close"].pct_change(self.ivw_lookback).shift(1)
            both_avail = (
                prices["ivw_close"].notna()
                & prices["ive_close"].notna()
                & ivw_ret.notna()
                & ive_ret.notna()
            )
            growth_on = both_avail & (ivw_ret > ive_ret)
            value_on = both_avail & (ive_ret >= ivw_ret)
            scale = pd.Series(
                np.select(
                    [growth_on, value_on],
                    [self.growth_scale, self.value_scale],
                    default=1.0,
                ),
                index=prices.index,
            )

        return pd.DataFrame(
            {"SPY": (w_spy * scale).clip(upper=1.0), "TLT": (w_tlt * scale).clip(upper=1.0)},
            index=prices.index,
        )


class PolicyRateSectorRotate(MultiAssetStrategy):
    """H422: H021 with yield-curve tilt into rate-sensitive defensive sectors.

    When TLT 12m return > SHY 12m return (rates falling, bonds outperforming):
      Reduce SPY, add XLU + XLP weight.
    Policy rate changes most directly affect utility/consumer staples sectors.
    """

    name = "policy_rate_sector_rotate"

    def __init__(
        self,
        vol_lookback: int = 60,
        curve_lookback: int = 252,
        tilt_pct: float = 0.15,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.curve_lookback = curve_lookback
        self.tilt_pct = tilt_pct

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        has_shy = "shy_close" in prices.columns
        has_xlp = "xlp_close" in prices.columns
        has_xlu = "xlu_close" in prices.columns
        tlt_ret = prices["tlt_close"] / prices["tlt_close"].shift(self.curve_lookback) - 1
        shy_ret = (
            prices["shy_close"] / prices["shy_close"].shift(self.curve_lookback) - 1
            if has_shy
            else pd.Series(0.0, index=prices.index)
        )
        shy_avail = prices["shy_close"].notna() if has_shy else pd.Series(False, index=prices.index)
        xlp_avail = prices["xlp_close"].notna() if has_xlp else pd.Series(False, index=prices.index)
        xlu_avail = prices["xlu_close"].notna() if has_xlu else pd.Series(False, index=prices.index)
        rate_falling = (
            (tlt_ret > shy_ret) & tlt_avail & shy_avail & xlp_avail & xlu_avail & tlt_ret.notna()
        )
        half_tilt = self.tilt_pct / 2.0
        w_spy_adj = w_spy.where(~rate_falling, w_spy * (1.0 - self.tilt_pct))
        w_xlp = pd.Series(0.0, index=prices.index).where(~rate_falling, w_spy * half_tilt)
        w_xlu = pd.Series(0.0, index=prices.index).where(~rate_falling, w_spy * half_tilt)
        w_spy_adj = w_spy_adj.where(tlt_avail, 1.0)
        w_tlt_final = w_tlt.where(tlt_avail, 0.0)
        return pd.DataFrame(
            {"SPY": w_spy_adj, "TLT": w_tlt_final, "XLP": w_xlp, "XLU": w_xlu}, index=prices.index
        )


class CreditCycleSectorFilter(MultiAssetStrategy):
    """H423: Credit cycle gates sector tilt into defensive XLP/XLU.

    When HYG 63d return < LQD 63d return (credit spreading → risk-off):
      Reduce SPY, add XLP + XLU as defensive sectors.
    Credit improving: pure H021. Cohen et al. (2013): credit stress = policy risk.
    """

    name = "credit_cycle_sector_filter"

    def __init__(
        self,
        vol_lookback: int = 60,
        credit_lookback: int = 63,
        tilt_pct: float = 0.20,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.credit_lookback = credit_lookback
        self.tilt_pct = tilt_pct

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        has_hyg = "hyg_close" in prices.columns
        has_lqd = "lqd_close" in prices.columns
        has_xlp = "xlp_close" in prices.columns
        has_xlu = "xlu_close" in prices.columns
        hyg_ret = (
            prices["hyg_close"].pct_change(self.credit_lookback)
            if has_hyg
            else pd.Series(np.nan, index=prices.index)
        )
        lqd_ret = (
            prices["lqd_close"].pct_change(self.credit_lookback)
            if has_lqd
            else pd.Series(np.nan, index=prices.index)
        )
        hyg_avail = prices["hyg_close"].notna() if has_hyg else pd.Series(False, index=prices.index)
        lqd_avail = prices["lqd_close"].notna() if has_lqd else pd.Series(False, index=prices.index)
        xlp_avail = prices["xlp_close"].notna() if has_xlp else pd.Series(False, index=prices.index)
        xlu_avail = prices["xlu_close"].notna() if has_xlu else pd.Series(False, index=prices.index)
        credit_stress = (
            (hyg_ret < lqd_ret) & hyg_avail & lqd_avail & xlp_avail & xlu_avail & hyg_ret.notna()
        )
        half_tilt = self.tilt_pct / 2.0
        w_spy_adj = w_spy.where(~credit_stress, w_spy * (1.0 - self.tilt_pct))
        w_xlp = pd.Series(0.0, index=prices.index).where(~credit_stress, w_spy * half_tilt)
        w_xlu = pd.Series(0.0, index=prices.index).where(~credit_stress, w_spy * half_tilt)
        w_spy_adj = w_spy_adj.where(tlt_avail, 1.0)
        w_tlt_final = w_tlt.where(tlt_avail, 0.0)
        return pd.DataFrame(
            {"SPY": w_spy_adj, "TLT": w_tlt_final, "XLP": w_xlp, "XLU": w_xlu}, index=prices.index
        )


class CommodityExposureTiming(MultiAssetStrategy):
    """H424: DBC momentum signals commodity regime; tilt SPY vs XLP defensively.

    DBC downtrend (DBC < 63d MA): reduce SPY, add XLP (defensive staples).
    DBC uptrend: standard H021 (commodity inflation = broad growth environment).
    """

    name = "commodity_exposure_timing"

    def __init__(
        self,
        vol_lookback: int = 60,
        dbc_ma: int = 63,
        tilt_pct: float = 0.15,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.dbc_ma = dbc_ma
        self.tilt_pct = tilt_pct

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        has_dbc = "dbc_close" in prices.columns
        has_xlp = "xlp_close" in prices.columns
        if has_dbc:
            dbc_ma_s = prices["dbc_close"].rolling(self.dbc_ma, min_periods=self.dbc_ma // 2).mean()
            dbc_avail = prices["dbc_close"].notna()
            xlp_avail = (
                prices["xlp_close"].notna() if has_xlp else pd.Series(False, index=prices.index)
            )
            dbc_downtrend = (
                (prices["dbc_close"] < dbc_ma_s) & dbc_avail & xlp_avail & dbc_ma_s.notna()
            )
        else:
            dbc_downtrend = pd.Series(False, index=prices.index)
        w_spy_adj = w_spy.where(~dbc_downtrend, w_spy * (1.0 - self.tilt_pct))
        w_xlp = pd.Series(0.0, index=prices.index).where(~dbc_downtrend, w_spy * self.tilt_pct)
        w_spy_adj = w_spy_adj.where(tlt_avail, 1.0)
        w_tlt_final = w_tlt.where(tlt_avail, 0.0)
        return pd.DataFrame(
            {"SPY": w_spy_adj, "TLT": w_tlt_final, "XLP": w_xlp}, index=prices.index
        )


class MarketQualityRegimePort(MultiAssetStrategy):
    """H481: H021 overlaid with market-wide GP/TA quality regime signal.

    Signal: rolling 252d median GP/TA across S&P 500 universe (pre-computed).
    Signal > its own 252d trailing mean → equity-favorable → SPY weight × 1.20.
    Signal ≤ mean → defensive → SPY weight × 0.80.
    Freed/added weight shifts to/from TLT. Capped at 1.0 total exposure.
    """

    name = "market_quality_regime_port"

    def __init__(
        self,
        market_gpta: pd.Series,
        vol_lookback: int = 60,
        signal_window: int = 252,
        bull_scale: float = 1.20,
        bear_scale: float = 0.80,
    ) -> None:
        self.market_gpta = market_gpta
        self.vol_lookback = vol_lookback
        self.signal_window = signal_window
        self.bull_scale = bull_scale
        self.bear_scale = bear_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        sig = self.market_gpta.reindex(prices.index).ffill()
        sig_ma = (
            sig.rolling(self.signal_window, min_periods=self.signal_window // 2).mean().shift(1)
        )
        bull = (sig.shift(1) > sig_ma) & tlt_avail
        scale = pd.Series(1.0, index=prices.index)
        scale = scale.where(~bull, self.bull_scale)
        scale = (
            scale.where(bull, scale)
            .where(bull | ~tlt_avail, self.bear_scale)
            .where(bull, self.bull_scale)
        )
        # Simpler: bull → scale up SPY, else scale down
        spy_scaled = (w_spy * self.bull_scale).where(bull, w_spy * self.bear_scale)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        spy_final = spy_scaled.where(tlt_avail, 1.0)
        tlt_final = tlt_adj.where(tlt_avail, 0.0)
        return pd.DataFrame({"SPY": spy_final, "TLT": tlt_final}, index=prices.index)


class MarketAccrualTimingPort(MultiAssetStrategy):
    """H482: H021 overlaid with market-wide accrual quality signal.

    Signal: rolling 252d mean of (NI-OCF)/TA across universe (pre-computed).
    High accruals (above 252d trailing mean) → earnings quality risk → SPY × 0.80.
    Low accruals → cash-backed earnings → SPY × 1.20.
    """

    name = "market_accrual_timing_port"

    def __init__(
        self,
        market_accruals: pd.Series,
        vol_lookback: int = 60,
        signal_window: int = 252,
        bull_scale: float = 1.20,
        bear_scale: float = 0.80,
    ) -> None:
        self.market_accruals = market_accruals
        self.vol_lookback = vol_lookback
        self.signal_window = signal_window
        self.bull_scale = bull_scale
        self.bear_scale = bear_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        sig = self.market_accruals.reindex(prices.index).ffill()
        sig_ma = (
            sig.rolling(self.signal_window, min_periods=self.signal_window // 2).mean().shift(1)
        )
        # High accruals (sig > ma) = BAD → bear_scale; low accruals = bull_scale
        high_accruals = (sig.shift(1) > sig_ma) & tlt_avail
        spy_scaled = (w_spy * self.bear_scale).where(high_accruals, w_spy * self.bull_scale)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        spy_final = spy_scaled.where(tlt_avail, 1.0)
        tlt_final = tlt_adj.where(tlt_avail, 0.0)
        return pd.DataFrame({"SPY": spy_final, "TLT": tlt_final}, index=prices.index)


class CSReturnDispersionAAPort(MultiAssetStrategy):
    """H483: H021 overlaid with cross-sectional return dispersion regime.

    Signal: rolling 21d mean of daily CS std of 5d returns (pre-computed).
    Low dispersion (below 252d median) → passive/ETF regime → SPY × 0.75.
    High dispersion → active/fundamental regime → normal H021.
    Pure price signal, full 12-fold coverage.
    """

    name = "cs_return_dispersion_aa_port"

    def __init__(
        self,
        cs_dispersion: pd.Series,
        vol_lookback: int = 60,
        signal_window: int = 252,
        low_disp_scale: float = 0.75,
    ) -> None:
        self.cs_dispersion = cs_dispersion
        self.vol_lookback = vol_lookback
        self.signal_window = signal_window
        self.low_disp_scale = low_disp_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        sig = self.cs_dispersion.reindex(prices.index).ffill()
        sig_median = (
            sig.rolling(self.signal_window, min_periods=self.signal_window // 2).median().shift(1)
        )
        low_disp = (sig.shift(1) < sig_median) & tlt_avail
        spy_scaled = (w_spy * self.low_disp_scale).where(low_disp, w_spy)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        spy_final = spy_scaled.where(tlt_avail, 1.0)
        tlt_final = tlt_adj.where(tlt_avail, 0.0)
        return pd.DataFrame({"SPY": spy_final, "TLT": tlt_final}, index=prices.index)


class QualityBreadthAAPort(MultiAssetStrategy):
    """H484: H021 overlaid with cross-sectional quality improvement breadth.

    Signal: fraction of S&P 500 stocks with GP/TA above own 252d GP/TA median (pre-computed).
    Breadth > 0.60 → broad quality improvement → SPY × 1.20.
    Breadth < 0.40 → quality deterioration → SPY × 0.80.
    """

    name = "quality_breadth_aa_port"

    def __init__(
        self,
        quality_breadth: pd.Series,
        vol_lookback: int = 60,
        bull_thresh: float = 0.60,
        bear_thresh: float = 0.40,
        bull_scale: float = 1.20,
        bear_scale: float = 0.80,
    ) -> None:
        self.quality_breadth = quality_breadth
        self.vol_lookback = vol_lookback
        self.bull_thresh = bull_thresh
        self.bear_thresh = bear_thresh
        self.bull_scale = bull_scale
        self.bear_scale = bear_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        breadth = self.quality_breadth.reindex(prices.index).ffill().shift(1)
        bull = (breadth > self.bull_thresh) & tlt_avail
        bear = (breadth < self.bear_thresh) & tlt_avail
        spy_scaled = w_spy.copy()
        spy_scaled = (w_spy * self.bull_scale).where(bull, spy_scaled)
        spy_scaled = (w_spy * self.bear_scale).where(bear, spy_scaled)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        spy_final = spy_scaled.where(tlt_avail, 1.0)
        tlt_final = tlt_adj.where(tlt_avail, 0.0)
        return pd.DataFrame({"SPY": spy_final, "TLT": tlt_final}, index=prices.index)


class SectorFundamentalRotation(MultiAssetStrategy):
    """H486: H021 overlaid with cross-sector GP/TA quality dispersion signal.

    Signal: rolling 63d cross-sector std of median GP/TA (pre-computed).
    High dispersion (technology/quality sectors dominating) → SPY × 1.10.
    Low dispersion (quality flattening across sectors) → SPY × 0.90.
    """

    name = "sector_fundamental_rotation"

    def __init__(
        self,
        sector_quality_dispersion: pd.Series,
        vol_lookback: int = 60,
        signal_window: int = 252,
        bull_scale: float = 1.10,
        bear_scale: float = 0.90,
    ) -> None:
        self.sector_quality_dispersion = sector_quality_dispersion
        self.vol_lookback = vol_lookback
        self.signal_window = signal_window
        self.bull_scale = bull_scale
        self.bear_scale = bear_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        sig = self.sector_quality_dispersion.reindex(prices.index).ffill()
        sig_ma = (
            sig.rolling(self.signal_window, min_periods=self.signal_window // 2).mean().shift(1)
        )
        high_disp = (sig.shift(1) > sig_ma) & tlt_avail
        spy_scaled = (w_spy * self.bull_scale).where(high_disp, w_spy * self.bear_scale)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        spy_final = spy_scaled.where(tlt_avail, 1.0)
        tlt_final = tlt_adj.where(tlt_avail, 0.0)
        return pd.DataFrame({"SPY": spy_final, "TLT": tlt_final}, index=prices.index)


class MomentumBreadthAAPort(MultiAssetStrategy):
    """H487: H021 overlaid with percentage of S&P 500 stocks above their 200d MA.

    Classic market breadth indicator applied as rule-based AA overlay.
    Breadth > 0.60 → broad market trend → SPY × 1.20.
    Breadth < 0.40 → trend deterioration → SPY × 0.80.
    Pure price signal, full 12-fold coverage.
    """

    name = "momentum_breadth_aa_port"

    def __init__(
        self,
        breadth_signal: pd.Series,
        vol_lookback: int = 60,
        bull_thresh: float = 0.60,
        bear_thresh: float = 0.40,
        bull_scale: float = 1.20,
        bear_scale: float = 0.80,
    ) -> None:
        self.breadth_signal = breadth_signal
        self.vol_lookback = vol_lookback
        self.bull_thresh = bull_thresh
        self.bear_thresh = bear_thresh
        self.bull_scale = bull_scale
        self.bear_scale = bear_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        breadth = self.breadth_signal.reindex(prices.index).ffill().shift(1)
        bull = (breadth > self.bull_thresh) & tlt_avail
        bear = (breadth < self.bear_thresh) & tlt_avail
        spy_scaled = w_spy.copy()
        spy_scaled = (w_spy * self.bull_scale).where(bull, spy_scaled)
        spy_scaled = (w_spy * self.bear_scale).where(bear, spy_scaled)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        spy_final = spy_scaled.where(tlt_avail, 1.0)
        tlt_final = tlt_adj.where(tlt_avail, 0.0)
        return pd.DataFrame({"SPY": spy_final, "TLT": tlt_final}, index=prices.index)


class CreditSpreadContinuousPort(MultiAssetStrategy):
    """H493: H021 with linear SPY-weight scaling from HYG-LQD return spread z-score.

    Signal: trailing 252d HYG return minus LQD return (credit spread proxy).
    Positive z → HYG outperforms → tight credit spreads → equity tilt.
    Negative z → HYG underperforms → wide spreads → defensive.
    Continuous analog of H308/H312 binary credit-spread signals.
    """

    name = "credit_spread_continuous_port"

    def __init__(
        self,
        vol_lookback: int = 60,
        spread_window: int = 252,
        scale_lo: float = 0.60,
        scale_hi: float = 1.40,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.spread_window = spread_window
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()

        hyg_ret = prices["hyg_close"].pct_change(self.spread_window)
        lqd_ret = prices["lqd_close"].pct_change(self.spread_window)
        spread = (hyg_ret - lqd_ret).shift(1)

        spread_mean = spread.rolling(self.spread_window, min_periods=self.spread_window // 2).mean()
        spread_std = spread.rolling(self.spread_window, min_periods=self.spread_window // 2).std()
        z = (spread - spread_mean) / spread_std.replace(0, np.nan)
        z = z.clip(-2.0, 2.0).fillna(0.0)

        mid = (self.scale_lo + self.scale_hi) / 2.0
        half_range = (self.scale_hi - self.scale_lo) / 2.0
        scale = (mid + z * half_range / 2.0).clip(self.scale_lo, self.scale_hi)

        spy_scaled = (w_spy * scale).clip(0.0, 1.0)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        return pd.DataFrame(
            {"SPY": spy_scaled.where(tlt_avail, 1.0), "TLT": tlt_adj.where(tlt_avail, 0.0)},
            index=prices.index,
        )


class MultiplexMacroComposite(MultiAssetStrategy):
    """H494: H021 with SPY-weight scaling from 3-signal continuous macro composite.

    Composite: average of (1) VIX percentile scaler, (2) yield-curve z-score scaler,
    (3) credit-spread z-score scaler. Each maps to [scale_lo, scale_hi]; average applied.
    Avoids binary-threshold whipsaw by keeping all signals continuous.
    """

    name = "multiplex_macro_composite"

    def __init__(
        self,
        vol_lookback: int = 60,
        signal_window: int = 252,
        scale_lo: float = 0.60,
        scale_hi: float = 1.40,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.signal_window = signal_window
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi

    def _compute_scale(self, z: pd.Series) -> pd.Series:
        z = z.clip(-2.0, 2.0).fillna(0.0)
        mid = (self.scale_lo + self.scale_hi) / 2.0
        half_range = (self.scale_hi - self.scale_lo) / 2.0
        return (mid + z * half_range / 2.0).clip(self.scale_lo, self.scale_hi)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        w = self.signal_window
        min_p = w // 2

        # (1) VIX percentile scaler
        vix = prices["vix_close"].shift(1)
        vix_pct = vix.rolling(w, min_periods=min_p).rank(pct=True)
        scale_vix = (
            (1.0 + (0.5 - vix_pct) * (self.scale_hi - self.scale_lo))
            .clip(self.scale_lo, self.scale_hi)
            .fillna(1.0)
        )

        # (2) Yield curve z-score scaler
        tlt_ret = prices["tlt_close"].pct_change(w)
        shy_ret = prices["shy_close"].pct_change(w)
        yc_spread = (tlt_ret - shy_ret).shift(1)
        yc_mean = yc_spread.rolling(w, min_periods=min_p).mean()
        yc_std = yc_spread.rolling(w, min_periods=min_p).std()
        yc_z = (yc_spread - yc_mean) / yc_std.replace(0, np.nan)
        scale_yc = self._compute_scale(yc_z)

        # (3) Credit spread z-score scaler
        hyg_ret = prices["hyg_close"].pct_change(w)
        lqd_ret = prices["lqd_close"].pct_change(w)
        cs_spread = (hyg_ret - lqd_ret).shift(1)
        cs_mean = cs_spread.rolling(w, min_periods=min_p).mean()
        cs_std = cs_spread.rolling(w, min_periods=min_p).std()
        cs_z = (cs_spread - cs_mean) / cs_std.replace(0, np.nan)
        scale_cs = self._compute_scale(cs_z)

        # Average: weight VIX and YC equally; CS falls back to neutral when HYG unavailable
        cs_avail = prices["hyg_close"].notna() & prices["lqd_close"].notna()
        scale_cs_safe = scale_cs.where(cs_avail, 1.0)
        scale = (scale_vix + scale_yc + scale_cs_safe) / 3.0

        spy_scaled = (w_spy * scale).clip(0.0, 1.0)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        return pd.DataFrame(
            {"SPY": spy_scaled.where(tlt_avail, 1.0), "TLT": tlt_adj.where(tlt_avail, 0.0)},
            index=prices.index,
        )
