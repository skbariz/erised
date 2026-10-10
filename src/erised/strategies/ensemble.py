"""Fixed blends and signal combinations of other allocation strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import _h021_weights, _vrp_series
from .base import MultiAssetStrategy


class H001H021DirectBlend(MultiAssetStrategy):
    """H088: 50/50 blend of H001 (MA trend) and H021 (risk parity) daily.

    H001 portion contributes pure SPY exposure (long or flat).
    H021 portion contributes SPY+TLT via inverse-vol weights.
    Combined: w_spy = 0.5*h001 + 0.5*w021_spy; w_tlt = 0.5*w021_tlt.
    Inherits H001 bull participation and H021 crisis diversification.
    """

    name = "h001_h021_direct_blend"

    def __init__(
        self,
        vol_lookback: int = 60,
        fast: int = 50,
        slow: int = 200,
        blend: float = 0.5,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.fast = fast
        self.slow = slow
        self.blend = blend

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy = prices["adj_close"]
        spy_r = spy.pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        # H001: MA50/200 signal (shift 1 to avoid lookahead — MA uses current-day close)
        fast_ma = spy.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = spy.rolling(self.slow, min_periods=self.slow).mean()
        h001_pos = (fast_ma > slow_ma).astype(float).shift(1).fillna(0.0)

        # H021: inverse-vol weights
        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy_h021 = (inv_spy / total).fillna(0.0)
        w_tlt_h021 = (inv_tlt / total).fillna(0.0)

        w_spy = self.blend * h001_pos + (1.0 - self.blend) * w_spy_h021
        w_tlt = (1.0 - self.blend) * w_tlt_h021

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class H043H021RegimeBlend(MultiAssetStrategy):
    """H089: 50/50 blend of H043 (Sell-in-May bond confirm) and H021 (risk parity).

    H043 is long SPY in Nov-Apr when TLT is NOT in uptrend, flat otherwise.
    H021 runs continuously with inverse-vol SPY+TLT weights.
    During May-Oct (H043 flat): portfolio is 50% H021 → 50% cash (natural de-risk).
    Expected MaxDD between H043's -10% and H021's -18%.
    """

    name = "h043_h021_regime_blend"

    def __init__(
        self,
        vol_lookback: int = 60,
        tlt_ma_period: int = 200,
        favorable_months: tuple = (11, 12, 1, 2, 3, 4),
        blend: float = 0.5,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.tlt_ma_period = tlt_ma_period
        self.favorable_months = frozenset(favorable_months)
        self.blend = blend

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        tlt = prices["tlt_close"]
        spy_r = prices["adj_close"].pct_change()
        tlt_r = tlt.pct_change()

        # H043 signal: Nov-Apr AND TLT NOT in uptrend (risk-on filter)
        tlt_ma = tlt.rolling(self.tlt_ma_period, min_periods=self.tlt_ma_period).mean().shift(1)
        in_season = pd.Series(
            prices.index.month.isin(self.favorable_months).astype(float), index=prices.index
        )
        bond_riskoff = (tlt.shift(1) > tlt_ma).astype(float)
        h043_pos = in_season * (1.0 - bond_riskoff)
        h043_pos = h043_pos.where(tlt_ma.notna(), 0.0)

        # H021: inverse-vol weights
        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_spy + inv_tlt
        w_spy_h021 = (inv_spy / total).fillna(0.0)
        w_tlt_h021 = (inv_tlt / total).fillna(0.0)

        w_spy = self.blend * h043_pos + (1.0 - self.blend) * w_spy_h021
        w_tlt = (1.0 - self.blend) * w_tlt_h021

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class MacroCombinationOverlay(MultiAssetStrategy):
    """H314: Ensemble of 3 macro signals for H021 exposure adjustment.

    Signal scores (+1 bullish, -1 bearish):
      VRP signal: VRP > 0 = +1, ≤ 0 = -1.
      Term spread: TLT 252d > SHY 252d = +1, else -1.
      Credit spread: HYG 63d > LQD 63d = +1, else -1 (0 if unavailable).
    Sum ≥ 2: 1.20× scale; sum = 1: 1.05×; sum = 0: 0.90×; sum ≤ -1: 0.75×.
    """

    name = "macro_combination_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
    ) -> None:
        self.vol_lookback = vol_lookback

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

        scale_arr = np.select(
            [total >= 2, total == 1, total == 0],
            [1.20, 1.05, 0.90],
            default=0.75,
        )
        scale = pd.Series(scale_arr, index=prices.index)
        return pd.DataFrame({"SPY": w_spy * scale, "TLT": w_tlt * scale}, index=prices.index)


class MacroRegimeSectorBlend(MultiAssetStrategy):
    """H425: 3-signal macro regime → sector-blended allocation.

    Signals: (1) VIX < 25, (2) TLT > 63d MA, (3) HYG > HYG.shift(63).
    3 bullish: 80% SPY / 20% TLT.
    2 bullish: 55% SPY / 25% TLT / 10% XLP / 10% XLU.
    0-1 bullish: 30% SPY / 40% TLT / 15% XLP / 15% XLU.
    """

    name = "macro_regime_sector_blend"

    def __init__(
        self,
        vix_threshold: float = 25.0,
        tlt_ma: int = 63,
        credit_lookback: int = 63,
    ) -> None:
        self.vix_threshold = vix_threshold
        self.tlt_ma = tlt_ma
        self.credit_lookback = credit_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        tlt_avail = prices["tlt_close"].notna()
        has_hyg = "hyg_close" in prices.columns
        has_xlp = "xlp_close" in prices.columns
        has_xlu = "xlu_close" in prices.columns
        vix_low = prices["vix_close"] < self.vix_threshold
        tlt_ma_s = prices["tlt_close"].rolling(self.tlt_ma, min_periods=self.tlt_ma // 2).mean()
        tlt_up = (prices["tlt_close"] > tlt_ma_s) & tlt_avail
        if has_hyg:
            hyg_avail = prices["hyg_close"].notna()
            hyg_up = (
                prices["hyg_close"] > prices["hyg_close"].shift(self.credit_lookback)
            ) & hyg_avail
        else:
            hyg_up = pd.Series(False, index=prices.index)
        xlp_avail = prices["xlp_close"].notna() if has_xlp else pd.Series(False, index=prices.index)
        xlu_avail = prices["xlu_close"].notna() if has_xlu else pd.Series(False, index=prices.index)
        can_sector = tlt_avail & xlp_avail & xlu_avail
        n_bull = vix_low.astype(int) + tlt_up.astype(int) + hyg_up.astype(int)
        w_spy = pd.Series(1.0, index=prices.index)
        w_tlt = pd.Series(0.0, index=prices.index)
        w_xlp = pd.Series(0.0, index=prices.index)
        w_xlu = pd.Series(0.0, index=prices.index)
        cond3 = (n_bull == 3) & can_sector
        w_spy = w_spy.where(~cond3, 0.80)
        w_tlt = w_tlt.where(~cond3, 0.20)
        cond2 = (n_bull == 2) & can_sector
        w_spy = w_spy.where(~cond2, 0.55)
        w_tlt = w_tlt.where(~cond2, 0.25)
        w_xlp = w_xlp.where(~cond2, 0.10)
        w_xlu = w_xlu.where(~cond2, 0.10)
        cond01 = (n_bull <= 1) & can_sector
        w_spy = w_spy.where(~cond01, 0.30)
        w_tlt = w_tlt.where(~cond01, 0.40)
        w_xlp = w_xlp.where(~cond01, 0.15)
        w_xlu = w_xlu.where(~cond01, 0.15)
        w_spy = w_spy.where(tlt_avail, 1.0)
        w_tlt = w_tlt.where(tlt_avail, 0.0)
        w_xlp = w_xlp.where(can_sector, 0.0)
        w_xlu = w_xlu.where(can_sector, 0.0)
        return pd.DataFrame(
            {"SPY": w_spy, "TLT": w_tlt, "XLP": w_xlp, "XLU": w_xlu}, index=prices.index
        )
