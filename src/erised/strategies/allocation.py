"""Multi-asset rotation and risk-parity allocation strategies across SPY, bonds, gold, and ETFs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import _h021_base_weights, _h021_weights
from .base import MultiAssetStrategy, Strategy


class SPYTLTDualMomentum(MultiAssetStrategy):
    """H018: Antonacci GEM-style dual momentum across SPY and TLT.

    Hold SPY when its 12m return is positive. If SPY is negative, hold TLT when
    TLT's 12m return is positive. If both are negative, go to cash. The
    absolute-momentum filter (cash when all assets are negative) is the key
    distinguisher from cross-sectional-only rotation.
    """

    def __init__(self, lookback_months: int = 12, skip_months: int = 1):
        self.lookback_days = int(lookback_months * 21)
        self.skip_days = int(skip_months * 21)
        self.name = f"spy_tlt_dual_mom_{lookback_months}m"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy = prices["adj_close"]
        tlt = prices["tlt_close"]

        spy_mom = spy.shift(self.skip_days) / spy.shift(self.lookback_days) - 1
        tlt_mom = tlt.shift(self.skip_days) / tlt.shift(self.lookback_days) - 1

        w_spy = pd.Series(0.0, index=prices.index)
        w_tlt = pd.Series(0.0, index=prices.index)

        spy_positive = spy_mom > 0
        tlt_positive = tlt_mom > 0
        both_valid = spy_mom.notna() & tlt_mom.notna()

        w_spy = w_spy.where(~(both_valid & spy_positive), 1.0)
        w_tlt = w_tlt.where(~(both_valid & ~spy_positive & tlt_positive), 1.0)

        weights = pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)
        return weights


class SPYGLDRotation(MultiAssetStrategy):
    """H019: SPY/GLD rotation — always allocated, no cash.

    Hold SPY when its 12m momentum is positive; hold GLD when SPY momentum is
    negative. Stays fully invested at all times. GLD historically appreciates
    during equity bear markets, particularly in inflationary regimes.
    """

    def __init__(self, lookback_months: int = 12, skip_months: int = 1):
        self.lookback_days = int(lookback_months * 21)
        self.skip_days = int(skip_months * 21)
        self.name = f"spy_gld_rotation_{lookback_months}m"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy = prices["adj_close"]
        spy_mom = spy.shift(self.skip_days) / spy.shift(self.lookback_days) - 1

        w_spy = pd.Series(0.0, index=prices.index)
        w_gld = pd.Series(0.0, index=prices.index)

        valid = spy_mom.notna()
        spy_positive = valid & (spy_mom > 0)
        spy_negative = valid & (spy_mom <= 0)

        w_spy = w_spy.where(~spy_positive, 1.0)
        w_gld = w_gld.where(~spy_negative, 1.0)

        weights = pd.DataFrame({"SPY": w_spy, "GLD": w_gld}, index=prices.index)
        return weights


class ThreeAssetMomentum(MultiAssetStrategy):
    """H020: Best-of-3 absolute momentum across SPY, TLT, GLD.

    Hold the single best 12m performer. Go to cash if the winner's momentum
    is negative (universal absolute-momentum filter). Broadest opportunity
    set in Batch 3 — captures regimes where equities AND bonds decline
    but commodities rise (e.g. inflationary bear markets).
    """

    def __init__(self, lookback_months: int = 12, skip_months: int = 1):
        self.lookback_days = int(lookback_months * 21)
        self.skip_days = int(skip_months * 21)
        self.name = f"three_asset_mom_{lookback_months}m"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        assets = {
            "SPY": prices["adj_close"],
            "TLT": prices["tlt_close"],
            "GLD": prices["gld_close"],
        }
        moms = pd.DataFrame(
            {
                name: px.shift(self.skip_days) / px.shift(self.lookback_days) - 1
                for name, px in assets.items()
            }
        )

        weights = pd.DataFrame(0.0, index=prices.index, columns=list(assets.keys()))
        valid_rows = moms.notna().all(axis=1)

        if valid_rows.any():
            best = pd.Series(np.nan, index=prices.index, dtype=object)
            best_mom = pd.Series(np.nan, index=prices.index)
            best[valid_rows] = moms[valid_rows].idxmax(axis=1)
            best_mom[valid_rows] = moms[valid_rows].max(axis=1)

            for asset in assets:
                weights[asset] = np.where(valid_rows & (best == asset) & (best_mom > 0), 1.0, 0.0)

        return weights


class RiskParity(MultiAssetStrategy):
    """H021: SPY/TLT equal-risk contribution weighting.

    Weights each asset by the inverse of its 60-day realized volatility,
    normalized to sum to 1. Both assets are held simultaneously at all times.
    Diversification benefit: SPY and TLT have historically low/negative
    correlation, so equal-risk allocation approximates minimum-variance.
    """

    def __init__(self, vol_lookback: int = 60):
        self.vol_lookback = vol_lookback
        self.name = f"risk_parity_{vol_lookback}d"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        # Annualized vol; shift 1 bar to prevent lookahead
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

        weights = pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)
        return weights


class DefensiveEquityHedge(MultiAssetStrategy):
    """H022: Trend-following with a bond overlay in downtrends.

    100% SPY when MA(fast, slow) is bullish. Shifts to 50% SPY + 50% TLT when
    the trend turns bearish — maintains equity participation but hedges with
    duration. Avoids the full whipsaw cost of going to cash (H001) or going
    short (H013) during slow downtrends.
    """

    def __init__(self, fast: int = 50, slow: int = 200):
        self.fast = fast
        self.slow = slow
        self.name = f"defensive_hedge_{fast}_{slow}"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        px = prices["adj_close"]
        fast_ma = px.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = px.rolling(self.slow, min_periods=self.slow).mean()

        trend_on = fast_ma > slow_ma
        valid = slow_ma.notna()

        w_spy = pd.Series(0.0, index=prices.index)
        w_tlt = pd.Series(0.0, index=prices.index)

        w_spy = w_spy.where(~(valid & trend_on), 1.0)
        w_spy = w_spy.where(~(valid & ~trend_on), 0.5)
        w_tlt = w_tlt.where(~(valid & ~trend_on), 0.5)

        weights = pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)
        return weights


class BondTrendEquityTimer(Strategy):
    """H033: Long SPY when TLT is below its MA (bonds weak = risk-on).

    Flight-to-safety bond rallies occur when investors fear equity losses.
    TLT trending above its 200MA signals bond demand = risk-off environment.
    TLT below its 200MA = no flight to safety = risk-on.

    Decorrelation thesis: bond trend and equity trend diverge in crises —
    exactly the regime where H001 is most likely to be wrong or late.
    Requires 'tlt_close' column in the prices DataFrame.
    """

    def __init__(self, ma_period: int = 200):
        self.ma_period = ma_period
        self.name = f"bond_trend_equity_{ma_period}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "tlt_close" not in prices.columns:
            raise ValueError("BondTrendEquityTimer requires 'tlt_close' column")
        tlt = prices["tlt_close"]
        tlt_ma = tlt.rolling(self.ma_period, min_periods=self.ma_period).mean()
        pos = (tlt < tlt_ma).astype(float)
        pos = pos.where(tlt_ma.notna(), 0.0)
        pos.name = "position"
        return pos


class SPYTLTCorrelationRegime(Strategy):
    """H034: Long SPY only when rolling SPY-TLT correlation is negative.

    In normal markets, equities and bonds have negative return correlation
    (flight-to-safety effect). During crisis contagion, both assets fall
    together and correlation flips positive. Flat when correlation is
    positive avoids the regime where diversification fails entirely.

    Decorrelation thesis: this signal uses cross-asset covariance structure,
    not price levels. It turns off at exactly the time H001 gets hurt most.
    Requires 'tlt_close' column in the prices DataFrame.
    """

    def __init__(self, corr_window: int = 60):
        self.corr_window = corr_window
        self.name = f"spy_tlt_corr_regime_{corr_window}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "tlt_close" not in prices.columns:
            raise ValueError("SPYTLTCorrelationRegime requires 'tlt_close' column")
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        rolling_corr = spy_r.rolling(self.corr_window, min_periods=self.corr_window).corr(tlt_r)
        pos = (rolling_corr < 0).astype(float)
        pos = pos.where(rolling_corr.notna(), 0.0)
        pos.name = "position"
        return pos


class GoldFearGauge(Strategy):
    """H035: Long SPY when GLD is below its MA (no panic bid for gold).

    Gold outperformance signals fear and inflation anxiety that historically
    precede equity weakness. GLD trending above its MA = investors seeking
    the gold safe haven = risk-off environment. GLD below MA = no fear
    premium in the gold market = risk-on signal for equities.

    Requires 'gld_close' column in the prices DataFrame.
    """

    def __init__(self, ma_period: int = 50):
        self.ma_period = ma_period
        self.name = f"gold_fear_gauge_{ma_period}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "gld_close" not in prices.columns:
            raise ValueError("GoldFearGauge requires 'gld_close' column")
        gld = prices["gld_close"]
        gld_ma = gld.rolling(self.ma_period, min_periods=self.ma_period).mean()
        pos = (gld < gld_ma).astype(float)
        pos = pos.where(gld_ma.notna(), 0.0)
        pos.name = "position"
        return pos


class EquityVsSafeHavenMomentum(Strategy):
    """H036: Long SPY when SPY 3m return > average of TLT and GLD 3m returns.

    Risk-on regime: equity outperforms safe havens. Risk-off: safe havens
    outperform equity. This relative momentum signal captures regime shifts
    earlier than absolute momentum because safe-haven outperformance often
    precedes equity declines by days to weeks.

    Decorrelation thesis: the signal is relative (SPY vs TLT/GLD), not
    absolute SPY trend. In bull markets both agree; in transitions, this
    signal flips before H001's MA crossover fires.
    Requires 'tlt_close' and 'gld_close' columns in the prices DataFrame.
    """

    def __init__(self, lookback_days: int = 63):
        self.lookback_days = lookback_days
        self.name = f"equity_vs_safe_haven_{lookback_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "tlt_close" not in prices.columns or "gld_close" not in prices.columns:
            raise ValueError(
                "EquityVsSafeHavenMomentum requires 'tlt_close' and 'gld_close' columns"
            )
        spy_mom = prices["adj_close"] / prices["adj_close"].shift(self.lookback_days) - 1
        tlt_mom = prices["tlt_close"] / prices["tlt_close"].shift(self.lookback_days) - 1
        gld_mom = prices["gld_close"] / prices["gld_close"].shift(self.lookback_days) - 1
        safe_haven_avg = (tlt_mom + gld_mom) / 2
        valid = spy_mom.notna() & tlt_mom.notna() & gld_mom.notna()
        pos = (spy_mom > safe_haven_avg).astype(float)
        pos = pos.where(valid, 0.0)
        pos.name = "position"
        return pos


class DualSafeHavenConfirmation(Strategy):
    """H037: Long SPY only when BOTH TLT and GLD are below their 200MAs.

    Maximum risk-on confirmation: no flight to either bonds or gold.
    When either safe haven is trending up, equity exposure is reduced
    to zero. The dual-filter is intentionally restrictive — it targets
    only the highest-quality risk-on windows where both safe havens
    are dormant, structurally avoiding the crises that hurt all trend
    strategies simultaneously.

    Requires 'tlt_close' and 'gld_close' columns in the prices DataFrame.
    """

    def __init__(self, ma_period: int = 200):
        self.ma_period = ma_period
        self.name = f"dual_safe_haven_{ma_period}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "tlt_close" not in prices.columns or "gld_close" not in prices.columns:
            raise ValueError(
                "DualSafeHavenConfirmation requires 'tlt_close' and 'gld_close' columns"
            )
        tlt = prices["tlt_close"]
        gld = prices["gld_close"]
        tlt_ma = tlt.rolling(self.ma_period, min_periods=self.ma_period).mean()
        gld_ma = gld.rolling(self.ma_period, min_periods=self.ma_period).mean()
        valid = tlt_ma.notna() & gld_ma.notna()
        pos = ((tlt < tlt_ma) & (gld < gld_ma)).astype(float)
        pos = pos.where(valid, 0.0)
        pos.name = "position"
        return pos


class ThreeAssetRiskParity(MultiAssetStrategy):
    """H051: Vol-equalized risk parity across SPY, TLT, and GLD.

    Extends H021 (SPY+TLT) by adding GLD as a third asset. Each asset
    receives weight = (1/realized_vol) / sum(1/realized_vol), normalized
    to sum to 1.0. Daily rebalancing. Falls back to SPY+TLT when GLD
    data is unavailable (folds 0-2, pre-2004).

    Thesis: GLD diversifies a distinct regime — inflation/stagflation —
    where SPY and TLT both suffer but gold holds. SPY+TLT handles
    deflation crises; adding GLD covers inflationary crises.

    Requires: tlt_close, gld_close columns.
    """

    name = "three_asset_risk_parity"

    def __init__(self, vol_lookback: int = 60):
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = (
            prices["tlt_close"].pct_change()
            if "tlt_close" in prices.columns
            else pd.Series(np.nan, index=prices.index)
        )
        gld_r = (
            prices["gld_close"].pct_change()
            if "gld_close" in prices.columns
            else pd.Series(np.nan, index=prices.index)
        )

        def _inv_vol(r: pd.Series) -> pd.Series:
            v = r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
                1
            ) * np.sqrt(252)
            return (1.0 / v.replace(0, np.nan)).fillna(0.0)

        inv_spy = _inv_vol(spy_r)
        inv_tlt = _inv_vol(tlt_r)
        inv_gld = _inv_vol(gld_r)

        # Zero out assets whose price series is missing
        if "tlt_close" not in prices.columns:
            inv_tlt = pd.Series(0.0, index=prices.index)
        if "gld_close" not in prices.columns:
            inv_gld = pd.Series(0.0, index=prices.index)

        # When GLD NaN (pre-2004), treat as 0 weight → falls back to SPY+TLT parity
        inv_gld = inv_gld.where(
            prices.get("gld_close", pd.Series(np.nan, index=prices.index)).notna(), 0.0
        )

        total = inv_spy + inv_tlt + inv_gld
        valid = total > 0

        w_spy = (inv_spy / total).where(valid, 0.0)
        w_tlt = (inv_tlt / total).where(valid, 0.0)
        w_gld = (inv_gld / total).where(valid, 0.0)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt, "GLD": w_gld}, index=prices.index)


class YieldCurveProxy(Strategy):
    """H054: Long SPY when the yield curve is not inverted (SHY not outperforming TLT).

    The yield curve slope (short vs. long rates) is the classic recession
    predictor. Proxied via ETF relative performance: when SHY (1-3yr Treasury)
    outperforms TLT (20yr Treasury) on a 3-month basis, short rates are rising
    relative to long rates — a flattening/inverting yield curve signalling
    recession risk.

    Long SPY when TLT 3m return >= SHY 3m return (curve positive or steepening).
    Flat when SHY outperforms (curve inverted). Defaults to long SPY when data
    unavailable.

    Requires: shy_close column. tlt_close column used if available.
    """

    def __init__(self, mom_days: int = 63):
        self.mom_days = mom_days
        self.name = f"yield_curve_proxy_{mom_days}d"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        if "shy_close" not in prices.columns:
            return pd.Series(1.0, index=prices.index, name="position")
        shy = prices["shy_close"]
        tlt = prices["tlt_close"] if "tlt_close" in prices.columns else shy
        shy_mom = shy / shy.shift(self.mom_days) - 1
        tlt_mom = tlt / tlt.shift(self.mom_days) - 1
        valid = shy_mom.notna() & tlt_mom.notna()
        # Curve NOT inverted when TLT return >= SHY return (long duration not lagging short)
        curve_ok = tlt_mom >= shy_mom
        pos = curve_ok.astype(float)
        pos = pos.where(valid, 0.0)
        pos.name = "position"
        return pos


class FiveAssetRotation(MultiAssetStrategy):
    """H055: Cross-sectional momentum rotation across SPY, TLT, GLD, DBC, VNQ.

    Ranks all available assets by 12-1m momentum. Holds the top-ranked asset
    if its momentum is positive; goes to cash if the top performer has negative
    momentum (absolute momentum filter). Broadens H020 (3-asset) to 5 assets,
    adding commodities (DBC) and real estate (VNQ).

    Requires: tlt_close, gld_close, dbc_close, vnq_close columns.
    Assets with unavailable data are excluded from ranking for that period.
    """

    name = "five_asset_rotation"

    def __init__(self, long_days: int = 252, skip_days: int = 21):
        self.long_days = long_days
        self.skip_days = skip_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {
            "SPY": "adj_close",
            "TLT": "tlt_close",
            "GLD": "gld_close",
            "DBC": "dbc_close",
            "VNQ": "vnq_close",
        }
        # Only include assets whose column exists
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        moms = pd.DataFrame(index=prices.index)
        for asset, col in available.items():
            px = prices[col]
            moms[asset] = px.shift(self.skip_days) / px.shift(self.long_days) - 1

        weights = pd.DataFrame(0.0, index=prices.index, columns=list(available.keys()))
        valid_rows = moms.notna().any(axis=1)

        for i in range(len(prices)):
            if not valid_rows.iloc[i]:
                continue
            day_mom = moms.iloc[i].dropna()
            if day_mom.empty:
                continue
            best = day_mom.idxmax()
            if day_mom[best] > 0:
                weights.at[weights.index[i], best] = 1.0

        return weights


class YieldCurveRotation(MultiAssetStrategy):
    """H056: Rotate SPY→TLT on yield curve inversion (SHY outperforms TLT).

    Normal curve (TLT outperforms SHY): long SPY.
    Inverted curve (SHY outperforms TLT): long TLT (flight-to-safety).
    Before SHY/TLT data: flat.
    """

    name = "yield_curve_rotation"

    def __init__(self, mom_days: int = 63) -> None:
        self.mom_days = mom_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt = prices.get("tlt_close", pd.Series(np.nan, index=prices.index))

        shy_r = shy.shift(1) / shy.shift(1 + self.mom_days) - 1
        tlt_r = tlt.shift(1) / tlt.shift(1 + self.mom_days) - 1

        valid = shy_r.notna() & tlt_r.notna()
        inverted = valid & (shy_r > tlt_r)

        w_spy = pd.Series(0.0, index=prices.index)
        w_tlt = pd.Series(0.0, index=prices.index)
        w_spy[valid & ~inverted] = 1.0
        w_tlt[inverted] = 1.0

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class BondCreditDualConfirm(Strategy):
    """H057: Long SPY when TLT below 200d MA AND HYG outperforms LQD.

    Falls back to TLT-only filter when HYG/LQD data unavailable (pre-2007).
    Both signals together block equity exposure during 2008-style crises.
    """

    def __init__(self, tlt_ma: int = 200, credit_days: int = 63) -> None:
        self.tlt_ma = tlt_ma
        self.credit_days = credit_days

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        tlt = prices.get("tlt_close", pd.Series(np.nan, index=prices.index))
        hyg = prices.get("hyg_close", pd.Series(np.nan, index=prices.index))
        lqd = prices.get("lqd_close", pd.Series(np.nan, index=prices.index))

        tlt_ma = tlt.shift(1).rolling(self.tlt_ma).mean()
        tlt_signal = tlt.shift(1) < tlt_ma

        hyg_r = hyg.shift(1) / hyg.shift(1 + self.credit_days) - 1
        lqd_r = lqd.shift(1) / lqd.shift(1 + self.credit_days) - 1
        credit_ok = hyg_r >= lqd_r

        has_credit = hyg_r.notna() & lqd_r.notna()

        pos = pd.Series(0.0, index=prices.index)
        pos[tlt_signal & (~has_credit)] = 1.0
        pos[tlt_signal & has_credit & credit_ok] = 1.0
        return pos


class InvertedCurveTreasuryLadder(MultiAssetStrategy):
    """H058: 75% TLT + 25% IEF when yield curve inverted for 21+ days.

    Sustained inversion (> sustain_days) signals a genuine recession risk.
    Brief inversions (<= sustain_days) are noise — hold SPY.
    Normal curve: 100% SPY.
    """

    name = "inverted_curve_treasury_ladder"

    def __init__(self, mom_days: int = 63, sustain_days: int = 21) -> None:
        self.mom_days = mom_days
        self.sustain_days = sustain_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt = prices.get("tlt_close", pd.Series(np.nan, index=prices.index))
        ief = prices.get("ief_close", pd.Series(np.nan, index=prices.index))

        shy_r = shy.shift(1) / shy.shift(1 + self.mom_days) - 1
        tlt_r = tlt.shift(1) / tlt.shift(1 + self.mom_days) - 1

        valid = shy_r.notna() & tlt_r.notna()
        raw_inv = valid & (shy_r > tlt_r)

        # Count consecutive inversion days
        inv_arr = raw_inv.to_numpy()
        consec = np.zeros(len(inv_arr), dtype=int)
        for i in range(1, len(inv_arr)):
            consec[i] = (consec[i - 1] + 1) if inv_arr[i] else 0
        sustained = pd.Series(consec >= self.sustain_days, index=prices.index)

        has_ief = ief.notna()

        w_spy = pd.Series(0.0, index=prices.index)
        w_tlt = pd.Series(0.0, index=prices.index)
        w_ief = pd.Series(0.0, index=prices.index)

        w_spy[valid & ~sustained] = 1.0
        w_tlt[sustained & has_ief] = 0.75
        w_ief[sustained & has_ief] = 0.25
        w_tlt[sustained & ~has_ief] = 1.0

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt, "IEF": w_ief}, index=prices.index)


class CarryMomentumSPYTLT(MultiAssetStrategy):
    """H060: SPY/TLT/flat with yield curve + SPY trend gate.

    Curve steep (TLT > SHY on mom_days) AND SPY above spy_ma → 100% SPY.
    Curve inverted (SHY > TLT on mom_days) → 100% TLT (defensive carry).
    Otherwise (mixed signals or pre-data) → flat.
    """

    name = "carry_momentum_spy_tlt"

    def __init__(self, mom_days: int = 63, spy_ma: int = 200) -> None:
        self.mom_days = mom_days
        self.spy_ma = spy_ma

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt = prices.get("tlt_close", pd.Series(np.nan, index=prices.index))

        shy_r = shy.shift(1) / shy.shift(1 + self.mom_days) - 1
        tlt_r = tlt.shift(1) / tlt.shift(1 + self.mom_days) - 1
        spy_ma = prices["adj_close"].shift(1).rolling(self.spy_ma).mean()

        valid = shy_r.notna() & tlt_r.notna()
        inverted = valid & (shy_r > tlt_r)
        spy_trend_up = prices["adj_close"].shift(1) > spy_ma

        w_spy = pd.Series(0.0, index=prices.index)
        w_tlt = pd.Series(0.0, index=prices.index)

        w_spy[valid & ~inverted & spy_trend_up] = 1.0
        w_tlt[inverted] = 1.0

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class ValueGrowthRotation(MultiAssetStrategy):
    """H061: Rotate IVW/IVE/SPY based on yield curve + SPY trend.

    Normal curve (TLT > SHY mom) AND SPY uptrend → IVW (growth).
    Inverted curve (SHY > TLT mom) → IVE (value/defensive).
    Otherwise (no data or mixed) → SPY (baseline).
    """

    name = "value_growth_rotation"

    def __init__(self, mom_days: int = 63, spy_ma: int = 200) -> None:
        self.mom_days = mom_days
        self.spy_ma = spy_ma

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt = prices.get("tlt_close", pd.Series(np.nan, index=prices.index))
        ive = prices.get("ive_close", pd.Series(np.nan, index=prices.index))
        ivw = prices.get("ivw_close", pd.Series(np.nan, index=prices.index))

        shy_r = shy.shift(1) / shy.shift(1 + self.mom_days) - 1
        tlt_r = tlt.shift(1) / tlt.shift(1 + self.mom_days) - 1
        spy_ma = prices["adj_close"].shift(1).rolling(self.spy_ma).mean()

        curve_valid = shy_r.notna() & tlt_r.notna()
        inverted = curve_valid & (shy_r > tlt_r)
        spy_up = prices["adj_close"].shift(1) > spy_ma

        has_ivw = ivw.notna()
        has_ive = ive.notna()

        w_spy = pd.Series(1.0, index=prices.index)
        w_ivw = pd.Series(0.0, index=prices.index)
        w_ive = pd.Series(0.0, index=prices.index)

        # Normal curve + SPY uptrend → IVW (if available, else SPY)
        growth_regime = curve_valid & ~inverted & spy_up
        w_spy[growth_regime & has_ivw] = 0.0
        w_ivw[growth_regime & has_ivw] = 1.0

        # Inverted curve → IVE (if available, else SPY)
        w_spy[inverted & has_ive] = 0.0
        w_ive[inverted & has_ive] = 1.0

        return pd.DataFrame({"SPY": w_spy, "IVW": w_ivw, "IVE": w_ive}, index=prices.index)


class TechDefensiveMomentum(MultiAssetStrategy):
    """H064: 12-1m cross-sectional momentum between QQQ (tech) and XLU (defensive).

    Long QQQ if QQQ momentum > XLU momentum and QQQ > 0.
    Long XLU if XLU momentum > QQQ momentum and XLU > 0.
    Both negative → long SPY (neutral).
    """

    name = "tech_defensive_momentum"

    def __init__(self, long_days: int = 252, skip_days: int = 21) -> None:
        self.long_days = long_days
        self.skip_days = skip_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        qqq = prices.get("qqq_close", pd.Series(np.nan, index=prices.index))
        xlu = prices.get("xlu_close", pd.Series(np.nan, index=prices.index))

        qqq_mom = qqq.shift(1 + self.skip_days) / qqq.shift(1 + self.long_days) - 1
        xlu_mom = xlu.shift(1 + self.skip_days) / xlu.shift(1 + self.long_days) - 1
        spy_mom = (
            prices["adj_close"].shift(1 + self.skip_days)
            / prices["adj_close"].shift(1 + self.long_days)
            - 1
        )

        w_spy = pd.Series(0.0, index=prices.index)
        w_qqq = pd.Series(0.0, index=prices.index)
        w_xlu = pd.Series(0.0, index=prices.index)

        both_valid = qqq_mom.notna() & xlu_mom.notna()
        qqq_leads = both_valid & (qqq_mom >= xlu_mom) & (qqq_mom > 0)
        xlu_leads = both_valid & (xlu_mom > qqq_mom) & (xlu_mom > 0)
        both_neg = both_valid & (qqq_mom <= 0) & (xlu_mom <= 0)

        w_qqq[qqq_leads] = 1.0
        w_xlu[xlu_leads] = 1.0
        # Both negative: fallback to SPY if SPY positive, else flat
        w_spy[both_neg & (spy_mom > 0)] = 1.0

        return pd.DataFrame({"SPY": w_spy, "QQQ": w_qqq, "XLU": w_xlu}, index=prices.index)


class FourAssetRiskParity(MultiAssetStrategy):
    """H076: SPY+TLT+GLD+DBC inverse-vol risk parity (full macro quadrant).

    SPY (growth), TLT (deflation), GLD (stagflation/fear), DBC (inflation).
    Weights each available asset by inverse realized vol, normalized to 1.
    Gracefully degrades to fewer assets when data unavailable.
    """

    name = "four_asset_risk_parity"

    def __init__(self, vol_lookback: int = 60) -> None:
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {
            "SPY": "adj_close",
            "TLT": "tlt_close",
            "GLD": "gld_close",
            "DBC": "dbc_close",
        }
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        vols: dict[str, pd.Series] = {}
        for asset, col in available.items():
            r = prices[col].pct_change()
            vols[asset] = r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
                1
            ) * np.sqrt(252)

        inv_vols = {a: (1.0 / v.replace(0, np.nan)) for a, v in vols.items()}
        total = sum(inv_vols.values())
        weights = pd.DataFrame(index=prices.index, columns=list(available.keys()), dtype=float)
        for asset in available:
            weights[asset] = (inv_vols[asset] / total).fillna(0.0)

        return weights


class RiskParityMomentumFilter(MultiAssetStrategy):
    """H077: H021 risk parity with per-asset 6m momentum filter.

    Each asset's weight is zeroed if its 6m return is negative; the freed
    risk budget is reallocated to remaining positive-momentum assets.
    If all assets have negative 6m momentum, goes to cash (all weights zero).
    """

    name = "risk_parity_momentum_filter"

    def __init__(self, vol_lookback: int = 60, mom_days: int = 126) -> None:
        self.vol_lookback = vol_lookback
        self.mom_days = mom_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        spy_mom = prices["adj_close"].shift(1) / prices["adj_close"].shift(1 + self.mom_days) - 1
        tlt_mom = prices["tlt_close"].shift(1) / prices["tlt_close"].shift(1 + self.mom_days) - 1

        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)

        # Zero out assets with negative or unavailable momentum
        spy_ok = (spy_mom > 0) & spy_mom.notna()
        tlt_ok = (tlt_mom > 0) & tlt_mom.notna()
        inv_spy_filtered = inv_spy.where(spy_ok, 0.0)
        inv_tlt_filtered = inv_tlt.where(tlt_ok, 0.0)

        total = inv_spy_filtered + inv_tlt_filtered
        w_spy = (inv_spy_filtered / total.replace(0, np.nan)).fillna(0.0)
        w_tlt = (inv_tlt_filtered / total.replace(0, np.nan)).fillna(0.0)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class CrisisProtocolRiskParity(MultiAssetStrategy):
    """H079: H021 risk parity with VIX+drawdown emergency override.

    Normal: inverse-vol weights (identical to H021).
    Crisis: when SPY drawdown from 20d high < crisis_dd AND VIX > vix_spike,
    override to crisis_tlt% TLT + (1-crisis_tlt)% SPY.
    Resume normal when drawdown recovers past resume_dd AND VIX < vix_resume.
    Uses hysteresis to prevent rapid toggling.
    """

    name = "crisis_protocol_risk_parity"

    def __init__(
        self,
        vol_lookback: int = 60,
        drawdown_window: int = 20,
        crisis_dd: float = -0.10,
        resume_dd: float = -0.07,
        vix_spike: float = 35.0,
        vix_resume: float = 30.0,
        crisis_tlt: float = 0.90,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.drawdown_window = drawdown_window
        self.crisis_dd = crisis_dd
        self.resume_dd = resume_dd
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

        # SPY drawdown from rolling high (shift 1 to avoid lookahead)
        spy_px = prices["adj_close"].shift(1)
        rolling_high = spy_px.rolling(self.drawdown_window, min_periods=1).max()
        drawdown = (spy_px / rolling_high - 1).fillna(0.0)
        vix = prices["vix_close"].shift(1)

        w_spy = w_spy_rp.copy()
        w_tlt = w_tlt_rp.copy()

        in_crisis = False
        for i in range(len(prices)):
            dd = drawdown.iloc[i]
            v = vix.iloc[i] if not np.isnan(vix.iloc[i]) else 0.0
            if not in_crisis and dd < self.crisis_dd and v > self.vix_spike:
                in_crisis = True
            elif in_crisis and dd > self.resume_dd and v < self.vix_resume:
                in_crisis = False
            if in_crisis:
                w_spy.iloc[i] = 1.0 - self.crisis_tlt
                w_tlt.iloc[i] = self.crisis_tlt

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class RiskParityYieldCurveOverlay(MultiAssetStrategy):
    """H080: H021 risk parity with yield-curve inversion overlay.

    Normal: H021 standard inverse-vol weights.
    Curve inverted (SHY curve_days return > TLT curve_days return):
    multiply SPY weight by spy_scale (0.50) and redistribute freed weight
    to TLT. Adds macro recession awareness on top of vol-based allocation.
    """

    name = "risk_parity_yield_curve_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
        curve_days: int = 63,
        spy_scale: float = 0.50,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.curve_days = curve_days
        self.spy_scale = spy_scale

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

        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt_px = prices["tlt_close"]
        shy_ret = shy.shift(1) / shy.shift(1 + self.curve_days) - 1
        tlt_ret = tlt_px.shift(1) / tlt_px.shift(1 + self.curve_days) - 1
        inverted = (shy_ret > tlt_ret) & shy_ret.notna() & tlt_ret.notna()

        # When inverted: scale SPY weight down, redistribute to TLT
        spy_cut = w_spy * (1.0 - self.spy_scale)
        w_spy_adj = w_spy.where(~inverted, w_spy - spy_cut)
        w_tlt_adj = w_tlt.where(~inverted, w_tlt + spy_cut)

        return pd.DataFrame({"SPY": w_spy_adj, "TLT": w_tlt_adj}, index=prices.index)


class RiskAdjustedMomentum(MultiAssetStrategy):
    """H082: Long the SPY/TLT/GLD asset with highest (12-1m return / vol) score.

    Risk-adjusts cross-sectional momentum to avoid picking high-vol winners that
    have outsized drawdowns. Normalizes each asset's 12-1m return by its realized
    vol over the same period. Flat if all scores non-positive.
    """

    name = "risk_adjusted_momentum"

    def __init__(self, mom_days: int = 252, skip_days: int = 21, vol_days: int = 252) -> None:
        self.mom_days = mom_days
        self.skip_days = skip_days
        self.vol_days = vol_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {"SPY": "adj_close", "TLT": "tlt_close", "GLD": "gld_close"}
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        scores: dict[str, pd.Series] = {}
        for asset, col in available.items():
            px = prices[col]
            ret = px.pct_change()
            mom = px.shift(1 + self.skip_days) / px.shift(1 + self.mom_days) - 1
            vol = ret.rolling(self.vol_days, min_periods=self.vol_days).std().shift(1) * np.sqrt(
                252
            )
            scores[asset] = (mom / vol.replace(0, np.nan)).shift(0)

        score_df = pd.DataFrame(scores, index=prices.index)
        weights = pd.DataFrame(0.0, index=prices.index, columns=list(available.keys()))
        for i in range(len(prices)):
            row = score_df.iloc[i]
            valid = row.dropna()
            if valid.empty or valid.max() <= 0:
                continue
            winner = valid.idxmax()
            weights.loc[prices.index[i], winner] = 1.0

        return weights


class ContrarianMultiAssetRotation(MultiAssetStrategy):
    """H083: Long the WORST 12-1m performer among SPY/TLT/GLD if its 3m trend reversed.

    De Bondt-Thaler long-term reversal applied to ETFs: the worst-performing asset
    over the prior year is a long candidate, but only when its 3m return turns positive
    (reversal has started — avoids falling knives).
    """

    name = "contrarian_multi_asset_rotation"

    def __init__(self, mom_days: int = 252, skip_days: int = 21, confirm_days: int = 63) -> None:
        self.mom_days = mom_days
        self.skip_days = skip_days
        self.confirm_days = confirm_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {"SPY": "adj_close", "TLT": "tlt_close", "GLD": "gld_close"}
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        moms: dict[str, pd.Series] = {}
        confirms: dict[str, pd.Series] = {}
        for asset, col in available.items():
            px = prices[col]
            moms[asset] = px.shift(1 + self.skip_days) / px.shift(1 + self.mom_days) - 1
            confirms[asset] = px.shift(1) / px.shift(1 + self.confirm_days) - 1

        mom_df = pd.DataFrame(moms, index=prices.index)
        confirm_df = pd.DataFrame(confirms, index=prices.index)
        weights = pd.DataFrame(0.0, index=prices.index, columns=list(available.keys()))

        for i in range(len(prices)):
            m_row = mom_df.iloc[i].dropna()
            c_row = confirm_df.iloc[i].dropna()
            common = m_row.index.intersection(c_row.index)
            if len(common) < 2:
                continue
            loser = m_row[common].idxmin()
            if c_row[loser] > 0:
                weights.loc[prices.index[i], loser] = 1.0

        return weights


class MomentumCarryBlend(MultiAssetStrategy):
    """H084: 50% cross-sectional momentum + 50% carry (term premium proxy).

    Momentum sub-signal: 100% in best 12-1m return asset (SPY/TLT/GLD) if positive.
    Carry sub-signal: 100% TLT if TLT 12m return > SHY 12m return (positive term premium),
    else 100% SPY. Average the two sub-portfolios 50/50.
    """

    name = "momentum_carry_blend"

    def __init__(
        self, mom_days: int = 252, skip_days: int = 21, carry_days: int = 252, blend: float = 0.5
    ) -> None:
        self.mom_days = mom_days
        self.skip_days = skip_days
        self.carry_days = carry_days
        self.blend = blend

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {"SPY": "adj_close", "TLT": "tlt_close", "GLD": "gld_close"}
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        # Momentum sub-signal: top 12-1m return asset (if positive)
        moms: dict[str, pd.Series] = {}
        for asset, col in available.items():
            px = prices[col]
            moms[asset] = px.shift(1 + self.skip_days) / px.shift(1 + self.mom_days) - 1
        mom_df = pd.DataFrame(moms, index=prices.index)

        # Carry sub-signal: TLT if term premium positive (TLT return > SHY return)
        tlt_ret = prices["tlt_close"].shift(1) / prices["tlt_close"].shift(1 + self.carry_days) - 1
        shy_px = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        shy_ret = shy_px.shift(1) / shy_px.shift(1 + self.carry_days) - 1
        term_premium = (tlt_ret > shy_ret) & tlt_ret.notna() & shy_ret.notna()

        cols = list(available.keys())
        w_mom = pd.DataFrame(0.0, index=prices.index, columns=cols)
        w_carry = pd.DataFrame(0.0, index=prices.index, columns=cols)

        idx_val = prices.index
        for i in range(len(prices)):
            row = mom_df.iloc[i].dropna()
            if not row.empty and row.max() > 0:
                w_mom.loc[idx_val[i], row.idxmax()] = 1.0
            # Carry: TLT if term premium, else SPY (when both available)
            if (
                "TLT" in available
                and "SPY" in available
                and pd.notna(tlt_ret.iloc[i])
                and pd.notna(shy_ret.iloc[i])
            ):
                if term_premium.iloc[i]:
                    w_carry.loc[idx_val[i], "TLT"] = 1.0
                else:
                    w_carry.loc[idx_val[i], "SPY"] = 1.0

        return self.blend * w_mom + (1.0 - self.blend) * w_carry


class MomentumMacroGate(MultiAssetStrategy):
    """H085: Cross-sectional momentum gated by yield curve inversion.

    Normally: long the top 12-1m return asset among available SPY/TLT/GLD/DBC/VNQ.
    During yield curve inversion (SHY 3m return > TLT 3m return): force 100% TLT
    regardless of momentum ranking (macro override).
    """

    name = "momentum_macro_gate"

    def __init__(self, mom_days: int = 252, skip_days: int = 21, curve_days: int = 63) -> None:
        self.mom_days = mom_days
        self.skip_days = skip_days
        self.curve_days = curve_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {
            "SPY": "adj_close",
            "TLT": "tlt_close",
            "GLD": "gld_close",
            "DBC": "dbc_close",
            "VNQ": "vnq_close",
        }
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        moms: dict[str, pd.Series] = {}
        for asset, col in available.items():
            px = prices[col]
            moms[asset] = px.shift(1 + self.skip_days) / px.shift(1 + self.mom_days) - 1
        mom_df = pd.DataFrame(moms, index=prices.index)

        tlt_px = prices["tlt_close"]
        shy_px = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt_curve = tlt_px.shift(1) / tlt_px.shift(1 + self.curve_days) - 1
        shy_curve = shy_px.shift(1) / shy_px.shift(1 + self.curve_days) - 1
        inverted = (shy_curve > tlt_curve) & shy_curve.notna() & tlt_curve.notna()

        cols = list(available.keys())
        weights = pd.DataFrame(0.0, index=prices.index, columns=cols)

        idx_val = prices.index
        for i in range(len(prices)):
            if inverted.iloc[i] and "TLT" in available:
                weights.loc[idx_val[i], "TLT"] = 1.0
                continue
            row = mom_df.iloc[i].dropna()
            if row.empty or row.max() <= 0:
                continue
            weights.loc[idx_val[i], row.idxmax()] = 1.0

        return weights


class MonthlyRebalancingRiskParity(MultiAssetStrategy):
    """H090: H021 risk parity with monthly (not daily) weight recalculation.

    Computes H021 inverse-vol weights on the first trading day of each calendar
    month and holds them constant until the next month-start. Reduces turnover
    ~20× vs daily H021 (from ~4845 to ~240 rebalances/year) at the cost of
    slower adaptation to volatility regime changes.
    """

    name = "monthly_rebalancing_risk_parity"

    def __init__(self, vol_lookback: int = 60) -> None:
        self.vol_lookback = vol_lookback

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
        w_spy_daily = (inv_spy / total).fillna(0.0)
        w_tlt_daily = (inv_tlt / total).fillna(0.0)

        # Identify first trading day of each calendar month
        idx = prices.index
        month_starts = set()
        seen: set = set()
        for d in idx:
            key = (d.year, d.month)
            if key not in seen:
                seen.add(key)
                month_starts.add(d)

        # Keep weights only at month-starts, forward-fill all other days
        w_spy_m = w_spy_daily.copy().where(pd.Series(idx.isin(month_starts), index=idx), np.nan)
        w_tlt_m = w_tlt_daily.copy().where(pd.Series(idx.isin(month_starts), index=idx), np.nan)
        w_spy_m = w_spy_m.ffill().fillna(0.0)
        w_tlt_m = w_tlt_m.ffill().fillna(0.0)

        return pd.DataFrame({"SPY": w_spy_m, "TLT": w_tlt_m}, index=prices.index)


class PCARiskRegime(MultiAssetStrategy):
    """H096: SPY-TLT-GLD principal component regime.

    Fits rolling PCA to identify the dominant risk factor across SPY, TLT, GLD.
    PC1 (forced positive SPY loading) captures risk-on/risk-off sentiment.
    Signal = projected 21-day return onto PC1; positive → long SPY, negative → long TLT.
    """

    name = "pca_risk_regime"

    def __init__(self, pca_lookback: int = 252, signal_days: int = 21) -> None:
        self.pca_lookback = pca_lookback
        self.signal_days = signal_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        has_gld = "gld_close" in prices.columns
        gld_r = (
            prices["gld_close"].pct_change() if has_gld else pd.Series(np.nan, index=prices.index)
        )

        spy_cum = prices["adj_close"].pct_change(self.signal_days)
        tlt_cum = prices["tlt_close"].pct_change(self.signal_days)
        gld_cum = (
            prices["gld_close"].pct_change(self.signal_days)
            if has_gld
            else pd.Series(np.nan, index=prices.index)
        )

        n = len(prices)
        score_arr = np.full(n, np.nan)

        for i in range(self.pca_lookback + self.signal_days, n):
            w_spy = spy_r.values[i - self.pca_lookback : i]
            w_tlt = tlt_r.values[i - self.pca_lookback : i]
            w_gld = gld_r.values[i - self.pca_lookback : i]

            n_gld_valid = int(np.sum(~np.isnan(w_gld)))
            if n_gld_valid > self.pca_lookback // 2:
                ret_mat = np.column_stack([w_spy, w_tlt, w_gld])
                sig = np.array(
                    [spy_cum.values[i - 1], tlt_cum.values[i - 1], gld_cum.values[i - 1]]
                )
            else:
                ret_mat = np.column_stack([w_spy, w_tlt])
                sig = np.array([spy_cum.values[i - 1], tlt_cum.values[i - 1]])

            valid = ~np.isnan(ret_mat).any(axis=1)
            clean = ret_mat[valid]
            if len(clean) < 60 or np.isnan(sig).any():
                continue

            cov = np.cov(clean.T)
            if cov.ndim < 2:
                cov = np.array([[float(cov)]])
            eigvals, eigvecs = np.linalg.eigh(cov)
            pc1 = eigvecs[:, -1]
            if pc1[0] < 0:
                pc1 = -pc1

            score_arr[i] = float(np.dot(sig, pc1))

        scores = pd.Series(score_arr, index=prices.index).shift(1)
        tlt_ok = prices["tlt_close"].notna()
        valid_sig = scores.notna()

        w_spy_out = pd.Series(0.0, index=prices.index)
        w_tlt_out = pd.Series(0.0, index=prices.index)
        w_spy_out[valid_sig & (scores > 0)] = 1.0
        w_tlt_out[valid_sig & (scores <= 0) & tlt_ok] = 1.0

        return pd.DataFrame({"SPY": w_spy_out, "TLT": w_tlt_out}, index=prices.index)


class CreditRegimeRiskParity(MultiAssetStrategy):
    """H308: H021 with HYG vs LQD credit regime signal.

    HYG 63d return > LQD 63d return → credit risk-on → H021 full.
    LQD outperforming → credit risk-off → SPY weight * spy_scale; excess to TLT.
    Pre-HYG era (pre-2007): H021 unmodified.
    """

    name = "credit_regime_risk_parity"

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

        credit_risk_off = (hyg_ret < lqd_ret) & hyg_avail & hyg_ret.notna()

        freed = w_spy * (1.0 - self.spy_scale)
        new_spy = w_spy.where(~credit_risk_off, w_spy * self.spy_scale)
        new_tlt = w_tlt.where(~credit_risk_off, (w_tlt + freed).clip(upper=1.0))

        return pd.DataFrame({"SPY": new_spy, "TLT": new_tlt}, index=prices.index)


class TermSpreadSPYTimer(MultiAssetStrategy):
    """H311: H021 with TLT vs SHY term spread SPY overlay.

    Proxy term spread: TLT lookback-day return minus SHY lookback-day return.
    Positive (normal curve) → H021 full.
    Negative (inverted curve proxy) → SPY weight * spy_scale; freed weight to TLT.
    """

    name = "term_spread_spy_timer"

    def __init__(
        self,
        vol_lookback: int = 60,
        spread_lookback: int = 252,
        spy_scale: float = 0.50,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.spread_lookback = spread_lookback
        self.spy_scale = spy_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        tlt_ret = prices["tlt_close"].pct_change(self.spread_lookback).shift(1)
        shy_ret = prices["shy_close"].pct_change(self.spread_lookback).shift(1)
        inverted = (shy_ret > tlt_ret) & tlt_ret.notna() & shy_ret.notna()

        freed = w_spy * (1.0 - self.spy_scale)
        new_spy = w_spy.where(~inverted, w_spy * self.spy_scale)
        new_tlt = w_tlt.where(~inverted, (w_tlt + freed).clip(upper=1.0))
        return pd.DataFrame({"SPY": new_spy, "TLT": new_tlt}, index=prices.index)


class FundingStressH077(MultiAssetStrategy):
    """H337: RiskParityMomentumFilter (H077) + funding stress override.

    H077 base: H021 weights if SPY 12m momentum > 0; else 100% TLT.
    Stress override: if VIX > vix_threshold AND HYG 63d < LQD 63d →
                     force to crisis_tlt% TLT / (1-crisis_tlt)% SPY,
                     regardless of the momentum gate.

    Pre-HYG era: stress override requires only VIX > vix_threshold.

    The momentum gate and stress override address different regimes: momentum
    filters out secular bear markets; funding stress catches sudden liquidity
    crises where momentum may still be positive at the time of the shock.
    """

    name = "funding_stress_h077"

    def __init__(
        self,
        vol_lookback: int = 60,
        mom_days: int = 252,
        vix_threshold: float = 30.0,
        crisis_tlt: float = 0.80,
        credit_lookback: int = 63,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.mom_days = mom_days
        self.vix_threshold = vix_threshold
        self.crisis_tlt = crisis_tlt
        self.credit_lookback = credit_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy_rp, w_tlt_rp = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # H077 momentum gate: SPY 12m > 0 → H021; else 100% TLT
        spy_mom = prices["adj_close"].pct_change(self.mom_days).shift(1)
        mom_ok = (spy_mom > 0) & spy_mom.notna()
        tlt_avail = prices["tlt_close"].notna()

        w_spy = w_spy_rp.where(mom_ok, 0.0)
        w_tlt = w_tlt_rp.where(mom_ok, tlt_avail.astype(float))

        # Funding stress override
        vix = prices["vix_close"].shift(1).fillna(0.0)
        vix_stress = vix > self.vix_threshold

        if "hyg_close" in prices.columns and "lqd_close" in prices.columns:
            hyg_ret = prices["hyg_close"].pct_change(self.credit_lookback).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(self.credit_lookback).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            credit_off = (hyg_ret < lqd_ret) & hyg_avail & hyg_ret.notna()
            stress_flag = (vix_stress & credit_off).where(hyg_avail, vix_stress)
        else:
            stress_flag = vix_stress

        # Override to crisis allocation
        w_spy = w_spy.where(~stress_flag, 1.0 - self.crisis_tlt)
        w_tlt = w_tlt.where(~stress_flag, self.crisis_tlt)

        return pd.DataFrame({"SPY": w_spy, "TLT": w_tlt}, index=prices.index)


class DoubleCheckH080(MultiAssetStrategy):
    """H338: RiskParityYieldCurveOverlay (H080) + credit spread double-confirm.

    H080 base: if SHY 63d return > TLT 63d return (curve inverted) →
               reduce SPY weight by 50%; else H021.
    Double-confirm logic:
      Both conditions (curve inverted AND HYG 63d < LQD 63d) → SPY × 0.60 (strong cut).
      Only H080 curve signal → SPY × 0.85 (partial cut, cautious).
      Only credit stress → H021 unchanged (credit stress alone not enough to cut SPY).
      Neither → H021 full.

    Pre-HYG era: falls back to H080 base (curve signal only, partial cut).
    """

    name = "double_check_h080"

    def __init__(
        self,
        vol_lookback: int = 60,
        curve_days: int = 63,
        credit_lookback: int = 63,
        both_scale: float = 0.60,
        curve_only_scale: float = 0.85,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.curve_days = curve_days
        self.credit_lookback = credit_lookback
        self.both_scale = both_scale
        self.curve_only_scale = curve_only_scale

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)

        # H080 curve signal
        shy = prices.get("shy_close", pd.Series(np.nan, index=prices.index))
        tlt_px = prices["tlt_close"]
        shy_ret = shy.shift(1) / shy.shift(1 + self.curve_days) - 1
        tlt_ret_curve = tlt_px.shift(1) / tlt_px.shift(1 + self.curve_days) - 1
        curve_inverted = (shy_ret > tlt_ret_curve) & shy_ret.notna() & tlt_ret_curve.notna()

        # Credit signal
        has_credit = "hyg_close" in prices.columns and "lqd_close" in prices.columns
        credit_stress = pd.Series(False, index=prices.index)
        if has_credit:
            hyg_ret = prices["hyg_close"].pct_change(self.credit_lookback).shift(1)
            lqd_ret = prices["lqd_close"].pct_change(self.credit_lookback).shift(1)
            hyg_avail = prices["hyg_close"].notna()
            credit_stress = ((hyg_ret < lqd_ret) & hyg_avail & hyg_ret.notna()).fillna(False)

        both = curve_inverted & credit_stress
        curve_only = curve_inverted & ~credit_stress

        # Apply tiered SPY cut
        freed_both = w_spy * (1.0 - self.both_scale)
        freed_curve = w_spy * (1.0 - self.curve_only_scale)

        new_spy = pd.Series(
            np.select(
                [both, curve_only],
                [w_spy * self.both_scale, w_spy * self.curve_only_scale],
                default=w_spy,
            ),
            index=prices.index,
        )
        new_tlt = pd.Series(
            np.select(
                [both, curve_only],
                [(w_tlt + freed_both).clip(upper=1.0), (w_tlt + freed_curve).clip(upper=1.0)],
                default=w_tlt,
            ),
            index=prices.index,
        )

        return pd.DataFrame({"SPY": new_spy, "TLT": new_tlt}, index=prices.index)


class RegulatedSectorTiming(MultiAssetStrategy):
    """H421: H021 + TLT trend gate to tilt into regulated defensive sectors.

    When TLT > 200d MA (rates falling = policy tailwind for defensives):
      SPY weight reduced by tilt_pct, split evenly between XLP and XLU.
    Otherwise: pure H021. Cohen, Diether & Malloy (2013): policy events
    benefit regulated utility and consumer staples sectors.
    """

    name = "regulated_sector_timing"

    def __init__(
        self,
        vol_lookback: int = 60,
        tlt_ma: int = 200,
        tilt_pct: float = 0.15,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.tlt_ma = tlt_ma
        self.tilt_pct = tilt_pct

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        w_spy, w_tlt = _h021_weights(spy_r, tlt_r, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        xlp_avail = (
            prices["xlp_close"].notna()
            if "xlp_close" in prices.columns
            else pd.Series(False, index=prices.index)
        )
        xlu_avail = (
            prices["xlu_close"].notna()
            if "xlu_close" in prices.columns
            else pd.Series(False, index=prices.index)
        )
        tlt_ma_s = prices["tlt_close"].rolling(self.tlt_ma, min_periods=self.tlt_ma // 2).mean()
        rate_falling = (prices["tlt_close"] > tlt_ma_s) & tlt_avail & xlp_avail & xlu_avail
        half_tilt = self.tilt_pct / 2.0
        w_spy_adj = w_spy.where(~rate_falling, w_spy * (1.0 - self.tilt_pct))
        w_xlp = pd.Series(0.0, index=prices.index).where(~rate_falling, w_spy * half_tilt)
        w_xlu = pd.Series(0.0, index=prices.index).where(~rate_falling, w_spy * half_tilt)
        w_spy_adj = w_spy_adj.where(tlt_avail, 1.0)
        w_tlt_final = w_tlt.where(tlt_avail, 0.0)
        return pd.DataFrame(
            {"SPY": w_spy_adj, "TLT": w_tlt_final, "XLP": w_xlp, "XLU": w_xlu}, index=prices.index
        )


class TLTMomentumOverlay(MultiAssetStrategy):
    """H488: H021 with TLT replaced by SHY during TLT downtrends.

    TLT 6m return > 0 → maintain standard risk-parity TLT allocation.
    TLT 6m return ≤ 0 → replace TLT with SHY (reduce duration, preserve low-risk allocation).
    Preserves H021 diversification without bond duration drag in rising rate environments.
    """

    name = "tlt_momentum_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
        mom_window: int = 126,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.mom_window = mom_window

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w_spy, w_tlt = _h021_base_weights(prices, self.vol_lookback)
        tlt_avail = prices["tlt_close"].notna()
        has_shy = "shy_close" in prices.columns
        tlt_mom = prices["tlt_close"].shift(1) / prices["tlt_close"].shift(1 + self.mom_window) - 1
        tlt_trending = (tlt_mom >= 0) | ~tlt_avail
        if has_shy:
            shy_avail = prices["shy_close"].notna()
            use_shy = ~tlt_trending & tlt_avail & shy_avail
            w_tlt_out = w_tlt.where(~use_shy, 0.0)
            w_shy_out = w_tlt.where(use_shy, 0.0)
            return pd.DataFrame(
                {"SPY": w_spy.where(tlt_avail, 1.0), "TLT": w_tlt_out, "SHY": w_shy_out},
                index=prices.index,
            )
        return pd.DataFrame(
            {"SPY": w_spy.where(tlt_avail, 1.0), "TLT": w_tlt.where(tlt_trending, w_tlt, 0.0)},
            index=prices.index,
        )


class ThreeAssetMomentumBridge(MultiAssetStrategy):
    """H489: SPY/TLT/GLD risk-parity base with 3-month momentum overlay.

    Base: H021-style inverse-vol weights across 3 assets.
    Momentum overlay: asset with highest 3m return gets weight × 1.5;
    lowest gets × 0.5; middle unchanged. Weights renormalized to sum ≤ 1.
    Bridges trend-following (H007) with multi-asset risk parity (H076).
    """

    name = "three_asset_momentum_bridge"

    def __init__(
        self,
        vol_lookback: int = 60,
        mom_window: int = 63,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.mom_window = mom_window

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        gld_r = prices["gld_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        gld_vol = gld_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        inv_gld = 1.0 / gld_vol.replace(0, np.nan)

        tlt_avail = prices["tlt_close"].notna()
        gld_avail = prices["gld_close"].notna()
        all_avail = tlt_avail & gld_avail

        total = (inv_spy + inv_tlt.where(tlt_avail, 0.0) + inv_gld.where(gld_avail, 0.0)).replace(
            0, np.nan
        )
        w_spy = (inv_spy / total).fillna(1.0)
        w_tlt = (inv_tlt.where(tlt_avail, 0.0) / total).fillna(0.0)
        w_gld = (inv_gld.where(gld_avail, 0.0) / total).fillna(0.0)

        # 3-month momentum
        spy_mom = prices["adj_close"].shift(1) / prices["adj_close"].shift(1 + self.mom_window) - 1
        tlt_mom = prices["tlt_close"].shift(1) / prices["tlt_close"].shift(1 + self.mom_window) - 1
        gld_mom = prices["gld_close"].shift(1) / prices["gld_close"].shift(1 + self.mom_window) - 1

        moms = pd.DataFrame({"SPY": spy_mom, "TLT": tlt_mom, "GLD": gld_mom})
        ranks = moms.rank(axis=1, method="first")  # 1=lowest, 3=highest
        spy_mult = ranks["SPY"].map({1: 0.5, 2: 1.0, 3: 1.5}).fillna(1.0)
        tlt_mult = ranks["TLT"].map({1: 0.5, 2: 1.0, 3: 1.5}).fillna(1.0)
        gld_mult = ranks["GLD"].map({1: 0.5, 2: 1.0, 3: 1.5}).fillna(1.0)

        w_spy_m = (w_spy * spy_mult).where(all_avail, w_spy)
        w_tlt_m = (w_tlt * tlt_mult).where(all_avail, w_tlt)
        w_gld_m = (w_gld * gld_mult).where(all_avail, w_gld)

        total_m = (w_spy_m + w_tlt_m + w_gld_m).replace(0, np.nan)
        w_spy_f = (w_spy_m / total_m).fillna(w_spy)
        w_tlt_f = (w_tlt_m / total_m).fillna(w_tlt)
        w_gld_f = (w_gld_m / total_m).fillna(w_gld)

        return pd.DataFrame({"SPY": w_spy_f, "TLT": w_tlt_f, "GLD": w_gld_f}, index=prices.index)


class ContinuousYieldCurveScaler(MultiAssetStrategy):
    """H491: H021 with linear SPY-weight scaling from TLT-SHY return spread z-score.

    Signal: trailing 252d TLT return minus SHY return (yield-curve steepness proxy).
    Z-scored over rolling 252d window → SPY scale in [scale_lo, scale_hi].
    High z (steep curve) → equity tilt; low z (flat/inverted) → defensive.
    Continuous analog of H311's binary yield-curve signal.
    """

    name = "continuous_yield_curve_scaler"

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

        tlt_ret = prices["tlt_close"].pct_change(self.spread_window)
        shy_ret = prices["shy_close"].pct_change(self.spread_window)
        spread = (tlt_ret - shy_ret).shift(1)

        spread_mean = spread.rolling(self.spread_window, min_periods=self.spread_window // 2).mean()
        spread_std = spread.rolling(self.spread_window, min_periods=self.spread_window // 2).std()
        z = (spread - spread_mean) / spread_std.replace(0, np.nan)
        z = z.clip(-2.0, 2.0).fillna(0.0)

        # z in [-2, 2] → scale in [scale_lo, scale_hi] via linear mapping
        mid = (self.scale_lo + self.scale_hi) / 2.0
        half_range = (self.scale_hi - self.scale_lo) / 2.0
        scale = (mid + z * half_range / 2.0).clip(self.scale_lo, self.scale_hi)

        spy_scaled = (w_spy * scale).clip(0.0, 1.0)
        tlt_adj = (1.0 - spy_scaled).clip(0.0, 1.0)
        return pd.DataFrame(
            {"SPY": spy_scaled.where(tlt_avail, 1.0), "TLT": tlt_adj.where(tlt_avail, 0.0)},
            index=prices.index,
        )


class GLDMomentumOverlay(MultiAssetStrategy):
    """H495: SPY/TLT/GLD 3-asset parity with GLD 6m momentum tilt.

    Base: H076-style inverse-vol weights across SPY, TLT, GLD.
    GLD 6m return > 0 → GLD weight × mom_up_mult (default 1.2); renormalize.
    GLD 6m return ≤ 0 → GLD weight × mom_dn_mult (default 0.5), floor at gld_floor.
    Tests whether GLD trend improves H076 multi-asset parity.
    """

    name = "gld_momentum_overlay"

    def __init__(
        self,
        vol_lookback: int = 60,
        mom_window: int = 126,
        mom_up_mult: float = 1.2,
        mom_dn_mult: float = 0.5,
        gld_floor: float = 0.10,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.mom_window = mom_window
        self.mom_up_mult = mom_up_mult
        self.mom_dn_mult = mom_dn_mult
        self.gld_floor = gld_floor

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy_r = prices["adj_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()
        gld_r = prices["gld_close"].pct_change()

        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        gld_vol = gld_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        tlt_avail = prices["tlt_close"].notna()
        gld_avail = prices["gld_close"].notna()

        inv_spy = 1.0 / spy_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        inv_gld = 1.0 / gld_vol.replace(0, np.nan)

        total_base = (
            inv_spy + inv_tlt.where(tlt_avail, 0.0) + inv_gld.where(gld_avail, 0.0)
        ).replace(0, np.nan)
        w_spy = (inv_spy / total_base).fillna(1.0)
        w_tlt = (inv_tlt.where(tlt_avail, 0.0) / total_base).fillna(0.0)
        w_gld = (inv_gld.where(gld_avail, 0.0) / total_base).fillna(0.0)

        gld_mom = prices["gld_close"].shift(1) / prices["gld_close"].shift(1 + self.mom_window) - 1
        gld_up = (gld_mom >= 0) & gld_avail

        gld_mult = pd.Series(self.mom_up_mult, index=prices.index).where(gld_up, self.mom_dn_mult)
        w_gld_adj = (w_gld * gld_mult).where(gld_avail, 0.0)
        # Apply floor when GLD is down-trending
        w_gld_floor = pd.Series(self.gld_floor, index=prices.index).where(gld_avail & ~gld_up, 0.0)
        w_gld_adj = w_gld_adj.clip(lower=w_gld_floor)

        total_adj = (w_spy + w_tlt + w_gld_adj).replace(0, np.nan)
        w_spy_f = (w_spy / total_adj).fillna(w_spy)
        w_tlt_f = (w_tlt / total_adj).fillna(w_tlt)
        w_gld_f = (w_gld_adj / total_adj).fillna(0.0)

        return pd.DataFrame({"SPY": w_spy_f, "TLT": w_tlt_f, "GLD": w_gld_f}, index=prices.index)


class QQQTLTRiskParity(MultiAssetStrategy):
    """H571: QQQ/TLT inverse-vol risk parity.

    Identical mechanism to H021 (RiskParity) but uses QQQ (Nasdaq-100) instead
    of SPY for the equity leg. QQQ has higher historical CAGR and deeper crashes
    than SPY, giving TLT a more valuable hedge role. QQQ available from 1999-03;
    folds 0-1 mostly cash (no QQQ+TLT overlap until fold 2).
    """

    name = "qqq_tlt_risk_parity"

    def __init__(self, vol_lookback: int = 60) -> None:
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        qqq_r = prices["qqq_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        qqq_vol = qqq_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        inv_qqq = 1.0 / qqq_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_qqq + inv_tlt

        w_qqq = (inv_qqq / total).fillna(0.0)
        w_tlt = (inv_tlt / total).fillna(0.0)

        return pd.DataFrame({"QQQ": w_qqq, "TLT": w_tlt}, index=prices.index)


class IWMTLTRiskParity(MultiAssetStrategy):
    """H572: IWM/TLT inverse-vol risk parity.

    H021-style risk parity but using IWM (Russell 2000 small-cap) instead of
    SPY. Small-cap has lower correlation with TLT than large-cap in normal
    regimes; deeper crashes in recessions. IWM available from 2000-05;
    folds 0-1 mostly cash.
    """

    name = "iwm_tlt_risk_parity"

    def __init__(self, vol_lookback: int = 60) -> None:
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        iwm_r = prices["iwm_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        iwm_vol = iwm_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        inv_iwm = 1.0 / iwm_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_iwm + inv_tlt

        w_iwm = (inv_iwm / total).fillna(0.0)
        w_tlt = (inv_tlt / total).fillna(0.0)

        return pd.DataFrame({"IWM": w_iwm, "TLT": w_tlt}, index=prices.index)


class QQQIWMTLTRiskParity(MultiAssetStrategy):
    """H573: QQQ/IWM/TLT three-asset inverse-vol risk parity.

    Growth (QQQ) + value/small-cap (IWM) + bonds (TLT). Gracefully degrades
    to fewer assets when data is unavailable. Both QQQ (1999-03) and IWM
    (2000-05) start after TLT (2002-07); folds 0-2 mostly cash.
    """

    name = "qqq_iwm_tlt_risk_parity"

    def __init__(self, vol_lookback: int = 60) -> None:
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        asset_cols = {"QQQ": "qqq_close", "IWM": "iwm_close", "TLT": "tlt_close"}
        available = {k: v for k, v in asset_cols.items() if v in prices.columns}

        vols: dict[str, pd.Series] = {}
        for asset, col in available.items():
            r = prices[col].pct_change()
            vols[asset] = r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
                1
            ) * np.sqrt(252)

        inv_vols = {a: 1.0 / v.replace(0, np.nan) for a, v in vols.items()}
        total = sum(inv_vols.values())
        weights = pd.DataFrame(index=prices.index, columns=list(available.keys()), dtype=float)
        for asset in available:
            weights[asset] = (inv_vols[asset] / total).fillna(0.0)

        return weights


class VNQTLTRiskParity(MultiAssetStrategy):
    """H578: VNQ/TLT inverse-vol risk parity.

    REITs (VNQ) paired with long-duration Treasuries (TLT). REITs benefit from
    falling rates like TLT but also from rising real estate values (anti-TLT).
    The ambiguous VNQ-TLT correlation may produce a distinctive risk profile.
    VNQ available from 2004-09; folds 0-3 mostly cash.
    """

    name = "vnq_tlt_risk_parity"

    def __init__(self, vol_lookback: int = 60) -> None:
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        vnq_r = prices["vnq_close"].pct_change()
        tlt_r = prices["tlt_close"].pct_change()

        vnq_vol = vnq_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback).std().shift(
            1
        ) * np.sqrt(252)

        inv_vnq = 1.0 / vnq_vol.replace(0, np.nan)
        inv_tlt = 1.0 / tlt_vol.replace(0, np.nan)
        total = inv_vnq + inv_tlt

        w_vnq = (inv_vnq / total).fillna(0.0)
        w_tlt = (inv_tlt / total).fillna(0.0)

        return pd.DataFrame({"VNQ": w_vnq, "TLT": w_tlt}, index=prices.index)
