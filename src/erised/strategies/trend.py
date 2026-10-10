"""Trend-following and time-series momentum strategies on SPY."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import MultiAssetStrategy, Strategy


class MovingAverageCrossover(Strategy):
    """Classic dual-MA trend-following.

    Long when fast MA > slow MA, flat otherwise. By default uses
    50/200, which is the textbook "golden cross" / "death cross".
    """

    def __init__(self, fast: int = 50, slow: int = 200, allow_short: bool = False):
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short
        self.name = f"ma_cross_{fast}_{slow}{'_ls' if allow_short else ''}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        fast_ma = px.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = px.rolling(self.slow, min_periods=self.slow).mean()
        long_signal = (fast_ma > slow_ma).astype(float)
        if self.allow_short:
            short_signal = (fast_ma < slow_ma).astype(float)
            pos = long_signal - short_signal
        else:
            pos = long_signal
        # Before we have enough history, stay flat.
        pos = pos.where(slow_ma.notna(), 0.0)
        pos.name = "position"
        return pos


class LongShortMACrossover(MovingAverageCrossover):
    """H013: MA(50,200) long/short — short below the death cross instead of flat.

    Identical signal to H001 (MovingAverageCrossover) but sets allow_short=True,
    going −1 when fast MA < slow MA rather than 0. Captures bear-market returns
    that H001 abandons by going to cash.
    """

    def __init__(self, fast: int = 50, slow: int = 200):
        super().__init__(fast=fast, slow=slow, allow_short=True)


class DualMALongShort(MovingAverageCrossover):
    """H015: MA(20,100) long/short — faster signal than H013's 50/200 pair.

    Tests whether catching trend turns earlier improves Sharpe or erodes gains
    through excess turnover. allow_short=True: −1 below the 100-day MA.
    """

    def __init__(self, fast: int = 20, slow: int = 100):
        super().__init__(fast=fast, slow=slow, allow_short=True)


class LongShortTSMomentum(Strategy):
    """H014: 12-1 month time-series momentum, long/short.

    Identical lookback logic to H007 (TimeSeriesMomentum) but goes short (−1)
    when the 12-1 month return is negative rather than going flat. Tests whether
    the momentum signal has predictive power on the short side.
    """

    def __init__(self, long_months: int = 12, skip_months: int = 1):
        self.long_days = int(long_months * 21)
        self.skip_days = int(skip_months * 21)
        self.name = f"ls_ts_mom_{long_months}m_{skip_months}skip"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        mom = px.shift(self.skip_days) / px.shift(self.long_days) - 1
        pos = mom.apply(lambda x: 0.0 if pd.isna(x) else float(np.sign(x)))
        pos.name = "position"
        return pos


class TimeSeriesMomentum(Strategy):
    """Hypothesis 007: 12-1 month time-series momentum (skip-month convention).

    Long when the return from 12 months ago to 1 month ago is positive.
    The 1-month skip avoids the short-term reversal at lag 1 (Jegadeesh & Titman).
    """

    def __init__(self, long_months: int = 12, skip_months: int = 1):
        self.long_days = int(long_months * 21)
        self.skip_days = int(skip_months * 21)
        self.name = f"ts_mom_{long_months}m_{skip_months}skip"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        # Return from long_days ago to skip_days ago (both purely past data)
        mom = px.shift(self.skip_days) / px.shift(self.long_days) - 1

        pos = (mom > 0).astype(float)
        pos = pos.where(mom.notna(), 0.0)
        pos.name = "position"
        return pos


class AbsoluteMomentumStrategy(Strategy):
    """Hypothesis 008: Antonacci's absolute momentum.

    Long when the trailing lookback_months return is positive; flat (cash) when
    negative. The simplest crash-protection filter — exits sustained downtrends.
    """

    def __init__(self, lookback_months: int = 12):
        self.lookback_days = int(lookback_months * 21)
        self.name = f"abs_mom_{lookback_months}m"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        mom = px / px.shift(self.lookback_days) - 1

        pos = (mom > 0).astype(float)
        pos = pos.where(mom.notna(), 0.0)
        pos.name = "position"
        return pos


class MACDCrossover(Strategy):
    """Hypothesis 009: MACD signal-line crossover.

    Long when MACD (EMA_fast - EMA_slow) is above its signal line (EMA of MACD).
    All EMAs are computed from data through t — no lookahead.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.name = f"macd_{fast}_{slow}_{signal}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        ema_fast = px.ewm(span=self.fast, adjust=False).mean()
        ema_slow = px.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()

        pos = (macd_line > signal_line).astype(float)
        # Zero out the warm-up window where EMAs are unreliable
        warmup = self.slow + self.signal
        if len(pos) > warmup:
            pos.iloc[:warmup] = 0.0
        pos.name = "position"
        return pos


class FastMACrossover(MovingAverageCrossover):
    """Hypothesis 010: MA(10/50) — faster trend-following.

    Tests whether catching trend turns earlier than H001's MA(50/200)
    improves Sharpe, or whether the extra turnover erodes gains.
    """

    name = "fast_ma_cross_10_50"

    def __init__(self, fast: int = 10, slow: int = 50):
        super().__init__(fast=fast, slow=slow)


class ADXFilteredMACrossover(Strategy):
    """Hypothesis 011: MA(50,200) crossover confirmed by ADX trend strength.

    Takes H001 signals only when ADX(period) >= adx_threshold, filtering
    out false crossovers in ranging, low-momentum markets.
    """

    def __init__(
        self,
        fast: int = 50,
        slow: int = 200,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
    ):
        self.fast = fast
        self.slow = slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.name = f"adx_ma_{fast}_{slow}_{adx_period}_{int(adx_threshold)}"

    @staticmethod
    def _adx(prices: pd.DataFrame, period: int) -> pd.Series:
        hi = prices["high"]
        lo = prices["low"]
        cl = prices["adj_close"]

        tr = pd.concat([hi - lo, (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()], axis=1).max(
            axis=1
        )

        up = hi - hi.shift(1)
        dn = lo.shift(1) - lo
        plus_dm = ((up > dn) & (up > 0)) * up
        minus_dm = ((dn > up) & (dn > 0)) * dn

        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr

        di_sum = plus_di + minus_di
        dx = (100 * (plus_di - minus_di).abs() / di_sum).where(di_sum > 0, 0.0)
        adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        return adx

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        fast_ma = px.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = px.rolling(self.slow, min_periods=self.slow).mean()

        ma_signal = (fast_ma > slow_ma).astype(float)
        ma_signal = ma_signal.where(slow_ma.notna(), 0.0)

        adx = self._adx(prices, self.adx_period)
        trend_on = (adx >= self.adx_threshold).astype(float)
        trend_on = trend_on.where(adx.notna(), 0.0)

        pos = ma_signal * trend_on
        pos.name = "position"
        return pos


class DualFrequencyMomentum(Strategy):
    """H081: Long SPY only when both 12-1m AND 1m momentum are positive.

    12-1m (trend): return from 252 days ago to 21 days ago — confirms durable uptrend.
    1m (short-term): return over last 21 days — avoids entry during pullbacks in uptrend.
    Both must be positive to hold; flat when either is negative.
    """

    name = "dual_frequency_momentum"

    def __init__(self, mom_long: int = 252, mom_skip: int = 21, mom_short: int = 21) -> None:
        self.mom_long = mom_long
        self.mom_skip = mom_skip
        self.mom_short = mom_short

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        spy = prices["adj_close"]
        # 12-1m: return from mom_long days ago to mom_skip days ago (skip recent month)
        mom_long_sig = spy.shift(self.mom_skip) / spy.shift(self.mom_long) - 1
        # 1m: return over last mom_short days (current short-term trend)
        mom_short_sig = spy / spy.shift(self.mom_short) - 1
        pos = pd.Series(0.0, index=prices.index, name="position")
        valid = mom_long_sig.notna() & mom_short_sig.notna()
        pos[valid & (mom_long_sig > 0) & (mom_short_sig > 0)] = 1.0
        return pos


class SectorETFMomentum(MultiAssetStrategy):
    """H201: Rotate into top-N equity ETFs by 6m momentum, risk-parity weighted.

    ETF universe: XLP, XLU, IVE, IVW, QQQ, IWM, VNQ.
    Signal: 126-day trailing total return (6 months), shifted 1 bar.
    Portfolio: top-N by 6m return, inverse-vol (vol_lookback-day) risk-parity.
    Falls back to SPY when fewer ETFs have sufficient history.

    Prices DataFrame must contain {symbol.lower()}_close columns for each ETF.
    """

    name = "sector_etf_momentum"

    _EQUITY_ETFS = ["XLP", "XLU", "IVE", "IVW", "QQQ", "IWM", "VNQ"]

    def __init__(
        self,
        mom_window: int = 126,
        vol_lookback: int = 60,
        top_n: int = 3,
    ) -> None:
        self.mom_window = mom_window
        self.vol_lookback = vol_lookback
        self.top_n = top_n

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        col_map = {a: f"{a.lower()}_close" for a in self._EQUITY_ETFS}
        available = {a: col for a, col in col_map.items() if col in prices.columns}

        if not available:
            return pd.DataFrame({"SPY": pd.Series(1.0, index=prices.index)})

        # 6m momentum and rolling vol for each available ETF
        mom_df = pd.DataFrame(
            {a: prices[col].pct_change(self.mom_window) for a, col in available.items()}
        )
        ret_df = pd.DataFrame({a: prices[col].pct_change() for a, col in available.items()})
        vol_df = ret_df.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std()
        # Fill NaN vol with 1.0 so inv-vol is equal-weight when history is short
        inv_vol_df = vol_df.replace(0, np.nan).rdiv(1).fillna(1.0)

        # Rank ETFs by 6m momentum each day (1 = best); NaN = insufficient history
        mom_rank = mom_df.rank(axis=1, ascending=False, na_option="keep")
        top_mask = (mom_rank <= self.top_n) & mom_rank.notna()

        # ETF weights: inv-vol within top-N, normalized to sum to 1
        raw = inv_vol_df.where(top_mask, 0.0)
        total = raw.sum(axis=1).replace(0, np.nan)
        etf_w = raw.div(total, axis=0).fillna(0.0)

        # SPY fallback when no ETF has enough history
        no_etf = top_mask.sum(axis=1) == 0
        etf_w["SPY"] = no_etf.astype(float)

        return etf_w.shift(1).fillna(0.0)


class SectorMomentumRiskParity(MultiAssetStrategy):
    """H205: H021 variant — top equity ETF by 6m momentum paired with TLT.

    Selects the single best-performing equity ETF (XLP, XLU, IVE, IVW, QQQ,
    IWM, VNQ) by 126-day trailing return, then risk-parity weights it against
    TLT. Hypothesis: concentrating the equity leg in the strongest sector while
    keeping the TLT hedge should improve on H021's fixed SPY exposure.

    Falls back to SPY when no ETF has 6m history. Falls back to 100% winner
    equity ETF when TLT is unavailable.
    """

    name = "sector_momentum_risk_parity"

    _EQUITY_ETFS = ["XLP", "XLU", "IVE", "IVW", "QQQ", "IWM", "VNQ"]

    def __init__(
        self,
        mom_window: int = 126,
        vol_lookback: int = 60,
    ) -> None:
        self.mom_window = mom_window
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        col_map = {a: f"{a.lower()}_close" for a in self._EQUITY_ETFS}
        available = {a: col for a, col in col_map.items() if col in prices.columns}

        tlt_col = "tlt_close"
        has_tlt = tlt_col in prices.columns

        # 6m momentum for equity ETFs
        mom_df = pd.DataFrame(
            {a: prices[col].pct_change(self.mom_window) for a, col in available.items()}
        )

        # 60d vol for equity ETFs, TLT, SPY
        ret_df = pd.DataFrame({a: prices[col].pct_change() for a, col in available.items()})
        vol_etf = ret_df.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std()

        tlt_r = prices[tlt_col].pct_change() if has_tlt else pd.Series(np.nan, index=prices.index)
        tlt_vol = tlt_r.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std()
        spy_r = prices["adj_close"].pct_change()
        spy_vol = spy_r.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std()

        # Top-1 equity ETF by 6m momentum (all-NaN rows handled explicitly)
        winner_valid = mom_df.notna().any(axis=1)
        winner = pd.Series("SPY", index=prices.index, dtype=object)
        if winner_valid.any():
            winner.loc[winner_valid] = mom_df.loc[winner_valid].idxmax(axis=1, skipna=True)

        # Build output weight DataFrame covering all possible assets
        all_assets = list(
            dict.fromkeys(list(available.keys()) + (["TLT"] if has_tlt else []) + ["SPY"])
        )
        wdf = pd.DataFrame(0.0, index=prices.index, columns=all_assets)

        for date in prices.index:
            w_asset = winner.loc[date]

            # Equity leg vol
            if w_asset == "SPY":
                e_v = spy_vol.loc[date]
            elif w_asset in vol_etf.columns:
                e_v = vol_etf.loc[date, w_asset]
            else:
                e_v = np.nan
            inv_e = (1.0 / e_v) if (not np.isnan(e_v) and e_v > 0) else 1.0

            # TLT leg
            t_v = tlt_vol.loc[date] if has_tlt else np.nan
            tlt_ok = has_tlt and not np.isnan(prices[tlt_col].loc[date])

            if tlt_ok and not np.isnan(t_v) and t_v > 0:
                inv_t = 1.0 / t_v
                tot = inv_e + inv_t
                wdf.loc[date, w_asset] = inv_e / tot
                wdf.loc[date, "TLT"] = inv_t / tot
            else:
                wdf.loc[date, w_asset] = 1.0

        return wdf.shift(1).fillna(0.0)
