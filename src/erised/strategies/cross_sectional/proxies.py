"""Proxy constructions of governance, ESG, ownership, attention, and product-market signals from price and fundamental data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import (
    _align_fund,
    _cs_zscore,
    _equal_weight_top_quantile,
    _monthly_rebalance,
    _periodic_rebalance,
    _sector_neutral_weights,
)
from .base import CrossSectionalStrategy


class RetainedEarningsProxy(CrossSectionalStrategy):
    """H247: YoY book equity growth / total assets = retained earnings proxy."""

    name = "retained_earnings_proxy"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        window: int = 252,
        within_sector_quantile: float = 0.20,
    ):
        self.book_equity = book_equity
        self.total_assets = total_assets
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        be = _align_fund(self.book_equity, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        be_growth = (be - be.shift(self.window)) / ta.shift(self.window).where(
            ta.shift(self.window) > 0
        )
        be_growth = be_growth.clip(-0.5, 1.0)
        daily_w = _sector_neutral_weights(
            be_growth, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class IntangibleQuality(CrossSectionalStrategy):
    """H248: GP/book_equity -- intangible asset intensity proxy.

    High GP relative to book equity signals brand, IP, or other intangibles
    not captured on the balance sheet.
    """

    name = "intangible_quality"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ):
        self.gross_profit = gross_profit
        self.book_equity = book_equity
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        gp_be = gp / be.where(be > 0)
        gp_be = gp_be.clip(-1.0, 10.0)
        daily_w = _sector_neutral_weights(
            gp_be, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class RecentEpochOnlyGP(CrossSectionalStrategy):
    """H366: GP/TA rank multiplied by recent-period price confirmation.

    GP/TA z-score x (1 + clip(z(21d_return), -1, 1)). Tilts quality toward
    names with recent positive confirmation, down-weighting quality traps.
    McLean-Pontiff 2016: weighting towards recent evidence reduces stale signal.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "recent_epoch_only_gp"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        recent_window: int = 21,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.recent_window = recent_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        recent_ret = prices_panel.pct_change(self.recent_window)
        tilt = _cs_zscore(recent_ret).clip(-1.0, 1.0).fillna(0.0)
        score = _cs_zscore(gp_ta).fillna(0.0) * (1.0 + 0.3 * tilt)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class PostPublicationHaircutComposite(CrossSectionalStrategy):
    """H367: Composite of GP/TA, ROE, accruals with inverse-volatility shrinkage.

    Each signal z-score shrunk by its trailing cross-sectional standard deviation:
    signal_weight_i = 1 / (1 + rolling_std(z_i)). Noisier signals get more haircut.
    Approximates post-publication decay (McLean-Pontiff 2016): attenuates weak/noisy
    signals automatically. Sector-neutral top 20%, quarterly rebalance.
    """

    name = "post_publication_haircut_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        shrink_window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.shrink_window = shrink_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)

        z1 = _cs_zscore(gp_ta)
        z2 = _cs_zscore(roe)
        z3 = _cs_zscore(-accruals)

        def _haircut_weight(z: pd.DataFrame) -> pd.DataFrame:
            row_std = z.std(axis=1).rolling(self.shrink_window, min_periods=63).mean()
            weight = 1.0 / (1.0 + row_std.clip(lower=0.1))
            return z.multiply(weight, axis=0)

        score = (
            _haircut_weight(z1).fillna(0.0)
            + _haircut_weight(z2).fillna(0.0)
            + _haircut_weight(z3).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class IncrementalAlphaFilter(CrossSectionalStrategy):
    """H368: Keep only cross-sectional signals with t-stat > threshold vs GP+momentum.

    For each quarterly rebalance, regresses 63d forward returns on (candidate signal,
    GP/TA, momentum) using trailing 756d data. Candidate signal kept if |t-stat| > 1.5.
    Passing signals equal-weighted into composite. Feng-Giglio-Xiu 2020.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "incremental_alpha_filter"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        train_window: int = 756,
        pred_horizon: int = 63,
        t_threshold: float = 1.5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.train_window = train_window
        self.pred_horizon = pred_horizon
        self.t_threshold = t_threshold
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        mom = prices_panel.pct_change(252).shift(21).clip(-1.0, 5.0)
        asset_growth = ((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0)

        candidates = {
            "roe": roe,
            "neg_accruals": -accruals,
            "neg_asset_growth": -asset_growth,
        }
        controls = [gp_ta, mom]

        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        score_df = gp_ta.reindex(index=all_dates, columns=tickers)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos - self.pred_horizon
            train_start = max(0, train_end - self.train_window)
            if train_end - train_start < 40:
                continue

            passing_signals = []
            for cand_panel in candidates.values():
                X_rows, y_rows = [], []
                for tp in range(
                    train_start, train_end - self.rebalance_period, self.rebalance_period
                ):
                    t_d = all_dates[tp]
                    t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                    try:
                        c_vals = cand_panel.loc[t_d, tickers].values
                        ctrl_vals = np.column_stack([p.loc[t_d, tickers].values for p in controls])
                        X = np.column_stack([c_vals.reshape(-1, 1), ctrl_vals])
                    except Exception:
                        continue
                    mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                    fwd = (
                        (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
                    )
                    valid = mem_t & np.isfinite(fwd) & np.isfinite(X).all(axis=1)
                    if valid.sum() < 20:
                        continue
                    X_rows.append(X[valid])
                    y_rows.append(fwd[valid])

                if not X_rows:
                    continue
                X_tr = np.vstack(X_rows)
                y_tr = np.concatenate(y_rows)
                try:
                    from numpy.linalg import lstsq

                    X_aug = np.column_stack([np.ones(len(X_tr)), X_tr])
                    coef, _, _, _ = lstsq(X_aug, y_tr, rcond=None)
                    residuals = y_tr - X_aug @ coef
                    sigma2 = np.var(residuals)
                    XtX_inv = np.linalg.inv(X_aug.T @ X_aug + 1e-8 * np.eye(X_aug.shape[1]))
                    se = np.sqrt(sigma2 * XtX_inv[1, 1])
                    t_stat = abs(coef[1] / (se + 1e-10))
                    if t_stat > self.t_threshold:
                        passing_signals.append(cand_panel.loc[rebal_date, tickers].values)
                except Exception:
                    continue

            if passing_signals:
                scores = np.mean(np.column_stack(passing_signals), axis=1)
            else:
                scores = gp_ta.loc[rebal_date, tickers].values
            score_df.loc[rebal_date, tickers] = scores

        score_df = score_df.ffill().fillna(gp_ta)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class EigenportfolioCharacteristic(CrossSectionalStrategy):
    """H369: PCA on monthly characteristic matrix; PC1 score as ranking signal.

    Characteristics: GP/TA, ROE, neg_accruals, neg_asset_growth, momentum.
    Each quarterly rebalance: fit PCA on past year's cross-sectional data,
    extract PC1 score for current date. Long top 20% by PC1.
    Green-Hand-Zhang 2017: PCA extracts the truly independent dimensions.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "eigenportfolio_characteristic"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        n_components: int = 1,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.n_components = n_components
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0)
        mom = prices_panel.pct_change(252).shift(21).clip(-1.0, 5.0)
        feat_panels = [gp_ta, roe, -accruals, -asset_growth, mom]

        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        score_df = gp_ta.reindex(index=all_dates, columns=tickers)

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            daily_w = _sector_neutral_weights(
                gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
            )
            return _periodic_rebalance(daily_w, self.rebalance_period)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos
            train_start = max(0, train_end - 252)
            if train_end - train_start < 30:
                continue
            rows = []
            for tp in range(train_start, train_end, self.rebalance_period):
                t_d = all_dates[tp]
                try:
                    X = np.column_stack([fp.loc[t_d, tickers].values for fp in feat_panels])
                except Exception:
                    continue
                mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                valid = mem_t & np.isfinite(X).all(axis=1)
                if valid.sum() < 20:
                    continue
                rows.append(X)
            if not rows:
                continue
            X_train = np.vstack(rows)
            try:
                sc = StandardScaler()
                Xs = sc.fit_transform(X_train)
                pca = PCA(n_components=self.n_components)
                pca.fit(Xs)
                X_cur = np.column_stack([fp.loc[rebal_date, tickers].values for fp in feat_panels])
                X_cur_scaled = sc.transform(X_cur)
                pc1 = pca.transform(X_cur_scaled)[:, 0]
                score_df.loc[rebal_date, tickers] = pc1
            except Exception:
                continue

        score_df = score_df.ffill().fillna(gp_ta)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class DecayAdjustedComposite(CrossSectionalStrategy):
    """H370: Equal-weight blend of batch-4 decay/alpha-control signals.

    Combines RecentEpochOnlyGP, PostPublicationHaircut, and GP/TA anchor
    via clipped z-score averaging. Acts as an ensemble with implicit
    regularization from the clipping. Sector-neutral top 20%, quarterly.
    """

    name = "decay_adjusted_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        clip_val: float = 2.0,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.clip_val = clip_val
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        recent_ret = prices_panel.pct_change(21)
        tilt = _cs_zscore(recent_ret).clip(-1.0, 1.0).fillna(0.0)
        # Signal 1: recent-epoch GP
        z1 = (_cs_zscore(gp_ta).fillna(0.0) * (1.0 + 0.3 * tilt)).clip(
            -self.clip_val, self.clip_val
        )
        # Signal 2: shrunk composite (GP + ROE + accruals)
        row_std_gp = _cs_zscore(gp_ta).std(axis=1).rolling(252, min_periods=63).mean()
        row_std_roe = _cs_zscore(roe).std(axis=1).rolling(252, min_periods=63).mean()
        row_std_acc = _cs_zscore(-accruals).std(axis=1).rolling(252, min_periods=63).mean()
        w_gp = 1.0 / (1.0 + row_std_gp.clip(lower=0.1))
        w_roe = 1.0 / (1.0 + row_std_roe.clip(lower=0.1))
        w_acc = 1.0 / (1.0 + row_std_acc.clip(lower=0.1))
        shrunk = (
            _cs_zscore(gp_ta).fillna(0.0).multiply(w_gp, axis=0)
            + _cs_zscore(roe).fillna(0.0).multiply(w_roe, axis=0)
            + _cs_zscore(-accruals).fillna(0.0).multiply(w_acc, axis=0)
        )
        z2 = _cs_zscore(shrunk).clip(-self.clip_val, self.clip_val).fillna(0.0)
        # Signal 3: plain GP/TA anchor
        z3 = _cs_zscore(gp_ta).clip(-self.clip_val, self.clip_val).fillna(0.0)
        score = z1 + z2 + z3
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class LowDilutionGovernance(CrossSectionalStrategy):
    """H381: Long stocks with lowest annual share dilution rate.

    dilution = (shares_t - shares_{t-252}) / shares_{t-252}.
    Low dilution = management not extracting value via issuance.
    Governance proxy (Gompers-Ishii-Metrick 2003). Sector-neutral, quarterly.
    """

    name = "low_dilution_governance"

    def __init__(
        self,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        sh = _align_fund(self.shares_outstanding, prices_panel)
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        daily_w = _sector_neutral_weights(
            dilution, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class BuybackQualityGovernance(CrossSectionalStrategy):
    """H382: Long stocks with declining share count (net buybacks).

    net_buyback_signal = max(0, -dilution) = value when shares decrease.
    Only positive values enter the score; no signal if shares increase.
    Bebchuk-Cohen-Ferrell 2009: management returning capital is measurable.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "buyback_quality_governance"

    def __init__(
        self,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        sh = _align_fund(self.shares_outstanding, prices_panel)
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        net_buyback = (-dilution).clip(lower=0.0)
        daily_w = _sector_neutral_weights(
            net_buyback, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CapitalDisciplineGovernance(CrossSectionalStrategy):
    """H383: Governance composite: z(-dilution) + z(net_buyback) + z(-accruals).

    Three independent governance proxies combined. Low dilution and active
    buybacks signal capital return focus; low accruals signal cash-backed quality.
    Edmans 2011, Bebchuk 2009: composite governance beats any single proxy.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "capital_discipline_governance"

    def __init__(
        self,
        shares_outstanding: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.shares_outstanding = shares_outstanding
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        net_buyback = (-dilution).clip(lower=0.0)
        avg_ta_safe = ((ta + ta.shift(self.lookback)) / 2).where(ta > 0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        score = (
            _cs_zscore(-dilution).fillna(0.0)
            + _cs_zscore(net_buyback).fillna(0.0)
            + _cs_zscore(-accruals).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class GovernanceQualityComposite(CrossSectionalStrategy):
    """H384: Capital discipline governance composite filtered by GP/TA quality gate.

    Requires GP/TA > median within membership before ranking by governance score.
    Governance signal only meaningful for fundamentally healthy firms.
    Edmans 2011: governance predicts returns especially when fundamentals strong.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "governance_quality_composite"

    def __init__(
        self,
        shares_outstanding: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        gross_profit: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.shares_outstanding = shares_outstanding
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.gross_profit = gross_profit
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(self.lookback)) / 2).where(ta > 0)
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        net_buyback = (-dilution).clip(lower=0.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        gov_score = (
            _cs_zscore(-dilution).fillna(0.0)
            + _cs_zscore(net_buyback).fillna(0.0)
            + _cs_zscore(-accruals).fillna(0.0)
        )
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        mem = membership.reindex_like(gp_ta).fillna(False)
        gp_median = gp_ta.where(mem).median(axis=1)
        quality_gate = gp_ta.ge(gp_median, axis=0) & mem
        gated = gov_score.where(quality_gate, np.nan)
        daily_w = _sector_neutral_weights(
            gated, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class GovernanceMDRComposite(CrossSectionalStrategy):
    """H385: Equal-weight composite of batch-7 governance proxy signals.

    Combines z(-dilution), z(net_buyback), z(-accruals), z(GP/TA) with
    governance composite gate. Tests whether governance signals diversify
    beyond pure quality signals. Sector-neutral top 20%, quarterly.
    """

    name = "governance_mdr_composite"

    def __init__(
        self,
        shares_outstanding: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        gross_profit: pd.DataFrame,
        sectors: pd.Series,
        clip_val: float = 2.0,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.shares_outstanding = shares_outstanding
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.gross_profit = gross_profit
        self.sectors = sectors
        self.clip_val = clip_val
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        sh_lag = sh.shift(252)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        net_buyback = (-dilution).clip(lower=0.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)

        z1 = _cs_zscore(-dilution).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z2 = _cs_zscore(net_buyback).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z3 = _cs_zscore(-accruals).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z4 = _cs_zscore(gp_ta).clip(-self.clip_val, self.clip_val).fillna(0.0)

        score = z1 + z2 + z3 + z4
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class SinExclusionPremium(CrossSectionalStrategy):
    """H386: Equal-weight S&P 500 excluding Consumer Staples sector.

    Proxy for sin-stock exclusion (Hong-Kacperczyk 2009): Consumer Staples
    includes tobacco, alcohol, and other socially screened categories.
    Tests whether sector-level exclusion pressure affects large-cap returns.
    Equal-weight, quarterly rebalance.
    """

    name = "sin_exclusion_premium"

    def __init__(
        self,
        sectors: pd.Series,
        excluded_sector: str = "Consumer Staples",
        rebalance_period: int = 63,
    ) -> None:
        self.sectors = sectors
        self.excluded_sector = excluded_sector
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        excluded_tickers = set(self.sectors[self.sectors == self.excluded_sector].index.tolist())
        mem = membership.copy()
        for t in excluded_tickers:
            if t in mem.columns:
                mem[t] = False
        n_inc = mem.sum(axis=1).replace(0, np.nan)
        daily_w = mem.astype(float).div(n_inc, axis=0).fillna(0.0)
        return _periodic_rebalance(daily_w, self.rebalance_period)


class LowCarbonSectorNeutral(CrossSectionalStrategy):
    """H387: Sector-neutral GP/TA quality excluding Energy sector.

    Proxy for low-carbon quality portfolio (Bolton-Kacperczyk 2021):
    Energy sector has highest carbon intensity. Excluding it and doing
    sector-neutral GP/TA rank captures carbon-risk-free quality signal.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "low_carbon_sector_neutral"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        excluded_sector: str = "Energy",
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.excluded_sector = excluded_sector
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        excluded_tickers = set(self.sectors[self.sectors == self.excluded_sector].index.tolist())
        filtered_mem = membership.copy()
        for t in excluded_tickers:
            if t in filtered_mem.columns:
                filtered_mem[t] = False
        daily_w = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class GreenProxyQuality(CrossSectionalStrategy):
    """H388: GP/TA quality ranked within Utilities + Information Technology.

    Green proxy: these two sectors have lowest carbon intensity in S&P 500.
    Tests whether a sector-concentrated quality signal within the green proxy
    earns distinct returns from full-universe quality.
    Pastor-Stambaugh-Taylor 2022. Equal-weight top 20%, quarterly.
    """

    name = "green_proxy_quality"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        green_sectors: tuple = ("Utilities", "Information Technology"),
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.green_sectors = green_sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        green_tickers = set(self.sectors[self.sectors.isin(self.green_sectors)].index.tolist())
        green_mem = pd.DataFrame(False, index=membership.index, columns=membership.columns)
        for t in green_tickers:
            if t in green_mem.columns:
                green_mem[t] = membership[t]
        n_green = green_mem.sum(axis=1).replace(0, np.nan)
        if (n_green < 2).all():
            n = membership.sum(axis=1).replace(0, np.nan)
            return _periodic_rebalance(
                membership.astype(float).div(n, axis=0).fillna(0.0), self.rebalance_period
            )
        daily_w = _equal_weight_top_quantile(
            gp_ta, green_mem, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class ControversyProxyAvoid(CrossSectionalStrategy):
    """H389: GP/TA quality excluding worst-decile controversy proxy.

    Controversy proxy = z(leverage) + z(dilution) + z(-GP/TA).
    Avoid bottom decile (most controversial). Long remaining stocks by GP/TA.
    Proxy for ESG exclusion without vendor data (Hong-Kacperczyk 2009 inspired).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "controversy_proxy_avoid"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        exclude_pct: float = 0.10,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.long_term_debt = long_term_debt
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.lookback = lookback
        self.exclude_pct = exclude_pct
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        leverage = (ltd / ta_safe).clip(0.0, 2.0)
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        controversy = (
            _cs_zscore(leverage).fillna(0.0)
            + _cs_zscore(dilution).fillna(0.0)
            + _cs_zscore(-gp_ta).fillna(0.0)
        )
        mem = membership.reindex_like(controversy).fillna(False)
        n_members = mem.sum(axis=1).replace(0, np.nan)
        excl_n = (n_members * self.exclude_pct).clip(lower=1)
        ranks_asc = controversy.where(mem).rank(axis=1, ascending=True, na_option="keep")
        excluded = (ranks_asc.ge(ranks_asc.max(axis=1) - excl_n, axis=0)) & ranks_asc.notna()
        filtered_mem = mem & ~excluded
        daily_w = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class ESGProxyComposite(CrossSectionalStrategy):
    """H390: Composite of batch-8 ESG/controversy proxy signals.

    Blends: ex-Consumer-Staples quality, ex-Energy quality, controversy-
    avoided quality, and plain GP/TA anchor. Tests whether sector-exclusion
    and controversy-avoidance signals add diversification.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "esg_proxy_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        clip_val: float = 2.0,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.long_term_debt = long_term_debt
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.clip_val = clip_val
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        leverage = (ltd / ta_safe).clip(0.0, 2.0)
        sh_lag = sh.shift(252)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)

        # ex-staples GP
        staples_t = set(self.sectors[self.sectors == "Consumer Staples"].index)
        mem_ex_st = membership.copy()
        for t in staples_t:
            if t in mem_ex_st.columns:
                mem_ex_st[t] = False
        daily_w1 = _sector_neutral_weights(
            gp_ta, mem_ex_st, self.sectors, self.within_sector_quantile, ascending=False
        )

        # ex-energy GP
        energy_t = set(self.sectors[self.sectors == "Energy"].index)
        mem_ex_en = membership.copy()
        for t in energy_t:
            if t in mem_ex_en.columns:
                mem_ex_en[t] = False
        daily_w2 = _sector_neutral_weights(
            gp_ta, mem_ex_en, self.sectors, self.within_sector_quantile, ascending=False
        )

        # controversy-avoided GP
        controversy = (
            _cs_zscore(leverage).fillna(0.0)
            + _cs_zscore(dilution).fillna(0.0)
            + _cs_zscore(-gp_ta).fillna(0.0)
        )
        mem = membership.reindex_like(controversy).fillna(False)
        n_members = mem.sum(axis=1).replace(0, np.nan)
        excl_n = (n_members * 0.10).clip(lower=1)
        ranks_asc = controversy.where(mem).rank(axis=1, ascending=True, na_option="keep")
        excluded = (ranks_asc.ge(ranks_asc.max(axis=1) - excl_n, axis=0)) & ranks_asc.notna()
        filtered_mem = mem & ~excluded
        daily_w3 = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )

        # plain GP/TA anchor
        daily_w4 = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )

        combined = daily_w1 + daily_w2 + daily_w3 + daily_w4
        row_sum = combined.sum(axis=1).replace(0, np.nan)
        daily_w = combined.div(row_sum, axis=0).fillna(0.0)
        return _periodic_rebalance(daily_w, self.rebalance_period)


class LowAttentionValuePort(CrossSectionalStrategy):
    """H391: GP/TA quality rank dampened by recent price-activity (attention proxy).

    Attention proxy = trailing 21d mean(|daily_return|) / trailing 63d mean(|daily_return|).
    High ratio = stock is drawing attention; low ratio = neglected.
    Signal = z(GP/TA) * (1 - clip(z(attention_ratio), 0, 2) * 0.3).
    Neglected quality stocks reprice more slowly without event-risk noise.
    Antweiler-Frank 2004. Sector-neutral top 20%, monthly rebalance.
    """

    name = "low_attention_value_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        short_window: int = 21,
        long_window: int = 63,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.short_window = short_window
        self.long_window = long_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)

        daily_abs_ret = prices_panel.pct_change().abs()
        short_act = daily_abs_ret.rolling(self.short_window, min_periods=5).mean()
        long_act = daily_abs_ret.rolling(self.long_window, min_periods=20).mean()
        attention_ratio = (short_act / long_act.replace(0, np.nan)).clip(0.0, 5.0)

        z_gp = _cs_zscore(gp_ta).fillna(0.0)
        z_att = _cs_zscore(attention_ratio).clip(0.0, 2.0).fillna(0.0)
        score = z_gp * (1.0 - 0.3 * z_att)

        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class ConsistentMomentumPort(CrossSectionalStrategy):
    """H392: 12-1 momentum filtered by 3-month momentum agreement (consistency gate).

    Only hold momentum stocks where both 12m and 3m momentum are positive and
    in the top half of the cross-section. Consistent trend = "crowd-confirmed"
    signal. Chen-De-Hu-Hwang 2014: crowd agreement strengthens anomaly signals.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "consistent_momentum_port"

    def __init__(
        self,
        sectors: pd.Series,
        long_window: int = 252,
        skip_window: int = 21,
        short_window: int = 63,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.sectors = sectors
        self.long_window = long_window
        self.skip_window = skip_window
        self.short_window = short_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        mom_long = prices_panel.pct_change(self.long_window).shift(self.skip_window)
        mom_short = prices_panel.pct_change(self.short_window)
        mem = membership.reindex_like(mom_long).fillna(False)
        med_long = mom_long.where(mem).median(axis=1)
        med_short = mom_short.where(mem).median(axis=1)
        # Gate: both signals must be above their cross-sectional median
        confirmed = mom_long.ge(med_long, axis=0) & mom_short.ge(med_short, axis=0)
        gated_mom = mom_long.where(confirmed, np.nan)
        daily_w = _sector_neutral_weights(
            gated_mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class LowVolDisagreementAvoid(CrossSectionalStrategy):
    """H393: GP/TA quality weighted by inverse realized volatility.

    Position weight inversely proportional to 21d realized vol (high vol =
    high disagreement → reduce weight). Antweiler-Frank 2004: disagreement
    predicts volatility. Penalising high-vol names improves Sharpe.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "low_vol_disagreement_avoid"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        vol_window: int = 21,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.vol_window = vol_window
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        daily_ret = prices_panel.pct_change()
        realized_vol = daily_ret.rolling(self.vol_window, min_periods=5).std().shift(1)
        inv_vol = (1.0 / realized_vol.replace(0, np.nan)).clip(0.0, 100.0)
        base_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        scaled = (base_w * inv_vol).fillna(0.0)
        row_sum = scaled.sum(axis=1).replace(0, np.nan)
        rescaled = scaled.div(row_sum, axis=0).fillna(0.0)
        return _monthly_rebalance(rescaled, prices_panel)


class ReturnSpikeRiskFilter(CrossSectionalStrategy):
    """H394: GP/TA quality with 50% weight cut when |return| spikes > 2 sigma.

    Detects event risk via absolute return z-score. If a stock's 5d mean
    absolute return exceeds 2 sigma of its trailing 63d distribution, cut
    position weight by 50%. Reduces exposure to M&A, earnings gaps, short
    squeezes. Sector-neutral top 20%, monthly rebalance.
    """

    name = "return_spike_risk_filter"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        spike_window: int = 5,
        lookback_window: int = 63,
        spike_threshold: float = 2.0,
        cut_factor: float = 0.5,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.spike_window = spike_window
        self.lookback_window = lookback_window
        self.spike_threshold = spike_threshold
        self.cut_factor = cut_factor
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0)
        # Sustainability filter (H265-style) then GP rank
        sust = _cs_zscore(-accruals).fillna(0.0) + _cs_zscore(-asset_growth).fillna(0.0)
        mem = membership.reindex_like(gp_ta).fillna(False)
        sust_rank = sust.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        filtered_mem = membership & (sust_rank >= 0.50).fillna(False)
        base_w = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        # Spike detection
        abs_ret = prices_panel.pct_change().abs()
        recent_act = abs_ret.rolling(self.spike_window, min_periods=1).mean()
        hist_mean = (
            abs_ret.rolling(self.lookback_window, min_periods=20).mean().shift(self.spike_window)
        )
        hist_std = (
            abs_ret.rolling(self.lookback_window, min_periods=20).std().shift(self.spike_window)
        )
        z_spike = (recent_act - hist_mean) / hist_std.replace(0, np.nan)
        spike_flag = z_spike.gt(self.spike_threshold).fillna(False)
        scale = spike_flag.astype(float) * (self.cut_factor - 1.0) + 1.0
        scaled = (base_w * scale).clip(lower=0.0)
        row_sum = scaled.sum(axis=1).replace(0, np.nan)
        rescaled = scaled.div(row_sum, axis=0).fillna(0.0)
        return _monthly_rebalance(rescaled, prices_panel)


class VolatilityAwareComposite(CrossSectionalStrategy):
    """H395: Equal-weight blend of batch-9 volatility/attention-aware signals.

    Combines: neglected quality (GP dampened by activity), consistent momentum
    (12m+3m agreement), GP x inverse-vol, and spike-filtered GP.
    Tests whether price-based attention proxies diversify vs pure EDGAR quality.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "volatility_aware_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        clip_val: float = 2.0,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.clip_val = clip_val
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0)
        daily_abs_ret = prices_panel.pct_change().abs()
        short_act = daily_abs_ret.rolling(21, min_periods=5).mean()
        long_act = daily_abs_ret.rolling(63, min_periods=20).mean()
        attention_ratio = (short_act / long_act.replace(0, np.nan)).clip(0.0, 5.0)
        mom_long = prices_panel.pct_change(252).shift(21)
        mom_short = prices_panel.pct_change(63)
        realized_vol = prices_panel.pct_change().rolling(21, min_periods=5).std().shift(1)
        inv_vol = (1.0 / realized_vol.replace(0, np.nan)).clip(0.0, 100.0)

        mem = membership.reindex_like(gp_ta).fillna(False)
        med_long = mom_long.where(mem).median(axis=1)
        med_short = mom_short.where(mem).median(axis=1)
        confirmed_mom = mom_long.where(
            mom_long.ge(med_long, axis=0) & mom_short.ge(med_short, axis=0), np.nan
        )

        z_gp = _cs_zscore(gp_ta)
        z_att = _cs_zscore(attention_ratio).clip(0.0, 2.0).fillna(0.0)
        neglect_score = (z_gp * (1.0 - 0.3 * z_att)).fillna(0.0)

        sust = _cs_zscore(-accruals).fillna(0.0) + _cs_zscore(-asset_growth).fillna(0.0)
        sust_rank = sust.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        filtered_mem = membership & (sust_rank >= 0.50).fillna(False)

        z1 = _cs_zscore(neglect_score).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z2 = _cs_zscore(confirmed_mom).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z3 = _cs_zscore(gp_ta * inv_vol).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z4 = _cs_zscore(gp_ta.where(filtered_mem)).clip(-self.clip_val, self.clip_val).fillna(0.0)
        score = z1 + z2 + z3 + z4

        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class EarningsPersistencePort(CrossSectionalStrategy):
    """H406: Long stocks with most-stable NI/TA over 2 years (low earnings variance).

    Stability = 1 / rolling_std(NI/TA, 2 years). Stable earnings = higher-quality
    disclosure (Li 2008 readability proxy) → better future earnings persistence.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "earnings_persistence_port"

    def __init__(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        rolling_std = ni_ta.rolling(self.window, min_periods=self.window // 4).std()
        # Stability = inverse of variance; low std = high stability
        stability = 1.0 / rolling_std.replace(0, np.nan).clip(lower=1e-6)
        daily_w = _sector_neutral_weights(
            stability, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CashFlowReliabilityPort(CrossSectionalStrategy):
    """H407: Long stocks where OCF consistently exceeds NI over 2 years.

    Rolling fraction of periods with OCF > NI. High fraction = cash earnings
    reliably dominate accruals → durable financial quality. Proxy for Li (2010):
    positive forward-looking tone firms have more durable earnings.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "cash_flow_reliability_port"

    def __init__(
        self,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ocf_exceeds_ni = (ocf > ni).astype(float)
        reliability = ocf_exceeds_ni.rolling(self.window, min_periods=self.window // 4).mean()
        daily_w = _sector_neutral_weights(
            reliability, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class BookValueGrowthPort(CrossSectionalStrategy):
    """H408: Long top-quintile by annual book equity growth rate.

    Book equity growth = book_equity / book_equity.shift(252) - 1. Growing book
    equity = retained earnings quality = management reinvesting profits confidently.
    Proxy for Li (2010) forward-looking positive tone.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "book_value_growth_port"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.book_equity = book_equity
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        be = _align_fund(self.book_equity, prices_panel)
        be_lag = be.shift(self.lookback)
        be_safe_lag = be_lag.where(be_lag.abs() > 0)
        be_growth = ((be - be_lag) / be_safe_lag.abs()).clip(-1.0, 5.0)
        daily_w = _sector_neutral_weights(
            be_growth, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class BalanceSheetStrengthPort(CrossSectionalStrategy):
    """H409: Long top-quintile by composite balance sheet improvement.

    Z-score composite of: (1) debt/assets declining YoY, (2) OCF/TA improving YoY,
    (3) book equity/TA improving YoY. All three signals together proxy for positive
    disclosure trajectory (Li 2010 forward-looking statements content).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "balance_sheet_strength_port"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        operating_cf: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.book_equity = book_equity
        self.long_term_debt = long_term_debt
        self.operating_cf = operating_cf
        self.total_assets = total_assets
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        be = _align_fund(self.book_equity, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        # Ratios
        debt_ratio = (ltd / ta_safe).clip(0.0, 5.0)
        ocf_ratio = (ocf / ta_safe).clip(-2.0, 5.0)
        be_ratio = (be / ta_safe).clip(-1.0, 5.0)
        # YoY improvements
        debt_improve = -(debt_ratio - debt_ratio.shift(self.lookback))  # negative = improving
        ocf_improve = ocf_ratio - ocf_ratio.shift(self.lookback)
        be_improve = be_ratio - be_ratio.shift(self.lookback)
        composite = (
            _cs_zscore(debt_improve).fillna(0.0)
            + _cs_zscore(ocf_improve).fillna(0.0)
            + _cs_zscore(be_improve).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class WorkingCapitalHealthPort(CrossSectionalStrategy):
    """H410: Long top-quintile by improving (CA − CL) / Revenue ratio.

    Working capital efficiency = (CA − CL) / Revenue. YoY improvement = better
    operational discipline and cash conversion. Proxy for Li (2010) FLS: improving
    operational signals correlate with positive forward-looking tone.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "working_capital_health_port"

    def __init__(
        self,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.revenue = revenue
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ca = _align_fund(self.current_assets, prices_panel)
        cl = _align_fund(self.current_liabilities, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        wc = ca - cl
        wc_ratio = (wc / rev_safe).clip(-2.0, 5.0)
        wc_improve = wc_ratio - wc_ratio.shift(self.lookback)
        daily_w = _sector_neutral_weights(
            wc_improve, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class ConsistentGrowthStreak(CrossSectionalStrategy):
    """H411: Long stocks with persistent simultaneous revenue + NI growth.

    Rolling fraction of days over 2 years where YoY revenue growth > 0 AND
    YoY NI growth > 0. High fraction = persistent positive management tone proxy
    (Mayew & Venkatachalam 2012): confident management → better future performance.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "consistent_growth_streak"

    def __init__(
        self,
        revenue: pd.DataFrame,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        growth_lookback: int = 252,
        streak_window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.revenue = revenue
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.growth_lookback = growth_lookback
        self.streak_window = streak_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        rev_growth_pos = (rev > rev.shift(self.growth_lookback)).astype(float)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        ni_growth_pos = (ni_ta > ni_ta.shift(self.growth_lookback)).astype(float)
        both_growing = rev_growth_pos * ni_growth_pos
        streak = both_growing.rolling(
            self.streak_window, min_periods=self.streak_window // 4
        ).mean()
        daily_w = _sector_neutral_weights(
            streak, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class FundamentalDivergenceAvoid(CrossSectionalStrategy):
    """H412: Long stocks where fundamentals outpace price; avoid the reverse.

    Divergence score = z(NI/TA YoY growth) − z(6m price return). Positive divergence
    = fundamentals running ahead of price (value opportunity). Negative divergence
    = price running ahead of fundamentals (torpedo risk). Monthly rebalance to capture
    price dynamics. Sector-neutral top 20%.
    """

    name = "fundamental_divergence_avoid"

    def __init__(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        fundamental_lookback: int = 252,
        price_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.fundamental_lookback = fundamental_lookback
        self.price_lookback = price_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        ni_growth = (ni_ta - ni_ta.shift(self.fundamental_lookback)).clip(-2.0, 5.0)
        price_ret = prices_panel / prices_panel.shift(self.price_lookback) - 1
        divergence = _cs_zscore(ni_growth).fillna(0.0) - _cs_zscore(price_ret).fillna(0.0)
        daily_w = _sector_neutral_weights(
            divergence, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MarginExpansionPort(CrossSectionalStrategy):
    """H413: Long top-quintile by gross margin expansion + revenue growth confirm.

    GP/Revenue YoY change (margin expansion) with revenue growth > 0 filter.
    Expanding margins with growing revenue = pricing power + competitive advantage.
    Mayew & Venkatachalam proxy: confident management → improving margins.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "margin_expansion_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        gp_margin = (gp / rev_safe).clip(0.0, 2.0)
        margin_expansion = gp_margin - gp_margin.shift(self.lookback)
        rev_growth_pos = (rev > rev.shift(self.lookback)).astype(float)
        # Only consider stocks with positive revenue growth
        score = margin_expansion.where(rev_growth_pos.astype(bool))
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class EarningsVariabilityAvoid(CrossSectionalStrategy):
    """H414: Long stocks with lowest NI/TA rolling variance (stable earnings).

    Low NI volatility = predictable, stable earnings environment. Proxy for
    Mayew & Venkatachalam (2012): uncertain management (high vocal stress) predicts
    weaker future performance. Low NI variance = low management uncertainty proxy.
    Sector-neutral bottom-20%-variance (ascending=True), quarterly rebalance.
    """

    name = "earnings_variability_avoid"

    def __init__(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.window = window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        variability = ni_ta.rolling(self.window, min_periods=self.window // 4).std()
        # ascending=True: lowest variability gets selected (best quality)
        daily_w = _sector_neutral_weights(
            variability, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class StableMarginMomentum(CrossSectionalStrategy):
    """H415: Composite of low GP/Revenue volatility + 6m price momentum.

    Stable gross margins (low GP/Revenue rolling std) combined with positive
    6m price momentum. Stocks with both confirmed price trend AND stable business
    model have double confirmation. Proxy for credible, consistent management.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "stable_margin_momentum"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        margin_window: int = 504,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.sectors = sectors
        self.margin_window = margin_window
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        gp_margin = (gp / rev_safe).clip(0.0, 2.0)
        margin_std = gp_margin.rolling(
            self.margin_window, min_periods=self.margin_window // 4
        ).std()
        # Low margin std = stable; negate so high stability = high score
        margin_stability = -margin_std
        # 6m momentum
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        composite = _cs_zscore(margin_stability).fillna(0.0) + _cs_zscore(momentum).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class DifferentiatedMarginPort(CrossSectionalStrategy):
    """H416: Long firms with high AND stable gross margins (differentiation proxy).

    z(GP/Revenue level) + z(-std(GP/Revenue, 2yr)). Persistently high, stable
    margins = product differentiation (Hoberg & Phillips 2016 TNIC proxy).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "differentiated_margin_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        stability_window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.sectors = sectors
        self.stability_window = stability_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        gp_margin = (gp / rev_safe).clip(0.0, 2.0)
        margin_std = gp_margin.rolling(
            self.stability_window, min_periods=self.stability_window // 4
        ).std()
        composite = _cs_zscore(gp_margin).fillna(0.0) + _cs_zscore(-margin_std).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CompetitivePressureAvoid(CrossSectionalStrategy):
    """H417: Avoid margin compression + revenue decel; rank rest by GP/TA quality.

    Filter: GP/Revenue declining AND revenue growth decelerating = losing market
    position (Hoberg, Phillips & Prabhala 2014). Remove these, rank survivors.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "competitive_pressure_avoid"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        accel_lag: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.total_assets = total_assets
        self.sectors = sectors
        self.lookback = lookback
        self.accel_lag = accel_lag
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        rev_safe = rev.where(rev > 0)
        gp_margin = (gp / rev_safe).clip(0.0, 2.0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        margin_declining = gp_margin < gp_margin.shift(self.lookback)
        rev_growth = (rev - rev.shift(self.lookback)) / rev.shift(self.lookback).abs().clip(
            lower=1.0
        )
        rev_decelerating = rev_growth < rev_growth.shift(self.accel_lag)
        under_threat = (margin_declining & rev_decelerating).fillna(False)
        safe_mem = membership & ~under_threat
        daily_w = _sector_neutral_weights(
            gp_ta, safe_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class PeerGroupMomentum(CrossSectionalStrategy):
    """H418: Sector-relative 6m momentum winners with NI growth confirmation.

    Proxy for peer group information spillover: sector leaders with improving
    fundamentals likely benefiting from positive peer earnings signals
    (Hoberg & Phillips 2016). Sector-neutral top 20%, monthly rebalance.
    """

    name = "peer_group_momentum"

    def __init__(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        ni_lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.mom_lookback = mom_lookback
        self.ni_lookback = ni_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        ni_growing = (ni_ta > ni_ta.shift(self.ni_lookback)).fillna(False)
        mom = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        mem_f = membership.reindex_like(mom).fillna(False)
        rel_mom = pd.DataFrame(0.0, index=mom.index, columns=mom.columns)
        for sector in self.sectors.dropna().unique():
            tickers = [t for t in self.sectors[self.sectors == sector].index if t in mom.columns]
            if not tickers:
                continue
            sec_m = mom[tickers].where(mem_f[tickers])
            med = sec_m.median(axis=1)
            rel_mom[tickers] = sec_m.sub(med, axis=0).fillna(0.0).values
        quality_mem = membership & ni_growing
        daily_w = _sector_neutral_weights(
            rel_mom, quality_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CashDefenseProxy(CrossSectionalStrategy):
    """H419: Avoid firms building defensive cash under competitive threat.

    Filter: CA/TA rising AND GP/Revenue declining = defensive cash hoarding.
    Hoberg, Phillips & Prabhala (2014): product market threat triggers cash buildup.
    Long survivors ranked by GP/TA quality. Sector-neutral top 20%, quarterly.
    """

    name = "cash_defense_proxy"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        current_assets: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.total_assets = total_assets
        self.current_assets = current_assets
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ca = _align_fund(self.current_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        rev_safe = rev.where(rev > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        gp_margin = (gp / rev_safe).clip(0.0, 2.0)
        ca_ratio = (ca / ta_safe).clip(0.0, 2.0)
        ca_rising = (ca_ratio > ca_ratio.shift(self.lookback)).fillna(False)
        margin_declining = (gp_margin < gp_margin.shift(self.lookback)).fillna(False)
        defensive_hoarding = ca_rising & margin_declining
        safe_mem = membership & ~defensive_hoarding
        daily_w = _sector_neutral_weights(
            gp_ta, safe_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MarketShareProxy(CrossSectionalStrategy):
    """H420: Long top-quintile sector-relative revenue growth (market share gainers).

    Revenue growth - sector median revenue growth. Proxy for market share gains
    (Hoberg & Phillips 2016 without product text). Sector-neutral top 20%, quarterly.
    """

    name = "market_share_proxy"

    def __init__(
        self,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.revenue = revenue
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        rev_lag = rev.shift(self.lookback)
        rev_growth = ((rev - rev_lag) / rev_lag.abs().clip(lower=1.0)).clip(-1.0, 5.0)
        mem_f = membership.reindex_like(rev_growth).fillna(False)
        sec_rel_rev = pd.DataFrame(0.0, index=rev_growth.index, columns=rev_growth.columns)
        for sector in self.sectors.dropna().unique():
            tickers = [
                t for t in self.sectors[self.sectors == sector].index if t in rev_growth.columns
            ]
            if not tickers:
                continue
            sec_g = rev_growth[tickers].where(mem_f[tickers])
            med = sec_g.median(axis=1)
            sec_rel_rev[tickers] = sec_g.sub(med, axis=0).fillna(0.0).values
        daily_w = _sector_neutral_weights(
            sec_rel_rev, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class AssetEfficiencyGrowth(CrossSectionalStrategy):
    """H426: Long top-quintile by improving Revenue/TA (asset turnover growth).

    Innovation efficiency = output per input. Revenue/TA improvement proxies
    for increasing return on knowledge capital (Hirshleifer et al. 2013).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "asset_efficiency_growth"

    def __init__(
        self,
        revenue: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.revenue = revenue
        self.total_assets = total_assets
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        rev = _align_fund(self.revenue, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        at = (rev / ta_safe).clip(0.0, 10.0)
        at_growth = at - at.shift(self.lookback)
        daily_w = _sector_neutral_weights(
            at_growth, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class CapitalProductivityPort(CrossSectionalStrategy):
    """H427: Long top-quintile by GP/TA YoY growth (knowledge capital productivity).

    GP/TA growth rate (different from H239 GP/TA level). Captures improving
    return on capital — proxy for Kogan et al. (2017) patent-value shocks.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "capital_productivity_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        gp_ta_growth = gp_ta - gp_ta.shift(self.lookback)
        daily_w = _sector_neutral_weights(
            gp_ta_growth, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class GrowthReinvestmentQuality(CrossSectionalStrategy):
    """H428: Quality + growth + retention composite as disciplined reinvestment proxy.

    z(GP/TA) + z(revenue growth) + z(book equity growth). Three factors.
    Proxy for Eberhart et al. (2004): unexpected R&D with balance-sheet strength.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "growth_reinvestment_quality"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        revenue: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.revenue = revenue
        self.book_equity = book_equity
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        rev_growth = (
            (rev - rev.shift(self.lookback)) / rev.shift(self.lookback).abs().clip(lower=1.0)
        ).clip(-1.0, 5.0)
        be_growth = (
            (be - be.shift(self.lookback)) / be.shift(self.lookback).abs().clip(lower=1.0)
        ).clip(-1.0, 5.0)
        composite = (
            _cs_zscore(gp_ta).fillna(0.0)
            + _cs_zscore(rev_growth).fillna(0.0)
            + _cs_zscore(be_growth).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class SectorRelativeMarginProxy(CrossSectionalStrategy):
    """H429: Sector-relative GP/Revenue premium + 6m momentum composite.

    z(GP/Revenue - sector median) + z(6m momentum). Above-sector-median GP margin
    = innovation/intangible intensity vs peers (Hirshleifer et al. 2013 proxy).
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "sector_relative_margin_proxy"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        revenue: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.gross_profit = gross_profit
        self.revenue = revenue
        self.sectors = sectors
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        rev_safe = rev.where(rev > 0)
        gp_margin = (gp / rev_safe).clip(0.0, 2.0)
        mem_f = membership.reindex_like(gp_margin).fillna(False)
        sec_rel = pd.DataFrame(0.0, index=gp_margin.index, columns=gp_margin.columns)
        for sector in self.sectors.dropna().unique():
            tickers = [
                t for t in self.sectors[self.sectors == sector].index if t in gp_margin.columns
            ]
            if not tickers:
                continue
            sec_m = gp_margin[tickers].where(mem_f[tickers])
            med = sec_m.median(axis=1)
            sec_rel[tickers] = sec_m.sub(med, axis=0).fillna(0.0).values
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        composite = _cs_zscore(sec_rel).fillna(0.0) + _cs_zscore(momentum).fillna(0.0)
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class OperatingLeverageImprovement(CrossSectionalStrategy):
    """H430: Long top-quintile by GP/NI YoY improvement (operating leverage building).

    Rising GP/NI = gross profit growing faster than net income = building scalable
    operating leverage. Proxy for Kogan et al. (2017) firms investing in growth.
    Only defined where NI > 0. Sector-neutral top 20%, quarterly rebalance.
    """

    name = "operating_leverage_improvement"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.net_income = net_income
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ni_safe = ni.where(ni > 0)
        op_lev = (gp / ni_safe).clip(0.0, 20.0)
        op_lev_change = op_lev - op_lev.shift(self.lookback)
        daily_w = _sector_neutral_weights(
            op_lev_change, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class SectorEWDivergence(CrossSectionalStrategy):
    """H437: Long sector-relative 5d reversal when fundamentals are positive.

    Stocks that underperformed their sector equal-weight return over 5 days
    (ETF-flow-driven dislocation) and have positive NI growth should mean-revert.
    Israeli et al. (2017): ETF flows create non-fundamental dislocations.
    Quality filter: NI/TA YoY growth > 0. Sector-neutral top 20%, weekly rebalance.
    """

    name = "sector_ew_divergence"

    def __init__(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        reversal_window: int = 5,
        ni_lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 5,
    ) -> None:
        self.net_income = net_income
        self.total_assets = total_assets
        self.sectors = sectors
        self.reversal_window = reversal_window
        self.ni_lookback = ni_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        ni_growing = (ni_ta > ni_ta.shift(self.ni_lookback)).fillna(False)
        ret_5d = prices_panel.pct_change(self.reversal_window)
        # Sector-relative 5d return: stock return minus sector equal-weight
        mem_f = membership.reindex_like(ret_5d).fillna(False)
        sector_rel = pd.DataFrame(0.0, index=ret_5d.index, columns=ret_5d.columns)
        for sector in self.sectors.dropna().unique():
            tickers = [t for t in self.sectors[self.sectors == sector].index if t in ret_5d.columns]
            if not tickers:
                continue
            sec_r = ret_5d[tickers].where(mem_f[tickers])
            sec_mean = sec_r.mean(axis=1)
            sector_rel[tickers] = sec_r.sub(sec_mean, axis=0).fillna(0.0).values
        # Reverse: long most negative sector-relative return (biggest laggards)
        score = (-sector_rel).where(ni_growing, other=np.nan)
        daily_w = _sector_neutral_weights(
            score,
            membership & ni_growing,
            self.sectors,
            self.within_sector_quantile,
            ascending=False,
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MarketCapConcentrationAvoid(CrossSectionalStrategy):
    """H436: GP/TA quality in mid-cap S&P 500 names (exclude mega-cap passive-crowded).

    Market cap = prices * shares. Within each sector, exclude top cap_exclude_pct
    by market cap (most passive-owned) and apply GP/TA quality to remainder.
    Ben-David et al. (2018): high ETF ownership -> higher vol, less idiosyncratic signal.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "market_cap_concentration_avoid"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        cap_exclude_pct: float = 0.10,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.cap_exclude_pct = cap_exclude_pct
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        shares = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        mktcap = (prices_panel * shares.clip(lower=0.0)).replace(0, np.nan)
        mem_f = membership.reindex_like(mktcap).fillna(False)
        # Exclude top cap_exclude_pct within each sector (mega-caps)
        mega_cap = pd.DataFrame(False, index=prices_panel.index, columns=prices_panel.columns)
        for sector in self.sectors.dropna().unique():
            tickers = [
                t for t in self.sectors[self.sectors == sector].index if t in prices_panel.columns
            ]
            if not tickers:
                continue
            sec_cap = mktcap[tickers].where(mem_f[tickers])
            n_mem = mem_f[tickers].sum(axis=1)
            thresh = (n_mem * (1.0 - self.cap_exclude_pct)).clip(lower=1)
            rank = sec_cap.rank(axis=1, ascending=True, na_option="keep")
            mega_cap[tickers] = rank.gt(thresh, axis=0)
        mid_cap_mem = membership & ~mega_cap
        daily_w = _sector_neutral_weights(
            gp_ta, mid_cap_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class InverseIndexWeightMomentum(CrossSectionalStrategy):
    """H438: 6m momentum weighted inversely by market cap rank within sector.

    Smaller S&P 500 names get larger positions (less passive-owned).
    Ben-David et al. (2018): ETF-heavy names have less information advantage.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "inverse_index_weight_momentum"

    def __init__(
        self,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        shares = _align_fund(self.shares_outstanding, prices_panel)
        mktcap = (prices_panel * shares.clip(lower=0.0)).replace(0, np.nan)
        momentum = prices_panel / prices_panel.shift(self.mom_lookback) - 1
        mem_f = membership.reindex_like(mktcap).fillna(False)
        # Inverse cap rank score within sector
        inv_cap = pd.DataFrame(0.0, index=prices_panel.index, columns=prices_panel.columns)
        for sector in self.sectors.dropna().unique():
            tickers = [
                t for t in self.sectors[self.sectors == sector].index if t in prices_panel.columns
            ]
            if not tickers:
                continue
            sec_cap = mktcap[tickers].where(mem_f[tickers])
            n_mem = mem_f[tickers].sum(axis=1).replace(0, np.nan)
            cap_rank = sec_cap.rank(axis=1, ascending=False, na_option="keep")
            inv_cap[tickers] = (cap_rank / n_mem.values[:, None]).fillna(0.0)
        # Score = momentum * inverse_cap_weight
        score = momentum * inv_cap.where(inv_cap > 0, np.nan)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class SectorCorrelationBreak(CrossSectionalStrategy):
    """H439: GP/TA quality with position sizing reduced when within-sector correlation spikes.

    High rolling within-sector correlation = ETF arbitrage dominating = shrink positions.
    Low correlation = stocks moving independently = full position.
    Israeli et al. (2017): high return synchronicity = ETF crowding.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "sector_correlation_break"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        corr_window: int = 63,
        high_corr_scale: float = 0.50,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.corr_window = corr_window
        self.high_corr_scale = high_corr_scale
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        base_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        daily_rets = prices_panel.pct_change()
        # Mean within-sector pairwise correlation per day
        mem_f = membership.reindex_like(prices_panel).fillna(False)
        sector_corr = pd.Series(0.0, index=prices_panel.index)
        for sector in self.sectors.dropna().unique():
            tickers = [
                t for t in self.sectors[self.sectors == sector].index if t in prices_panel.columns
            ]
            if len(tickers) < 2:
                continue
            sec_ret = daily_rets[tickers].where(mem_f[tickers])
            roll_corr = (
                sec_ret.rolling(self.corr_window, min_periods=20)
                .corr()
                .groupby(level=0)
                .mean()
                .mean(axis=1)
            )
            sector_corr = sector_corr.add(
                roll_corr.reindex(prices_panel.index).fillna(0.0), fill_value=0.0
            )
        # Normalize by n_sectors
        n_sectors = len(self.sectors.dropna().unique())
        mean_corr = (sector_corr / max(n_sectors, 1)).clip(0.0, 1.0)
        # Scale factor: reduce to high_corr_scale when corr > 75th pctile
        corr_thresh = mean_corr.rolling(252, min_periods=63).quantile(0.75)
        high_corr_regime = mean_corr.gt(corr_thresh.fillna(1.0))
        scale = pd.Series(1.0, index=prices_panel.index).where(
            ~high_corr_regime, self.high_corr_scale
        )
        scaled = base_w.mul(scale, axis=0)
        row_sum = scaled.sum(axis=1).replace(0, np.nan)
        rescaled = scaled.div(row_sum, axis=0).fillna(0.0)
        return _periodic_rebalance(rescaled, self.rebalance_period)


class LowIndexBetaQuality(CrossSectionalStrategy):
    """H440: GP/TA quality prioritized for low-beta (less passive-crowded) names.

    Score = GP/TA * (1 / beta) where beta = rolling 252d market beta.
    Low-beta S&P 500 names = less ETF-dominated = stronger fundamentals-to-price link.
    Ben-David et al. (2018): ETF-heavy names have higher index beta.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "low_index_beta_quality"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        spy_rets: pd.Series,
        sectors: pd.Series,
        beta_window: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.spy_rets = spy_rets
        self.sectors = sectors
        self.beta_window = beta_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        daily_rets = prices_panel.pct_change()
        spy = self.spy_rets.reindex(prices_panel.index).fillna(0.0)
        min_p = self.beta_window // 2
        cov = daily_rets.rolling(self.beta_window, min_periods=min_p).cov(spy)
        var = spy.rolling(self.beta_window, min_periods=min_p).var()
        beta = (cov / var.replace(0, np.nan)).clip(0.01, 5.0).shift(1)
        # Low beta = high inv_beta weight bonus
        inv_beta = (1.0 / beta).clip(0.0, 10.0)
        score = gp_ta * inv_beta
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class ShareholderAlignedPort(CrossSectionalStrategy):
    """H441: Low dilution + positive OCF + 6m momentum = dedicated-holder proxy.

    Shares YoY growth <= 0 (no dilution) AND OCF/TA > 0 AND 6m price > 0.
    All three conditions. Rank qualifiers by GP/TA quality.
    Yan & Zhang (2009): dedicated long-horizon institutions prefer these firms.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "shareholder_aligned_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        mom_lookback: int = 126,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.shares_outstanding = shares_outstanding
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.lookback = lookback
        self.mom_lookback = mom_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        shares = _align_fund(self.shares_outstanding, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        no_dilution = (shares <= shares.shift(self.lookback)).fillna(False)
        ocf_pos = (ocf > 0).fillna(False)
        mom_pos = (prices_panel > prices_panel.shift(self.mom_lookback)).fillna(False)
        aligned_mem = membership & no_dilution & ocf_pos & mom_pos
        daily_w = _sector_neutral_weights(
            gp_ta, aligned_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class TransientOwnershipAvoid(CrossSectionalStrategy):
    """H442: Avoid high revenue volatility + share issuance (transient ownership signal).

    Filter: rolling std(rev growth) high AND shares YoY increasing -> avoid.
    Bushee (1998): transient institutions pressure short-term EPS -> revenue instability.
    Long survivors ranked by GP/TA quality. Sector-neutral top 20%, quarterly.
    """

    name = "transient_ownership_avoid"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        revenue: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        vol_window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.revenue = revenue
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.lookback = lookback
        self.vol_window = vol_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        rev = _align_fund(self.revenue, prices_panel)
        shares = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        rev_growth = (
            (rev - rev.shift(self.lookback)) / rev.shift(self.lookback).abs().clip(lower=1.0)
        ).clip(-1.0, 5.0)
        rev_vol = rev_growth.rolling(self.vol_window, min_periods=self.vol_window // 4).std()
        # High revenue volatility: above cross-sectional median
        mem_f = membership.reindex_like(rev_vol).fillna(False)
        vol_median = rev_vol.where(mem_f).median(axis=1)
        high_rev_vol = rev_vol.gt(vol_median, axis=0).fillna(False)
        # Shares increasing (dilution)
        share_diluting = (shares > shares.shift(self.lookback)).fillna(False)
        transient_signal = high_rev_vol & share_diluting
        safe_mem = membership & ~transient_signal
        daily_w = _sector_neutral_weights(
            gp_ta, safe_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class DedicatedOwnerProxy(CrossSectionalStrategy):
    """H443: Consistent buyback + positive NI trend + GP/TA quality = dedicated-owner proxy.

    z(-shares_growth) + z(NI/TA growth) + z(GP/TA level). Three-factor composite.
    Yan & Zhang (2009): dedicated institutions accumulate in these firms.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "dedicated_owner_proxy"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        shares = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        ni_growth = ni_ta - ni_ta.shift(self.lookback)
        shares_lag = shares.shift(self.lookback)
        shares_growth = ((shares - shares_lag) / shares_lag.abs().clip(lower=1.0)).clip(-1.0, 5.0)
        composite = (
            _cs_zscore(-shares_growth).fillna(0.0)
            + _cs_zscore(ni_growth).fillna(0.0)
            + _cs_zscore(gp_ta).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class SmartAccumulationProxy(CrossSectionalStrategy):
    """H444: Price > 200d MA + NI/TA improving + GP/TA improving = smart accumulation.

    All three conditions required. Rank qualifiers by composite improvement score.
    Yan & Zhang (2009): short-term institutional accumulation predicts earnings surprises.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "smart_accumulation_proxy"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        ma_window: int = 200,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.sectors = sectors
        self.ma_window = ma_window
        self.lookback = lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        ni_ta = (ni / ta_safe).clip(-2.0, 5.0)
        price_ma = prices_panel.rolling(self.ma_window, min_periods=self.ma_window // 2).mean()
        price_above_ma = (prices_panel > price_ma).fillna(False)
        ni_improving = (ni_ta > ni_ta.shift(self.lookback)).fillna(False)
        gp_ta_improving = (gp_ta > gp_ta.shift(self.lookback)).fillna(False)
        accumulation_mem = membership & price_above_ma & ni_improving & gp_ta_improving
        score = _cs_zscore(ni_ta - ni_ta.shift(self.lookback)).fillna(0.0) + _cs_zscore(
            gp_ta - gp_ta.shift(self.lookback)
        ).fillna(0.0)
        daily_w = _sector_neutral_weights(
            score, accumulation_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class OwnerMixQualityPort(CrossSectionalStrategy):
    """H445: Low dilution + OCF reliability + book equity growth = owner mix quality.

    z(-shares_growth) + z(OCF/NI fraction) + z(BE growth). Three-factor composite.
    Bushee & Yan & Zhang: mix of owner horizons reflected in these fundamental signals.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "owner_mix_quality_port"

    def __init__(
        self,
        book_equity: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        reliability_window: int = 504,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.book_equity = book_equity
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.lookback = lookback
        self.reliability_window = reliability_window
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        be = _align_fund(self.book_equity, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        shares = _align_fund(self.shares_outstanding, prices_panel)
        shares_lag = shares.shift(self.lookback)
        shares_growth = ((shares - shares_lag) / shares_lag.abs().clip(lower=1.0)).clip(-1.0, 5.0)
        ocf_exceeds_ni = (ocf > ni).astype(float)
        ocf_reliability = ocf_exceeds_ni.rolling(
            self.reliability_window, min_periods=self.reliability_window // 4
        ).mean()
        be_lag = be.shift(self.lookback)
        be_growth = ((be - be_lag) / be_lag.abs().clip(lower=1.0)).clip(-1.0, 5.0)
        composite = (
            _cs_zscore(-shares_growth).fillna(0.0)
            + _cs_zscore(ocf_reliability).fillna(0.0)
            + _cs_zscore(be_growth).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            composite, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)
