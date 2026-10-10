"""Machine-learning and nonlinear characteristic models (elastic net, gradient boosting, binned scores)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import (
    _align_fund,
    _cs_zscore,
    _monthly_rebalance,
    _periodic_rebalance,
    _sector_neutral_weights,
)
from .base import CrossSectionalStrategy


class PercentileBinnedGP(CrossSectionalStrategy):
    """H361: GP/TA scored by quintile bins instead of continuous z-score.

    Assigns scores 1-5 by quintile. Tests whether the top GP quintile genuinely
    outperforms quintile 4 (nonlinear shape; Freyberger et al. 2020).
    Sector-neutral, quarterly rebalance.
    """

    name = "percentile_binned_gp"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        n_bins: int = 5,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.n_bins = n_bins
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
        mem = membership.reindex_like(gp_ta).fillna(False)
        pct_rank = gp_ta.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        bin_score = (
            (pct_rank * self.n_bins)
            .clip(upper=self.n_bins)
            .fillna(0.0)
            .apply(lambda x: x.astype(int))
            .astype(float)
        )
        daily_w = _sector_neutral_weights(
            bin_score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MonotonicityVotingComposite(CrossSectionalStrategy):
    """H362: Vote across GP/TA, ROE, accruals -- include only if >=2/3 signals agree.

    Inspired by Green-Hand-Zhang: if signals disagree, the stock is excluded.
    Stocks where 2+ signals place in top quintile are high-conviction longs.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "monotonicity_voting_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        vote_quantile: float = 0.20,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.vote_quantile = vote_quantile
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
        mem = membership.reindex_like(gp_ta).fillna(False)
        n_members = mem.sum(axis=1).replace(0, np.nan)
        top_n = (n_members * self.vote_quantile).clip(lower=1)
        gp_rank = gp_ta.where(mem).rank(axis=1, ascending=False, na_option="keep")
        roe_rank = roe.where(mem).rank(axis=1, ascending=False, na_option="keep")
        acc_rank = (-accruals).where(mem).rank(axis=1, ascending=False, na_option="keep")
        votes = (
            (gp_rank.le(top_n, axis=0) & gp_rank.notna()).astype(int)
            + (roe_rank.le(top_n, axis=0) & roe_rank.notna()).astype(int)
            + (acc_rank.le(top_n, axis=0) & acc_rank.notna()).astype(int)
        ).astype(float)
        score = votes.where(votes >= 2, np.nan)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class ElasticNetReturnForecast(CrossSectionalStrategy):
    """H363: Elastic-net regularized cross-sectional expected-return ranker.

    For each quarterly rebalance date, fits sklearn ElasticNet on a rolling
    3-year window of (feature, next-quarter-return) pairs.
    Features: GP/TA, ROE, neg_accruals, neg_asset_growth, momentum, OCF/TA.
    Falls back to GP/TA rank if sklearn unavailable.
    Freyberger et al. (2020) + Kozak et al. (2020).
    """

    name = "elastic_net_return_forecast"

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
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.train_window = train_window
        self.pred_horizon = pred_horizon
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            from sklearn.linear_model import ElasticNet
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            gp = _align_fund(self.gross_profit, prices_panel)
            ta = _align_fund(self.total_assets, prices_panel)
            gp_ta = (gp / ta.where(ta > 0)).clip(0.0, 5.0)
            daily_w = _sector_neutral_weights(
                gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
            )
            return _periodic_rebalance(daily_w, self.rebalance_period)

        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        feat_panels = [
            (gp / ta_safe).clip(0.0, 5.0),
            (ni / be.where(be > 0)).clip(-2.0, 5.0),
            -((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0),
            -((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0),
            prices_panel.pct_change(252).shift(21).clip(-1.0, 5.0),
            (ocf / ta_safe).clip(-1.0, 2.0),
        ]

        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        # Initialise with GP/TA fallback so early folds behave like H239
        score_df = feat_panels[0].reindex(index=all_dates, columns=tickers)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos - self.pred_horizon
            train_start = max(0, train_end - self.train_window)
            if train_end - train_start < 40:
                continue
            X_rows, y_rows = [], []
            for tp in range(train_start, train_end - self.rebalance_period, self.rebalance_period):
                t_d = all_dates[tp]
                t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                try:
                    X = np.column_stack([fp.loc[t_d, tickers].values for fp in feat_panels])
                except Exception:
                    continue
                mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                fwd = (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
                valid = mem_t & np.isfinite(fwd) & np.isfinite(X).all(axis=1)
                if valid.sum() < 10:
                    continue
                X_rows.append(X[valid])
                y_rows.append(fwd[valid])
            if not X_rows:
                continue
            X_tr = np.vstack(X_rows)
            y_tr = np.concatenate(y_rows)
            scaler = StandardScaler()
            try:
                Xs = scaler.fit_transform(X_tr)
                model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=2000)
                model.fit(Xs, y_tr)
                X_pred = np.column_stack([fp.loc[rebal_date, tickers].values for fp in feat_panels])
                preds = model.predict(scaler.transform(X_pred))
                score_df.loc[rebal_date, tickers] = preds
            except Exception:
                continue

        score_df = score_df.ffill().fillna(feat_panels[0])
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class NonlinearMomentumBins(CrossSectionalStrategy):
    """H364: Long only the top decile of 12-1 momentum (J-shaped selection).

    Tests whether momentum alpha is concentrated in the top decile, not quintile.
    Freyberger et al. (2020) nonlinear shape diagnostics.
    Sector-neutral top 10%, monthly rebalance.
    """

    name = "nonlinear_momentum_bins"

    def __init__(
        self,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        top_decile: float = 0.10,
    ) -> None:
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_window = skip_window
        self.top_decile = top_decile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        mom = prices_panel.pct_change(self.mom_window).shift(self.skip_window)
        daily_w = _sector_neutral_weights(
            mom, membership, self.sectors, self.top_decile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


class ShrunkSignalComposite(CrossSectionalStrategy):
    """H365: Clipped z-score blend of batch-3 nonlinear and regularized signals.

    Combines four signals with each z-score clipped to [-clip_val, clip_val]:
      clip(z(GP_quintile_bin)) + clip(z(voting_votes)) +
      clip(z(top_decile_momentum)) + clip(z(GP/TA))
    Kozak et al. (2020): clipping acts as implicit regularization.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "shrunk_signal_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        clip_val: float = 2.0,
        n_bins: int = 5,
        vote_quantile: float = 0.20,
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
        self.n_bins = n_bins
        self.vote_quantile = vote_quantile
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
        mom = prices_panel.pct_change(252).shift(21)
        mem = membership.reindex_like(gp_ta).fillna(False)
        n_members = mem.sum(axis=1).replace(0, np.nan)
        top_n = (n_members * self.vote_quantile).clip(lower=1)

        pct_rank = gp_ta.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        bin_score = (
            (pct_rank * self.n_bins)
            .clip(upper=self.n_bins)
            .fillna(0.0)
            .apply(lambda x: x.astype(int))
            .astype(float)
        )
        z1 = _cs_zscore(bin_score).clip(-self.clip_val, self.clip_val).fillna(0.0)

        gp_rank = gp_ta.where(mem).rank(axis=1, ascending=False, na_option="keep")
        roe_rank = roe.where(mem).rank(axis=1, ascending=False, na_option="keep")
        acc_rank = (-accruals).where(mem).rank(axis=1, ascending=False, na_option="keep")
        votes = (
            (gp_rank.le(top_n, axis=0) & gp_rank.notna()).astype(int)
            + (roe_rank.le(top_n, axis=0) & roe_rank.notna()).astype(int)
            + (acc_rank.le(top_n, axis=0) & acc_rank.notna()).astype(int)
        ).astype(float)
        z2 = _cs_zscore(votes).clip(-self.clip_val, self.clip_val).fillna(0.0)

        mom_rank = mom.where(mem).rank(axis=1, ascending=False, na_option="keep")
        top_dec_n = (n_members * 0.10).clip(lower=1)
        in_top_mom = (mom_rank.le(top_dec_n, axis=0) & mom_rank.notna()).astype(float)
        z3 = _cs_zscore(in_top_mom).clip(-self.clip_val, self.clip_val).fillna(0.0)

        z4 = _cs_zscore(gp_ta).clip(-self.clip_val, self.clip_val).fillna(0.0)

        score = z1 + z2 + z3 + z4
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class LinearCrossSectionalRank(CrossSectionalStrategy):
    """H371: OLS cross-sectional regression of forward return on 8 characteristics.

    Features: GP/TA, ROE, neg_accruals, neg_asset_growth, momentum,
    OCF/TA, leverage (D/A), book-to-market. OLS, no regularization.
    Required ML ladder baseline (Chen-Pelger-Zhu 2023).
    Falls back to GP/TA when training data insufficient. Quarterly rebalance.
    """

    name = "linear_cross_sectional_rank"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        sectors: pd.Series,
        train_window: int = 756,
        pred_horizon: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.long_term_debt = long_term_debt
        self.sectors = sectors
        self.train_window = train_window
        self.pred_horizon = pred_horizon
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def _build_feat_panels(self, prices_panel: pd.DataFrame) -> list[pd.DataFrame]:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        return [
            (gp / ta_safe).clip(0.0, 5.0),
            (ni / be.where(be > 0)).clip(-2.0, 5.0),
            -((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0),
            -((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0),
            prices_panel.pct_change(252).shift(21).clip(-1.0, 5.0),
            (ocf / ta_safe).clip(-1.0, 2.0),
            (ltd / ta_safe).clip(0.0, 2.0),
            (be / ta_safe).clip(0.0, 3.0),
        ]

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        feat_panels = self._build_feat_panels(prices_panel)
        gp_ta = feat_panels[0]
        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        score_df = gp_ta.reindex(index=all_dates, columns=tickers)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos - self.pred_horizon
            train_start = max(0, train_end - self.train_window)
            if train_end - train_start < 40:
                continue
            X_rows, y_rows = [], []
            for tp in range(train_start, train_end - self.rebalance_period, self.rebalance_period):
                t_d = all_dates[tp]
                t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                try:
                    X = np.column_stack([fp.loc[t_d, tickers].values for fp in feat_panels])
                except Exception:
                    continue
                mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                fwd = (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
                valid = mem_t & np.isfinite(fwd) & np.isfinite(X).all(axis=1)
                if valid.sum() < 10:
                    continue
                X_rows.append(X[valid])
                y_rows.append(fwd[valid])
            if not X_rows:
                continue
            X_tr = np.vstack(X_rows)
            y_tr = np.concatenate(y_rows)
            try:
                from sklearn.preprocessing import StandardScaler

                sc = StandardScaler()
                Xs = sc.fit_transform(X_tr)
                coef, _, _, _ = np.linalg.lstsq(
                    np.column_stack([np.ones(len(Xs)), Xs]), y_tr, rcond=None
                )
                X_pred = np.column_stack([fp.loc[rebal_date, tickers].values for fp in feat_panels])
                X_pred_s = sc.transform(X_pred)
                preds = np.column_stack([np.ones(len(X_pred_s)), X_pred_s]) @ coef
                score_df.loc[rebal_date, tickers] = preds
            except Exception:
                continue

        score_df = score_df.ffill().fillna(gp_ta)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class ElasticNetRankStrong(CrossSectionalStrategy):
    """H372: Elastic net with stronger regularization (alpha=0.3) on 8 features.

    Same feature set as H371. Higher alpha enforces more shrinkage toward zero,
    reduces overfitting in the 500-stock universe. Falls back to GP/TA.
    Quarterly rebalance. Freyberger et al. (2020) + Kozak et al. (2020).
    """

    name = "elastic_net_rank_strong"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        sectors: pd.Series,
        train_window: int = 756,
        pred_horizon: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
        alpha: float = 0.30,
        l1_ratio: float = 0.5,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.long_term_debt = long_term_debt
        self.sectors = sectors
        self.train_window = train_window
        self.pred_horizon = pred_horizon
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _build_feat_panels(self, prices_panel: pd.DataFrame) -> list[pd.DataFrame]:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        return [
            (gp / ta_safe).clip(0.0, 5.0),
            (ni / be.where(be > 0)).clip(-2.0, 5.0),
            -((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0),
            -((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0),
            prices_panel.pct_change(252).shift(21).clip(-1.0, 5.0),
            (ocf / ta_safe).clip(-1.0, 2.0),
            (ltd / ta_safe).clip(0.0, 2.0),
            (be / ta_safe).clip(0.0, 3.0),
        ]

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            from sklearn.linear_model import ElasticNet
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            gp = _align_fund(self.gross_profit, prices_panel)
            ta = _align_fund(self.total_assets, prices_panel)
            gp_ta = (gp / ta.where(ta > 0)).clip(0.0, 5.0)
            daily_w = _sector_neutral_weights(
                gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
            )
            return _periodic_rebalance(daily_w, self.rebalance_period)

        feat_panels = self._build_feat_panels(prices_panel)
        gp_ta = feat_panels[0]
        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        score_df = gp_ta.reindex(index=all_dates, columns=tickers)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos - self.pred_horizon
            train_start = max(0, train_end - self.train_window)
            if train_end - train_start < 40:
                continue
            X_rows, y_rows = [], []
            for tp in range(train_start, train_end - self.rebalance_period, self.rebalance_period):
                t_d = all_dates[tp]
                t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                try:
                    X = np.column_stack([fp.loc[t_d, tickers].values for fp in feat_panels])
                except Exception:
                    continue
                mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                fwd = (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
                valid = mem_t & np.isfinite(fwd) & np.isfinite(X).all(axis=1)
                if valid.sum() < 10:
                    continue
                X_rows.append(X[valid])
                y_rows.append(fwd[valid])
            if not X_rows:
                continue
            X_tr = np.vstack(X_rows)
            y_tr = np.concatenate(y_rows)
            scaler = StandardScaler()
            try:
                Xs = scaler.fit_transform(X_tr)
                model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=2000)
                model.fit(Xs, y_tr)
                X_pred = np.column_stack([fp.loc[rebal_date, tickers].values for fp in feat_panels])
                preds = model.predict(scaler.transform(X_pred))
                score_df.loc[rebal_date, tickers] = preds
            except Exception:
                continue

        score_df = score_df.ffill().fillna(gp_ta)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class GradientBoostingRank(CrossSectionalStrategy):
    """H373: Gradient Boosting cross-sectional ranker (8 features: price + EDGAR).

    Uses price features (momentum, reversal, vol -- always available) plus
    EDGAR features (GP/TA, ROE, accruals, OCF/TA, asset_growth) when available.
    Price features enable non-zero predictions in pre-EDGAR folds. Shallow trees
    (max_depth=3) limit overfitting in the 500-stock universe.
    Chen-Pelger-Zhu 2023. Quarterly rebalance.
    """

    name = "gradient_boosting_rank"

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
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
        max_depth: int = 3,
        n_estimators: int = 100,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.sectors = sectors
        self.train_window = train_window
        self.pred_horizon = pred_horizon
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period
        self.max_depth = max_depth
        self.n_estimators = n_estimators

    def _build_feat_panels(self, prices_panel: pd.DataFrame) -> list[pd.DataFrame]:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        return [
            # Price features (always available)
            prices_panel.pct_change(252).shift(21).fillna(0.0).clip(-1.0, 5.0),
            prices_panel.pct_change(21).fillna(0.0).clip(-1.0, 2.0),
            prices_panel.pct_change(63).rolling(63).std().shift(1).fillna(0.0).clip(0.0, 0.5),
            # EDGAR features (NaN in pre-EDGAR folds, model handles missing)
            (gp / ta_safe).clip(0.0, 5.0).fillna(0.0),
            (ni / be.where(be > 0)).clip(-2.0, 5.0).fillna(0.0),
            -((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0).fillna(0.0),
            -((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0).fillna(0.0),
            (ocf / ta_safe).clip(-1.0, 2.0).fillna(0.0),
        ]

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            gp = _align_fund(self.gross_profit, prices_panel)
            ta = _align_fund(self.total_assets, prices_panel)
            gp_ta = (gp / ta.where(ta > 0)).clip(0.0, 5.0)
            daily_w = _sector_neutral_weights(
                gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
            )
            return _periodic_rebalance(daily_w, self.rebalance_period)

        feat_panels = self._build_feat_panels(prices_panel)
        mom_panel = feat_panels[0]  # price fallback
        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        score_df = mom_panel.reindex(index=all_dates, columns=tickers)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos - self.pred_horizon
            train_start = max(0, train_end - self.train_window)
            if train_end - train_start < 40:
                continue
            X_rows, y_rows = [], []
            for tp in range(train_start, train_end - self.rebalance_period, self.rebalance_period):
                t_d = all_dates[tp]
                t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                try:
                    X = np.column_stack([fp.loc[t_d, tickers].values for fp in feat_panels])
                except Exception:
                    continue
                mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                fwd = (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
                valid = mem_t & np.isfinite(fwd) & np.isfinite(X).all(axis=1)
                if valid.sum() < 10:
                    continue
                X_rows.append(X[valid])
                y_rows.append(fwd[valid])
            if not X_rows:
                continue
            X_tr = np.vstack(X_rows)
            y_tr = np.concatenate(y_rows)
            scaler = StandardScaler()
            try:
                Xs = scaler.fit_transform(X_tr)
                model = GradientBoostingRegressor(
                    max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=42
                )
                model.fit(Xs, y_tr)
                X_pred = np.column_stack([fp.loc[rebal_date, tickers].values for fp in feat_panels])
                preds = model.predict(scaler.transform(X_pred))
                score_df.loc[rebal_date, tickers] = preds
            except Exception:
                continue

        score_df = score_df.ffill().fillna(mom_panel)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MLEnsembleRank(CrossSectionalStrategy):
    """H374: Equal-weight blend of OLS, ElasticNet, and GB cross-sectional predictions.

    Model averaging smooths overfitting from any single model. Implements the
    full ML model ladder (Chen-Pelger-Zhu 2023). Falls back to GP/TA when
    training data insufficient. Quarterly rebalance.
    """

    name = "ml_ensemble_rank"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        long_term_debt: pd.DataFrame,
        sectors: pd.Series,
        train_window: int = 756,
        pred_horizon: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.long_term_debt = long_term_debt
        self.sectors = sectors
        self.train_window = train_window
        self.pred_horizon = pred_horizon
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def _build_feat_panels(self, prices_panel: pd.DataFrame) -> list[pd.DataFrame]:
        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ltd = _align_fund(self.long_term_debt, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        return [
            prices_panel.pct_change(252).shift(21).fillna(0.0).clip(-1.0, 5.0),
            prices_panel.pct_change(21).fillna(0.0).clip(-1.0, 2.0),
            (gp / ta_safe).clip(0.0, 5.0).fillna(0.0),
            (ni / be.where(be > 0)).clip(-2.0, 5.0).fillna(0.0),
            -((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0).fillna(0.0),
            -((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0).fillna(0.0),
            (ocf / ta_safe).clip(-1.0, 2.0).fillna(0.0),
            (ltd / ta_safe).clip(0.0, 2.0).fillna(0.0),
        ]

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.linear_model import ElasticNet
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            gp = _align_fund(self.gross_profit, prices_panel)
            ta = _align_fund(self.total_assets, prices_panel)
            gp_ta = (gp / ta.where(ta > 0)).clip(0.0, 5.0)
            daily_w = _sector_neutral_weights(
                gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
            )
            return _periodic_rebalance(daily_w, self.rebalance_period)

        feat_panels = self._build_feat_panels(prices_panel)
        mom_panel = feat_panels[0]
        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        score_df = mom_panel.reindex(index=all_dates, columns=tickers)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos - self.pred_horizon
            train_start = max(0, train_end - self.train_window)
            if train_end - train_start < 40:
                continue
            X_rows, y_rows = [], []
            for tp in range(train_start, train_end - self.rebalance_period, self.rebalance_period):
                t_d = all_dates[tp]
                t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                try:
                    X = np.column_stack([fp.loc[t_d, tickers].values for fp in feat_panels])
                except Exception:
                    continue
                mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                fwd = (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
                valid = mem_t & np.isfinite(fwd) & np.isfinite(X).all(axis=1)
                if valid.sum() < 10:
                    continue
                X_rows.append(X[valid])
                y_rows.append(fwd[valid])
            if not X_rows:
                continue
            X_tr = np.vstack(X_rows)
            y_tr = np.concatenate(y_rows)
            scaler = StandardScaler()
            try:
                Xs = scaler.fit_transform(X_tr)
                m1_coef, _, _, _ = np.linalg.lstsq(
                    np.column_stack([np.ones(len(Xs)), Xs]), y_tr, rcond=None
                )
                m2 = ElasticNet(alpha=0.3, l1_ratio=0.5, max_iter=2000)
                m2.fit(Xs, y_tr)
                m3 = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=42)
                m3.fit(Xs, y_tr)
                X_pred = np.column_stack([fp.loc[rebal_date, tickers].values for fp in feat_panels])
                Xps = scaler.transform(X_pred)
                p1 = np.column_stack([np.ones(len(Xps)), Xps]) @ m1_coef
                p2 = m2.predict(Xps)
                p3 = m3.predict(Xps)
                preds = (p1 + p2 + p3) / 3.0
                score_df.loc[rebal_date, tickers] = preds
            except Exception:
                continue

        score_df = score_df.ffill().fillna(mom_panel)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


class MacroStateGBRank(CrossSectionalStrategy):
    """H375: Gradient Boosting ranker with SPY 200d MA macro-state feature.

    9 features: price+EDGAR features from H373 plus a binary macro-state flag
    (SPY > SPY_200d_MA = 1, else 0). Macro state conditions the entire model.
    Chen-Pelger-Zhu 2023: macro-state conditioning improves asset pricing models.
    Quarterly rebalance.
    """

    name = "macro_state_gb_rank"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        operating_cf: pd.DataFrame,
        spy_prices: pd.Series,
        sectors: pd.Series,
        train_window: int = 756,
        pred_horizon: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
        max_depth: int = 3,
        n_estimators: int = 100,
        ma_window: int = 200,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.operating_cf = operating_cf
        self.spy_prices = spy_prices
        self.sectors = sectors
        self.train_window = train_window
        self.pred_horizon = pred_horizon
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.ma_window = ma_window

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            gp = _align_fund(self.gross_profit, prices_panel)
            ta = _align_fund(self.total_assets, prices_panel)
            gp_ta = (gp / ta.where(ta > 0)).clip(0.0, 5.0)
            daily_w = _sector_neutral_weights(
                gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
            )
            return _periodic_rebalance(daily_w, self.rebalance_period)

        gp = _align_fund(self.gross_profit, prices_panel)
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)

        spy_ma = self.spy_prices.reindex(prices_panel.index).ffill()
        macro_state = (
            (spy_ma > spy_ma.rolling(self.ma_window, min_periods=100).mean())
            .astype(float)
            .fillna(0.5)
        )

        feat_panels = [
            prices_panel.pct_change(252).shift(21).fillna(0.0).clip(-1.0, 5.0),
            prices_panel.pct_change(21).fillna(0.0).clip(-1.0, 2.0),
            prices_panel.pct_change(63).rolling(63).std().shift(1).fillna(0.0).clip(0.0, 0.5),
            (gp / ta_safe).clip(0.0, 5.0).fillna(0.0),
            (ni / be.where(be > 0)).clip(-2.0, 5.0).fillna(0.0),
            -((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0).fillna(0.0),
            -((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0).fillna(0.0),
            (ocf / ta_safe).clip(-1.0, 2.0).fillna(0.0),
        ]
        mom_panel = feat_panels[0]
        all_dates = prices_panel.index
        tickers = prices_panel.columns.tolist()
        score_df = mom_panel.reindex(index=all_dates, columns=tickers)

        for pos in range(0, len(all_dates), self.rebalance_period):
            rebal_date = all_dates[pos]
            train_end = pos - self.pred_horizon
            train_start = max(0, train_end - self.train_window)
            if train_end - train_start < 40:
                continue
            X_rows, y_rows = [], []
            for tp in range(train_start, train_end - self.rebalance_period, self.rebalance_period):
                t_d = all_dates[tp]
                t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                ms = macro_state.loc[t_d] if t_d in macro_state.index else 0.5
                try:
                    X_base = np.column_stack([fp.loc[t_d, tickers].values for fp in feat_panels])
                    X = np.column_stack([X_base, np.full(len(tickers), ms)])
                except Exception:
                    continue
                mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                fwd = (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
                valid = mem_t & np.isfinite(fwd) & np.isfinite(X).all(axis=1)
                if valid.sum() < 10:
                    continue
                X_rows.append(X[valid])
                y_rows.append(fwd[valid])
            if not X_rows:
                continue
            X_tr = np.vstack(X_rows)
            y_tr = np.concatenate(y_rows)
            scaler = StandardScaler()
            try:
                Xs = scaler.fit_transform(X_tr)
                model = GradientBoostingRegressor(
                    max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=42
                )
                model.fit(Xs, y_tr)
                ms_cur = macro_state.loc[rebal_date] if rebal_date in macro_state.index else 0.5
                X_pred_base = np.column_stack(
                    [fp.loc[rebal_date, tickers].values for fp in feat_panels]
                )
                X_pred = np.column_stack([X_pred_base, np.full(len(tickers), ms_cur)])
                preds = model.predict(scaler.transform(X_pred))
                score_df.loc[rebal_date, tickers] = preds
            except Exception:
                continue

        score_df = score_df.ffill().fillna(mom_panel)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)
