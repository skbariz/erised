"""Staging: H366-H390 strategy classes (Batches 4-8, Set 5)."""
# ---------------------------------------------------------------------------
# H366: RecentEpochOnlyGP  (GP rank tilted by recent price confirmation)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H367: PostPublicationHaircutComposite  (shrinkage-weighted quality composite)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H368: IncrementalAlphaFilter  (regression-based signal selection)
# ---------------------------------------------------------------------------


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
            for cname, cand_panel in candidates.items():
                X_rows, y_rows = [], []
                for tp in range(train_start, train_end - self.rebalance_period, self.rebalance_period):
                    t_d = all_dates[tp]
                    t_f = all_dates[min(tp + self.rebalance_period, len(all_dates) - 1)]
                    try:
                        c_vals = cand_panel.loc[t_d, tickers].values
                        ctrl_vals = np.column_stack([p.loc[t_d, tickers].values for p in controls])
                        X = np.column_stack([c_vals.reshape(-1, 1), ctrl_vals])
                    except Exception:
                        continue
                    mem_t = membership.loc[t_d].reindex(tickers).fillna(False).values
                    fwd = (prices_panel.loc[t_f] / prices_panel.loc[t_d] - 1).reindex(tickers).values
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


# ---------------------------------------------------------------------------
# H369: EigenportfolioCharacteristic  (PCA on cross-sectional characteristics)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H370: DecayAdjustedComposite  (equal-weight blend of batch-4 signals)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H371: LinearCrossSectionalRank  (OLS linear ranker baseline)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H372: ElasticNetRankStrong  (elastic net alpha=0.3 -- stronger regularization)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H373: GradientBoostingRank  (GB with price + EDGAR features)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H374: MLEnsembleRank  (equal-weight blend of OLS + elastic net + GB)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H375: MacroStateGBRank  (GB with SPY 200d MA macro-state feature)
# ---------------------------------------------------------------------------


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
        macro_state = (spy_ma > spy_ma.rolling(self.ma_window, min_periods=100).mean()).astype(float).fillna(0.5)

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
                X_pred_base = np.column_stack([fp.loc[rebal_date, tickers].values for fp in feat_panels])
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


# ---------------------------------------------------------------------------
# Shared helper: detect filing event windows from fundamental panel changes
# ---------------------------------------------------------------------------


def _filing_window(fund_panel: pd.DataFrame, lookback_days: int = 5) -> pd.DataFrame:
    """Boolean mask: True in the lookback_days after any filing event.

    Filing event = date when forward-filled fundamental panel changes value.
    Uses past-only window (causal, no lookahead).
    """
    changed = fund_panel.diff().abs().fillna(0.0) > 1e-6
    return changed.rolling(lookback_days, min_periods=1).max().astype(bool)


# ---------------------------------------------------------------------------
# H376: EarningsWindowQualityBoost  (GP quality with post-filing size boost)
# ---------------------------------------------------------------------------


class EarningsWindowQualityBoost(CrossSectionalStrategy):
    """H376: GP/TA quality rank with 1.5x weight boost in post-filing window.

    Within 5 trading days of an EDGAR filing (detected from NI panel change),
    boost position weight by filing_boost factor. Outside window: base_scale.
    Engelberg-McLean-Pontiff 2018: anomaly payoffs concentrated around news.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "earnings_window_quality_boost"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        filing_window_days: int = 5,
        filing_boost: float = 1.5,
        base_scale: float = 0.8,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.sectors = sectors
        self.filing_window_days = filing_window_days
        self.filing_boost = filing_boost
        self.base_scale = base_scale
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

        in_event = _filing_window(ni, self.filing_window_days)
        scale = in_event.astype(float) * (self.filing_boost - self.base_scale) + self.base_scale

        base_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        scaled_w = (base_w * scale).clip(upper=1.0)
        row_sum = scaled_w.sum(axis=1).replace(0, np.nan)
        scaled_w = scaled_w.div(row_sum, axis=0).fillna(0.0)
        return _periodic_rebalance(scaled_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H377: EarningsSurpriseDrift  (PEAD scaled by NI change magnitude)
# ---------------------------------------------------------------------------


class EarningsSurpriseDrift(CrossSectionalStrategy):
    """H377: Post-earnings announcement drift scaled by earnings surprise magnitude.

    Surprise = |delta_NI / total_assets| at most recent filing date.
    Position sized by surprise rank: larger surprises get proportionally larger
    weight when NI change is positive. Engelberg-McLean-Pontiff 2018.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "earnings_surprise_drift"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ta_safe = ta.where(ta > 0)

        delta_ni = ni.diff()
        surprise_raw = (delta_ni / ta_safe).clip(-0.5, 0.5)
        positive_surprise = surprise_raw.where(surprise_raw > 0, 0.0)

        daily_w = _sector_neutral_weights(
            positive_surprise, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H378: QuietPeriodMomentum  (momentum avoiding post-filing window)
# ---------------------------------------------------------------------------


class QuietPeriodMomentum(CrossSectionalStrategy):
    """H378: 12-1 momentum reduced to zero in post-filing window.

    During the 5 trading days after an EDGAR filing (from NI panel changes),
    set position weight to zero for that stock and redistribute to remaining
    members. Reduces gap risk around earnings. Engelberg-McLean-Pontiff 2018.
    Sector-neutral top 20%, monthly rebalance.
    """

    name = "quiet_period_momentum"

    def __init__(
        self,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        filing_window_days: int = 5,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.net_income = net_income
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_window = skip_window
        self.filing_window_days = filing_window_days
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        mom = prices_panel.pct_change(self.mom_window).shift(self.skip_window)
        in_event = _filing_window(ni, self.filing_window_days)
        quiet_mom = mom.where(~in_event, np.nan)
        daily_w = _sector_neutral_weights(
            quiet_mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H379: EarningsConfirmedQuality  (GP rank filtered by positive NI surprise)
# ---------------------------------------------------------------------------


class EarningsConfirmedQuality(CrossSectionalStrategy):
    """H379: GP/TA quality rank filtered by non-negative NI change at last filing.

    Only hold GP-ranked stocks where the most recent EDGAR filing showed
    NI change >= 0 (positive earnings confirmation). Exit when NI disappoints.
    Tetlock et al. 2008: positive news tone confirms fundamental quality.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "earnings_confirmed_quality"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.sectors = sectors
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
        ni_confirmed = ni.diff() >= 0
        gated_gp = gp_ta.where(ni_confirmed, np.nan)
        daily_w = _sector_neutral_weights(
            gated_gp, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H380: EventAwareComposite  (batch-6 earnings-event composite)
# ---------------------------------------------------------------------------


class EventAwareComposite(CrossSectionalStrategy):
    """H380: Equal-weight composite of batch-6 earnings-event-aware signals.

    Blends: GP quality (baseline), earnings-confirmed quality, quiet-period
    momentum, positive-surprise drift. Each signal clipped before combining.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "event_aware_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        sectors: pd.Series,
        clip_val: float = 2.0,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
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
        ta_safe = ta.where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        ni_confirmed = ni.diff() >= 0
        delta_ni = ni.diff()
        positive_surprise = (delta_ni / ta_safe).clip(0.0, 0.5)
        mom = prices_panel.pct_change(252).shift(21)
        in_event = _filing_window(ni, 5)
        quiet_mom = mom.where(~in_event, np.nan)
        confirmed_gp = gp_ta.where(ni_confirmed, np.nan)

        z1 = _cs_zscore(gp_ta).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z2 = _cs_zscore(confirmed_gp).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z3 = _cs_zscore(positive_surprise).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z4 = _cs_zscore(quiet_mom).clip(-self.clip_val, self.clip_val).fillna(0.0)

        score = z1 + z2 + z3 + z4
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H381: LowDilutionGovernance  (low share issuance = governance proxy)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H382: BuybackQualityGovernance  (negative dilution = net buyback quality)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H383: CapitalDisciplineGovernance  (dilution + buyback + accruals composite)
# ---------------------------------------------------------------------------


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
        ta_safe = ta.where(ta > 0)
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


# ---------------------------------------------------------------------------
# H384: GovernanceQualityComposite  (governance x GP quality gate)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H385: GovernanceMDRComposite  (batch-7 governance ensemble)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H386: SinExclusionPremium  (ex-Consumer Staples equal-weight)
# ---------------------------------------------------------------------------


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
        excluded_tickers = set(
            self.sectors[self.sectors == self.excluded_sector].index.tolist()
        )
        mem = membership.copy()
        for t in excluded_tickers:
            if t in mem.columns:
                mem[t] = False
        n_inc = mem.sum(axis=1).replace(0, np.nan)
        daily_w = mem.astype(float).div(n_inc, axis=0).fillna(0.0)
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H387: LowCarbonSectorNeutral  (ex-Energy sector-neutral GP quality)
# ---------------------------------------------------------------------------


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
        excluded_tickers = set(
            self.sectors[self.sectors == self.excluded_sector].index.tolist()
        )
        filtered_mem = membership.copy()
        for t in excluded_tickers:
            if t in filtered_mem.columns:
                filtered_mem[t] = False
        daily_w = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H388: GreenProxyQuality  (Utilities + IT sector quality portfolio)
# ---------------------------------------------------------------------------


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
        green_tickers = set(
            self.sectors[self.sectors.isin(self.green_sectors)].index.tolist()
        )
        green_mem = pd.DataFrame(False, index=membership.index, columns=membership.columns)
        for t in green_tickers:
            if t in green_mem.columns:
                green_mem[t] = membership[t]
        n_green = green_mem.sum(axis=1).replace(0, np.nan)
        if (n_green < 5).all():
            n = membership.sum(axis=1).replace(0, np.nan)
            return _periodic_rebalance(
                membership.astype(float).div(n, axis=0).fillna(0.0), self.rebalance_period
            )
        daily_w = _equal_weight_top_quantile(
            gp_ta, green_mem, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H389: ControversyProxyAvoid  (avoid high-leverage + dilution + low GP)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H390: ESGProxyComposite  (batch-8 ESG proxy ensemble)
# ---------------------------------------------------------------------------


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

        combined = (daily_w1 + daily_w2 + daily_w3 + daily_w4)
        row_sum = combined.sum(axis=1).replace(0, np.nan)
        daily_w = combined.div(row_sum, axis=0).fillna(0.0)
        return _periodic_rebalance(daily_w, self.rebalance_period)
