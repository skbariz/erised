"""Temporary staging: H351-H365 strategy classes to be appended to cross_sectional.py."""
# ---------------------------------------------------------------------------
# H351: QInvestmentRank  (q-factor: low investment = high expected return)
# ---------------------------------------------------------------------------


class QInvestmentRank(CrossSectionalStrategy):
    """H351: Long lowest-quartile of asset-growth rate (q-factor investment signal).

    investment_rate = (total_assets_t - total_assets_{t-252bd}) / total_assets_{t-252bd}
    Low investment signals capital discipline and predicts higher future returns.
    Sector-neutral top 20% by lowest investment rate. Quarterly rebalance.
    """

    name = "q_investment_rank"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
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
        ta = _align_fund(self.total_assets, prices_panel)
        ta_safe = ta.where(ta > 0)
        inv_rate = (ta - ta.shift(self.lookback)) / ta_safe.shift(self.lookback)
        inv_rate = inv_rate.clip(-1.0, 5.0)
        daily_w = _sector_neutral_weights(
            inv_rate, membership, self.sectors, self.within_sector_quantile, ascending=True
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H352: QROELevel  (q-factor: high profitability / ROE)
# ---------------------------------------------------------------------------


class QROELevel(CrossSectionalStrategy):
    """H352: Long top-quartile by ROE = net_income / book_equity (q-factor profitability).

    Persistent high ROE predicts higher future returns (Hou-Xue-Zhang 2015).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "q_roe_level"

    def __init__(
        self,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.net_income = net_income
        self.book_equity = book_equity
        self.sectors = sectors
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        be_safe = be.where(be > 0)
        roe = (ni / be_safe).clip(-2.0, 5.0)
        daily_w = _sector_neutral_weights(
            roe, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H353: QInvestProfCombo  (q-factor: low investment + high ROE composite)
# ---------------------------------------------------------------------------


class QInvestProfCombo(CrossSectionalStrategy):
    """H353: z(-investment_rate) + z(ROE) composite -- dual q-factor signal.

    Combines capital discipline (low asset growth) and profitability (high ROE).
    Tests whether the q-factor framework adds incremental value vs single signals.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "q_invest_prof_combo"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
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
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ta_safe = ta.where(ta > 0)
        inv_rate = ((ta - ta.shift(self.lookback)) / ta_safe.shift(self.lookback)).clip(-1.0, 5.0)
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        score = _cs_zscore(-inv_rate).fillna(0.0) + _cs_zscore(roe).fillna(0.0)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H354: QResidualMomentum  (momentum filtered by q-factor quality gate)
# ---------------------------------------------------------------------------


class QResidualMomentum(CrossSectionalStrategy):
    """H354: 12-1 momentum only for stocks where q-factor composite > median.

    Quality gate (low investment + high ROE) ensures momentum is backed by
    fundamentals, not deteriorating firms. Sector-neutral top 20%, monthly rebalance.
    """

    name = "q_residual_momentum"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        sectors: pd.Series,
        mom_window: int = 252,
        skip_window: int = 21,
        ta_lookback: int = 252,
        within_sector_quantile: float = 0.20,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.sectors = sectors
        self.mom_window = mom_window
        self.skip_window = skip_window
        self.ta_lookback = ta_lookback
        self.within_sector_quantile = within_sector_quantile

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        ta_safe = ta.where(ta > 0)
        inv_rate = ((ta - ta.shift(self.ta_lookback)) / ta_safe.shift(self.ta_lookback)).clip(-1.0, 5.0)
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        q_score = _cs_zscore(-inv_rate).fillna(0.0) + _cs_zscore(roe).fillna(0.0)
        q_median = q_score.median(axis=1)
        q_gate = q_score.ge(q_median, axis=0)
        mom = prices_panel.pct_change(self.mom_window).shift(self.skip_window)
        gated_mom = mom.where(q_gate, np.nan)
        daily_w = _sector_neutral_weights(
            gated_mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H355: QFactorComposite  (batch-1 ensemble: q-factors + GP)
# ---------------------------------------------------------------------------


class QFactorComposite(CrossSectionalStrategy):
    """H355: z(-investment) + z(ROE) + z(GP/TA) three-signal q-factor composite.

    Combines q-factor investment and profitability pillars with gross profitability.
    Tests whether q-factors add over stand-alone GP (H239 baseline).
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "q_factor_composite"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        book_equity: pd.DataFrame,
        gross_profit: pd.DataFrame,
        sectors: pd.Series,
        ta_lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.book_equity = book_equity
        self.gross_profit = gross_profit
        self.sectors = sectors
        self.ta_lookback = ta_lookback
        self.within_sector_quantile = within_sector_quantile
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        be = _align_fund(self.book_equity, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        ta_safe = ta.where(ta > 0)
        inv_rate = ((ta - ta.shift(self.ta_lookback)) / ta_safe.shift(self.ta_lookback)).clip(-1.0, 5.0)
        roe = (ni / be.where(be > 0)).clip(-2.0, 5.0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        score = (
            _cs_zscore(-inv_rate).fillna(0.0)
            + _cs_zscore(roe).fillna(0.0)
            + _cs_zscore(gp_ta).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H356: ManagementMispricing  (Stambaugh-Yuan management cluster)
# ---------------------------------------------------------------------------


class ManagementMispricing(CrossSectionalStrategy):
    """H356: Management-related mispricing composite (Stambaugh-Yuan 2017).

    z(-accruals) + z(-asset_growth) + z(-dilution).
    Proxies management actions that reduce shareholder value.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "management_mispricing"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
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
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(self.lookback)) / 2).where(ta > 0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(self.lookback)) / ta_safe.shift(self.lookback)).clip(-1.0, 5.0)
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        score = (
            _cs_zscore(-accruals).fillna(0.0)
            + _cs_zscore(-asset_growth).fillna(0.0)
            + _cs_zscore(-dilution).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H357: PerformanceMispricing  (Stambaugh-Yuan performance cluster)
# ---------------------------------------------------------------------------


class PerformanceMispricing(CrossSectionalStrategy):
    """H357: Performance-related mispricing composite (Stambaugh-Yuan 2017).

    z(ROA) + z(delta_ROA) + z(OCF/TA).
    Proxies improving operating performance being under-priced by the market.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "performance_mispricing"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
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
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        roa = (ni / ta_safe).clip(-1.0, 2.0)
        delta_roa = (roa - roa.shift(self.lookback)).clip(-1.0, 1.0)
        ocf_ta = (ocf / ta_safe).clip(-1.0, 2.0)
        score = (
            _cs_zscore(roa).fillna(0.0)
            + _cs_zscore(delta_roa).fillna(0.0)
            + _cs_zscore(ocf_ta).fillna(0.0)
        )
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H358: MispricingCombo  (50/50 management + performance mispricing)
# ---------------------------------------------------------------------------


class MispricingCombo(CrossSectionalStrategy):
    """H358: 50/50 blend of management and performance mispricing composites.

    Combines Stambaugh-Yuan's two mispricing clusters. Tests whether the two
    clusters together are more robust than either alone.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "mispricing_combo"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
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
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(self.lookback)) / 2).where(ta > 0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(self.lookback)) / ta_safe.shift(self.lookback)).clip(-1.0, 5.0)
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        mgmt = (
            _cs_zscore(-accruals).fillna(0.0)
            + _cs_zscore(-asset_growth).fillna(0.0)
            + _cs_zscore(-dilution).fillna(0.0)
        )
        roa = (ni / ta_safe).clip(-1.0, 2.0)
        delta_roa = (roa - roa.shift(self.lookback)).clip(-1.0, 1.0)
        ocf_ta = (ocf / ta_safe).clip(-1.0, 2.0)
        perf = (
            _cs_zscore(roa).fillna(0.0)
            + _cs_zscore(delta_roa).fillna(0.0)
            + _cs_zscore(ocf_ta).fillna(0.0)
        )
        score = 0.5 * _cs_zscore(mgmt).fillna(0.0) + 0.5 * _cs_zscore(perf).fillna(0.0)
        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H359: MispricingAvoidFilter  (equal-weight ex worst-decile mispricing)
# ---------------------------------------------------------------------------


class MispricingAvoidFilter(CrossSectionalStrategy):
    """H359: Equal-weight S&P 500 excluding worst-decile combined mispricing.

    Long-only avoid strategy: exclude the bottom 10% by combined mispricing score
    (management + performance), hold remaining equal-weight.
    Tests whether mispricing avoidance alone adds value. Quarterly rebalance.
    """

    name = "mispricing_avoid_filter"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        exclude_pct: float = 0.10,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.shares_outstanding = shares_outstanding
        self.sectors = sectors
        self.lookback = lookback
        self.exclude_pct = exclude_pct
        self.rebalance_period = rebalance_period

    def generate_weights(
        self,
        prices_panel: pd.DataFrame,
        membership: pd.DataFrame,
    ) -> pd.DataFrame:
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(self.lookback)) / 2).where(ta > 0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(self.lookback)) / ta_safe.shift(self.lookback)).clip(-1.0, 5.0)
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        roa = (ni / ta_safe).clip(-1.0, 2.0)
        delta_roa = (roa - roa.shift(self.lookback)).clip(-1.0, 1.0)
        ocf_ta = (ocf / ta_safe).clip(-1.0, 2.0)
        combined = (
            _cs_zscore(-accruals).fillna(0.0)
            + _cs_zscore(-asset_growth).fillna(0.0)
            + _cs_zscore(-dilution).fillna(0.0)
            + _cs_zscore(roa).fillna(0.0)
            + _cs_zscore(delta_roa).fillna(0.0)
            + _cs_zscore(ocf_ta).fillna(0.0)
        )
        mem = membership.reindex_like(combined).fillna(False)
        n_members = mem.sum(axis=1).replace(0, np.nan)
        excl_n = (n_members * self.exclude_pct).clip(lower=1)
        ranks_asc = combined.where(mem).rank(axis=1, ascending=True, na_option="keep")
        excluded = ranks_asc.le(excl_n, axis=0) & ranks_asc.notna()
        included = mem & ~excluded
        n_inc = included.sum(axis=1).replace(0, np.nan)
        daily_w = included.astype(float).div(n_inc, axis=0).fillna(0.0)
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H360: MispricingQualityComposite  (GP filter + mispricing ensemble)
# ---------------------------------------------------------------------------


class MispricingQualityComposite(CrossSectionalStrategy):
    """H360: Mispricing combo (H358) filtered by GP/TA quality gate.

    Requires GP/TA > median within membership before applying mispricing rank.
    Tests whether quality x mispricing interaction outperforms either alone.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "mispricing_quality_composite"

    def __init__(
        self,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        gross_profit: pd.DataFrame,
        sectors: pd.Series,
        lookback: int = 252,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.shares_outstanding = shares_outstanding
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
        ta = _align_fund(self.total_assets, prices_panel)
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        sh = _align_fund(self.shares_outstanding, prices_panel)
        gp = _align_fund(self.gross_profit, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(self.lookback)) / 2).where(ta > 0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(self.lookback)) / ta_safe.shift(self.lookback)).clip(-1.0, 5.0)
        sh_lag = sh.shift(self.lookback)
        dilution = ((sh - sh_lag) / sh_lag.where(sh_lag > 0)).clip(-0.5, 2.0)
        roa = (ni / ta_safe).clip(-1.0, 2.0)
        delta_roa = (roa - roa.shift(self.lookback)).clip(-1.0, 1.0)
        ocf_ta = (ocf / ta_safe).clip(-1.0, 2.0)
        mispricing = (
            _cs_zscore(-accruals).fillna(0.0)
            + _cs_zscore(-asset_growth).fillna(0.0)
            + _cs_zscore(-dilution).fillna(0.0)
            + _cs_zscore(roa).fillna(0.0)
            + _cs_zscore(delta_roa).fillna(0.0)
            + _cs_zscore(ocf_ta).fillna(0.0)
        )
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        mem = membership.reindex_like(gp_ta).fillna(False)
        gp_median = gp_ta.where(mem).median(axis=1)
        quality_gate = gp_ta.ge(gp_median, axis=0) & mem
        gated = mispricing.where(quality_gate, np.nan)
        daily_w = _sector_neutral_weights(
            gated, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H361: PercentileBinnedGP  (nonlinear quintile bin scoring of GP/TA)
# ---------------------------------------------------------------------------


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
            (pct_rank * self.n_bins).clip(upper=self.n_bins).fillna(0.0)
            .apply(lambda x: x.astype(int))
            .astype(float)
        )
        daily_w = _sector_neutral_weights(
            bin_score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H362: MonotonicityVotingComposite  (vote across GP, ROE, accruals)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H363: ElasticNetReturnForecast  (regularized ML cross-sectional ranker)
# ---------------------------------------------------------------------------


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
        score_df = pd.DataFrame(np.nan, index=all_dates, columns=tickers)

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

        score_df = score_df.ffill().fillna(0.0)
        daily_w = _sector_neutral_weights(
            score_df, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _periodic_rebalance(daily_w, self.rebalance_period)


# ---------------------------------------------------------------------------
# H364: NonlinearMomentumBins  (extreme-only momentum, top decile)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H365: ShrunkSignalComposite  (batch-3 ensemble with clipped z-scores)
# ---------------------------------------------------------------------------


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
            (pct_rank * self.n_bins).clip(upper=self.n_bins).fillna(0.0)
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
