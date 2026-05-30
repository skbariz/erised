"""Staging: H391-H399 strategy classes (Batches 9-10 non-MDR, Set 5)."""
# ---------------------------------------------------------------------------
# H391: LowAttentionValuePort  (neglected quality via |return| attention proxy)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H392: ConsistentMomentumPort  (both 12m and 3m momentum agree = confirmed)
# ---------------------------------------------------------------------------


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
        confirmed = (
            mom_long.ge(med_long, axis=0)
            & mom_short.ge(med_short, axis=0)
        )
        gated_mom = mom_long.where(confirmed, np.nan)
        daily_w = _sector_neutral_weights(
            gated_mom, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H393: LowVolDisagreementAvoid  (GP quality weighted by inverse realized vol)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# H394: ReturnSpikeRiskFilter  (GP composite with large-return event risk cut)
# ---------------------------------------------------------------------------


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
        hist_mean = abs_ret.rolling(self.lookback_window, min_periods=20).mean().shift(self.spike_window)
        hist_std = abs_ret.rolling(self.lookback_window, min_periods=20).std().shift(self.spike_window)
        z_spike = (recent_act - hist_mean) / hist_std.replace(0, np.nan)
        spike_flag = z_spike.gt(self.spike_threshold).fillna(False)
        scale = spike_flag.astype(float) * (self.cut_factor - 1.0) + 1.0
        scaled = (base_w * scale).clip(lower=0.0)
        row_sum = scaled.sum(axis=1).replace(0, np.nan)
        rescaled = scaled.div(row_sum, axis=0).fillna(0.0)
        return _monthly_rebalance(rescaled, prices_panel)


# ---------------------------------------------------------------------------
# H395: VolatilityAwareComposite  (batch-9 attention ensemble)
# ---------------------------------------------------------------------------


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
        spike_cut_w = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )

        z1 = _cs_zscore(neglect_score).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z2 = _cs_zscore(confirmed_mom).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z3 = _cs_zscore(gp_ta * inv_vol).clip(-self.clip_val, self.clip_val).fillna(0.0)
        z4 = _cs_zscore(gp_ta).clip(-self.clip_val, self.clip_val).fillna(0.0)
        score = z1 + z2 + z3 + z4

        daily_w = _sector_neutral_weights(
            score, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        return _monthly_rebalance(daily_w, prices_panel)


# ---------------------------------------------------------------------------
# H396: LowTurnoverGP  (threshold rebalancing for GP rank)
# ---------------------------------------------------------------------------


class LowTurnoverGP(CrossSectionalStrategy):
    """H396: GP/TA rank with threshold rebalancing (only when rank moves >2 deciles).

    Rebalances a stock's weight only when its GP/TA rank changes by more than
    threshold_deciles × (1/n_deciles) of the cross-section. Average rebalance
    frequency halved vs quarterly. Hendershott et al. 2011: turnover costs
    compound; threshold rebalancing captures most signal at lower cost.
    Sector-neutral, threshold rebalance.
    """

    name = "low_turnover_gp"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        n_deciles: int = 10,
        threshold_deciles: int = 2,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 21,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.n_deciles = n_deciles
        self.threshold_deciles = threshold_deciles
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
        pct_rank = gp_ta.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep").fillna(0.0)
        decile_rank = (pct_rank * self.n_deciles).clip(upper=self.n_deciles).apply(
            lambda x: x.astype(int)
        )

        # Compute base sector-neutral weights at monthly frequency
        base_w = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        base_w_monthly = _periodic_rebalance(base_w, self.rebalance_period)

        # Apply threshold: only update if rank changed by > threshold_deciles
        threshold = self.threshold_deciles / self.n_deciles
        prev_rank = pct_rank.shift(self.rebalance_period)
        rank_change = (pct_rank - prev_rank).abs()
        rebalance_needed = rank_change.gt(threshold) | rank_change.isna()

        # Build thresholded weights: carry forward when no rebalance needed
        final_w = base_w_monthly.copy()
        for i in range(1, len(final_w)):
            for t in final_w.columns:
                if not rebalance_needed.iloc[i][t]:
                    final_w.iloc[i][t] = final_w.iloc[i - 1][t]

        # Renormalize rows
        row_sum = final_w.sum(axis=1).replace(0, np.nan)
        return final_w.div(row_sum, axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# H397: SpreadAwarePositionSizer  (GP quality weighted by inverse vol-proxy spread)
# ---------------------------------------------------------------------------


class SpreadAwarePositionSizer(CrossSectionalStrategy):
    """H397: GP/TA quality with position sizes inversely proportional to realized vol.

    Realized vol proxies for effective spread (higher vol = wider spread for
    large caps). Weights scaled by inv_vol within the selected top-quality basket.
    Hendershott et al. 2011: effective spread varies even within S&P 500.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "spread_aware_position_sizer"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        vol_window: int = 63,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
        self.sectors = sectors
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
        ni = _align_fund(self.net_income, prices_panel)
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0)
        sust = _cs_zscore(-accruals).fillna(0.0) + _cs_zscore(-asset_growth).fillna(0.0)
        mem = membership.reindex_like(gp_ta).fillna(False)
        sust_rank = sust.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        filtered_mem = membership & (sust_rank >= 0.50).fillna(False)
        base_w = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        realized_vol = prices_panel.pct_change().rolling(self.vol_window, min_periods=20).std().shift(1)
        inv_vol = (1.0 / realized_vol.replace(0, np.nan)).clip(0.0, 100.0)
        # Scale within selected basket
        selected = base_w > 0
        scaled = base_w * inv_vol.where(selected, 0.0)
        row_sum = scaled.sum(axis=1).replace(0, np.nan)
        rescaled = scaled.div(row_sum, axis=0).fillna(0.0)
        return _periodic_rebalance(rescaled, self.rebalance_period)


# ---------------------------------------------------------------------------
# H398: LiquidityRegimeCapPort  (ML ensemble with low-vol regime cap)
# ---------------------------------------------------------------------------


class LiquidityRegimeCapPort(CrossSectionalStrategy):
    """H398: GP/TA quality with liquidity-regime participation cap.

    When a stock's realized vol drops below its 10th historical percentile
    (unusual quiet = potential liquidity event ahead), cap position at 50%.
    When realized vol is above 90th percentile (illiquid stress), cap at 30%.
    Normal regime: full weight. Hendershott et al. 2011: liquidity regime matters.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "liquidity_regime_cap_port"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        sectors: pd.Series,
        vol_window: int = 21,
        lookback: int = 252,
        low_pct: float = 0.10,
        high_pct: float = 0.90,
        low_cap: float = 0.50,
        high_cap: float = 0.30,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.sectors = sectors
        self.vol_window = vol_window
        self.lookback = lookback
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.low_cap = low_cap
        self.high_cap = high_cap
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
        realized_vol = prices_panel.pct_change().rolling(self.vol_window, min_periods=5).std().shift(1)
        vol_low = realized_vol.rolling(self.lookback, min_periods=63).quantile(self.low_pct)
        vol_high = realized_vol.rolling(self.lookback, min_periods=63).quantile(self.high_pct)
        in_low_regime = realized_vol.lt(vol_low)
        in_high_regime = realized_vol.gt(vol_high)
        cap = pd.DataFrame(1.0, index=prices_panel.index, columns=prices_panel.columns)
        cap = cap.where(~in_low_regime, self.low_cap)
        cap = cap.where(~in_high_regime, self.high_cap)
        scaled = (base_w * cap).clip(lower=0.0)
        row_sum = scaled.sum(axis=1).replace(0, np.nan)
        rescaled = scaled.div(row_sum, axis=0).fillna(0.0)
        return _periodic_rebalance(rescaled, self.rebalance_period)


# ---------------------------------------------------------------------------
# H399: ExecutionOptimizedComposite  (batch-10 execution ensemble)
# ---------------------------------------------------------------------------


class ExecutionOptimizedComposite(CrossSectionalStrategy):
    """H399: Normalized blend of batch-10 execution-refined signals.

    Combines: spread-aware GP (vol-scaled), liquidity-regime capped GP, and
    plain GP/TA anchor. Tests whether execution cost adjustments collectively
    improve net-of-cost risk-adjusted returns.
    Sector-neutral top 20%, quarterly rebalance.
    """

    name = "execution_optimized_composite"

    def __init__(
        self,
        gross_profit: pd.DataFrame,
        total_assets: pd.DataFrame,
        net_income: pd.DataFrame,
        operating_cf: pd.DataFrame,
        sectors: pd.Series,
        within_sector_quantile: float = 0.20,
        rebalance_period: int = 63,
    ) -> None:
        self.gross_profit = gross_profit
        self.total_assets = total_assets
        self.net_income = net_income
        self.operating_cf = operating_cf
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
        ocf = _align_fund(self.operating_cf, prices_panel)
        ta_safe = ta.where(ta > 0)
        avg_ta_safe = ((ta + ta.shift(252)) / 2).where(ta > 0)
        gp_ta = (gp / ta_safe).clip(0.0, 5.0)
        accruals = ((ni - ocf) / avg_ta_safe).clip(-1.0, 1.0)
        asset_growth = ((ta - ta.shift(252)) / ta_safe.shift(252)).clip(-1.0, 5.0)
        sust = _cs_zscore(-accruals).fillna(0.0) + _cs_zscore(-asset_growth).fillna(0.0)
        mem = membership.reindex_like(gp_ta).fillna(False)
        sust_rank = sust.where(mem).rank(axis=1, pct=True, ascending=True, na_option="keep")
        filtered_mem = membership & (sust_rank >= 0.50).fillna(False)
        # Component 1: spread-aware (inv-vol scaled) within quality filter
        realized_vol = prices_panel.pct_change().rolling(63, min_periods=20).std().shift(1)
        inv_vol = (1.0 / realized_vol.replace(0, np.nan)).clip(0.0, 100.0)
        base_filt = _sector_neutral_weights(
            gp_ta, filtered_mem, self.sectors, self.within_sector_quantile, ascending=False
        )
        spread_w = base_filt * inv_vol.where(base_filt > 0, 0.0)
        rs1 = spread_w.sum(axis=1).replace(0, np.nan)
        w1 = spread_w.div(rs1, axis=0).fillna(0.0)
        # Component 2: liquidity-regime capped plain GP
        vol_low = realized_vol.rolling(252, min_periods=63).quantile(0.10)
        vol_high = realized_vol.rolling(252, min_periods=63).quantile(0.90)
        in_low = realized_vol.lt(vol_low); in_high = realized_vol.gt(vol_high)
        cap = pd.DataFrame(1.0, index=prices_panel.index, columns=prices_panel.columns)
        cap = cap.where(~in_low, 0.50).where(~in_high, 0.30)
        base_plain = _sector_neutral_weights(
            gp_ta, membership, self.sectors, self.within_sector_quantile, ascending=False
        )
        regime_w = (base_plain * cap).clip(lower=0.0)
        rs2 = regime_w.sum(axis=1).replace(0, np.nan)
        w2 = regime_w.div(rs2, axis=0).fillna(0.0)
        # Component 3: plain GP anchor
        w3 = base_plain.copy()
        combined = w1 + w2 + w3
        row_sum = combined.sum(axis=1).replace(0, np.nan)
        daily_w = combined.div(row_sum, axis=0).fillna(0.0)
        return _periodic_rebalance(daily_w, self.rebalance_period)
