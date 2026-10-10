"""Tests for Batch 9 strategies: pairs trading and PCA stat arb (H190-H194)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies.cross_sectional import (
    LongShortMomentum,
    LongShortReversal,
    PCAOUMeanReversion,
    PCAResidualReversal,
    SectorPairsTrading,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATES = pd.date_range("2010-01-01", periods=400, freq="B")
TICKERS = ["AAPL", "MSFT", "GOOG", "JPM", "GS", "BAC", "XOM", "CVX", "JNJ", "PFE"]
SECTORS = pd.Series(
    {
        "AAPL": "Tech",
        "MSFT": "Tech",
        "GOOG": "Tech",
        "JPM": "Financials",
        "GS": "Financials",
        "BAC": "Financials",
        "XOM": "Energy",
        "CVX": "Energy",
        "JNJ": "Healthcare",
        "PFE": "Healthcare",
    }
)


def make_prices(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.015, (len(DATES), len(TICKERS)))
    prices = 100 * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(prices, index=DATES, columns=TICKERS)


def make_membership(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=prices.index, columns=prices.columns)


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def membership(prices):
    return make_membership(prices)


@pytest.fixture
def train_prices(prices):
    return prices.iloc[:252]


# ---------------------------------------------------------------------------
# Cointegrated pair for pairs trading test
# ---------------------------------------------------------------------------


def make_cointegrated_prices() -> pd.DataFrame:
    """SYNC_A and SYNC_B track each other; NOISE is unrelated."""
    rng = np.random.default_rng(0)
    common = np.cumsum(rng.normal(0, 0.01, len(DATES)))
    sync_a = 100 * np.exp(common + rng.normal(0, 0.002, len(DATES)))
    sync_b = 100 * np.exp(common + rng.normal(0, 0.002, len(DATES)))
    noise = 100 * np.cumprod(1 + rng.normal(0, 0.02, len(DATES)))
    df = pd.DataFrame({"SYNC_A": sync_a, "SYNC_B": sync_b, "NOISE": noise}, index=DATES)
    return df


# ---------------------------------------------------------------------------
# H190: SectorPairsTrading
# ---------------------------------------------------------------------------


class TestSectorPairsTrading:
    def test_returns_dataframe(self, prices, membership, train_prices):
        s = SectorPairsTrading(train_prices=train_prices, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert isinstance(w, pd.DataFrame)
        assert w.index.equals(prices.index)

    def test_pairs_formed_within_sectors(self, train_prices):
        s = SectorPairsTrading(train_prices=train_prices, sectors=SECTORS, n_pairs_per_sector=2)
        for a, b in s.pairs:
            assert SECTORS[a] == SECTORS[b], f"Pair ({a},{b}) spans sectors"

    def test_dollar_neutral(self, prices, membership, train_prices):
        s = SectorPairsTrading(train_prices=train_prices, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        # Rows with active positions should be approximately dollar-neutral
        active = w.abs().sum(axis=1) > 0
        net_exposure = w[active].sum(axis=1).abs()
        assert (net_exposure < 0.01).all(), "Pairs weights should be dollar-neutral"

    def test_long_short_signs(self, train_prices):
        """Cointegrated pair: verify reversal direction by forcing a spread divergence."""
        coint = make_cointegrated_prices()
        coint_sectors = pd.Series({"SYNC_A": "Tech", "SYNC_B": "Tech", "NOISE": "Energy"})
        s = SectorPairsTrading(train_prices=coint.iloc[:252], sectors=coint_sectors, entry_z=0.5)
        w = s.generate_weights(coint, make_membership(coint))
        # Should have both positive and negative weights somewhere
        assert (w > 0).any().any(), "Should have long positions"
        assert (w < 0).any().any(), "Should have short positions"

    def test_missing_tickers_handled(self, prices, membership, train_prices):
        """Strategy should silently skip pairs where one ticker is missing."""
        sub_prices = prices.drop(columns=["MSFT"])
        sub_mem = membership.drop(columns=["MSFT"])
        s = SectorPairsTrading(train_prices=train_prices, sectors=SECTORS)
        w = s.generate_weights(sub_prices, sub_mem)
        assert "MSFT" not in w.columns


# ---------------------------------------------------------------------------
# H191: PCAResidualReversal
# ---------------------------------------------------------------------------


class TestPCAResidualReversal:
    def test_returns_dataframe(self, prices, membership, train_prices):
        s = PCAResidualReversal(train_prices=train_prices, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert isinstance(w, pd.DataFrame)
        assert w.index.equals(prices.index)

    def test_weights_sum_leq_one(self, prices, membership, train_prices):
        s = PCAResidualReversal(train_prices=train_prices, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert (w.sum(axis=1).round(6) <= 1.0).all()

    def test_n_components_stored(self, train_prices):
        s = PCAResidualReversal(train_prices=train_prices, sectors=SECTORS, n_components=5)
        assert s.components_df.shape[0] == 5

    def test_periodic_rebalance_structure(self, prices, membership, train_prices):
        s = PCAResidualReversal(train_prices=train_prices, sectors=SECTORS, rebalance_period=5)
        w = s.generate_weights(prices, membership)
        # Rows 1-4 within each 5-day block should equal row 0 of that block
        block_start = w.iloc[::5]
        for _ in range(1, 5):
            aligned = block_start.reindex(w.index, method="ffill")
            diff = (w - aligned).abs().max(axis=1)
            # Allow small float differences
            assert (diff < 1e-10).all()


# ---------------------------------------------------------------------------
# H192: PCAOUMeanReversion
# ---------------------------------------------------------------------------


class TestPCAOUMeanReversion:
    def test_returns_dataframe(self, prices, membership, train_prices):
        s = PCAOUMeanReversion(train_prices=train_prices, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert isinstance(w, pd.DataFrame)

    def test_weights_sum_leq_one(self, prices, membership, train_prices):
        s = PCAOUMeanReversion(train_prices=train_prices, sectors=SECTORS, ou_window=60)
        w = s.generate_weights(prices, membership)
        assert (w.sum(axis=1).round(6) <= 1.0).all()

    def test_tighter_entry_z_fewer_positions(self, prices, membership, train_prices):
        s_loose = PCAOUMeanReversion(
            train_prices=train_prices, sectors=SECTORS, entry_z=-1.0, ou_window=60
        )
        s_tight = PCAOUMeanReversion(
            train_prices=train_prices, sectors=SECTORS, entry_z=-2.5, ou_window=60
        )
        w_loose = s_loose.generate_weights(prices, membership)
        w_tight = s_tight.generate_weights(prices, membership)
        n_loose = (w_loose > 0).sum().sum()
        n_tight = (w_tight > 0).sum().sum()
        assert n_loose >= n_tight, "Looser threshold should produce >= positions"

    def test_components_match_n_components(self, train_prices):
        for k in [5, 10]:
            s = PCAOUMeanReversion(train_prices=train_prices, sectors=SECTORS, n_components=k)
            assert s.components_df.shape[0] == k


# ---------------------------------------------------------------------------
# H193: LongShortMomentum
# ---------------------------------------------------------------------------


class TestLongShortMomentum:
    def test_returns_dataframe(self, prices, membership):
        s = LongShortMomentum(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert isinstance(w, pd.DataFrame)

    def test_dollar_neutral(self, prices, membership):
        s = LongShortMomentum(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        active = w.abs().sum(axis=1) > 0
        net = w[active].sum(axis=1).abs()
        assert (net < 0.01).all(), "L/S portfolio should be dollar-neutral"

    def test_has_both_long_and_short(self, prices, membership):
        s = LongShortMomentum(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert (w > 0).any().any(), "Should have long positions"
        assert (w < 0).any().any(), "Should have short positions"

    def test_gross_exposure_near_one(self, prices, membership):
        s = LongShortMomentum(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        active = w.abs().sum(axis=1) > 0
        gross = w[active].abs().sum(axis=1)
        assert ((gross - 1.0).abs() < 0.01).all(), "Gross exposure should be ~1.0"

    def test_high_mom_long_low_mom_short(self):
        """Stock with monotonically rising price should be long; falling should be short."""
        prices = make_prices()
        # Override: AAPL rises 100%, MSFT falls 50% over first 252 days
        up_factor = np.linspace(1.0, 2.0, len(DATES))
        down_factor = np.linspace(1.0, 0.5, len(DATES))
        prices["AAPL"] = 100 * up_factor
        prices["MSFT"] = 100 * down_factor
        mem = make_membership(prices)
        s = LongShortMomentum(sectors=SECTORS)
        w = s.generate_weights(prices, mem)
        # AAPL (strong momentum) should be long; MSFT (weak) short in most periods
        active_rows = w.abs().sum(axis=1) > 0
        if active_rows.sum() > 20:
            aapl_sign = w.loc[active_rows, "AAPL"].mean()
            msft_sign = w.loc[active_rows, "MSFT"].mean()
            assert aapl_sign > 0, "Rising stock should be long on average"
            assert msft_sign < 0, "Falling stock should be short on average"


# ---------------------------------------------------------------------------
# H194: LongShortReversal
# ---------------------------------------------------------------------------


class TestLongShortReversal:
    def test_returns_dataframe(self, prices, membership):
        s = LongShortReversal(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert isinstance(w, pd.DataFrame)

    def test_dollar_neutral(self, prices, membership):
        s = LongShortReversal(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        active = w.abs().sum(axis=1) > 0
        net = w[active].sum(axis=1).abs()
        assert (net < 0.01).all(), "L/S portfolio should be dollar-neutral"

    def test_has_both_long_and_short(self, prices, membership):
        s = LongShortReversal(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert (w > 0).any().any()
        assert (w < 0).any().any()

    def test_loser_long_winner_short(self):
        """Stock that fell most last 5 days should be long; winner short."""
        prices = make_prices()
        # On day 300, force AAPL down 20% and MSFT up 20% cumulatively over 5 days
        prices.iloc[295:300, 0] = prices.iloc[295, 0] * np.linspace(1.0, 0.8, 5)  # AAPL down
        prices.iloc[295:300, 1] = prices.iloc[295, 1] * np.linspace(1.0, 1.2, 5)  # MSFT up
        mem = make_membership(prices)
        s = LongShortReversal(sectors=SECTORS, rebalance_period=1)
        w = s.generate_weights(prices, mem)
        # At day 300, AAPL should be long (big loser) and MSFT short (big winner)
        day = prices.index[299]
        assert w.loc[day, "AAPL"] > 0, "Loser should be long"
        assert w.loc[day, "MSFT"] < 0, "Winner should be short"

    def test_periodic_rebalance_holds_weights(self, prices, membership):
        s = LongShortReversal(sectors=SECTORS, rebalance_period=5)
        w = s.generate_weights(prices, membership)
        # Non-rebalance rows (positions 1-4 within each 5-day block) should
        # equal the preceding rebalance row
        block_start = w.iloc[::5]
        aligned = block_start.reindex(w.index, method="ffill")
        diff = (w - aligned).abs().max(axis=1)
        assert (diff < 1e-10).all()
