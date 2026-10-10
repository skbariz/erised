"""H653 DeployableInverseVol653: inverse-vol weights are a valid, vol-monotone simplex point."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "runners"))

import numpy as np
import pandas as pd
from run_h653 import inverse_vol_weights


def test_inverse_vol_weights_valid_and_monotone():
    rng = np.random.RandomState(0)
    idx = pd.date_range("2000-01-01", periods=500, freq="B")
    # three sleeves with increasing volatility
    clean = pd.DataFrame(
        {
            "low": 0.005 * rng.randn(500),
            "mid": 0.015 * rng.randn(500),
            "high": 0.030 * rng.randn(500),
        },
        index=idx,
    )
    w = inverse_vol_weights(clean)
    assert np.all(w >= 0) and np.isclose(w.sum(), 1.0)  # valid simplex point
    assert w[0] > w[1] > w[2]  # lower vol -> higher weight


def test_inverse_vol_ignores_nan_history():
    rng = np.random.RandomState(1)
    idx = pd.date_range("2000-01-01", periods=100, freq="B")
    clean = pd.DataFrame(
        {"a": np.r_[np.full(50, np.nan), 0.01 * rng.randn(50)], "b": 0.01 * rng.randn(100)},
        index=idx,
    )
    # std() is NaN-aware, so the 50 NaN rows of "a" don't deflate its vol toward zero
    w = inverse_vol_weights(clean)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(np.isfinite(w))
