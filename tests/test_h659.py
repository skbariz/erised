"""H659 DeployableERC659: equal-risk-contribution weights equalize per-sleeve variance contributions."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "runners"))

import numpy as np
from run_h659 import erc_weights


def test_erc_equalizes_risk_contributions():
    rng = np.random.RandomState(0)
    # three uncorrelated sleeves with different vols
    x = rng.randn(3000, 3) * np.array([0.005, 0.01, 0.02])
    w = erc_weights(x)
    assert np.all(w > 0) and np.isclose(w.sum(), 1.0)  # valid simplex point
    cov = np.cov(x.T) * 252
    rc = w * (cov @ w)
    assert np.allclose(rc, rc.mean(), rtol=0.05)  # risk contributions ~equal


def test_erc_downweights_high_vol_sleeve():
    rng = np.random.RandomState(1)
    x = rng.randn(3000, 2) * np.array([0.005, 0.02])  # second sleeve 4x the vol
    w = erc_weights(x)
    assert w[0] > w[1]  # lower-vol sleeve gets more weight
