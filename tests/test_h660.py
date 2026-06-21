"""H660 DeployableMinVarLW660: Ledoit-Wolf shrinkage is valid/trace-preserving; GMV prefers low variance."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
from run_h660 import gmv_weights, ledoit_wolf_cov


def test_lw_intensity_in_unit_interval_and_trace_preserved():
    rng = np.random.RandomState(0)
    x = rng.randn(1000, 5) * np.array([0.01, 0.02, 0.01, 0.03, 0.02])
    cov_lw, delta = ledoit_wolf_cov(x)
    assert 0.0 <= delta <= 1.0
    s = (x - x.mean(0)).T @ (x - x.mean(0)) / x.shape[0]  # MLE sample cov
    assert np.isclose(np.trace(cov_lw), np.trace(s), rtol=1e-8)  # shrinkage to mu*I preserves trace
    assert np.allclose(cov_lw, cov_lw.T)  # symmetric


def test_lw_shrinks_off_diagonals_toward_zero():
    rng = np.random.RandomState(1)
    base = rng.randn(800, 1)
    x = np.hstack([base + 0.1 * rng.randn(800, 1) for _ in range(4)])  # strongly correlated
    cov_lw, delta = ledoit_wolf_cov(x)
    s = (x - x.mean(0)).T @ (x - x.mean(0)) / x.shape[0]
    off = ~np.eye(4, dtype=bool)
    assert np.abs(cov_lw[off]).mean() < np.abs(s[off]).mean()  # off-diagonals pulled toward target


def test_gmv_prefers_low_variance_assets():
    cov = np.diag([1.0, 4.0, 9.0])  # uncorrelated, increasing variance
    w = gmv_weights(cov)
    assert np.isclose(w.sum(), 1.0) and np.all(w >= 0)
    assert w[0] > w[1] > w[2]  # inverse-variance ordering
