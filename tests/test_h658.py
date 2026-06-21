"""H658 DeployableMAD658: the return-to-MAD ratio penalizes higher dispersion at equal mean."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
from run_h658 import lev_mad_neg, mad_ratio


def test_mad_ratio_penalizes_higher_dispersion_at_equal_mean():
    sign = np.where(np.arange(300) % 2 == 0, 1.0, -1.0)
    a = 0.001 + 0.005 * sign  # small alternating dispersion, mean 0.001
    b = 0.001 + 0.02 * sign  # larger dispersion, identical mean 0.001
    assert np.isclose(a.mean(), b.mean())
    assert mad_ratio(a) > mad_ratio(b)  # tighter dispersion -> higher ratio


def test_mad_ratio_positive_for_positive_drift():
    rng = np.random.RandomState(0)
    assert mad_ratio(0.0008 + 0.01 * rng.randn(500)) > 0.0


def test_lev_mad_neg_is_negative_ratio_at_unit_leverage():
    rng = np.random.RandomState(1)
    r = 0.0005 + 0.01 * rng.randn(300)
    idx = pd.date_range("2000-01-01", periods=300, freq="B")
    neg = lev_mad_neg(r.reshape(-1, 1), idx, 100.0, 1.0)(np.array([1.0]))
    assert np.isclose(-neg, mad_ratio(r), rtol=1e-6)
