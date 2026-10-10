"""H657 DeployableOmega657: Omega rewards shifting probability mass from losses to gains."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "runners"))

import numpy as np
import pandas as pd
from run_h657 import lev_omega_neg, omega_ratio


def test_omega_rewards_shrinking_a_loss():
    rng = np.random.RandomState(0)
    r = 0.0005 + 0.01 * rng.randn(400)
    r2 = r.copy()
    i = int(r.argmin())
    r2[i] = r[i] / 2  # halve the worst loss -> strictly more upside mass relative to downside
    assert omega_ratio(r2) > omega_ratio(r)


def test_omega_above_one_for_positive_drift():
    rng = np.random.RandomState(1)
    r = 0.0006 + 0.008 * rng.randn(500)
    assert omega_ratio(r) > 1.0  # positive expected return -> gains mass exceeds losses mass


def test_lev_omega_neg_is_negative_ratio_at_unit_leverage():
    rng = np.random.RandomState(2)
    r = 0.0005 + 0.01 * rng.randn(300)
    idx = pd.date_range("2000-01-01", periods=300, freq="B")
    neg = lev_omega_neg(r.reshape(-1, 1), idx, 100.0, 1.0)(np.array([1.0]))
    assert neg < 0.0 and np.isclose(-neg, omega_ratio(r), rtol=1e-6)
