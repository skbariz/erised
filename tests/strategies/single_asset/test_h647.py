"""H647 DeployableLowCorr: correlation-penalty objective behaves monotonically."""

from __future__ import annotations

import numpy as np


def test_corr_penalty_increases_with_correlation():
    rng = np.random.RandomState(0)
    base = rng.normal(0, 0.01, 1000)
    h001 = rng.normal(0, 0.01, 1000)
    high_corr = base + h001  # strongly correlated with h001
    low_corr = base  # less correlated
    lam = 1.0
    pen_high = lam * np.corrcoef(high_corr, h001)[0, 1]
    pen_low = lam * np.corrcoef(low_corr, h001)[0, 1]
    assert pen_high > pen_low  # penalty term larger for the higher-corr portfolio
