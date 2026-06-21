"""H656 DeployableCVaR656: the return-to-CVaR ratio penalizes a fat left tail at equal mean."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
from run_h656 import cvar_ratio, lev_cvar_neg


def test_cvar_ratio_penalizes_fat_left_tail():
    # Two streams with IDENTICAL mean: 'thin' spreads losses over 20 shallow days,
    # 'fat' concentrates the same total loss into 4 deep days -> worse 5% expected shortfall.
    thin = np.full(300, 0.0015)
    thin[10:30] -= 0.01  # 20 x -0.01 of extra loss
    fat = np.full(300, 0.0015)
    fat[10:14] -= 0.05  # 4 x -0.05 of extra loss (same total -0.20)
    assert np.isclose(thin.mean(), fat.mean())  # equal mean by construction

    idx = pd.date_range("2000-01-01", periods=300, freq="B")
    # cap=1.0 with a huge target vol pins leverage at ~1.0 so raw returns pass through.
    neg_fat = lev_cvar_neg(fat.reshape(-1, 1), idx, 100.0, 1.0)(np.array([1.0]))
    neg_thin = lev_cvar_neg(thin.reshape(-1, 1), idx, 100.0, 1.0)(np.array([1.0]))
    assert neg_fat > neg_thin  # fat tail -> lower ratio -> higher (worse) minimized objective


def test_cvar_ratio_positive_for_positive_drift():
    rng = np.random.RandomState(0)
    r = 0.0008 + 0.01 * rng.randn(500)
    assert cvar_ratio(r) > 0.0
