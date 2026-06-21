"""H661 DeployableFoldRobust661: the objective penalizes dispersion of per-fold Sharpe."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
from run_h661 import _fold_sharpes, lev_foldrobust_neg


def test_objective_penalizes_fold_dispersion():
    idx = pd.date_range("2000-01-01", periods=200, freq="B")
    m1 = np.zeros(200, bool)
    m1[:100] = True
    masks = [m1, ~m1]
    rng = np.random.RandomState(0)
    steady = 0.001 + 0.005 * rng.randn(200)
    disp = steady.copy()
    disp[:100] += 0.004  # first half much stronger
    disp[100:] -= 0.004  # second half much weaker -> high fold-Sharpe dispersion
    w = np.array([1.0])
    neg_steady = lev_foldrobust_neg(steady.reshape(-1, 1), idx, 100.0, 1.0, masks, lam=1.0)(w)
    neg_disp = lev_foldrobust_neg(disp.reshape(-1, 1), idx, 100.0, 1.0, masks, lam=1.0)(w)
    assert neg_disp > neg_steady  # more fold dispersion -> worse (higher) minimized objective


def test_fold_sharpes_skips_short_folds():
    short = np.zeros(50, bool)
    short[:5] = True  # only 5 days -> below the 10-day minimum, dropped
    full = ~short
    fss = _fold_sharpes(0.001 + 0.005 * np.random.RandomState(0).randn(50), [short, full])
    assert len(fss) == 1  # the 5-day fold is excluded
