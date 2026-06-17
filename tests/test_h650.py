"""H650 GrandDeployable650: lagged-cap construction (scalar vs Series target_vol)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
from set8_lev import two_state_cap, vix_momentum_mask


def _lagged_caps(vix, idx, thr, asym):
    bull = vix_momentum_mask(vix, idx, thr, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    tv = two_state_cap(bull, 0.08, 0.04, idx) if asym else 0.065
    return tv, cap


def test_lagged_caps_scalar_vs_series_tv():
    idx = pd.date_range("2000-01-01", periods=60, freq="B")
    vix = pd.Series(np.linspace(25.0, 10.0, 60), index=idx)
    tv_plain, cap_plain = _lagged_caps(vix, idx, 16.0, asym=False)
    tv_asym, cap_asym = _lagged_caps(vix, idx, 16.0, asym=True)
    assert np.isscalar(tv_plain) and tv_plain == 0.065
    assert isinstance(tv_asym, pd.Series)
    assert set(np.unique(tv_asym.to_numpy())) <= {0.08, 0.04}
    assert len(cap_plain) == 60 and len(cap_asym) == 60
