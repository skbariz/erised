"""Runtime configuration and synthetic data."""

from __future__ import annotations

import importlib
from pathlib import Path

import erised.config
from erised.data.schema import validate_bars
from erised.data.synthetic import synthetic_bars


def test_data_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ERISED_DATA_DIR", str(tmp_path))
    try:
        assert tmp_path == importlib.reload(erised.config).DATA_DIR
    finally:
        monkeypatch.delenv("ERISED_DATA_DIR")
        importlib.reload(erised.config)


def test_data_dir_defaults_to_repo_data_in_source_checkout(monkeypatch):
    monkeypatch.delenv("ERISED_DATA_DIR", raising=False)
    repo_root = Path(__file__).resolve().parents[2]
    assert repo_root / "data" == importlib.reload(erised.config).DATA_DIR


def test_synthetic_bars_conform_to_schema_and_are_deterministic():
    a = synthetic_bars(start="2010-01-01", end="2012-12-31", seed=7)
    b = synthetic_bars(start="2010-01-01", end="2012-12-31", seed=7)
    validate_bars(a, "SYNTH")
    assert a.equals(b)
