"""End-to-end CLI checks against an isolated synthetic data directory."""

from __future__ import annotations

import importlib

import pytest

import erised.config


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Reload the data stack against tmp_path so the CLI never touches the real store."""
    monkeypatch.setenv("ERISED_DATA_DIR", str(tmp_path))
    modules = [
        "erised.config",
        "erised.data.loader",
        "erised.data.stocks",
        "erised.data.holdout",
        "erised.data.synthetic",
        "erised.cli",
    ]
    reloaded = [importlib.reload(importlib.import_module(m)) for m in modules]
    yield reloaded[-1], tmp_path
    monkeypatch.delenv("ERISED_DATA_DIR")
    for m in modules:
        importlib.reload(importlib.import_module(m))
    assert tmp_path != erised.config.DATA_DIR


def test_synthetic_seal_backtest_roundtrip(cli, capsys):
    module, data_dir = cli
    assert module.main(["data", "synthetic"]) == 0
    assert module.main(["data", "seal"]) == 0
    assert (data_dir / "SPY_research.parquet").exists()
    assert (data_dir / "HOLDOUT_HASH.txt").exists()

    capsys.readouterr()
    assert module.main(["backtest", "MovingAverageCrossover", "--param", "fast=20"]) == 0
    out = capsys.readouterr().out
    assert "12 folds" in out and "Sharpe 95% CI" in out

    assert module.main(["backtest", "RiskParity", "--assets", "TLT"]) == 0


def test_backtest_reports_missing_asset_column(cli):
    module, _ = cli
    module.main(["data", "synthetic"])
    module.main(["data", "seal"])
    with pytest.raises(SystemExit, match="--assets"):
        module.main(["backtest", "RiskParity"])


def test_unknown_strategy_is_rejected(cli):
    module, _ = cli
    with pytest.raises(SystemExit, match="Unknown strategy"):
        module.main(["backtest", "NotAStrategy"])


def test_strategy_listing(cli, capsys):
    module, _ = cli
    assert module.main(["strategies"]) == 0
    assert "MovingAverageCrossover" in capsys.readouterr().out
    assert module.main(["strategies", "--cross-sectional"]) == 0
    assert "BettingAgainstBeta" in capsys.readouterr().out
