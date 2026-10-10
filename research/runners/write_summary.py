"""write_summary.py - Write a compact JSON summary after any hypothesis run.

Usage (add to end of any run_h*.py):
    from scripts.write_summary import write_summary
    write_summary(
        hypothesis_id="021",
        name="risk-parity",
        strategy_class="RiskParity",
        family="risk parity",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=corr,
        fold_sharpes=[round(sharpe(f.net_returns), 3) for f in wf.fold_results],
        decision="keep",  # or "kill"
        notes="",
    )
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def write_summary(
    hypothesis_id: str,
    name: str,
    strategy_class: str,
    family: str,
    sharpe: float,
    ci_low: float,
    ci_high: float,
    cagr: float,
    max_drawdown: float,
    n_trades: int,
    h001_corr: float,
    fold_sharpes: list[float],
    decision: str,
    notes: str = "",
    canonical_params: dict | None = None,
) -> Path:
    """Write results/<NNN>-<name>/summary.json and return the path."""

    results_dir = Path(__file__).resolve().parents[1] / "results" / f"{hypothesis_id}-{name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "hypothesis": f"H{hypothesis_id}",
        "name": name,
        "strategy_class": strategy_class,
        "family": family,
        "run_date": date.today().isoformat(),
        "oos_sharpe": round(sharpe, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(max_drawdown, 4),
        "n_trades": n_trades,
        "h001_corr": round(h001_corr, 4),
        "fold_sharpes": [round(s, 3) for s in fold_sharpes],
        "decision": decision.lower(),
        "kept": decision.lower() == "keep",
        "notes": notes,
        "canonical_params": canonical_params or {},
        "spy_benchmark": {"sharpe": 0.51, "cagr": 0.0835, "max_drawdown": -0.5592},
        "beats_benchmark": sharpe > 0.51 and ci_low > 0.0,
    }

    out = results_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written: {out}")
    return out
