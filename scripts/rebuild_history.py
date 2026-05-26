#!/usr/bin/env python3
"""Rebuild git history: 4 platform commits + 4 per hypothesis (H001-H012) = 52 total.

Run from project root:
    python scripts/rebuild_history.py

Creates branch `clean-main` with the clean history.  After review:
    git checkout clean-main
    git branch -D main
    git branch -m clean-main main
    git push --force origin main
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ─── Git helpers ────────────────────────────────────────────────────────────


def g(*args, check=True):
    r = subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"\nFAIL: git {' '.join(args)}\n{r.stderr.strip()}")
        sys.exit(1)
    return r.stdout.strip()


def write(rel, content):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def rm(rel):
    p = ROOT / rel
    if p.exists():
        p.unlink()


def commit(msg, *add_files, rm_files=()):
    for f in rm_files:
        g("rm", "--cached", "--", f, check=False)
        rm(f)
    for f in add_files:
        g("add", "--", f)
    g("commit", "-m", msg)
    print(f"  {g('rev-parse', '--short', 'HEAD')}  {msg[:72]}")


# ─── Source extractors ──────────────────────────────────────────────────────


def load_classes():
    """Return dict name→source extracted from base.py via AST."""
    src = (ROOT / "src/quant/strategies/base.py").read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    nodes = sorted(
        [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)],
        key=lambda n: n.lineno,
    )
    out = {}
    for i, node in enumerate(nodes):
        s = node.lineno - 1
        e = nodes[i + 1].lineno - 1 if i + 1 < len(nodes) else len(lines)
        out[node.name] = "".join(lines[s:e]).rstrip("\n")
    # Header = everything before first class
    out["__header__"] = "".join(lines[: nodes[0].lineno - 1]).rstrip("\n")
    return out


def load_test_funcs():
    """Return dict name→source for every function in test_backtest.py."""
    src = (ROOT / "tests/test_backtest.py").read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    nodes = sorted(
        [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)],
        key=lambda n: n.lineno,
    )
    out = {}
    for i, node in enumerate(nodes):
        s = node.lineno - 1
        e = nodes[i + 1].lineno - 1 if i + 1 < len(nodes) else len(lines)
        out[node.name] = "".join(lines[s:e]).rstrip("\n")
    return out


def load_batch_sections():
    """Return ordered list of (label, body) for each H-section in run_batch1.py."""
    src = (ROOT / "scripts/run_batch1.py").read_text(encoding="utf-8")
    header_end = src.find("\n\ndef main()")
    header = src[:header_end]
    # Split main() body on the section markers
    main_src = src[header_end:]
    # markers are "    # ------ H0XX" or "    # Correlation"
    sections = re.split(r"(\n    # -{10,} H\d{3}|\n    # -{10,} Correlation)", main_src)
    # sections[0] = main() header up to first marker
    # sections[1,3,5,...] = markers; sections[2,4,6,...] = bodies
    result = [("__header__", header), ("__main_top__", sections[0])]
    for i in range(1, len(sections) - 1, 2):
        label = sections[i].strip().lstrip("# -").strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""
        result.append((label, sections[i] + body))
    return result


def load_summary_rows():
    """Return header, list of table rows, and footer from SUMMARY.md."""
    src = (ROOT / "results/SUMMARY.md").read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    # Find the table: lines starting with | 0
    rows = {}
    for line in lines:
        m = re.match(r"\| (\d{3}) \|", line)
        if m:
            rows[m.group(1)] = line
    return rows


# ─── File builders ──────────────────────────────────────────────────────────

# Ordered list: (hypothesis_id, class_name, slug, family, status, sharpe, corr)
HYPO_META = [
    ("001", "MovingAverageCrossover", "ma-50-200-baseline", "trend", "kept (baseline)", 0.61, "—"),
    (
        "002",
        "MeanReversionStrategy",
        "mean-reversion-spy",
        "mean reversion",
        "killed",
        0.24,
        "+0.55",
    ),
    (
        "003",
        "ShortHoldMeanReversion",
        "short-hold-mean-reversion",
        "mean reversion",
        "killed",
        0.01,
        "+0.44",
    ),
    ("004", "RSIReversionStrategy", "rsi-reversion", "mean reversion", "killed", 0.06, "+0.31"),
    (
        "005",
        "ConsecutiveDownDays",
        "consecutive-down-days",
        "mean reversion",
        "killed",
        0.44,
        "+0.40",
    ),
    (
        "006",
        "VIXFilteredReversionStrategy",
        "vix-filtered-reversion",
        "mean reversion",
        "killed",
        0.01,
        "+0.40",
    ),
    ("007", "TimeSeriesMomentum", "ts-momentum", "trend", "kept", 0.62, "+0.89"),
    ("008", "AbsoluteMomentumStrategy", "absolute-momentum", "trend", "kept", 0.62, "+0.86"),
    ("009", "MACDCrossover", "macd-crossover", "trend", "killed", 0.37, "+0.32"),
    ("010", "FastMACrossover", "fast-ma-crossover", "trend", "killed", 0.45, "+0.58"),
    ("011", "ADXFilteredMACrossover", "adx-filtered-ma", "trend", "killed", 0.27, "+0.72"),
    ("012", "VolatilityRegime", "vol-regime", "volatility", "killed", 0.33, "+0.51"),
]

PLATFORM_CLASSES = ["Strategy", "BuyAndHold"]

# Test functions per hypothesis
PLATFORM_TESTS = [
    "test_buy_and_hold_matches_benchmark",
    "test_no_lookahead_position_shifted",
    "test_costs_charged_on_position_change",
    "test_short_position_inverts_returns",
    "test_position_magnitude_validated",
]
H_TESTS = {
    "001": ["test_ma_crossover_no_lookahead", "test_buy_and_hold_compounds_correctly"],
    "002": [
        "test_mean_reversion_no_lookahead",
        "test_mean_reversion_fires_on_down_move_and_holds",
        "test_mean_reversion_flat_before_warmup",
    ],
    "003": [
        "test_short_hold_mean_reversion_no_lookahead",
        "test_short_hold_mean_reversion_hold_expires_at_2",
    ],
    "004": [
        "test_rsi_reversion_no_lookahead",
        "test_rsi_reversion_flat_before_warmup",
        "test_rsi_reversion_exits_above_exit_rsi",
    ],
    "005": [
        "test_consecutive_down_days_no_lookahead",
        "test_consecutive_down_days_fires_on_streak",
        "test_consecutive_down_days_no_signal_after_up_break",
    ],
    "006": [
        "test_vix_filtered_reversion_no_lookahead",
        "test_vix_filtered_reversion_suppressed_in_low_vix",
        "test_vix_filtered_reversion_requires_vix_column",
    ],
    "007": ["test_ts_momentum_no_lookahead", "test_ts_momentum_flat_before_warmup"],
    "008": ["test_absolute_momentum_no_lookahead", "test_absolute_momentum_long_in_uptrend"],
    "009": ["test_macd_no_lookahead", "test_macd_flat_during_warmup"],
    "010": ["test_fast_ma_no_lookahead", "test_fast_ma_flat_before_slow_ma_warm"],
    "011": ["test_adx_filtered_ma_no_lookahead", "test_adx_filtered_ma_suppresses_weak_trend"],
    "012": ["test_vol_regime_no_lookahead", "test_vol_regime_flat_before_warmup"],
}
# Helpers that appear in test_backtest.py, added at these stages
TEST_HELPERS = {
    "platform": ["_make_prices"],
    "006": ["_make_prices_with_vix"],
}

# Classes added to strategies/__init__.py imports per hypothesis
INIT_CLASSES = {
    "001": ["MovingAverageCrossover"],
    "002": ["MeanReversionStrategy"],
    "003": ["ShortHoldMeanReversion"],
    "004": ["RSIReversionStrategy"],
    "005": ["ConsecutiveDownDays"],
    "006": ["VIXFilteredReversionStrategy"],
    "007": ["TimeSeriesMomentum"],
    "008": ["AbsoluteMomentumStrategy"],
    "009": ["MACDCrossover"],
    "010": ["FastMACrossover"],
    "011": ["ADXFilteredMACrossover"],
    "012": ["VolatilityRegime"],
}

# run_batch1 section labels in order (from split)
BATCH_SECTION_IDS = [
    "H003",
    "H004",
    "H005",
    "H006",
    "H007",
    "H008",
    "H009",
    "H010",
    "H011",
    "H012",
    "Correlation",
]


def build_base_py(classes, include_hyp_ids):
    """Build base.py with platform classes + hypothesis classes for given ids."""
    parts = [classes["__header__"]]
    for cls in PLATFORM_CLASSES:
        parts.append("\n\n" + classes[cls])
    # Add in hypothesis order
    hyp_class_map = {hid: cls for hid, cls, *_ in HYPO_META}
    for hid in [h[0] for h in HYPO_META]:
        if hid in include_hyp_ids:
            parts.append("\n\n" + classes[hyp_class_map[hid]])
    return "\n".join(parts) + "\n"


def build_strategies_init(include_hyp_ids):
    """Build strategies/__init__.py exports."""
    all_names = sorted(
        ["Strategy", "BuyAndHold"]
        + [c for hid in include_hyp_ids for c in INIT_CLASSES.get(hid, [])]
    )
    imports = ",\n    ".join(all_names)
    dunder = "\n".join(f'    "{n}",' for n in all_names)
    return f"""from .base import (
    {imports},
)

__all__ = [
{dunder}
]
"""


def build_test_backtest(test_funcs, include_hyp_ids):
    """Build test_backtest.py for given hypothesis ids."""
    # Determine which strategy classes are imported at this stage
    strat_imports = ["BuyAndHold"]
    for hid in [h[0] for h in HYPO_META]:
        if hid in include_hyp_ids:
            strat_imports += INIT_CLASSES.get(hid, [])
    strat_imports = sorted(strat_imports)
    imports_str = ",\n    ".join(strat_imports)

    lines = [
        '"""Tests for the backtest engine."""',
        "",
        "from __future__ import annotations",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import pytest",
        "",
        "from quant.backtest.engine import assert_no_lookahead, run_backtest",
        "from quant.strategies.base import (",
        f"    {imports_str},",
        ")",
        "",
        "",
    ]

    # Platform helper
    lines.append(test_funcs["_make_prices"])
    lines.append("")
    lines.append("")

    # H006 vix helper (add when H006 is included)
    if "006" in include_hyp_ids:
        lines.append(test_funcs["_make_prices_with_vix"])
        lines.append("")
        lines.append("")

    # Platform tests
    for name in PLATFORM_TESTS:
        lines.append(test_funcs[name])
        lines.append("")
        lines.append("")

    # Per-hypothesis tests in order
    for hid, *_ in HYPO_META:
        if hid not in include_hyp_ids:
            continue
        cls_name = [cls for hid2, cls, *_ in HYPO_META if hid2 == hid][0]
        _, _, slug, family, *_ = [m for m in HYPO_META if m[0] == hid][0]
        lines.append(f"# {'─' * 75}")
        lines.append(f"# H{hid} – {cls_name}")
        lines.append(f"# {'─' * 75}")
        lines.append("")
        for tname in H_TESTS.get(hid, []):
            if tname in test_funcs:
                lines.append(test_funcs[tname])
                lines.append("")
                lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def build_run_batch1(batch_sections, include_up_to_hid):
    """Build run_batch1.py including sections up to and including include_up_to_hid."""
    # Determine which strategy classes to import at this stage
    needed_classes = {"MovingAverageCrossover"}  # always needed for h001_baseline()
    for hid, cls, *_ in HYPO_META:
        if "003" <= hid <= include_up_to_hid:
            needed_classes.add(cls)
    imports_str = ",\n    ".join(sorted(needed_classes))

    sec_map = {label: body for label, body in batch_sections}

    # Rewrite header: replace the ALL-imports block with just what exists so far
    header = re.sub(
        r"from quant\.strategies\.base import \([^)]+\)",
        f"from quant.strategies.base import (\n    {imports_str},\n)",
        sec_map["__header__"],
    )

    # Build main_top: include VIX setup only once H006 is in scope
    main_top = sec_map["__main_top__"]
    vix_marker = "\n    # VIX for H006"
    if vix_marker in main_top and include_up_to_hid < "006":
        before_vix = main_top[: main_top.index(vix_marker)]
        folds_idx = main_top.index("\n    folds = make_folds(", main_top.index(vix_marker))
        main_top = before_vix + "\n" + main_top[folds_idx:]

    parts = [header, main_top]

    # Add H-sections in order up to include_up_to_hid
    for hid in [h[0] for h in HYPO_META if h[0] >= "003"]:
        if hid > include_up_to_hid:
            break
        label = f"H{hid}"
        if label in sec_map:
            parts.append(sec_map[label])

    # Add correlation section at H012; otherwise close with __main__ guard
    if include_up_to_hid == "012" and "Correlation" in sec_map:
        parts.append(sec_map["Correlation"])
    else:
        parts.append('\n\n\nif __name__ == "__main__":\n    main()\n')

    return "".join(parts)


def build_summary_md(include_hyp_ids):
    """Build SUMMARY.md with rows for given hypothesis ids."""
    rows = load_summary_rows()
    header = """\
# Strategy Results — Master Leaderboard

Updated after each hypothesis run. All metrics are **walk-forward out-of-sample**
(research period 1993–2022, holdout sealed and untouched). Folds: 5-year train /
2-year test, 12 non-overlapping folds. Costs: 0.5 bps commission + 1.0 bps slippage.

SPY benchmark (same OOS windows): Sharpe 0.51 | CAGR 8.35% | Max DD −55.92%

---

## Leaderboard

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr (returns) | Notes |
|---|------|--------|--------|-----------|--------|------|--------|----------------------|-------|
"""
    table_rows = ""
    for hid in [h[0] for h in HYPO_META]:
        if hid in include_hyp_ids and hid in rows:
            table_rows += rows[hid]

    # Goal trackers (simplified, updated with each hypothesis)
    n_tested = len(include_hyp_ids)
    kept = [
        hid
        for hid in include_hyp_ids
        if any(m[4] == "kept (baseline)" or m[4] == "kept" for m in HYPO_META if m[0] == hid)
    ]
    n_kept = len(kept)
    best_sharpe = max((m[5] for m in HYPO_META if m[0] in include_hyp_ids), default=0.0)

    footer = f"""
---

## Goal trackers

| Target | Current best | Status |
|--------|-------------|--------|
| OOS Sharpe ≥ 1.0 | {best_sharpe:.2f} (H{kept[0] if kept else "—"}) | in progress |
| Max DD < SPY (−55.92%) | −33.72% (H001) | {"**met**" if "001" in include_hyp_ids else "in progress"} |
| Strategies tested ≥ 20 | {n_tested} | in progress |
| Strategies kept (ensemble) | {n_kept} | in progress |

---

_See `results/<id>-<name>/stats.md` for full metrics on each hypothesis._
"""
    return header + table_rows + footer


# ─── Main sequence ──────────────────────────────────────────────────────────


def main():
    print("Loading source files...")
    classes = load_classes()
    test_funcs = load_test_funcs()
    batch_sections = load_batch_sections()

    # Save originals of files we'll overwrite incrementally
    orig = {
        "backtest/__init__": (ROOT / "src/quant/backtest/__init__.py").read_text(encoding="utf-8"),
        "metrics/__init__": (ROOT / "src/quant/metrics/__init__.py").read_text(encoding="utf-8"),
        "validation/__init__": (ROOT / "src/quant/validation/__init__.py").read_text(
            encoding="utf-8"
        ),
        "research_log/001": (ROOT / "research_log/001-ma-50-200-baseline.md").read_text(
            encoding="utf-8"
        ),
        "research_log/002": (ROOT / "research_log/002-mean-reversion-spy.md").read_text(
            encoding="utf-8"
        ),
    }
    for hid, cls_name, slug, *_ in HYPO_META:
        stats_path = ROOT / f"results/{hid}-{slug}/stats.md"
        orig[f"stats/{hid}"] = stats_path.read_text(encoding="utf-8")

    print("Creating orphan branch 'clean-main'...")
    g("checkout", "--orphan", "clean-main")
    g("rm", "-rf", "--cached", ".")
    print()

    # ── Platform commit 1: scaffold ─────────────────────────────────────────
    print("Platform commits:")
    # Write empty versions of __init__.py files whose modules don't exist yet
    for rel in (
        "src/quant/backtest/__init__.py",
        "src/quant/metrics/__init__.py",
        "src/quant/strategies/__init__.py",
        "src/quant/validation/__init__.py",
    ):
        write(rel, "")
    commit(
        "chore: project scaffold — pyproject.toml, uv.lock, src layout",
        "pyproject.toml",
        "uv.lock",
        ".gitignore",
        "README.md",
        "SETUP.md",
        "data/.gitkeep",
        "src/quant/__init__.py",
        "src/quant/backtest/__init__.py",
        "src/quant/data/__init__.py",
        "src/quant/execution/__init__.py",
        "src/quant/execution/paper_trader.py",
        "src/quant/features/__init__.py",
        "src/quant/metrics/__init__.py",
        "src/quant/strategies/__init__.py",
        "src/quant/validation/__init__.py",
    )

    # ── Platform commit 2: data pipeline ────────────────────────────────────
    commit(
        "platform: data pipeline — schema, ingest, loader, fetch scripts",
        "src/quant/data/schema.py",
        "src/quant/data/ingest.py",
        "src/quant/data/loader.py",
        "scripts/fetch_data.py",
        "scripts/generate_synthetic_data.py",
        "scripts/sanity_check.py",
        "tests/__init__.py",
        "tests/test_data.py",
    )

    # ── Platform commit 3: engine + metrics + holdout ───────────────────────
    # Restore proper __init__.py content now that the modules are being committed
    write("src/quant/backtest/__init__.py", orig["backtest/__init__"])
    write("src/quant/metrics/__init__.py", orig["metrics/__init__"])
    write("src/quant/validation/__init__.py", orig["validation/__init__"])
    commit(
        "platform: backtest engine, walk-forward validator, performance metrics",
        "src/quant/backtest/__init__.py",
        "src/quant/backtest/engine.py",
        "src/quant/metrics/__init__.py",
        "src/quant/metrics/performance.py",
        "src/quant/validation/__init__.py",
        "src/quant/validation/walk_forward.py",
        "tests/test_metrics.py",
        "scripts/seal_holdout.py",
        "data/HOLDOUT_HASH.txt",
    )

    # ── Platform commit 4: base strategy interface + platform tests ─────────
    write(
        "src/quant/strategies/base.py",
        classes["__header__"]
        + "\n\n"
        + classes["Strategy"]
        + "\n\n"
        + classes["BuyAndHold"]
        + "\n",
    )
    write("src/quant/strategies/__init__.py", build_strategies_init([]))
    write("tests/test_backtest.py", build_test_backtest(test_funcs, []))
    commit(
        "platform: Strategy base class, BuyAndHold, engine tests, research log template",
        "src/quant/strategies/base.py",
        "src/quant/strategies/__init__.py",
        "tests/test_backtest.py",
        "research_log/_TEMPLATE.md",
        "research_log/README.md",
    )

    print()

    # ── Per-hypothesis commits ───────────────────────────────────────────────
    hyp_ids_done: list[str] = []

    for hid, cls_name, slug, family, status, sharpe, corr in HYPO_META:
        results_dir = f"results/{hid}-{slug}"
        print(f"H{hid} ({cls_name}):")

        # ── Commit 1: open ───────────────────────────────────────────────────
        write(f"{results_dir}/.gitkeep", "")
        commit(
            f"H{hid}: open — {cls_name} hypothesis (issue #{int(hid)})", f"{results_dir}/.gitkeep"
        )

        # ── Commit 2: implement signal ───────────────────────────────────────
        hyp_ids_done.append(hid)
        write("src/quant/strategies/base.py", build_base_py(classes, hyp_ids_done))
        write("src/quant/strategies/__init__.py", build_strategies_init(hyp_ids_done))
        commit(
            f"H{hid}: implement {cls_name}",
            "src/quant/strategies/base.py",
            "src/quant/strategies/__init__.py",
        )

        # ── Commit 3: tests + run script ─────────────────────────────────────
        write("tests/test_backtest.py", build_test_backtest(test_funcs, hyp_ids_done))
        add_test_files = ["tests/test_backtest.py"]
        if hid == "002":
            add_test_files.append("scripts/run_hypothesis_002.py")
        elif hid >= "003":
            write("scripts/run_batch1.py", build_run_batch1(batch_sections, hid))
            add_test_files.append("scripts/run_batch1.py")
        commit(
            f"H{hid}: no-lookahead + structural tests; add to run script",
            *add_test_files,
        )

        # ── Commit 4: results + close ─────────────────────────────────────────
        write(f"{results_dir}/stats.md", orig[f"stats/{hid}"])
        write("results/SUMMARY.md", build_summary_md(hyp_ids_done))
        add_result_files = [f"{results_dir}/stats.md", "results/SUMMARY.md"]
        if hid == "001":
            write("research_log/001-ma-50-200-baseline.md", orig["research_log/001"])
            add_result_files.append("research_log/001-ma-50-200-baseline.md")
        elif hid == "002":
            write("research_log/002-mean-reversion-spy.md", orig["research_log/002"])
            add_result_files.append("research_log/002-mean-reversion-spy.md")
        verdict = status.split()[0].upper()
        commit(
            f"H{hid}: results — OOS Sharpe {sharpe:.2f}, {verdict} (closes #{int(hid)})",
            *add_result_files,
            rm_files=(f"{results_dir}/.gitkeep",),
        )

        print()

    # ── Reorder base.py to hypothesis order (MovingAverageCrossover was at end) ──
    print("Finalising file order...")
    final_base = build_base_py(classes, [h[0] for h in HYPO_META])
    write("src/quant/strategies/base.py", final_base)
    write("src/quant/strategies/__init__.py", build_strategies_init([h[0] for h in HYPO_META]))
    write("tests/test_backtest.py", build_test_backtest(test_funcs, [h[0] for h in HYPO_META]))
    g(
        "add",
        "--",
        "src/quant/strategies/base.py",
        "src/quant/strategies/__init__.py",
        "tests/test_backtest.py",
    )
    st = g("status", "--porcelain")
    if st:
        g("commit", "-m", "chore: reorder strategy file to hypothesis order")
        print(
            f"  {g('rev-parse', '--short', 'HEAD')}  chore: reorder strategy file to hypothesis order"
        )

    print()
    total = g("rev-list", "--count", "HEAD")
    print(f"Done! Branch 'clean-main' has {total} commits.")
    print()
    print("Next steps:")
    print("  git checkout clean-main")
    print("  git branch -D main")
    print("  git branch -m clean-main main")
    print("  git push --force origin main")


if __name__ == "__main__":
    main()
