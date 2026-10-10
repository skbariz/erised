"""Strategy visualization dashboard.

Usage:
    uv run python scripts/visualize.py                  # metrics only (default)
    uv run python scripts/visualize.py --mode metrics   # fast, reads JSONs only
    uv run python scripts/visualize.py --mode equity    # re-runs top 10 strategies (~5-10 min)
    uv run python scripts/visualize.py --mode all       # both

Outputs:
    results/visualizations/metrics_dashboard.png
    results/visualizations/equity_curves.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — saves to file without requiring a display
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
VIZ_DIR = RESULTS_DIR / "visualizations"

SPY_SHARPE = 0.51
SPY_CAGR = 0.0835
SPY_MAXDD = -0.5592

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _parse_summary_md() -> list[dict]:
    """Parse the master SUMMARY.md leaderboard table into a list of dicts (all 50 strategies)."""
    path = RESULTS_DIR / "SUMMARY.md"
    rows = []
    with path.open(encoding="utf-8") as fh:
        in_table = False
        for line in fh:
            line = line.strip()
            if line.startswith("| #") and "Name" in line:
                in_table = True
                continue
            if in_table and line.startswith("|---"):
                continue
            if in_table and line.startswith("|"):
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) < 8:
                    continue
                hyp_id = parts[0].zfill(3)
                try:
                    sharpe_raw = parts[4].replace("**", "").replace("−", "-").replace("−", "-").strip()
                    sharpe = float(sharpe_raw)
                except ValueError:
                    continue
                status = parts[3].lower().replace("*", "").replace("★", "").strip()
                kept = "kept" in status
                # Max DD: strip % sign and leading −
                dd_raw = parts[7].replace("**", "").replace("−", "-").replace("−", "-").replace("%", "").strip()
                max_dd = float(dd_raw) / 100 if dd_raw and dd_raw not in ("—", "—") else None
                # CAGR
                cagr_raw = parts[6].replace("**", "").replace("−", "-").replace("−", "-").replace("%", "").strip()
                cagr = float(cagr_raw) / 100 if cagr_raw and cagr_raw not in ("—", "—") else None
                # H001 corr
                corr_raw = parts[8].replace("+", "").replace("**", "").replace("−", "-").replace("–", "-").strip()
                h001_corr = float(corr_raw) if corr_raw and corr_raw not in ("—", "–", "—") else None
                rows.append(
                    {
                        "hypothesis": f"H{hyp_id}",
                        "name": parts[1].strip().replace("**", ""),
                        "family": parts[2].strip(),
                        "status": status,
                        "oos_sharpe": sharpe,
                        "kept": kept,
                        "max_drawdown": max_dd,
                        "cagr": cagr,
                        "h001_corr": h001_corr,
                    }
                )
            elif in_table and not line.startswith("|"):
                in_table = False
    rows.sort(key=lambda r: int(r["hypothesis"].lstrip("H")))
    return rows


def load_summaries() -> list[dict]:
    """Return all 50 strategies from SUMMARY.md, enriched with fold data from summary.json where available."""
    all_strats = _parse_summary_md()
    # Build lookup of rich JSON data for kept strategies
    json_data: dict[str, dict] = {}
    for f in RESULTS_DIR.glob("*/summary.json"):
        with f.open() as fh:
            d = json.load(fh)
        json_data[d["hypothesis"]] = d
    # Merge: JSON fields take precedence for kept strategies
    merged = []
    for row in all_strats:
        hyp = row["hypothesis"]
        if hyp in json_data:
            merged.append(json_data[hyp])
        else:
            merged.append(row)
    merged.sort(key=lambda s: int(s["hypothesis"].lstrip("H")))
    return merged


# ---------------------------------------------------------------------------
# Metrics dashboard (fast — JSON only)
# ---------------------------------------------------------------------------


def plot_metrics(summaries: list[dict]) -> None:
    kept = [s for s in summaries if s.get("kept")]
    top10_kept = sorted(kept, key=lambda s: s["oos_sharpe"], reverse=True)[:10]
    top10_ids = {s["hypothesis"] for s in top10_kept}

    fig = plt.figure(figsize=(22, 26))
    fig.suptitle(
        "Strategy Dashboard: H001–H050 vs SPY\n"
        "Walk-forward OOS 1993–2022 | 5y train / 2y test | costs 1.5 bps/turn",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.38, top=0.96, bottom=0.04)

    # ---- Panel 1: All-50 Sharpe bar (full width) ----
    ax1 = fig.add_subplot(gs[0, :])
    bar_colors = ["#27ae60" if s.get("kept") else "#e74c3c" for s in summaries]
    bars = ax1.bar(
        range(len(summaries)),
        [s["oos_sharpe"] for s in summaries],
        color=bar_colors,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.6,
    )
    for i, s in enumerate(summaries):
        if s["hypothesis"] in top10_ids:
            bars[i].set_edgecolor("#1a1a1a")
            bars[i].set_linewidth(2.5)
            ax1.annotate(
                f"{s['oos_sharpe']:.2f}",
                xy=(i, s["oos_sharpe"]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                fontweight="bold",
                color="#1a1a1a",
            )

    ax1.axhline(
        SPY_SHARPE,
        color="navy",
        linestyle="--",
        linewidth=1.8,
        label=f"SPY benchmark ({SPY_SHARPE})",
    )
    ax1.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax1.set_xticks(range(len(summaries)))
    ax1.set_xticklabels([s["hypothesis"] for s in summaries], rotation=90, fontsize=7.5)
    ax1.set_ylabel("OOS Sharpe Ratio")
    ax1.set_title(
        "OOS Sharpe — All 50 Strategies  [green=kept | red=killed | bold outline=top 10 kept]",
        fontsize=11,
    )
    ax1.legend(loc="upper left", fontsize=9)

    # Legend patches
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#27ae60", label="Kept"),
        Patch(facecolor="#e74c3c", label="Killed"),
    ]
    ax1.legend(
        handles=legend_elements + ax1.get_legend_handles_labels()[0][-1:],
        loc="upper left",
        fontsize=8,
    )

    # ---- Panel 2: CI forest plot (kept with CI data, sorted by Sharpe) ----
    ax2 = fig.add_subplot(gs[1, 0])
    kept_ci = sorted([s for s in kept if "ci_low" in s], key=lambda s: s["oos_sharpe"])
    y = list(range(len(kept_ci)))
    for i, s in enumerate(kept_ci):
        ax2.plot([s["ci_low"], s["ci_high"]], [i, i], color="#3498db", linewidth=2, alpha=0.7)
        color = "#27ae60" if s["hypothesis"] in top10_ids else "#2c3e50"
        ax2.scatter(s["oos_sharpe"], i, color=color, zorder=5, s=45)

    ax2.axvline(SPY_SHARPE, color="red", linestyle="--", linewidth=1.5, label=f"SPY ({SPY_SHARPE})")
    ax2.axvline(0, color="gray", linewidth=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels([s["hypothesis"] for s in kept_ci], fontsize=8)
    ax2.set_xlabel("Sharpe Ratio")
    ax2.set_title("95% Confidence Intervals — Kept Strategies", fontsize=10)
    ax2.legend(fontsize=8)

    # ---- Panel 3: Max DD bar (kept, sorted best→worst) ----
    ax3 = fig.add_subplot(gs[1, 1])
    kept_dd = sorted(kept, key=lambda s: s["max_drawdown"], reverse=True)
    dd_vals = [s["max_drawdown"] * 100 for s in kept_dd]
    ax3.barh(range(len(kept_dd)), dd_vals, color="#3498db", alpha=0.8)
    ax3.axvline(
        SPY_MAXDD * 100,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"SPY ({SPY_MAXDD * 100:.1f}%)",
    )
    ax3.set_yticks(range(len(kept_dd)))
    ax3.set_yticklabels([s["hypothesis"] for s in kept_dd], fontsize=8)
    ax3.set_xlabel("Max Drawdown (%)")
    ax3.set_title("Max Drawdown — Kept Strategies (left = shallower)", fontsize=10)
    ax3.legend(fontsize=8)

    # ---- Panel 4: Sharpe vs Max DD scatter ----
    ax4 = fig.add_subplot(gs[2, 0])
    sharpes_k = [s["oos_sharpe"] for s in kept]
    maxdds_k = [s["max_drawdown"] * 100 for s in kept]
    cagrs_k = [s["cagr"] * 100 for s in kept]
    sc = ax4.scatter(
        maxdds_k,
        sharpes_k,
        c=cagrs_k,
        cmap="RdYlGn",
        s=80,
        alpha=0.85,
        edgecolors="gray",
        linewidth=0.5,
        vmin=0,
        vmax=max(cagrs_k) + 1,
    )
    plt.colorbar(sc, ax=ax4, label="CAGR (%)", shrink=0.85)
    ax4.scatter(
        [SPY_MAXDD * 100], [SPY_SHARPE], marker="X", color="red", s=160, zorder=6, label="SPY"
    )
    ax4.axhline(SPY_SHARPE, color="red", linestyle="--", linewidth=1, alpha=0.45)
    ax4.axvline(SPY_MAXDD * 100, color="red", linestyle="--", linewidth=1, alpha=0.45)
    for s in kept:
        ax4.annotate(
            s["hypothesis"],
            xy=(s["max_drawdown"] * 100, s["oos_sharpe"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    ax4.set_xlabel("Max Drawdown (%)")
    ax4.set_ylabel("OOS Sharpe")
    ax4.set_title("Sharpe vs Max DD (color = CAGR)", fontsize=10)
    ax4.legend(fontsize=8)

    # ---- Panel 5: H001 correlation ----
    ax5 = fig.add_subplot(gs[2, 1])
    kept_corr = sorted([s for s in kept if s.get("h001_corr") is not None], key=lambda s: s["h001_corr"])
    corr_vals = [s["h001_corr"] for s in kept_corr]
    norm = plt.Normalize(min(corr_vals), max(corr_vals))
    cmap = plt.cm.RdYlGn_r
    bar_colors5 = [cmap(norm(v)) for v in corr_vals]
    ax5.barh(range(len(kept_corr)), corr_vals, color=bar_colors5, alpha=0.9)
    ax5.axvline(0.30, color="purple", linestyle="--", linewidth=1.5, label="0.30 threshold")
    ax5.axvline(0.50, color="navy", linestyle=":", linewidth=1.2, alpha=0.7, label="0.50")
    ax5.set_yticks(range(len(kept_corr)))
    ax5.set_yticklabels([s["hypothesis"] for s in kept_corr], fontsize=8)
    ax5.set_xlabel("Pearson correlation with H001 baseline")
    ax5.set_title("H001 Correlation — Kept Strategies (lower = more diversifying)", fontsize=10)
    ax5.legend(fontsize=8)

    # ---- Panel 6: Fold Sharpe heatmap (top 10 kept with fold data × 12 folds) ----
    ax6 = fig.add_subplot(gs[3, :])
    top10_with_folds = [s for s in top10_kept if "fold_sharpes" in s]
    fold_data = np.array([s["fold_sharpes"] for s in top10_with_folds])
    fold_labels = [f"Fold {i}" for i in range(fold_data.shape[1])]
    strat_labels = [f"{s['hypothesis']} {s['name'][:18]}" for s in top10_with_folds]

    vmax = max(abs(fold_data.max()), abs(fold_data.min()), 2.5)
    im = ax6.imshow(fold_data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax6, label="Fold Sharpe", shrink=0.7)

    ax6.set_xticks(range(fold_data.shape[1]))
    ax6.set_xticklabels(fold_labels, fontsize=8, rotation=45)
    ax6.set_yticks(range(len(top10_with_folds)))
    ax6.set_yticklabels(strat_labels, fontsize=8)
    ax6.set_title(
        "Per-Fold Sharpe — Top 10 Kept Strategies (green=strong, red=weak, consistent rows = robust)",
        fontsize=10,
    )

    for r in range(fold_data.shape[0]):
        for c in range(fold_data.shape[1]):
            val = fold_data[r, c]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax6.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out = VIZ_DIR / "metrics_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Equity curves (slow — re-runs top 10 individual strategies)
# ---------------------------------------------------------------------------


def plot_equity_curves() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    from erised.data.loader import load_research
    from erised.strategies.base import (
        AbsoluteMomentumStrategy,
        DefensiveEquityHedge,
        HalloweenMomentumConfirm,
        MovingAverageCrossover,
        RiskParity,
        SeasonalityTrendCombo,
        SellInMayBondConfirm,
        SPYGLDRotation,
        TimeSeriesMomentum,
        VolScaledTrend,
    )
    from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

    print("Loading price data...")
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})

    spy_tlt = spy.join(tlt, how="left")
    spy_tlt["tlt_close"] = spy_tlt["tlt_close"].ffill()

    spy_gld = spy.join(gld, how="left")
    spy_gld["gld_close"] = spy_gld["gld_close"].ffill()

    folds_spy = make_folds(spy, train_years=5, test_years=2)
    folds_multi = make_folds(spy_tlt, train_years=5, test_years=2)

    strategies = [
        (
            "H021",
            "RiskParity",
            lambda: walk_forward_multi(spy_tlt, lambda _: RiskParity(), folds=folds_multi),
        ),
        (
            "H046",
            "HalloweenMomentumConfirm",
            lambda: walk_forward(spy, lambda _: HalloweenMomentumConfirm(), folds=folds_spy),
        ),
        (
            "H022",
            "DefensiveEquityHedge",
            lambda: walk_forward_multi(
                spy_tlt, lambda _: DefensiveEquityHedge(), folds=folds_multi
            ),
        ),
        (
            "H019",
            "SPYGLDRotation",
            lambda: walk_forward_multi(
                spy_gld,
                lambda _: SPYGLDRotation(),
                folds=make_folds(spy_gld, train_years=5, test_years=2),
            ),
        ),
        (
            "H007",
            "TimeSeriesMomentum",
            lambda: walk_forward(spy, lambda _: TimeSeriesMomentum(), folds=folds_spy),
        ),
        (
            "H008",
            "AbsoluteMomentum",
            lambda: walk_forward(spy, lambda _: AbsoluteMomentumStrategy(), folds=folds_spy),
        ),
        (
            "H001",
            "MA(50,200)",
            lambda: walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy),
        ),
        (
            "H027",
            "SeasonalityTrendCombo",
            lambda: walk_forward(spy, lambda _: SeasonalityTrendCombo(), folds=folds_spy),
        ),
        (
            "H016",
            "VolScaledTrend",
            lambda: walk_forward(spy, lambda _: VolScaledTrend(), folds=folds_spy),
        ),
        (
            "H043",
            "SellInMayBondConfirm",
            lambda: walk_forward(spy_tlt, lambda _: SellInMayBondConfirm(), folds=folds_multi),
        ),
    ]

    colors = [
        "#e74c3c",
        "#3498db",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
        "#34495e",
        "#e91e63",
        "#00bcd4",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(18, 14))
    fig.suptitle(
        "Cumulative Equity Curves — Top 10 Kept Strategies vs SPY\n"
        "Walk-forward OOS only (1993–2022) | Starting value = 1.0 | Costs included",
        fontsize=14,
        fontweight="bold",
    )

    ax_main = axes[0]
    ax_dd = axes[1]

    results_data = []
    for _idx, (hyp_id, label, runner) in enumerate(strategies):
        print(f"  Running {hyp_id}: {label}...")
        wf = runner()
        rets = wf.combined_returns.sort_index()
        equity = (1 + rets).cumprod()
        results_data.append((hyp_id, label, equity, rets))

    # SPY buy-and-hold aligned to the same OOS dates
    all_dates = results_data[0][2].index
    spy_oos_rets = spy["adj_close"].pct_change().reindex(all_dates).fillna(0)
    spy_equity = (1 + spy_oos_rets).cumprod()

    # --- Subplot 1: Equity curves ---
    ax_main.plot(
        spy_equity.index,
        spy_equity.values,
        color="black",
        linewidth=2.5,
        linestyle="-",
        label="SPY buy-and-hold",
        zorder=10,
        alpha=0.85,
    )

    for idx, (hyp_id, label, equity, _) in enumerate(results_data):
        summary_path = next(RESULTS_DIR.glob(f"*{hyp_id.lower()[1:].zfill(3)}*/summary.json"), None)
        sharpe_str = ""
        if summary_path:
            with summary_path.open() as fh:
                s = json.load(fh)
            sharpe_str = f" (Sh={s['oos_sharpe']:.2f}, DD={s['max_drawdown'] * 100:.1f}%)"

        ax_main.plot(
            equity.index,
            equity.values,
            color=colors[idx],
            linewidth=1.6,
            alpha=0.85,
            label=f"{hyp_id} {label}{sharpe_str}",
        )

    ax_main.set_ylabel("Portfolio Value (start = 1.0)")
    ax_main.set_title("Cumulative Returns", fontsize=11)
    ax_main.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax_main.set_yscale("log")
    ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # Shade recessions (approximate — 2000-2002, 2008-2009, 2020)
    recessions = [
        ("2000-03-01", "2002-10-31"),
        ("2007-12-01", "2009-06-30"),
        ("2020-02-01", "2020-04-30"),
    ]
    for start, end in recessions:
        ax_main.axvspan(
            np.datetime64(start), np.datetime64(end), color="lightcoral", alpha=0.15, zorder=0
        )

    # --- Subplot 2: Drawdown curves ---
    def compute_drawdown(equity):
        roll_max = equity.cummax()
        return (equity / roll_max - 1) * 100

    ax_dd.plot(
        spy_equity.index,
        compute_drawdown(spy_equity).values,
        color="black",
        linewidth=2.5,
        label="SPY",
        zorder=10,
        alpha=0.85,
    )

    for idx, (hyp_id, _label, equity, _) in enumerate(results_data):
        dd = compute_drawdown(equity)
        ax_dd.plot(dd.index, dd.values, color=colors[idx], linewidth=1.4, alpha=0.75, label=hyp_id)

    for start, end in recessions:
        ax_dd.axvspan(
            np.datetime64(start), np.datetime64(end), color="lightcoral", alpha=0.15, zorder=0
        )

    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Date")
    ax_dd.set_title("Rolling Drawdown", fontsize=11)
    ax_dd.legend(loc="lower left", fontsize=8, ncol=5)

    plt.tight_layout()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out = VIZ_DIR / "equity_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy visualization dashboard")
    parser.add_argument(
        "--mode",
        choices=["metrics", "equity", "all"],
        default="metrics",
        help="metrics = fast JSON-based charts | equity = re-run top 10 strategies | all = both",
    )
    args = parser.parse_args()

    summaries = load_summaries()
    print(
        f"Loaded {len(summaries)} strategy summaries ({sum(1 for s in summaries if s.get('kept'))} kept)"
    )

    if args.mode in ("metrics", "all"):
        print("\n--- Metrics Dashboard ---")
        plot_metrics(summaries)

    if args.mode in ("equity", "all"):
        print("\n--- Equity Curves (re-running top 10 strategies) ---")
        plot_equity_curves()


if __name__ == "__main__":
    main()
