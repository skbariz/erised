"""Generate chart of all 83 kept strategies (H001-H200)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ── collect & dedup ──────────────────────────────────────────────────────────
results: dict[int, dict] = {}
for d in sorted((Path(__file__).resolve().parents[1] / "results").iterdir()):
    f = d / "summary.json"
    if not f.exists():
        continue
    try:
        data = json.loads(f.read_text())
        hid = data.get("hypothesis", data.get("hypothesis_id", ""))
        if not hid:
            continue
        hid_num = int(str(hid).replace("H", ""))
        if hid_num > 200:
            continue
        if data.get("decision", "").lower() != "keep":
            continue
        sharpe = data.get("oos_sharpe", data.get("sharpe", 0))
        if hid_num not in results or sharpe > results[hid_num]["sharpe"]:
            results[hid_num] = {
                "id": f"H{hid_num:03d}",
                "num": hid_num,
                "name": data.get("name", ""),
                "family": data.get("family", ""),
                "sharpe": sharpe,
                "maxdd": abs(data.get("max_drawdown", 0)),
                "cagr": data.get("cagr", 0),
                "h001": data.get("h001_corr", 0),
            }
    except Exception:
        pass

rows = sorted(results.values(), key=lambda x: x["num"])


def get_color(r: dict) -> str:
    n = r["num"]
    fam = r["family"].lower()
    if n == 200:
        return "#FF1744"
    if "ensemble" in fam:
        return "#AB47BC"
    if n <= 100:
        if "trend" in fam or ("momentum" in fam and "multi" not in fam):
            return "#1565C0"
        if "calendar" in fam or "seasonal" in fam:
            return "#2E7D32"
        if "vol" in fam or "vix" in fam:
            return "#F57F17"
        return "#00838F"
    # H150+
    if "momentum" in fam:
        return "#42A5F5"
    if "reversal" in fam or "reversion" in fam or "stat arb" in fam:
        return "#26A69A"
    if "quality" in fam or "value" in fam or "earnings" in fam:
        return "#FFA726"
    if "anomaly" in fam or "defensive" in fam or "beta" in fam:
        return "#EC407A"
    return "#78909C"


SPY_SHARPE = 0.51
SPY_MAXDD = 55.92

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 24), facecolor="#0D1117")
gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[2.2, 1.6, 1.0], hspace=0.38)

# ── Panel 1: Sharpe vs MaxDD scatter ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor("#161B22")

ax1.scatter([SPY_SHARPE], [SPY_MAXDD], s=260, marker="*",
            color="#FFD600", zorder=9)
ax1.annotate("SPY", (SPY_SHARPE, SPY_MAXDD),
             xytext=(5, 4), textcoords="offset points",
             fontsize=9, color="#FFD600", fontweight="bold")

for r in rows:
    sz = 200 if r["num"] == 200 else 65
    ec = "white" if r["num"] == 200 else "none"
    ax1.scatter(r["sharpe"], r["maxdd"] * 100, s=sz, color=get_color(r),
                alpha=0.88, zorder=5, edgecolors=ec, linewidths=1.4)

labels = {
    21:  ("H021\nRisk Parity",           6,  5),
    86:  ("H086\nLowVol Target",          6,  5),
    89:  ("H089\nSeasonal+Parity",        6,  5),
    95:  ("H095\nMDR H1-100",             6, -14),
    78:  ("H078\nSeasonal RP",           -62,  5),
    192: ("H192\nPCA-OU",                  6,  5),
    179: ("H179\nLowBeta Quality",         6, -13),
    194: ("H194\nL/S Reversal",           6,  5),
    200: ("H200\nCross-Program MDR ★",    8, -16),
}
for r in rows:
    if r["num"] in labels:
        txt, xo, yo = labels[r["num"]]
        ax1.annotate(txt, (r["sharpe"], r["maxdd"] * 100),
                     xytext=(xo, yo), textcoords="offset points",
                     fontsize=7.8, color="white", alpha=0.93,
                     fontweight="bold" if r["num"] == 200 else "normal")

ax1.axvline(0.51, color="#FFD600", lw=1.0, ls="--", alpha=0.45)
ax1.set_xlabel("OOS Sharpe Ratio", color="#C9D1D9", fontsize=12)
ax1.set_ylabel("Max Drawdown (%)", color="#C9D1D9", fontsize=12)
ax1.set_title("83 Kept Strategies — Sharpe vs Max Drawdown",
              color="white", fontsize=14, fontweight="bold", pad=10)
ax1.tick_params(colors="#8B949E")
for spine in ax1.spines.values():
    spine.set_color("#30363D")

legend_items = [
    mpatches.Patch(color="#1565C0", label="H1-100  Trend / Momentum"),
    mpatches.Patch(color="#2E7D32", label="H1-100  Calendar / Seasonal"),
    mpatches.Patch(color="#F57F17", label="H1-100  Vol / VIX"),
    mpatches.Patch(color="#00838F", label="H1-100  Risk Parity / Rotation"),
    mpatches.Patch(color="#AB47BC", label="Ensemble strategies"),
    mpatches.Patch(color="#42A5F5", label="H150+  Cross-sectional Momentum"),
    mpatches.Patch(color="#26A69A", label="H150+  Reversal / Stat Arb"),
    mpatches.Patch(color="#FFA726", label="H150+  Quality / Earnings"),
    mpatches.Patch(color="#EC407A", label="H150+  Defensive / Low-Beta"),
    mpatches.Patch(color="#FF1744", label="H200  Terminal Champion"),
]
ax1.legend(handles=legend_items, loc="upper right", fontsize=8.5,
           framealpha=0.25, facecolor="#161B22", edgecolor="#30363D",
           labelcolor="#C9D1D9")

# ── Panel 2: horizontal bar chart ranked by sharpe ───────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#161B22")

sorted_rows = sorted(rows, key=lambda x: x["sharpe"])
y = np.arange(len(sorted_rows))
bar_colors = [get_color(r) for r in sorted_rows]
bars = ax2.barh(y, [r["sharpe"] for r in sorted_rows], height=0.72,
                color=bar_colors, alpha=0.88)

ax2.axvline(0.51, color="#FFD600", lw=1.0, ls="--", alpha=0.55, label="SPY (0.51)")
ax2.axvline(1.0,  color="#4AEDC4", lw=0.8, ls=":",  alpha=0.45, label="Sharpe = 1.0")

ax2.set_yticks(y)
ax2.set_yticklabels([r["id"] for r in sorted_rows], fontsize=5.0, color="#8B949E")
ax2.set_xlabel("OOS Sharpe Ratio", color="#C9D1D9", fontsize=11)
ax2.set_title("All 83 Kept Strategies — Ranked by Sharpe Ratio",
              color="white", fontsize=13, fontweight="bold", pad=8)
ax2.tick_params(colors="#8B949E")
for spine in ax2.spines.values():
    spine.set_color("#30363D")
ax2.legend(fontsize=8.5, framealpha=0.25, facecolor="#161B22",
           edgecolor="#30363D", labelcolor="#C9D1D9")

h200_idx = next(i for i, r in enumerate(sorted_rows) if r["num"] == 200)
bars[h200_idx].set_edgecolor("white")
bars[h200_idx].set_linewidth(1.5)

# ── Panel 3: summary tables ───────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor("#161B22")
ax3.axis("off")

top10 = sorted(rows, key=lambda x: x["sharpe"], reverse=True)[:10]
low_dd = sorted((r for r in rows if r["maxdd"] < 0.25), key=lambda x: x["maxdd"])[:10]

col_w = [0.09, 0.40, 0.14, 0.13, 0.11]
headers = ["ID", "Name", "Sharpe", "Max DD", "CAGR"]


def draw_table(ax: plt.Axes, title: str, data_rows: list, x0: float, y0: float, width: float = 0.46) -> None:
    ax.text(x0 + width / 2, y0 + 0.06, title, transform=ax.transAxes,
            color="#58A6FF", fontsize=9.5, fontweight="bold", ha="center", va="bottom")
    y = y0
    for j, h in enumerate(headers):
        xc = x0 + sum(col_w[:j]) * width
        ax.text(xc, y, h, transform=ax.transAxes,
                color="#C9D1D9", fontsize=7.5, fontweight="bold", va="top")
    y -= 0.075
    for r in data_rows:
        vals = [r["id"], r["name"][:30], f"{r['sharpe']:.3f}",
                f"-{r['maxdd']*100:.1f}%", f"{r['cagr']:.1%}"]
        c = "#FF1744" if r["num"] == 200 else ("#4AEDC4" if r["sharpe"] >= 1.0 else "#C9D1D9")
        for j, v in enumerate(vals):
            xc = x0 + sum(col_w[:j]) * width
            ax.text(xc, y, v, transform=ax.transAxes,
                    color=c, fontsize=7.0, va="top", clip_on=True)
        y -= 0.072


draw_table(ax3, "Top 10 by Sharpe Ratio", top10, 0.01, 0.98, 0.47)
draw_table(ax3, "Lowest Max Drawdown (< 25%)", low_dd, 0.52, 0.98, 0.47)

fig.suptitle(
    "Erised Quant  ·  H001–H200  ·  83 Kept Strategies out of 200 Tested",
    color="white", fontsize=16, fontweight="bold", y=0.998,
)

out = (Path(__file__).resolve().parents[1] / "results/visualizations/all_kept_strategies.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0D1117")
print(f"Saved: {out}")
