"""Figures for the revised manuscript, drawn only from stored summaries.

fig14_forest.png  pre-registered primary paired contrasts (mean ΔF1, 95% CI over 10 seeds); filled marker = Holm p < 0.05
fig15_timing.png  Step 8: B and async F1 by acquisition timing of the historical image (mean, 95% CI over 10 seeds),
                  SAR-only U-Net on the same 9 cities as a dashed reference
Palette: validated categorical slots 1-2 (dataviz reference palette, light mode, all checks pass).

Usage (repository root):  .venv/Scripts/python.exe experiments/robust/revision/make_revision_figures.py
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "experiments" / "robust"
OUT = ROOT / "figures" / "paper"
INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#d9d8d4", "#2a78d6", "#eb6834"
load = lambda p: json.loads((R / p).read_text(encoding="utf-8"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def forest() -> None:
    s6, s7, s4b, s9, s10, s11 = (load(p) for p in ("phase6/summary.json", "phase7/summary.json", "stage4b/stage4b_summary.json",
                                                   "step9/summary.json", "step10/summary.json", "realcloud/summary.json"))
    single = lambda r: {**r, "holm_p": r["t_p"]}  # one-test families
    groups = [
        ("Full input (Step 6)", [("DS-UNet A − SAR-only U-Net", s6["primary_rq4"]["A - SAR-only"]),
                                 ("DS-UNet A − optical-only U-Net", s6["primary_rq4"]["A - optical-only"]),
                                 ("DS-UNet A − input-fusion U-Net", s6["primary_rq4"]["A - input fusion"])]),
        ("No post-change optical (Step 6)", [("B@no_o2 − SAR-only", s6["primary_rq1"]["B@no_o2 - SAR-only"]),
                                             ("async@no_o2 − SAR-only", s6["primary_rq1"]["async@no_o2 - SAR-only"])]),
        ("Historical image within a model", [("B (late fusion): present − absent (Stage 4B)", single(s4b["primary"])),
                                             ("E (early fusion): present − absent (Step 9)", s9["primary"]["E: image present - absent"]),
                                             ("E effect − B effect (Step 9)",
                                              s9["primary"]["interaction: (E present - absent) - (B present - absent)"])]),
        ("SAR-only detector (Step 10, seeds 11–20)", [("E without optical − SAR-only U-Net", s10["primary"]["E@no_opt - SAR-only"])]),
        ("Polarisation, SAR-only U-Net (Step 7)", [("VV+VH − VV", s7["primary_rq3"]["VV+VH - VV"]),
                                                   ("VH − VV", s7["primary_rq3"]["VH - VV"])]),
        ("Real cloudy scenes, 7 cities (Step 11)", [("B: rb95 mask − no mask", s11["primary"]["B real_rb95 - B real"]),
                                                   ("B with rb95 mask − SAR-only", s11["primary"]["B real_rb95 - SAR-only"]),
                                                   ("B − A, no mask", s11["primary"]["B real - A real"])]),
    ]
    rows, y, ticks = [], 0, []
    for title, items in groups:
        ticks.append((y, title, True))
        y -= 1
        for name, r in items:
            rows.append((y, r))
            ticks.append((y, name, False))
            y -= 1
        y -= 0.4
    fig, ax = plt.subplots(figsize=(7.0, 0.24 * (-y) + 0.6))
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    for yy, r in rows:
        sig = r["holm_p"] < 0.05
        ax.plot(r["ci95"], [yy, yy], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(r["mean_diff"], yy, "o", ms=6, mfc=BLUE if sig else "white", mec=BLUE, mew=1.5, zorder=3)
        ax.text(1.02, yy, f"{r['mean_diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]", va="center", ha="left",
                fontsize=7, color=MUTED, transform=ax.get_yaxis_transform())
    ax.set_yticks([t[0] for t in ticks])
    ax.set_yticklabels([t[1] for t in ticks])
    for lab, t in zip(ax.get_yticklabels(), ticks):
        lab.set_fontweight("bold" if t[2] else "normal")
        lab.set_color(INK if t[2] else MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.32, 0.36)
    ax.set_ylim(y + 0.2, 0.8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Mean paired difference in test F1 (95% CI over 10 seeds)")
    ax.plot([], [], "o", mfc=BLUE, mec=BLUE, label="Holm p < 0.05")
    ax.plot([], [], "o", mfc="white", mec=BLUE, mew=1.5, label="not detectable")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7)
    fig.savefig(OUT / "fig14_forest.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ci(values) -> tuple[float, float]:
    v = np.asarray(values)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return float(v.mean()), float(h)


def timing() -> None:
    s8 = load("phase8/summary.json")
    months, tags = [1, 3, 6, 12, 18], ["gap01", "gap03", "gap06", "gap12", "gap18"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sar, _ = ci(s8["arms"]["p6_unet_sar"]["oscd"]["f1"]["values"])
    ax.axhline(sar, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(18.6, sar, "SAR-only U-Net", va="center", fontsize=7, color=MUTED)
    for arm, color, name, dx in (("p2_dropout", BLUE, "B (dropout)", -0.15), ("p6_async", ORANGE, "async", 0.15)):
        m = [ci(s8["arms"][arm][t]["f1"]["values"]) for t in tags]
        x = [mo + dx for mo in months]
        ax.errorbar(x, [a for a, _ in m], yerr=[h for _, h in m], color=color, lw=2, marker="o", ms=5, capsize=0,
                    elinewidth=1.2, mec="white", mew=1, zorder=3)
        ax.text(18.6, m[-1][0] + (0.006 if arm == "p2_dropout" else -0.006), name, va="center", fontsize=7, color=INK)
    ax.set_xticks(months)
    ax.set_xlim(0, 18.5)
    ax.set_xlabel("Historical image acquired before OSCD date 2 (months)")
    ax.set_ylabel("Test F1, 9 cities")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(OUT / "fig15_timing.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    forest()
    timing()
    print("wrote fig14_forest.png and fig15_timing.png")
