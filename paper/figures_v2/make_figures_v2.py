"""Figures for paper/final_submission_v2.md. Writes only into paper/figures_v2/; original figures are not touched.

fig01_workflow.png        schematic of the study workflow (no data)
fig02_architectures.png   schematic of the networks and input conditions, drawn from the code (no data)
fig03_forest.png          registered primary contrasts; row labels carry city set and threshold rule
fig04_timing.png          Step 8 timing figure, regenerated unchanged by experiments/robust/revision/make_revision_figures.py
fig05_error_maps.png      copy of figures/paper/fig13_error_maps.png (unchanged)
fig06_cloud_conditions.png  Step 11: A and B test F1 under each post-change input, 7 partly cloudy cities
fig07_cloud_masks.png     Step 11: real scenes and s2cloudless masks at the frozen bal and rb95 operating points

All numbers come from stored summaries (phase6/7, stage4b, step9/10, realcloud summary.json); masks come from the stored
detector probabilities and Step 5's frozen operating points. Palette: dataviz reference categorical slots 1-2 (validated,
light mode). Usage (repository root):  .venv/Scripts/python.exe paper/figures_v2/make_figures_v2.py
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
R = ROOT / "experiments" / "robust"
sys.path.insert(0, str(ROOT))
from src.phase5.composite import detector_mask  # noqa: E402

spec = importlib.util.spec_from_file_location("rev", R / "revision" / "make_revision_figures.py")
rev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rev)
rev.OUT = OUT  # reuse the stored-summary loaders and style; write here instead of figures/paper
INK, MUTED, GRID, BLUE, ORANGE, AQUA = rev.INK, rev.MUTED, rev.GRID, rev.BLUE, rev.ORANGE, "#1baf7a"
FILL = {"sar": "#cde2fb", "opt": "#d6f0e3", "fuse": "#fde3d6", "neutral": "#f0efec"}
load, ci = rev.load, rev.ci


def box(ax, x, y, w, h, text, fc, fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=MUTED, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, shrinkA=0, shrinkB=0))


def workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.set_xlim(-0.01, 1.04), ax.set_ylim(0, 1), ax.axis("off")
    cols = [
        ("Data", [("OSCD: 24 S2 L1C pairs,\nlabels; 10 train / 4 val /\n10 test cities", FILL["opt"]),
                  ("S1 VV 3-month means\n(DS-UNet release);\nVV/VH via Earth Engine", FILL["sar"]),
                  ("Historical S2 scenes\n(1–18 months);\nreal cloudy scenes", FILL["opt"])]),
        ("Training", [
                  ("Seeds 1–10 per arm;\nE and SAR-only U-Net\nalso seeds 11–20", FILL["neutral"]),
                  ("Arms A, B, C, async,\nU-Net baselines, E;\nconditions full / no_o2\n/ no_opt", FILL["neutral"]),
                  ("2500 steps; checkpoint\nby validation F1", FILL["neutral"])]),
        ("Selection\n(validation cities)", [
                  ("Per-condition threshold\n(argmax of validation F1,\n101-point grid), frozen", FILL["neutral"]),
                  ("Pre-registration with\ncode/input/checkpoint\nhashes, before test", FILL["fuse"])]),
        ("Test\n(10 fixed cities)", [
                  ("Pooled pixel counts:\nF1, precision, recall,\nfalse-alarm rate", FILL["neutral"]),
                  ("Corrupted inputs: missing,\nsynthetic / composite /\nreal cloud, masks", FILL["neutral"])]),
        ("Analysis", [("Paired over seeds:\nt-test, Holm; Wilcoxon\nalongside; 95% CI", FILL["neutral"]),
                      ("Exploratory: city\nresampling, leave-one-\ncity-out, land cover", FILL["neutral"])]),
    ]
    w, gap = 0.17, 0.045
    for i, (title, items) in enumerate(cols):
        x = 0.0 + i * (w + gap)
        ax.text(x + w / 2, 0.97, title, ha="center", va="top", fontsize=7.5, fontweight="bold", color=INK)
        h, top = 0.2, 0.76
        for j, (t, fc) in enumerate(items):
            box(ax, x, top - j * (h + 0.04) - h, w, h, t, fc, fs=6.3)
        if i < len(cols) - 1:
            arrow(ax, x + w + 0.006, 0.42, x + w + gap - 0.006, 0.42)
    ax.text(0.0, 0.0, "Schematic only; no data. Blue: SAR; green: optical; orange: integrity control.",
            fontsize=6.3, color=MUTED, va="bottom")
    fig.savefig(OUT / "fig01_workflow.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architectures() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.3), gridspec_kw={"width_ratios": [1.3, 1, 1.05]})
    for ax in axes:
        ax.set_xlim(0, 1), ax.set_ylim(-0.04, 1), ax.axis("off")
    # (a) DS-UNet
    ax = axes[0]
    ax.set_title("(a) DS-UNet (arms A, B, C, async)", fontsize=7.5, color=INK, loc="left")
    box(ax, 0.02, 0.70, 0.42, 0.16, "S1 VV,\ndates 1+2 (2 ch)", FILL["sar"], fs=6.2)
    box(ax, 0.54, 0.70, 0.44, 0.16, "S2 13 bands,\ndates 1+2 (26 ch)", FILL["opt"], fs=6.2)
    box(ax, 0.02, 0.43, 0.42, 0.16, "SAR U-Net\n[64,128,256,512]\n→ 64 features", FILL["sar"], fs=6.3)
    box(ax, 0.54, 0.43, 0.44, 0.16, "Optical U-Net\n[64,128,256,512]\n→ 64 features", FILL["opt"], fs=6.3)
    box(ax, 0.1, 0.17, 0.8, 0.14, "concatenate (128) → 1×1 conv\n(only point where streams meet)", FILL["fuse"], fs=6.1)
    box(ax, 0.3, -0.02, 0.4, 0.1, "change logit", FILL["neutral"], fs=6.3)
    for x in (0.23, 0.76):
        arrow(ax, x, 0.70, x, 0.59)
        arrow(ax, x, 0.43, 0.5 + (x - 0.5) * 0.4, 0.31)
    arrow(ax, 0.5, 0.17, 0.5, 0.08)
    # (b) single-stream U-Nets
    ax = axes[1]
    ax.set_title("(b) U-Net baselines (full input)", fontsize=7.5, color=INK, loc="left")
    for j, (t, fc) in enumerate((("SAR-only: VV dates 1+2\n(2 ch; VH/VV+VH in RQ3)", FILL["sar"]),
                                 ("Optical-only: S2 dates 1+2\n(26 ch)", FILL["opt"]),
                                 ("Input fusion: VV + S2,\ndates 1+2 (28 ch)", FILL["fuse"]))):
        box(ax, 0.04, 0.72 - j * 0.2, 0.92, 0.16, t, fc, fs=6.3)
    box(ax, 0.2, 0.08, 0.6, 0.14, "one U-Net\n[64,128,256,512]", FILL["neutral"], fs=6.3)
    arrow(ax, 0.5, 0.32, 0.5, 0.22)
    # (c) E and the conditions
    ax = axes[2]
    ax.set_title("(c) E: early fusion + dropout", fontsize=7.5, color=INK, loc="left")
    box(ax, 0.04, 0.80, 0.92, 0.12, "VV + S2 stacked at the input\n(28 ch), one U-Net", FILL["fuse"], fs=6.3)
    ax.text(0.04, 0.72, "Training conditions (A: full only;\nB, E: full 0.50 / no_o2 0.25 / no_opt 0.25;\n"
            "async: no_o2 only)", fontsize=6, color=MUTED, va="top")
    rows = [("full", (1, 1, 1, 1)), ("no_o2", (1, 1, 1, 0)), ("no_opt", (1, 0, 1, 0))]
    heads = ["S1 d1", "S2 d1", "S1 d2", "S2 d2"]
    x0, cw, y0 = 0.22, 0.185, 0.40
    for k, hname in enumerate(heads):
        ax.text(x0 + k * cw + cw / 2, y0 + 0.1, hname, ha="center", fontsize=5.6, color=MUTED)
    for r, (name, cells) in enumerate(rows):
        y = y0 - r * 0.11
        ax.text(0.2, y + 0.04, name, ha="right", va="center", fontsize=6.3, color=INK)
        for k, on in enumerate(cells):
            fc = (FILL["sar"] if k % 2 == 0 else FILL["opt"]) if on else "white"
            ax.add_patch(FancyBboxPatch((x0 + k * cw + 0.01, y), cw - 0.02, 0.08, boxstyle="round,pad=0,rounding_size=0.01",
                                        fc=fc, ec=MUTED, lw=0.6, hatch=None if on else "////"))
    ax.text(0.04, 0.02, "Hatched = channels set to zero.\nIn no_o2 the date-1 optical slot holds\n"
            "the historical image.", fontsize=6, color=MUTED, va="bottom")
    fig.text(0.01, -0.02, "Schematic drawn from src/training/train_robust.py, train_rrd.py and DS-UNet networks.py "
             "(commit 8bfa4be); no data.", fontsize=6.3, color=MUTED)
    fig.savefig(OUT / "fig02_architectures.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def forest() -> None:
    s6, s7, s4b, s9, s10, s11 = (load(p) for p in ("phase6/summary.json", "phase7/summary.json", "stage4b/stage4b_summary.json",
                                                   "step9/summary.json", "step10/summary.json", "realcloud/summary.json"))
    single = lambda r: {**r, "holm_p": r["t_p"]}  # one-test families
    groups = [
        ("Full input — 10 cities, per-condition thresholds (Step 6)", [
            ("DS-UNet A − SAR-only U-Net", s6["primary_rq4"]["A - SAR-only"]),
            ("DS-UNet A − optical-only U-Net", s6["primary_rq4"]["A - optical-only"]),
            ("DS-UNet A − input-fusion U-Net", s6["primary_rq4"]["A - input fusion"])]),
        ("No post-change optical — 10 cities, per-condition thresholds (Step 6)", [
            ("B@no_o2 − SAR-only U-Net", s6["primary_rq1"]["B@no_o2 - SAR-only"]),
            ("async@no_o2 − SAR-only U-Net", s6["primary_rq1"]["async@no_o2 - SAR-only"])]),
        ("Historical image within a model — 10 cities", [
            ("B: present − absent (full-input threshold; Stage 4B)", single(s4b["primary"])),
            ("E: present − absent (per-condition thresholds; Step 9)", s9["primary"]["E: image present - absent"]),
            ("E effect − B effect (Step 9)", s9["primary"]["interaction: (E present - absent) - (B present - absent)"])]),
        ("Seeds 11–20 — 10 cities, per-condition thresholds (Step 10)", [
            ("E without optical − SAR-only U-Net", s10["primary"]["E@no_opt - SAR-only"])]),
        ("Polarisation, SAR-only U-Net — 9 cities (Step 7)", [
            ("VV+VH − VV", s7["primary_rq3"]["VV+VH - VV"]), ("VH − VV", s7["primary_rq3"]["VH - VV"])]),
        ("Real cloudy scenes — 7 cities, full-input thresholds (Step 11)", [
            ("B: rb95 mask − no mask", s11["primary"]["B real_rb95 - B real"]),
            ("B with rb95 mask − SAR-only U-Net", s11["primary"]["B real_rb95 - SAR-only"]),
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
    fig, ax = plt.subplots(figsize=(7.2, 0.24 * (-y) + 0.6))
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
    ax.set_xlabel("Mean paired difference in test F1 (95% CI over 10 training seeds; fixed test cities)")
    ax.plot([], [], "o", mfc=BLUE, mec=BLUE, label="Holm p < 0.05 (single test: p < 0.05)")
    ax.plot([], [], "o", mfc="white", mec=BLUE, mew=1.5, label="no detectable difference")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7)
    fig.savefig(OUT / "fig03_forest.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


CONDS = [("oscd", "clear\n(OSCD)"), ("real", "cloudy,\nno mask"), ("real_bal", "cloudy,\nbal mask"),
         ("real_rb95", "cloudy,\nrb95 mask"), ("missing", "image\ndiscarded")]


def cloud_conditions() -> None:
    arms = load("realcloud/summary.json")["arms"]["cloudy"]
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    sar, sh = ci(arms["p6_unet_sar"]["oscd"]["f1"]["values"])
    ax.axhspan(sar - sh, sar + sh, color=FILL["neutral"], zorder=0)
    ax.axhline(sar, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(len(CONDS) - 0.45, sar, "SAR-only\nU-Net", va="center", fontsize=6.5, color=MUTED)
    x = np.arange(len(CONDS))
    for arm, color, name, dx in (("p2_full", ORANGE, "A (full-input training)", -0.1),
                                 ("p2_dropout", BLUE, "B (modality dropout)", 0.1)):
        m = [ci(arms[arm][c]["f1"]["values"]) for c, _ in CONDS]
        ax.errorbar(x + dx, [a for a, _ in m], yerr=[h for _, h in m], color=color, lw=0, elinewidth=1.4, marker="o",
                    ms=6, mec="white", mew=1, capsize=0, zorder=3, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in CONDS], fontsize=6.8)
    ax.set_xlim(-0.5, len(CONDS) - 0.1)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("Test F1, 7 partly cloudy cities")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7)
    fig.savefig(OUT / "fig06_cloud_conditions.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cloud_masks() -> None:
    """Partly cloudy cities with the lowest, median and highest city-box cloud fraction (rule fixed before drawing)."""
    man = json.loads((ROOT / "datasets" / "oscd_s2_realcloud" / "manifest.json").read_text(encoding="utf-8"))
    cloudy = sorted((r for r in man if r["tag"] == "cloudy"), key=lambda r: r["cloud"])
    pick = [cloudy[0], cloudy[len(cloudy) // 2], cloudy[-1]]
    ops = load("phase5/operating_points.json")["points"]
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 7.0))
    for i, r in enumerate(pick):
        with rasterio.open(ROOT / "datasets" / "oscd_s2_realcloud" / f"{r['city']}_cloudy.tif") as src:
            rgb = np.stack([src.read(b).astype(np.float32) for b in (4, 3, 2)], -1) / 10000.0  # B04, B03, B02
        prob = np.load(R / "realcloud" / "detector_outputs" / f"{r['city']}_cloudy.npz")["prob"]
        rgb = cv2.resize(rgb, (prob.shape[1], prob.shape[0]), interpolation=cv2.INTER_LINEAR)  # label grid, as in eval
        rgb = np.clip(rgb / 0.3, 0, 1)  # display stretch only
        panels = [(rgb, f"{r['city']} {r['day']} (Δ{r['days_from_date2']} d)\ncity-box cloud {r['cloud']:.2f}")]
        for op in ("bal", "rb95"):
            m = detector_mask(prob, ops[op]["t"], ops[op]["r"])
            panels.append((m, f"{op} mask: {m.mean():.2f} of pixels"))
        for j, (img, title) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img, cmap="Greys_r" if img.ndim == 2 else None, interpolation="nearest")
            ax.set_title(title, fontsize=6.8, color=INK)
            ax.set_xticks([]), ax.set_yticks([])  # keep the frame: an all-white mask must stay visible
            for sp in ax.spines.values():
                sp.set_color(MUTED)
    fig.text(0.01, 0.0, "True colour (B04/B03/B02, display stretch) resampled to the OSCD label grid; masks: white = masked "
             "(zero-filled before the change model).", fontsize=6.3, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "fig07_cloud_masks.png", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    workflow()
    architectures()
    forest()
    rev.OUT = OUT
    rev_timing_out = OUT / "fig15_timing.png"
    rev.timing()
    rev_timing_out.replace(OUT / "fig04_timing.png")
    cloud_conditions()
    cloud_masks()
    shutil.copyfile(ROOT / "figures" / "paper" / "fig13_error_maps.png", OUT / "fig05_error_maps.png")
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
