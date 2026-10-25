"""
Generate all figures for the research paper.

Figures produced in figures/paper/:
  fig01_architecture_comparison.png  - bar chart: F1 comparison across all models
  fig02_precision_recall.png         - precision vs recall scatter for all models
  fig03_training_curves.png          - validation F1 curves: all CNN models
  fig04_training_curves_transformer.png - validation F1 curves: transformers
  fig05_per_city_f1_heatmap.png      - per-city F1 heatmap across models
  fig06_far_vs_f1.png                - FAR vs F1 trade-off scatter
  fig07_scenario2_bar.png            - Scenario 2 F1 comparison
  fig08_qualitative_brasilia.png     - side-by-side visual comparison (Brasilia)
  fig09_confusion_breakdown.png      - TP/FP/FN count breakdown bar chart

Run from project root:
    .venv/Scripts/python.exe generate_paper_figures.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("matplotlib not installed — installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

try:
    import pandas as pd
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "-q"])
    import pandas as pd

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Data definitions
# ─────────────────────────────────────────────────────────────

# CNN rows are multi-seed means read from experiments/<name>_multiseed_summary.json
# (written by run_baseline_multiseed.py); Transformer rows are the pos_weight=41 runs.
CNN_EXPERIMENTS = {
    "SAR-only U-Net":       ("sar_baseline", 31.00),
    "Optical-only U-Net":   ("optical_baseline", 31.00),
    "Early Fusion":         ("fusion_baseline", 31.00),
    "Sync Dual-Stream CNN": ("dual_stream_baseline", 27.28),
    "Naive Missing-Optical CNN":      ("asynchronous_dual_stream", 27.28),
    "Clean Missing-Optical CNN":      ("asynchronous_dual_stream_clean", 27.28),
}
METRIC_KEYS = ["final_test_f1", "final_test_precision", "final_test_recall", "final_test_iou", "final_test_far"]


def load_summary(name: str) -> dict:
    with open(ROOT / "experiments" / f"{name}_multiseed_summary.json", encoding="utf-8") as fh:
        return json.load(fh)


MODELS_OSCD, STD_OSCD, SEED_F1 = [], [], []
for _label, (_name, _params) in CNN_EXPERIMENTS.items():
    _s = load_summary(_name)
    MODELS_OSCD.append((_label, *[_s[f"{k}_mean"] for k in METRIC_KEYS], _params))
    STD_OSCD.append([_s[f"{k}_std"] for k in METRIC_KEYS])
    SEED_F1.append(_s["final_test_f1_values"])
# Original Transformer configuration (pos_weight=41, weight decay 0)
for _label, _name, _params in (("Transformer Self-Attn", "transformer_self_attention", 1.36),
                               ("Transformer Cross-Attn", "transformer_cross_attention", 1.16)):
    _s = load_summary(_name)
    MODELS_OSCD.append((_label, *[_s[f"{k}_mean"] for k in METRIC_KEYS], _params))
    STD_OSCD.append([_s[f"{k}_std"] for k in METRIC_KEYS])
    SEED_F1.append(_s["final_test_f1_values"])
N_SEEDS = [len(v) for v in SEED_F1]
F1_STD = [s[0] for s in STD_OSCD]
IOU_STD = [s[3] for s in STD_OSCD]
FAR_STD = [s[4] for s in STD_OSCD]

LABELS    = [m[0] for m in MODELS_OSCD]
F1        = [m[1] for m in MODELS_OSCD]
PREC      = [m[2] for m in MODELS_OSCD]
REC       = [m[3] for m in MODELS_OSCD]
IOU       = [m[4] for m in MODELS_OSCD]
FAR       = [m[5] for m in MODELS_OSCD]
PARAMS    = [m[6] for m in MODELS_OSCD]

COLORS = [
    "#6baed6",  # SAR-only
    "#74c476",  # Optical-only
    "#fd8d3c",  # Early Fusion
    "#9ecae1",  # Sync Dual-Stream
    "#fdd49e",  # Naive Async
    "#e31a1c",  # Clean Missing-Optical CNN
    "#d9d9d9",  # Transformer Self
    "#bdbdbd",  # Transformer Cross
]

SCENARIO2 = {
    "Sync Dual-Stream CNN": 0.1102,
    "Clean Missing-Optical CNN":    0.0794,
    "Transformer Cross-Attn (tuned)": 0.0850,
}

# Base dirs are overwritten by every run, so per-run figures read the archived seed-42 copy.
EXPERIMENT_DIRS = {
    "SAR-only U-Net":         ROOT / "experiments" / "sar_baseline_seed42",
    "Optical-only U-Net":     ROOT / "experiments" / "optical_baseline_seed42",
    "Early Fusion":           ROOT / "experiments" / "fusion_baseline_seed42",
    "Sync Dual-Stream CNN":   ROOT / "experiments" / "dual_stream_baseline_seed42",
    "Naive Missing-Optical CNN":        ROOT / "experiments" / "asynchronous_dual_stream_seed42",
    "Clean Missing-Optical CNN":        ROOT / "experiments" / "asynchronous_dual_stream_clean_seed42",
    "Transformer Self-Attn":  ROOT / "experiments" / "transformer_self_attention_seed42",
    "Transformer Cross-Attn": ROOT / "experiments" / "transformer_cross_attention_seed42",
}


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def save(name: str):
    path = OUT / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/paper/{name}")


# ─────────────────────────────────────────────────────────────
# Fig 01 — F1 bar chart (main result)
# ─────────────────────────────────────────────────────────────

def fig01_f1_bar():
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(LABELS))
    bars = ax.bar(x, F1, yerr=F1_STD, capsize=4, color=COLORS, edgecolor="white",
                  linewidth=0.8, zorder=3, error_kw={"zorder": 5})
    for i, vals in enumerate(SEED_F1):
        ax.scatter([x[i]] * len(vals), vals, s=14, color="black", zorder=6)

    for i, (bar, v) in enumerate(zip(bars, F1)):
        ax.text(bar.get_x() + bar.get_width() / 2, max(SEED_F1[i] + [v + F1_STD[i]]) + 0.008,
                f"{v:.3f}\n(n={N_SEEDS[i]})", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score (τ = 0.5)", fontsize=11)
    ax.set_title("OSCD Test F1 — mean ± sample SD across seeds (dots = individual seeds)", fontsize=12)
    ax.set_ylim(0, 0.68)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotation: CNN vs Transformer loss note
    ax.annotate("pos_weight=1", xy=(4.5, 0.01), fontsize=7.5, color="#888888",
                ha="center", style="italic")
    ax.annotate("pos_weight=41", xy=(6.5, 0.01), fontsize=7.5, color="#888888",
                ha="center", style="italic")
    ax.axvline(5.5, color="#999999", linestyle=":", linewidth=1, alpha=0.5)

    plt.tight_layout()
    save("fig01_f1_bar.png")


# ─────────────────────────────────────────────────────────────
# Fig 02 — Precision vs Recall scatter
# ─────────────────────────────────────────────────────────────

def fig02_precision_recall():
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (label, p, r, f) in enumerate(zip(LABELS, PREC, REC, F1)):
        size = 120 + PARAMS[i] * 2
        ax.scatter(r, p, s=size, color=COLORS[i], edgecolors="black",
                   linewidth=0.8, zorder=4, label=label, alpha=0.9)
        ax.annotate(label.replace(" ", "\n"), (r, p), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, va="bottom")

    # F1 iso-curves
    f_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    x_line = np.linspace(0.01, 0.99, 200)
    for fval in f_vals:
        y_line = fval * x_line / (2 * x_line - fval)
        mask = (y_line > 0) & (y_line <= 1)
        ax.plot(x_line[mask], y_line[mask], ":", color="gray", linewidth=0.8, alpha=0.6)
        if np.any(mask):
            idx = np.where(mask)[0][len(np.where(mask)[0])//2]
            ax.text(x_line[idx], y_line[idx], f"F1={fval}", fontsize=7, color="gray", alpha=0.7)

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision vs Recall — OSCD Test Set\n(marker size ∝ parameter count)", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save("fig02_precision_recall.png")


# ─────────────────────────────────────────────────────────────
# Fig 03 — Training curves: CNN models
# ─────────────────────────────────────────────────────────────

def load_history(model_name: str) -> pd.DataFrame | None:
    exp_dir = EXPERIMENT_DIRS.get(model_name)
    if exp_dir is None:
        return None
    path = exp_dir / "training_history.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def fig03_training_curves_cnn():
    cnn_models = [
        "SAR-only U-Net", "Optical-only U-Net", "Early Fusion",
        "Sync Dual-Stream CNN", "Naive Missing-Optical CNN", "Clean Missing-Optical CNN"
    ]
    cnn_colors = dict(zip(LABELS, COLORS))

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for name in cnn_models:
        df = load_history(name)
        if df is None or "validation_f1" not in df.columns:
            continue
        vf1 = pd.to_numeric(df["validation_f1"], errors="coerce")
        ax.plot(df["epoch"], vf1, label=name, color=cnn_colors[name],
                linewidth=2 if name == "Clean Missing-Optical CNN" else 1.2,
                linestyle="-" if name == "Clean Missing-Optical CNN" else "--")
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "Training history CSV not found", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
    else:
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation F1", fontsize=11)
        ax.set_title("Training Progress — CNN Models (validation F1 per epoch)", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    save("fig03_training_curves_cnn.png")


# ─────────────────────────────────────────────────────────────
# Fig 04 — Training curves: Transformer models
# ─────────────────────────────────────────────────────────────

def fig04_training_curves_transformer():
    t_models = ["Transformer Self-Attn", "Transformer Cross-Attn"]
    t_colors = {"Transformer Self-Attn": "#d9d9d9", "Transformer Cross-Attn": "#bdbdbd"}

    # Check tuning dir too
    tuning_dir = ROOT / "experiments" / "transformer_tuning_2026-10-15"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, t_models):
        df = load_history(name)
        if df is not None and "validation_f1" in df.columns:
            vf1 = pd.to_numeric(df["validation_f1"], errors="coerce")
            ax.plot(df["epoch"], vf1, color="#636363", linewidth=2, label="Baseline (seed42)")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation F1")
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Training Progress — Transformer Models", fontsize=12, y=1.01)
    plt.tight_layout()
    save("fig04_training_curves_transformer.png")


# ─────────────────────────────────────────────────────────────
# Fig 05 — Per-city F1 heatmap
# ─────────────────────────────────────────────────────────────

def fig05_per_city_heatmap():
    show_models = [
        ("SAR-only U-Net",       "sar_baseline"),
        ("Optical-only U-Net",   "optical_baseline"),
        ("Early Fusion",         "fusion_baseline"),
        ("Sync Dual-Stream",     "dual_stream_baseline"),
        ("Clean Missing-Optical CNN",      "asynchronous_dual_stream_clean"),
    ]

    all_cities = None
    data = {}
    for label, dirname in show_models:
        paths = sorted((ROOT / "experiments").glob(f"{dirname}_seed*/per_city_metrics.csv"))
        if not paths:
            continue
        # per-city F1 averaged over seeds; a seed with undefined F1 for a city is skipped
        f1_col = pd.concat([pd.to_numeric(pd.read_csv(p).set_index("city")["f1"], errors="coerce")
                            for p in paths], axis=1).mean(axis=1)
        if all_cities is None:
            all_cities = list(f1_col.index)
        data[f"{label} (n={len(paths)})"] = f1_col.reindex(all_cities).values

    if not data or all_cities is None:
        print("  Skipping fig05: per-city CSVs not found")
        return

    matrix = np.array(list(data.values()))  # (n_models, n_cities)
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=0.7)

    ax.set_xticks(np.arange(len(all_cities)))
    ax.set_xticklabels([c.title() for c in all_cities], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(list(data.keys()), fontsize=9)

    for i in range(len(data)):
        for j in range(len(all_cities)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if val > 0.15 else "gray")

    plt.colorbar(im, ax=ax, label="F1 Score", fraction=0.015, pad=0.02)
    ax.set_title("Per-City F1 (mean over seeds) — OSCD Test Cities; red = low, green = high, blank = undefined",
                 fontsize=11, pad=10)
    plt.tight_layout()
    save("fig05_per_city_heatmap.png")


# ─────────────────────────────────────────────────────────────
# Fig 06 — FAR vs F1 scatter
# ─────────────────────────────────────────────────────────────

def fig06_far_vs_f1():
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, f1, prec, rec, iou, far, params) in enumerate(MODELS_OSCD):
        if far is None:
            continue
        ax.errorbar(far, f1, xerr=FAR_STD[i], yerr=F1_STD[i], fmt="none",
                    ecolor="#555555", elinewidth=0.8, capsize=3, zorder=3)
        ax.scatter(far, f1, s=100 + params * 2, color=COLORS[i],
                   edgecolors="black", linewidth=0.8, zorder=4, label=label)
        ax.annotate(f"{label} (n={N_SEEDS[i]})", (far, f1), textcoords="offset points",
                    xytext=(5, 3), fontsize=7.5)

    ax.set_xlabel("False Alarm Rate (FP / (FP + TN))", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("False Alarm Rate vs F1 — OSCD Test (mean ± sample SD across seeds)\n"
                 "upper-left is better (high F1, low FAR); marker size ∝ parameters", fontsize=10)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax.annotate("← Better (upper-left)", xy=(0.02, 0.97), xycoords="axes fraction",
                fontsize=8, color="#e31a1c", style="italic", alpha=0.8, va="top")

    plt.tight_layout()
    save("fig06_far_vs_f1.png")


# ─────────────────────────────────────────────────────────────
# Fig 07 — Scenario 2 bar chart
# ─────────────────────────────────────────────────────────────

def fig07_scenario2():
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(SCENARIO2.keys())
    vals = list(SCENARIO2.values())
    colors = ["#9ecae1", "#e31a1c", "#bdbdbd"]
    bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("F1 Score (τ = 0.5)", fontsize=11)
    ax.set_title("SEN12-MT Scenario 2 — Preliminary F1 Results\n(21-month gap, 4 test sites, proxy labels)", fontsize=10)
    ax.set_ylim(0, 0.16)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save("fig07_scenario2.png")


# ─────────────────────────────────────────────────────────────
# Fig 08 — Qualitative: Brasilia comparison (SAR vs Optical vs Clean Async)
# ─────────────────────────────────────────────────────────────

def fig08_qualitative_brasilia():
    sar_dir = ROOT / "figures" / "sar_baseline"
    opt_dir = ROOT / "figures" / "optical_baseline"

    sar_images = {
        "SAR T1": sar_dir / "brasilia_sar_t1.png",
        "SAR T2": sar_dir / "brasilia_sar_t2.png",
        "SAR Prediction": sar_dir / "brasilia_binary.png",
        "SAR Prob Map": sar_dir / "brasilia_probability.png",
    }
    opt_images = {
        "Optical T1": opt_dir / "brasilia_optical_t1.png",
        "Optical T2": opt_dir / "brasilia_optical_t2.png",
        "Optical Prediction": opt_dir / "brasilia_binary.png",
        "Ground Truth": sar_dir / "brasilia_ground_truth.png",
    }

    if not HAS_PIL:
        print("  Skipping fig08: Pillow not installed")
        return

    avail_sar = {k: v for k, v in sar_images.items() if v.exists()}
    avail_opt = {k: v for k, v in opt_images.items() if v.exists()}

    all_items = list(avail_sar.items()) + list(avail_opt.items())
    if not all_items:
        print("  Skipping fig08: no Brasilia figures found")
        return

    n = len(all_items)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx, (title, path) in enumerate(all_items):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        try:
            img = Image.open(path)
            ax.imshow(img, cmap="gray" if img.mode == "L" else None)
        except Exception:
            ax.text(0.5, 0.5, f"Error loading\n{title}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        ax.set_title(title, fontsize=9, pad=4)
        ax.axis("off")

    # Hide empty subplots
    for idx in range(len(all_items), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.suptitle("Qualitative Results — Brasilia (OSCD Test City)", fontsize=12, y=1.01)
    plt.tight_layout()
    save("fig08_qualitative_brasilia.png")


# ─────────────────────────────────────────────────────────────
# Fig 09 — TP/FP/FN confusion breakdown
# ─────────────────────────────────────────────────────────────

def fig09_confusion_breakdown():
    # Confusion counts from metrics.json files
    confusion = {}
    for label, dirname in [
        ("SAR-only U-Net",       "sar_baseline"),
        ("Optical-only U-Net",   "optical_baseline"),
        ("Early Fusion",         "fusion_baseline"),
        ("Sync Dual-Stream CNN", "dual_stream_baseline"),
        ("Naive Missing-Optical CNN",      "asynchronous_dual_stream"),
        ("Clean Missing-Optical CNN",      "asynchronous_dual_stream_clean"),
    ]:
        path = ROOT / "experiments" / f"{dirname}_seed42" / "metrics.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        counts = d.get("test_counts", {})
        if counts:
            confusion[label] = counts

    if not confusion:
        print("  Skipping fig09: no metrics.json found")
        return

    names = list(confusion.keys())
    tp = [confusion[n].get("tp", 0) for n in names]
    fp = [confusion[n].get("fp", 0) for n in names]
    fn = [confusion[n].get("fn", 0) for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - width, tp, width, label="True Positives", color="#41ab5d", edgecolor="white")
    b2 = ax.bar(x,         fp, width, label="False Positives", color="#fe9929", edgecolor="white")
    b3 = ax.bar(x + width, fn, width, label="False Negatives", color="#e34a33", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Pixel Count", fontsize=11)
    ax.set_title("Confusion Breakdown — OSCD Test Pixels (seed 42, pooled across 10 cities)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save("fig09_confusion_breakdown.png")


# ─────────────────────────────────────────────────────────────
# Fig 10 — Architecture diagram (text-based schematic)
# ─────────────────────────────────────────────────────────────

def fig10_architecture_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Architecture Schematics", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # Left: Clean Missing-Optical CNN
    ax = axes[0]
    ax.set_title("Clean Asynchronous Dual-Stream CNN\n(15-ch input: 2 SAR + 13 Optical T1)", fontsize=10)

    def box(ax, x, y, w, h, text, color, fontsize=8):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="#333333",
                                        linewidth=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=4)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))

    # SAR stream
    box(ax, 0.5, 7.5, 2, 0.8, "SAR Input\n2ch (VV+VH)", "#9ecae1")
    box(ax, 0.5, 5.8, 2, 1.4, "SAR Encoder\n(4 stages)\n64→512ch", "#9ecae1")
    arrow(ax, 1.5, 7.5, 1.5, 7.2)

    # Optical stream
    box(ax, 3.5, 7.5, 2, 0.8, "Optical Input\n13ch (S2-T1)", "#74c476")
    box(ax, 3.5, 5.8, 2, 1.4, "Optical Encoder\n(4 stages)\n64→512ch", "#74c476")
    arrow(ax, 4.5, 7.5, 4.5, 7.2)

    # Bottleneck
    box(ax, 1.8, 4.2, 2.5, 1.2, "Bottleneck\nConcatenate\n+ Conv1×1", "#fdd49e")
    arrow(ax, 1.5, 5.8, 2.5, 5.4)
    arrow(ax, 4.5, 5.8, 3.5, 5.4)

    # Decoder
    box(ax, 1.8, 2.5, 2.5, 1.4, "Shared Decoder\n+ Skip Connections\n(4 stages)", "#f4a460")
    arrow(ax, 3.05, 4.2, 3.05, 3.9)

    # Output
    box(ax, 2.2, 1.0, 1.7, 1.1, "Output\n1ch → σ → τ=0.5", "#e31a1c", fontsize=7)
    arrow(ax, 3.05, 2.5, 3.05, 2.1)

    _ca = list(CNN_EXPERIMENTS).index("Clean Missing-Optical CNN")
    ax.text(5.5, 5.5, f"27.28M params\nF1 = {F1[_ca]:.3f} ± {F1_STD[_ca]:.3f}\n(n={N_SEEDS[_ca]} seeds)", fontsize=9,
            va="center", ha="left",
            bbox=dict(boxstyle="round", facecolor="#fff3cd", edgecolor="#aaa"))

    # Right: Transformer Cross-Attention
    ax2 = axes[1]
    ax2.set_title("Lightweight Transformer Cross-Attention\n(15-ch input, patch embeddings)", fontsize=10)

    box(ax2, 0.5, 7.5, 2.5, 0.8, "SAR Patches\n4×4 tiles", "#9ecae1")
    box(ax2, 4.0, 7.5, 2.5, 0.8, "Optical Patches\n4×4 tiles", "#74c476")

    box(ax2, 0.5, 5.5, 2.5, 1.6, "SAR Embed\n+ Pos. Enc.", "#9ecae1")
    box(ax2, 4.0, 5.5, 2.5, 1.6, "Optical Embed\n+ Pos. Enc.", "#74c476")
    arrow(ax2, 1.75, 7.5, 1.75, 7.1)
    arrow(ax2, 5.25, 7.5, 5.25, 7.1)

    box(ax2, 1.8, 3.7, 3.5, 1.5, "Multi-Head\nCross-Attention\n(SAR↔Optical Q/K/V)", "#fdd49e")
    arrow(ax2, 1.75, 5.5, 2.5, 5.2)
    arrow(ax2, 5.25, 5.5, 4.5, 5.2)

    box(ax2, 2.3, 2.2, 2.5, 1.2, "FFN + Upsample\n→ Output 1ch", "#f4a460")
    arrow(ax2, 3.55, 3.7, 3.55, 3.4)

    box(ax2, 2.5, 0.8, 2.1, 1.0, "σ → τ=0.5\nBinary Map", "#e31a1c", fontsize=7)
    arrow(ax2, 3.55, 2.2, 3.55, 1.8)

    ax2.text(6.8, 5.5, "1.16M params\nF1 = 0.2477\nFAR = 0.2134", fontsize=9,
             va="center", ha="left",
             bbox=dict(boxstyle="round", facecolor="#fff3cd", edgecolor="#aaa"))

    plt.tight_layout()
    save("fig10_architecture_schematic.png")


# ─────────────────────────────────────────────────────────────
# Fig 11 — IoU comparison bar
# ─────────────────────────────────────────────────────────────

def fig11_iou_bar():
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(LABELS))
    bars = ax.bar(x, IOU, yerr=IOU_STD, capsize=4, color=COLORS, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("IoU (Jaccard Index)", fontsize=11)
    ax.set_title("OSCD IoU — mean ± sample SD across seeds", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for i, (bar, v) in enumerate(zip(bars, IOU)):
        ax.text(bar.get_x() + bar.get_width() / 2, v + IOU_STD[i] + 0.003, f"{v:.3f}\n(n={N_SEEDS[i]})",
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    save("fig11_iou_bar.png")


def fig12_matched_loss():
    """Matched-loss comparison: every model trained with pos_weight=1."""
    xf = load_summary("transformer_cross_attention_pw1")
    labels = list(CNN_EXPERIMENTS) + ["Transformer Cross-Attn (pw=1)"]
    means = F1[:len(CNN_EXPERIMENTS)] + [xf["final_test_f1_mean"]]
    stds = F1_STD[:len(CNN_EXPERIMENTS)] + [xf["final_test_f1_std"]]
    seeds = SEED_F1[:len(CNN_EXPERIMENTS)] + [xf["final_test_f1_values"]]
    colors = COLORS[:len(CNN_EXPERIMENTS)] + ["#bdbdbd"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="white", zorder=3)
    for i, vals in enumerate(seeds):
        ax.scatter([x[i]] * len(vals), vals, s=14, color="black", zorder=6)
        ax.text(x[i], max(vals + [means[i] + stds[i]]) + 0.008, f"{means[i]:.3f}\n(n={len(vals)})",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("OSCD test F1 (τ = 0.5)")
    ax.set_title("Matched-loss comparison (all models pos_weight=1; mean ± sample SD, dots = seeds)\n"
                 "Transformer: 1.16M params; its checkpoints were selected at epochs 1–5 before collapse",
                 fontsize=10)
    ax.set_ylim(0, 0.68)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save("fig12_controlled_comparison.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating figures in {OUT}")
    print()

    funcs = [
        ("Fig 01: F1 bar chart",               fig01_f1_bar),
        ("Fig 02: Precision-Recall scatter",   fig02_precision_recall),
        ("Fig 03: Training curves (CNNs)",     fig03_training_curves_cnn),
        ("Fig 04: Training curves (Transformers)", fig04_training_curves_transformer),
        ("Fig 05: Per-city F1 heatmap",        fig05_per_city_heatmap),
        ("Fig 06: FAR vs F1 scatter",          fig06_far_vs_f1),
        ("Fig 07: Scenario 2 bar chart",       fig07_scenario2),
        ("Fig 08: Qualitative Brasilia",       fig08_qualitative_brasilia),
        ("Fig 09: Confusion breakdown",        fig09_confusion_breakdown),
        ("Fig 10: Architecture schematics",    fig10_architecture_schematic),
        ("Fig 11: IoU bar chart",              fig11_iou_bar),
        ("Fig 12: Matched-loss comparison",    fig12_matched_loss),
    ]

    for name, fn in funcs:
        print(f"  {name}...")
        try:
            fn()
        except Exception as e:
            print(f"    ERROR: {e}")

    print()
    print("Done! All figures saved to figures/paper/")
    print()
    print("Files:")
    for p in sorted(OUT.glob("*.png")):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name}  ({size_kb:.0f} KB)")
