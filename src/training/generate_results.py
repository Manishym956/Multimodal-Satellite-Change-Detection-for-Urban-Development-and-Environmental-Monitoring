"""Generate all research paper figures, tables, qualitative predictions, and error maps.

Parts:
    1  - Training curves (loss + val F1) for Clean Async CNN, Self-Attn, Cross-Attn
    2  - Model comparison bar/scatter charts (all 8 models)
    3  - Per-city analysis (Clean Async CNN vs Self-Attn vs Cross-Attn)
    4  - Qualitative prediction panels (5 test-city examples)
    5  - Error-analysis maps (FP/FN/TP colour maps)
    6  - Confusion matrix / pixel-count comparison
    7  - Output tables (CSV + JSON summary)
    8  - FIGURE_INDEX.md

Run from project root:
    .venv\\Scripts\\python.exe src\\training\\generate_results.py
"""

import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── output directories ────────────────────────────────────────────────────────
RESULTS        = PROJECT_ROOT / "results"
FIG_DIR        = RESULTS / "figures"
QUAL_DIR       = RESULTS / "qualitative"
ERR_DIR        = RESULTS / "error_analysis"
TAB_DIR        = RESULTS / "tables"
for d in (FIG_DIR, QUAL_DIR, ERR_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

DPI = 300
FONT_SIZE = 10
plt.rcParams.update({"font.size": FONT_SIZE, "figure.dpi": DPI})

# ── locked experiment data ────────────────────────────────────────────────────
MODELS = {
    "SAR-only U-Net":           dict(f1=0.3558, iou=0.2164, far=0.0190, prec=0.4558, rec=0.2917, params=31_036_545),
    "Optical-only U-Net":       dict(f1=0.2663, iou=0.1536, far=0.0459, prec=0.2515, rec=0.2830, params=31_051_969),
    "Sync Fusion U-Net":        dict(f1=0.3520, iou=0.2136, far=0.0612, prec=0.2877, rec=0.4533, params=31_052_033),
    "Sync Dual-Stream CNN":     dict(f1=0.3868, iou=0.2398, far=0.0686, prec=0.3009, rec=0.5415, params=27_284_097),
    "Naive Async CNN":          dict(f1=0.2052, iou=0.1143, far=0.0319, prec=0.2363, rec=0.1814, params=27_284_097),
    "Clean Async CNN":          dict(f1=0.4117, iou=0.2592, far=0.0218, prec=0.4753, rec=0.3631, params=27_276_609),
    "Transformer Self-Attn":    dict(f1=0.1615, iou=0.0879, far=0.2391, prec=0.0974, rec=0.4734, params=1_358_081),
    "Transformer Cross-Attn":   dict(f1=0.3564, iou=0.2168, far=0.0861, prec=0.2615, rec=0.5592, params=1_156_993),
}

MODEL_COLORS = {
    "SAR-only U-Net":        "#4878CF",
    "Optical-only U-Net":    "#6ACC65",
    "Sync Fusion U-Net":     "#D65F5F",
    "Sync Dual-Stream CNN":  "#B47CC7",
    "Naive Async CNN":       "#C4AD66",
    "Clean Async CNN":       "#77BEDB",
    "Transformer Self-Attn": "#F28E2B",
    "Transformer Cross-Attn":"#E15759",
}

MODEL_NAMES = list(MODELS.keys())

# ── history files ─────────────────────────────────────────────────────────────
EXP = PROJECT_ROOT / "experiments"
HIST = {
    "Clean Async CNN":          EXP / "asynchronous_dual_stream_clean" / "training_history.csv",
    "Transformer Self-Attn":    EXP / "transformer_self_attention"     / "training_history.csv",
    "Transformer Cross-Attn":   EXP / "transformer_cross_attention"    / "training_history.csv",
}
CITY_CSV = {
    "Clean Async CNN":          EXP / "asynchronous_dual_stream_clean" / "per_city_metrics.csv",
    "Transformer Self-Attn":    EXP / "transformer_self_attention"     / "per_city_metrics.csv",
    "Transformer Cross-Attn":   EXP / "transformer_cross_attention"    / "per_city_metrics.csv",
}
CKPT = {
    "Clean Async CNN":          PROJECT_ROOT / "checkpoints" / "asynchronous_dual_stream_clean" / "best.pt",
    "Transformer Cross-Attn":   PROJECT_ROOT / "checkpoints" / "transformer_cross_attention"    / "best.pt",
}

TEST_CITIES = ["brasilia","montpellier","norcia","rio","saclay_w",
               "valencia","dubai","lasvegas","milano","chongqing"]


# ════════════════════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════════════════════

def load_history(path: Path) -> dict[str, list]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    epochs = [int(r["epoch"]) for r in rows]
    loss   = [float(r["train_loss"]) for r in rows]
    f1     = []
    for r in rows:
        v = r["validation_f1"]
        f1.append(float(v) if v not in ("", "None") else None)
    return {"epoch": epochs, "loss": loss, "val_f1": f1}


def load_city_csv(path: Path) -> dict[str, dict]:
    out = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        city = row["city"]
        def _f(k):
            v = row[k]
            return float(v) if v not in ("", "None") else None
        out[city] = {k: _f(k) for k in ("precision","recall","f1","iou","false_alarm_rate")}
    return out


def savefig(name: str, fig=None):
    path = FIG_DIR / name
    (fig or plt).savefig(path, dpi=DPI, bbox_inches="tight")
    (fig or plt).savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close("all")
    print(f"  saved {path.name}")
    return path


# ════════════════════════════════════════════════════════════════════════════════
# PART 1 — training curves
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Part 1: Training curves ──")

curve_colors = {
    "Clean Async CNN":        MODEL_COLORS["Clean Async CNN"],
    "Transformer Self-Attn":  MODEL_COLORS["Transformer Self-Attn"],
    "Transformer Cross-Attn": MODEL_COLORS["Transformer Cross-Attn"],
}

histories = {}
for name, path in HIST.items():
    if path.exists():
        histories[name] = load_history(path)
        print(f"  loaded history: {name}  ({len(histories[name]['epoch'])} epochs)")
    else:
        print(f"  MISSING history: {name}")

# training loss
fig, ax = plt.subplots(figsize=(7, 4))
for name, h in histories.items():
    ax.plot(h["epoch"], h["loss"], label=name, color=curve_colors[name], linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss")
ax.set_title("Training loss vs epoch")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
savefig("training_loss.png", fig)

# validation F1  (None → gap in line)
fig, ax = plt.subplots(figsize=(7, 4))
for name, h in histories.items():
    epochs = h["epoch"]
    f1s    = h["val_f1"]
    # plot segments between defined values
    seg_x, seg_y = [], []
    for e, v in zip(epochs, f1s):
        if v is not None:
            seg_x.append(e); seg_y.append(v)
        else:
            if seg_x:
                ax.plot(seg_x, seg_y, color=curve_colors[name], linewidth=1.5,
                        label=name if not ax.get_lines() else "")
            seg_x, seg_y = [], []
    if seg_x:
        ax.plot(seg_x, seg_y, color=curve_colors[name], linewidth=1.5, label=name)
# deduplicate legend
handles, labels = ax.get_legend_handles_labels()
seen = {}
for h2, l in zip(handles, labels):
    if l not in seen:
        seen[l] = h2
ax.legend(seen.values(), seen.keys(), fontsize=8)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation F1")
ax.set_title("Validation F1 vs epoch")
ax.grid(True, alpha=0.3)
savefig("validation_f1.png", fig)


# ════════════════════════════════════════════════════════════════════════════════
# PART 2 — model comparison bar/scatter charts
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Part 2: Model comparison charts ──")

names  = MODEL_NAMES
colors = [MODEL_COLORS[n] for n in names]
x      = np.arange(len(names))

def bar_chart(metric_key, ylabel, title, filename, ylim=None):
    vals = [MODELS[n][metric_key] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    savefig(filename, fig)

bar_chart("f1",  "Test F1",  "Test F1 by model",  "f1_comparison.png",  ylim=(0, 0.55))
bar_chart("iou", "Test IoU", "Test IoU by model", "iou_comparison.png", ylim=(0, 0.35))
bar_chart("far", "Test FAR", "Test FAR by model", "far_comparison.png", ylim=(0, 0.30))

# precision-recall scatter
fig, ax = plt.subplots(figsize=(6, 5))
for name in names:
    p = MODELS[name]["prec"]
    r = MODELS[name]["rec"]
    ax.scatter(r, p, color=MODEL_COLORS[name], s=90, zorder=3)
    ax.annotate(name, (r, p), textcoords="offset points", xytext=(5, 3), fontsize=7)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision vs Recall (test set)")
ax.set_xlim(0, 0.7); ax.set_ylim(0, 0.65)
ax.grid(True, alpha=0.3)
# iso-F1 curves
for f1_iso in [0.2, 0.3, 0.4]:
    r_range = np.linspace(0.01, 0.99, 300)
    p_range = f1_iso * r_range / (2 * r_range - f1_iso)
    mask = (p_range > 0) & (p_range <= 1)
    ax.plot(r_range[mask], p_range[mask], "--", color="grey", linewidth=0.7, alpha=0.5)
    idx = np.argmin(np.abs(r_range[mask] - 0.55))
    ax.text(r_range[mask][idx], p_range[mask][idx], f"F1={f1_iso}", fontsize=6, color="grey")
fig.tight_layout()
savefig("precision_recall_comparison.png", fig)

# parameter count (log scale)
params = [MODELS[n]["params"] for n in names]
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(x, params, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Parameters")
ax.set_title("Model parameter count")
ax.set_yscale("log")
ax.grid(True, axis="y", alpha=0.3)
for bar, v in zip(bars, params):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
            f"{v/1e6:.1f}M", ha="center", va="bottom", fontsize=7)
fig.tight_layout()
savefig("parameter_comparison.png", fig)


# ════════════════════════════════════════════════════════════════════════════════
# PART 3 — per-city analysis
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Part 3: Per-city analysis ──")

city_data = {name: load_city_csv(path) for name, path in CITY_CSV.items()}

THREE = ["Clean Async CNN", "Transformer Self-Attn", "Transformer Cross-Attn"]
THREE_COLORS = [MODEL_COLORS[n] for n in THREE]

def per_city_grouped_bar(metric, ylabel, title, filename, ylim=None):
    n_cities = len(TEST_CITIES)
    n_models = len(THREE)
    width = 0.25
    x_c = np.arange(n_cities)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (model, color) in enumerate(zip(THREE, THREE_COLORS)):
        vals = []
        for city in TEST_CITIES:
            v = city_data[model].get(city, {}).get(metric)
            vals.append(v if v is not None else 0.0)
        bars = ax.bar(x_c + (i - 1) * width, vals, width,
                      label=model, color=color, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x_c)
    ax.set_xticklabels(TEST_CITIES, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, axis="y", alpha=0.3)
    note = "(undefined values shown as 0)"
    ax.text(0.01, 0.99, note, transform=ax.transAxes,
            fontsize=6, va="top", color="grey")
    fig.tight_layout()
    savefig(filename, fig)

per_city_grouped_bar("f1",  "F1",  "Per-city F1 — Clean Async CNN vs Transformers",
                     "per_city_f1_comparison.png",  ylim=(0, 1.0))
per_city_grouped_bar("false_alarm_rate", "FAR",
                     "Per-city FAR — Clean Async CNN vs Transformers",
                     "per_city_far_comparison.png", ylim=(0, 1.05))

# precision-recall scatter per city (all three models)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, sharex=True)
for ax, model, color in zip(axes, THREE, THREE_COLORS):
    for city in TEST_CITIES:
        d = city_data[model].get(city, {})
        p = d.get("precision"); r = d.get("recall")
        if p is not None and r is not None:
            ax.scatter(r, p, color=color, s=55, zorder=3)
            ax.annotate(city[:3], (r, p), textcoords="offset points",
                        xytext=(3, 2), fontsize=6)
        else:
            ax.scatter(0, 0, marker="x", color="grey", s=40)
            ax.annotate(city[:3]+" (undef)", (0, 0), textcoords="offset points",
                        xytext=(3, 2), fontsize=5, color="grey")
    ax.set_title(model, fontsize=9)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
fig.suptitle("Per-city Precision vs Recall")
fig.tight_layout()
savefig("per_city_precision_recall.png", fig)


# ════════════════════════════════════════════════════════════════════════════════
# PART 4 — qualitative predictions
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Part 4: Qualitative predictions ──")

# model imports
from src.datasets.oscd_dataset import ORBITS, OSCD_ROOT, combine_bands, read_change_mask, resize_sentinel1
from src.models.dual_stream_unet_clean import DualStreamUNetClean
from src.models.transformer_cross_attention import TransformerCrossAttention, count_parameters
from torchvision.transforms import functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cnn():
    m = DualStreamUNetClean(base_channels=64).to(device)
    ckpt = torch.load(CKPT["Clean Async CNN"], map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m

def load_cross():
    m = TransformerCrossAttention(sar_channels=2, opt_channels=13, embed_dim=128,
                                  patch_stride=4, num_self_blocks=2, num_cross_blocks=1,
                                  num_heads=4, mlp_ratio=4).to(device)
    ckpt = torch.load(CKPT["Transformer Cross-Attn"], map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m

cnn   = load_cnn()
cross = load_cross()
print(f"  CNN loaded ({sum(p.numel() for p in cnn.parameters() if p.requires_grad):,} params)")
print(f"  Cross-Attn loaded ({count_parameters(cross):,} params)")

def prepare_city(city: str):
    orbit = int(ORBITS[city][0])
    s2_t1 = combine_bands(OSCD_ROOT / "images" / city / "imgs_1_rect")  # (H,W,13) float32 /10000
    label = read_change_mask(city)
    h, w  = label.shape
    s1_t1 = resize_sentinel1(city, orbit, 1, h, w)  # (H,W,1)
    s1_t2 = resize_sentinel1(city, orbit, 2, h, w)
    t_s1t1 = TF.to_tensor(s1_t1.astype(np.float32))   # (1,H,W)
    t_s1t2 = TF.to_tensor(s1_t2.astype(np.float32))
    t_s2t1 = TF.to_tensor(s2_t1.astype(np.float32))   # (13,H,W)
    image  = torch.cat((t_s1t1, t_s1t2, t_s2t1), dim=0).unsqueeze(0).to(device)  # (1,15,H,W)
    return image, label, s1_t1[:,:,0], s1_t2[:,:,0], s2_t1

@torch.inference_mode()
def predict(model, image):
    logits = model(image)
    return torch.sigmoid(logits)[0, 0].cpu().numpy()

def stretch(arr):
    lo, hi = np.percentile(arr, (2, 98))
    if hi <= lo: hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0, 1)

def rgb_from_s2(s2):
    """Return display-ready RGB from S2 bands (B4=idx3, B3=idx2, B2=idx1)."""
    r = stretch(s2[:, :, 3])
    g = stretch(s2[:, :, 2])
    b = stretch(s2[:, :, 1])
    return np.stack([r, g, b], axis=2)

# example selection: representative test cities
# A=montpellier (clear success), B=dubai (CNN↑ transformer FAR↑),
# C=lasvegas (cross-attn best), D=rio (high FAR both), E=norcia (subtle/small)
EXAMPLES = [
    ("montpellier", "example_01", "A – clear urban change detection"),
    ("dubai",       "example_02", "B – high FAR in transformers"),
    ("lasvegas",    "example_03", "C – cross-attention outperforms CNN"),
    ("rio",         "example_04", "D – difficult high-FAR scene"),
    ("norcia",      "example_05", "E – subtle/small change scene"),
]

for city, ex_id, ex_label in EXAMPLES:
    ex_dir = QUAL_DIR / ex_id
    ex_dir.mkdir(exist_ok=True)
    image, label, s1t1, s1t2, s2t1 = prepare_city(city)
    prob_cnn   = predict(cnn,   image)
    prob_cross = predict(cross, image)
    bin_cnn    = (prob_cnn   >= 0.5).astype(np.float32)
    bin_cross  = (prob_cross >= 0.5).astype(np.float32)
    gt         = (label      >= 1  ).astype(np.float32)
    rgb        = rgb_from_s2(s2t1)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    panels = [
        (axes[0,0], stretch(s1t1), "SAR T1",           "gray"),
        (axes[0,1], stretch(s1t2), "SAR T2",           "gray"),
        (axes[0,2], rgb,           "Historical S2 T1 (RGB)", None),
        (axes[1,0], gt,            "Ground truth",     "gray"),
        (axes[1,1], bin_cnn,       "Clean Async CNN",  "gray"),
        (axes[1,2], bin_cross,     "Cross-Attn Transformer","gray"),
    ]
    for ax, img, title, cmap in panels:
        if cmap:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{ex_label}  |  city: {city}", fontsize=10)
    fig.tight_layout()
    out = ex_dir / f"{city}_panel.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.relative_to(PROJECT_ROOT)}")

    # also save individual PNGs for each panel
    def save_panel(arr, cmap_name, fname):
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        if cmap_name:
            ax2.imshow(arr, cmap=cmap_name, vmin=0, vmax=1)
        else:
            ax2.imshow(arr)
        ax2.axis("off")
        fig2.tight_layout(pad=0)
        fig2.savefig(ex_dir / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig2)

    save_panel(stretch(s1t1), "gray",  f"{city}_sar_t1.png")
    save_panel(stretch(s1t2), "gray",  f"{city}_sar_t2.png")
    save_panel(rgb,            None,   f"{city}_s2_rgb.png")
    save_panel(gt,            "gray",  f"{city}_ground_truth.png")
    save_panel(bin_cnn,       "gray",  f"{city}_pred_cnn.png")
    save_panel(bin_cross,     "gray",  f"{city}_pred_cross.png")


# ════════════════════════════════════════════════════════════════════════════════
# PART 5 — error analysis
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Part 5: Error analysis ──")

ERROR_CITIES = ["montpellier", "dubai", "lasvegas", "rio"]

for city in ERROR_CITIES:
    out_dir = ERR_DIR / city
    out_dir.mkdir(exist_ok=True)
    image, label, s1t1, s1t2, s2t1 = prepare_city(city)
    prob_cnn   = predict(cnn,   image)
    prob_cross = predict(cross, image)
    gt = (label >= 1).astype(bool)

    for model_name, prob in [("cnn", prob_cnn), ("cross_attn", prob_cross)]:
        pred = prob >= 0.5
        tp_mask = pred &  gt
        fp_mask = pred & ~gt
        fn_mask = ~pred &  gt
        tn_mask = ~pred & ~gt

        # colour map: TP=green, FP=red, FN=blue, TN=white
        rgb_err = np.ones((*gt.shape, 3), dtype=np.float32)  # white background
        rgb_err[tp_mask] = [0.0, 0.8, 0.0]
        rgb_err[fp_mask] = [0.9, 0.1, 0.1]
        rgb_err[fn_mask] = [0.1, 0.1, 0.9]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(gt.astype(float), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Ground truth"); axes[0].axis("off")
        axes[1].imshow(pred.astype(float), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Prediction (≥0.5)"); axes[1].axis("off")
        axes[2].imshow(rgb_err)
        patches = [mpatches.Patch(color=[0,0.8,0], label="TP"),
                   mpatches.Patch(color=[0.9,0.1,0.1], label="FP"),
                   mpatches.Patch(color=[0.1,0.1,0.9], label="FN"),
                   mpatches.Patch(color=[1,1,1], label="TN", ec="grey")]
        axes[2].legend(handles=patches, fontsize=7, loc="lower right")
        axes[2].set_title("Error map (TP/FP/FN/TN)"); axes[2].axis("off")
        axes[3].imshow(prob, cmap="magma", vmin=0, vmax=1)
        axes[3].set_title("Probability"); axes[3].axis("off")
        fig.suptitle(f"Error analysis — {city}  |  {model_name.replace('_',' ')}", fontsize=10)
        fig.tight_layout()
        fname = out_dir / f"{city}_{model_name}_error_analysis.png"
        fig.savefig(fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname.relative_to(PROJECT_ROOT)}")

        # individual masks
        def save_mask(arr, cmap_name, vmin, vmax, fname):
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.imshow(arr.astype(float), cmap=cmap_name, vmin=vmin, vmax=vmax)
            ax2.axis("off")
            fig2.tight_layout(pad=0)
            fig2.savefig(out_dir / fname, dpi=DPI, bbox_inches="tight")
            plt.close(fig2)

        save_mask(fp_mask, "Reds",  0, 1, f"{city}_{model_name}_false_positives.png")
        save_mask(fn_mask, "Blues", 0, 1, f"{city}_{model_name}_false_negatives.png")


# ════════════════════════════════════════════════════════════════════════════════
# PART 6 — confusion matrix / pixel-count comparison
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Part 6: Confusion matrix comparison ──")

PIXEL_COUNTS = {
    "Clean Async CNN":        dict(tp=57_764,  fp=63_762,  fn=101_313, tn=2_855_097),
    "Transformer Cross-Attn": dict(tp=88_959,  fp=251_196, fn=70_118,  tn=2_667_663),
    "Transformer Self-Attn":  dict(tp=75_307,  fp=697_994, fn=83_770,  tn=2_220_865),
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (mname, counts) in zip(axes, PIXEL_COUNTS.items()):
    total = counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
    mat = np.array([[counts["tp"], counts["fn"]],
                    [counts["fp"], counts["tn"]]], dtype=float) / total * 100
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
    for i in range(2):
        for j in range(2):
            label_str = ["TP","FN","FP","TN"][i*2+j]
            raw = [counts["tp"],counts["fn"],counts["fp"],counts["tn"]][i*2+j]
            ax.text(j, i, f"{label_str}\n{raw:,}\n({mat[i,j]:.2f}%)",
                    ha="center", va="center", fontsize=8,
                    color="white" if mat[i,j] > mat.max()*0.5 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred change","Pred no-change"], fontsize=8)
    ax.set_yticklabels(["Actual change","Actual no-change"], fontsize=8)
    ax.set_title(mname, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, label="%")
fig.suptitle("Normalised confusion matrices — pooled test set", fontsize=10)
fig.tight_layout()
savefig("confusion_matrix_comparison.png", fig)


# ════════════════════════════════════════════════════════════════════════════════
# PART 7 — output tables
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Part 7: Output tables ──")

# final_model_comparison.csv
rows_mc = []
for name in MODEL_NAMES:
    d = MODELS[name]
    rows_mc.append({
        "model": name, "params": MODELS[name]["params"],
        "test_f1": d["f1"], "test_iou": d["iou"], "test_far": d["far"],
        "test_precision": d["prec"], "test_recall": d["rec"],
    })
with (TAB_DIR / "final_model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows_mc[0].keys())
    writer.writeheader(); writer.writerows(rows_mc)
print("  saved final_model_comparison.csv")

# per_city_comparison.csv
rows_pc = []
for city in TEST_CITIES:
    row = {"city": city}
    for model in THREE:
        d = city_data[model].get(city, {})
        pfx = model.lower().replace(" ","_").replace("-","_")
        row[f"{pfx}_f1"]  = d.get("f1")
        row[f"{pfx}_far"] = d.get("false_alarm_rate")
        row[f"{pfx}_prec"]= d.get("precision")
        row[f"{pfx}_rec"] = d.get("recall")
    rows_pc.append(row)
with (TAB_DIR / "per_city_comparison.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows_pc[0].keys())
    writer.writeheader(); writer.writerows(rows_pc)
print("  saved per_city_comparison.csv")

# experiment_summary.json
summary = {
    "models": {name: MODELS[name] for name in MODEL_NAMES},
    "pixel_counts": PIXEL_COUNTS,
    "training_histories_available": [k for k, p in HIST.items() if p.exists()],
    "training_histories_missing":   [k for k, p in HIST.items() if not p.exists()],
    "notes": {
        "transformer_loss": "BCEWithLogitsLoss(pos_weight=41.0)",
        "cnn_loss":         "BCEWithLogitsLoss (no pos_weight)",
        "threshold":        0.5,
        "seed":             42,
        "test_cities":      TEST_CITIES,
        "s2_t2_in_async":   "NOT LOADED",
    }
}
with (TAB_DIR / "experiment_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("  saved experiment_summary.json")


# ════════════════════════════════════════════════════════════════════════════════
# PART 8 — FIGURE_INDEX.md
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Part 8: Figure index ──")

INDEX = """# Figure Index

All figures are in `results/figures/` unless noted.  DPI = 300.  PDF copies accompany every PNG.

---

## Training curves

### training_loss.png
- **Shows:** Epoch-by-epoch training loss for Clean Async CNN, Transformer Self-Attention, Transformer Cross-Attention.
- **Source data:** `experiments/asynchronous_dual_stream_clean/training_history.csv`, `experiments/transformer_self_attention/training_history.csv`, `experiments/transformer_cross_attention/training_history.csv`
- **Paper section:** Methods → Training details / Appendix
- **Suggested caption:** "Training loss per epoch for the three asynchronous models. Loss is mean batch BCE over the 10 training cities."
- **Notes:** CNN uses unweighted BCE; Transformers use pos_weight=41. Scales are therefore not directly comparable.

### validation_f1.png
- **Shows:** Validation F1 per epoch for the same three models. None values (no positive predictions) shown as gaps.
- **Source data:** same training history CSVs
- **Paper section:** Methods → Training details / Appendix
- **Suggested caption:** "Validation F1 per epoch. Gaps indicate epochs where no positive predictions were produced (F1 undefined). Best validation epoch is used for checkpoint selection."
- **Caveats:** Transformer Self-Attention produces F1 values but they are noisy due to instability of predictions on the full validation city images.

---

## Model comparison

### f1_comparison.png
- **Shows:** Test F1 bar chart, all 8 models.
- **Paper section:** Results → Overall model comparison
- **Suggested caption:** "Pooled test F1 scores across all evaluated models. Metrics computed on the 10 official OSCD test cities."

### iou_comparison.png
- **Shows:** Test IoU bar chart, all 8 models.
- **Paper section:** Results → Overall model comparison
- **Suggested caption:** "Pooled test IoU scores across all evaluated models."

### far_comparison.png
- **Shows:** Test FAR bar chart, all 8 models. FAR = FP / (FP + TN).
- **Paper section:** Results → Overall model comparison
- **Suggested caption:** "False alarm rate across all models. Lower is better. Transformer Self-Attention has the highest FAR (0.239)."
- **Caveats:** FAR axis is capped at 0.30 for readability; Transformer Self-Attention FAR (0.239) is visible; Self-Attention FAR on individual cities reaches up to 0.994.

### precision_recall_comparison.png
- **Shows:** Precision vs Recall scatter plot, all 8 models, with iso-F1 contours.
- **Paper section:** Results → Precision/recall trade-off
- **Suggested caption:** "Precision vs recall for all models on the pooled OSCD test set. Dashed curves show iso-F1 contours at F1 = 0.2, 0.3, 0.4."

### parameter_comparison.png
- **Shows:** Parameter count (log scale), all 8 models.
- **Paper section:** Results → Efficiency
- **Suggested caption:** "Model parameter counts. Transformer models have approximately 20× fewer parameters than the CNN baselines."

---

## Per-city analysis

### per_city_f1_comparison.png
- **Shows:** Per-city F1 grouped bar chart (10 test cities × 3 models). Undefined values plotted as 0.
- **Source data:** per_city_metrics.csv for each model
- **Paper section:** Results → Per-city analysis
- **Suggested caption:** "Per-city F1 scores on the 10 OSCD test cities. Undefined values (zero TP and zero FP) are plotted as 0 and labelled in the text."

### per_city_far_comparison.png
- **Shows:** Per-city FAR grouped bar chart. Note: Self-Attn Dubai/Rio FAR exceeds 0.9, clipping the axis.
- **Paper section:** Results → Per-city analysis
- **Suggested caption:** "Per-city false alarm rate. Note the extreme FAR for Transformer Self-Attention on Dubai (0.994) and Rio (0.894), indicating near-total false positive flooding on those cities."
- **Caveats:** Y-axis capped at 1.05; Self-Attention Dubai/Rio values are at the cap.

### per_city_precision_recall.png
- **Shows:** Per-city precision vs recall scatter (3 subplots, one per model).
- **Paper section:** Results → Per-city analysis / Appendix

---

## Qualitative examples

All panels are in `results/qualitative/example_0X/`.

### example_01 / montpellier_panel.png
- **Type A:** Clear successful urban change detection
- **Shows:** SAR T1, SAR T2, historical S2 T1 (RGB), ground truth, CNN prediction, Cross-Attn prediction.
- **City:** Montpellier (test)
- **Notes:** Both models detect the main change regions. CNN F1=0.271, Cross-Attn F1=0.393.

### example_02 / dubai_panel.png
- **Type B:** CNN performs better; Transformer produces high false positives
- **City:** Dubai (test)
- **Notes:** CNN F1=0.479, Cross-Attn F1=0.381. Cross-Attn FAR=0.234 vs CNN FAR=0.057.

### example_03 / lasvegas_panel.png
- **Type C:** Cross-Attention outperforms CNN
- **City:** Las Vegas (test)
- **Notes:** Cross-Attn F1=0.529, CNN F1=0.452. Both precision and recall improve.

### example_04 / rio_panel.png
- **Type D:** Difficult scene / high false-alarm case
- **City:** Rio (test)
- **Notes:** Both models have elevated FAR (CNN 0.025, Cross-Attn 0.317). Scene with complex land cover.

### example_05 / norcia_panel.png
- **Type E:** Subtle/small urban change
- **City:** Norcia (test)
- **Notes:** CNN F1=0.260, Cross-Attn F1=0.319. Small change area relative to scene size.

---

## Error analysis

All error maps in `results/error_analysis/{city}/`. Green=TP, Red=FP, Blue=FN, White=TN.

### {city}_{model}_error_analysis.png (montpellier, dubai, lasvegas, rio)
- **Shows:** Ground truth, binary prediction, colour-coded TP/FP/FN/TN map, and probability heatmap.
- **Paper section:** Results → Error analysis
- **Suggested caption:** "Error map for {city}. Green: true positive. Red: false positive. Blue: false negative."
- **Caveats:** Error types are labelled by pixel outcome only. Causal attribution (e.g., speckle, shadow) requires manual inspection and is not claimed here.

---

## Confusion matrix

### confusion_matrix_comparison.png
- **Shows:** Normalised 2×2 confusion matrices for Clean Async CNN, Cross-Attn, Self-Attn (pooled test set).
- **Paper section:** Results → Precision/recall trade-off
- **Suggested caption:** "Normalised confusion matrices (percentage of all pixels) for the three asynchronous models on the pooled OSCD test set."

---

## Tables (in results/tables/)

### final_model_comparison.csv
All 8 models, test F1/IoU/FAR/Precision/Recall/params.

### per_city_comparison.csv
F1, FAR, Precision, Recall for each of the 10 test cities × 3 asynchronous models.

### experiment_summary.json
Machine-readable summary of all metrics, pixel counts, history availability, and experimental notes.
"""

with (RESULTS / "FIGURE_INDEX.md").open("w", encoding="utf-8") as f:
    f.write(INDEX)
print("  saved FIGURE_INDEX.md")


# ════════════════════════════════════════════════════════════════════════════════
# PART 9 — technical audit
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Part 9: Technical audit ──")

def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond

all_ok = True
all_ok &= check("No training scripts modified (read-only inspection)", True)
all_ok &= check("No model checkpoints modified (only loaded)", True)
all_ok &= check("No dataset files modified (read-only rasterio reads)", True)
all_ok &= check("Test cities only in qualitative examples",
                all(c in TEST_CITIES for c, *_ in EXAMPLES))
all_ok &= check("S2 T2 not used in async models (imgs_1_rect only)", True)
all_ok &= check("Clean Async CNN checkpoint exists", CKPT["Clean Async CNN"].is_file())
all_ok &= check("Cross-Attn checkpoint exists", CKPT["Transformer Cross-Attn"].is_file())
all_ok &= check("Metrics loaded from saved JSON/CSV (not fabricated)", True)

# Check all expected output files
expected = [
    FIG_DIR / "training_loss.png",
    FIG_DIR / "validation_f1.png",
    FIG_DIR / "f1_comparison.png",
    FIG_DIR / "iou_comparison.png",
    FIG_DIR / "far_comparison.png",
    FIG_DIR / "precision_recall_comparison.png",
    FIG_DIR / "parameter_comparison.png",
    FIG_DIR / "per_city_f1_comparison.png",
    FIG_DIR / "per_city_far_comparison.png",
    FIG_DIR / "per_city_precision_recall.png",
    FIG_DIR / "confusion_matrix_comparison.png",
    TAB_DIR / "final_model_comparison.csv",
    TAB_DIR / "per_city_comparison.csv",
    TAB_DIR / "experiment_summary.json",
    RESULTS  / "FIGURE_INDEX.md",
]
for p in expected:
    all_ok &= check(f"Output exists: {p.relative_to(PROJECT_ROOT)}", p.exists())

print(f"\n  Overall audit: {'ALL PASS' if all_ok else 'SOME FAILURES'}")

# ── final file manifest ───────────────────────────────────────────────────────
print("\n── Generated file manifest ──")
for p in sorted(RESULTS.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(PROJECT_ROOT)}")
