"""Scenario 2 valid-pixel inspection workflow.

READ-ONLY: never modifies checkpoints, split, labels, or reported metrics.

Tasks
-----
1. Compute per-site SAR validity masks (pre-fill NaN map) for val + test sites.
2. Report valid/invalid pixel counts per site, per SAR date.
3. Run inference with all three best.pt checkpoints and compute:
     (a) all-pixel metrics   -- matches original reported metrics exactly
     (b) valid-pixel metrics -- restricted to pixels valid in BOTH SAR dates
   Both protocols use threshold=0.5. No threshold tuning from test data.
4. Export overlay PNGs: SAR T1-VV, ground truth, probability, prediction,
   error map (TP/FP/FN/TN/invalid), FP-source breakdown.
5. Deep-dive on L15-1289E cross-attention high-FP site.

Outputs
-------
results/scenario_2/valid_pixel_inspection/
  pixel_validity_report.csv
  metrics_comparison.csv
  pooled_summary.csv
  {site}_{model}_overlay.png
  {site}_{model}_validity.png
  l15_1289_cross_attn_deep_dive.png

Run from project root:
    .venv\\Scripts\\python.exe src\\training\\inspect_valid_pixels.py
"""

import csv
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

import rasterio
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

from src.datasets.scenario2_dataset import (
    Scenario2Dataset, load_split,
    SAR_CHANNELS, OPT_CHANNELS,
    T1_YEAR, T1_MONTH, T2_YEAR, T2_MONTH,
    _s1_path,
)
from src.models.dual_stream_unet_clean import DualStreamUNetClean
from src.models.dual_stream_unet import DualStreamUNet
from src.models.transformer_cross_attention import TransformerCrossAttention
from src.utils.device import get_device

OUT_DIR = PROJECT_ROOT / "results" / "scenario_2" / "valid_pixel_inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI    = 150
device = get_device("auto")

split      = load_split()
VAL_SITES  = split["validation"]
TEST_SITES = split["test"]

EXPERIMENTS = {
    "async_cnn": {
        "ckpt":  PROJECT_ROOT / "checkpoints/scenario_2/async_dual_stream/best.pt",
        "mode":  "async",
        "label": "Async Dual-Stream CNN",
    },
    "sync_cnn": {
        "ckpt":  PROJECT_ROOT / "checkpoints/scenario_2/sync_dual_stream/best.pt",
        "mode":  "sync",
        "label": "Sync Dual-Stream CNN",
    },
    "cross_attn": {
        "ckpt":  PROJECT_ROOT / "checkpoints/scenario_2/cross_attention/best.pt",
        "mode":  "async",
        "label": "Cross-Attn Transformer",
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def read_s1_raw(path):
    """Return (H, W, 2) float32, NaN preserved (no fill)."""
    with rasterio.open(path) as ds:
        arr = ds.read().astype(np.float32)
    return arr.transpose(1, 2, 0)


def sar_validity_stats(site_id):
    """Return dict of NaN stats and (H,W) bool validity mask."""
    s1_t1 = read_s1_raw(_s1_path(site_id, T1_YEAR, T1_MONTH))
    s1_t2 = read_s1_raw(_s1_path(site_id, T2_YEAR, T2_MONTH))
    valid_t1 = np.isfinite(s1_t1).all(axis=2)   # (H,W) bool
    valid_t2 = np.isfinite(s1_t2).all(axis=2)
    valid_both = valid_t1 & valid_t2
    total = s1_t1.shape[0] * s1_t1.shape[1]
    return {
        "h": s1_t1.shape[0], "w": s1_t1.shape[1], "total": total,
        "inv_t1": int((~valid_t1).sum()),
        "inv_t2": int((~valid_t2).sum()),
        "invalid_either": int((~valid_both).sum()),
        "valid_both": int(valid_both.sum()),
        "pct_valid_both": 100.0 * valid_both.mean(),
        "mask": valid_both,
    }


def load_model(exp_key):
    cfg   = EXPERIMENTS[exp_key]
    saved = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    if exp_key == "async_cnn":
        m = DualStreamUNetClean(base_channels=64, sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS)
    elif exp_key == "sync_cnn":
        m = DualStreamUNet(base_channels=64, sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS * 2)
    else:
        m = TransformerCrossAttention(sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS)
    m.load_state_dict(saved["model_state_dict"])
    return m.to(device).eval()


def pixel_counts(pred_bool, gt_bool, mask_bool=None):
    """Return TP/TN/FP/FN, optionally restricted to mask pixels."""
    if mask_bool is not None:
        p = pred_bool[mask_bool]
        g = gt_bool[mask_bool]
    else:
        p, g = pred_bool, gt_bool
    return {
        "tp": int(( p &  g).sum()),
        "tn": int((~p & ~g).sum()),
        "fp": int(( p & ~g).sum()),
        "fn": int((~p &  g).sum()),
    }


def derived_rates(c):
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    def r(n, d): return n / d if d > 0 else None
    prec = r(tp, tp + fp)
    rec  = r(tp, tp + fn)
    f1   = (2 * prec * rec / (prec + rec)) if (prec and rec and prec + rec > 0) else None
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "iou": r(tp, tp + fp + fn),
        "false_alarm_rate": r(fp, fp + tn),
    }


def fmt(v, d=4):
    return "—" if v is None else f"{v:.{d}f}"


def pool_counts(rows, prefix):
    acc = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for row in rows:
        for k in acc:
            key = prefix + "_" + k
            val = row[key]
            if val is not None:
                acc[k] += val
    return acc


def prefix_dict(d, pfx):
    """Return a new dict with every key prefixed by pfx."""
    return {pfx + "_" + k: v for k, v in d.items()}


def stretch(arr):
    lo, hi = np.percentile(arr, (2, 98))
    if hi <= lo: hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  SAR validity report
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("STEP 1: SAR validity per site (validation + test)")
print("=" * 72)

validity_cache = {}   # site_id -> (H,W) bool mask
validity_rows  = []

hdr = ("Site", "Total px", "Inv T1", "Inv T2", "Inv either", "Valid both", "% valid")
print(f"\n  {hdr[0]:<38} {hdr[1]:>9} {hdr[2]:>8} {hdr[3]:>8} {hdr[4]:>11} {hdr[5]:>11} {hdr[6]:>8}")

for sname, sites in [("validation", VAL_SITES), ("test", TEST_SITES)]:
    print(f"\n  -- {sname} --")
    for sid in sites:
        s = sar_validity_stats(sid)
        validity_cache[sid] = s["mask"]
        short = sid[-20:]
        print(f"  {short:<38} {s['total']:>9,} {s['inv_t1']:>8,} {s['inv_t2']:>8,} "
              f"{s['invalid_either']:>11,} {s['valid_both']:>11,} {s['pct_valid_both']:>7.1f}%")
        row = {
            "split": sname, "site_id": sid,
            "total_pixels": s["total"],
            "invalid_t1": s["inv_t1"],
            "invalid_t2": s["inv_t2"],
            "invalid_either": s["invalid_either"],
            "valid_both": s["valid_both"],
            "pct_valid_both": round(s["pct_valid_both"], 3),
        }
        validity_rows.append(row)

vr_path = OUT_DIR / "pixel_validity_report.csv"
with vr_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(validity_rows[0].keys()))
    writer.writeheader()
    writer.writerows(validity_rows)
print(f"\n  Saved: pixel_validity_report.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  Inference + dual-protocol metrics on test sites
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("STEP 2: Inference — all-pixel vs valid-pixel metrics (test sites)")
print("=" * 72)

ds_async = Scenario2Dataset(split="test", mode="async")
ds_sync  = Scenario2Dataset(split="test", mode="sync")

metrics_rows = []

for exp_key, cfg in EXPERIMENTS.items():
    model = load_model(exp_key)
    saved = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    epoch = saved.get("epoch", "?")
    ds    = ds_async if cfg["mode"] == "async" else ds_sync
    print(f"\n  {cfg['label']}  (best.pt epoch={epoch})")

    for sid in TEST_SITES:
        idx    = ds.sites.index(sid)
        sample = ds[idx]
        image  = sample["image"].unsqueeze(0).to(device)
        label  = sample["label"].unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(image)

        prob  = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred  = (prob >= 0.5)
        gt    = (sample["label"][0].numpy() >= 0.5)
        valid = validity_cache[sid]

        all_c = pixel_counts(pred, gt)
        val_c = pixel_counts(pred, gt, valid)
        all_r = derived_rates(all_c)
        val_r = derived_rates(val_c)

        pct_v = validity_rows[[r["site_id"] for r in validity_rows].index(sid)]["pct_valid_both"]
        short = sid[-20:]
        print(f"    {short}:")
        print(f"      all-pixel : TP={all_c['tp']:5d} FP={all_c['fp']:6d} FN={all_c['fn']:5d}"
              f"  F1={fmt(all_r['f1'])}  FAR={fmt(all_r['false_alarm_rate'])}")
        print(f"      valid-px  : TP={val_c['tp']:5d} FP={val_c['fp']:6d} FN={val_c['fn']:5d}"
              f"  F1={fmt(val_r['f1'])}  FAR={fmt(val_r['false_alarm_rate'])}"
              f"  ({pct_v:.1f}% valid)")

        row = {"model": exp_key, "site_id": sid}
        row.update(prefix_dict(all_c, "all"))
        row.update(prefix_dict(all_r, "all"))
        row.update(prefix_dict(val_c, "val"))
        row.update(prefix_dict(val_r, "val"))
        row["valid_px"]   = int(valid.sum())
        row["invalid_px"] = int((~valid).sum())
        row["pct_valid"]  = round(100 * valid.mean(), 3)
        metrics_rows.append(row)

mc_path = OUT_DIR / "metrics_comparison.csv"
with mc_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
    writer.writeheader()
    writer.writerows(metrics_rows)
print(f"\n  Saved: metrics_comparison.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  Pooled comparison table
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("STEP 3: Pooled test metrics — all-pixel vs valid-pixel")
print("=" * 72)

print(f"\n  {'Model':<23} {'Protocol':<13} {'Prec':>8} {'Rec':>8} {'F1':>8} {'IoU':>8} {'FAR':>8}")
print("  " + "-" * 72)

pooled_rows = []
for exp_key in EXPERIMENTS:
    subset = [r for r in metrics_rows if r["model"] == exp_key]
    for proto, pfx in [("all-pixel", "all"), ("valid-px", "val")]:
        c = pool_counts(subset, pfx)
        r = derived_rates(c)
        print(f"  {EXPERIMENTS[exp_key]['label']:<23} {proto:<13} "
              f"{fmt(r['precision']):>8} {fmt(r['recall']):>8} "
              f"{fmt(r['f1']):>8} {fmt(r['iou']):>8} "
              f"{fmt(r['false_alarm_rate']):>8}")
        prow = {"model": exp_key, "protocol": proto}
        prow.update(c)
        prow.update(r)
        pooled_rows.append(prow)
    print()

ps_path = OUT_DIR / "pooled_summary.csv"
with ps_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(pooled_rows[0].keys()))
    writer.writeheader()
    writer.writerows(pooled_rows)
print(f"  Saved: pooled_summary.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  Overlay PNGs for all test sites x all models
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("STEP 4: Overlay PNGs (test sites)")
print("=" * 72)

for exp_key, cfg in EXPERIMENTS.items():
    model = load_model(exp_key)
    ds    = ds_async if cfg["mode"] == "async" else ds_sync

    for sid in TEST_SITES:
        idx    = ds.sites.index(sid)
        sample = ds[idx]
        image  = sample["image"].unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(image)

        prob  = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred  = (prob >= 0.5)
        gt    = (sample["label"][0].numpy() >= 0.5)
        valid = validity_cache[sid]
        sar_vv = sample["image"][0].numpy()   # VV band of SAR T1

        # Error map: TP=green, FP=red, FN=blue, TN=white, invalid=grey
        rgb = np.ones((*gt.shape, 3), dtype=np.float32)
        rgb[pred &  gt]  = [0.0,  0.75, 0.0]
        rgb[pred & ~gt]  = [0.85, 0.1,  0.1]
        rgb[~pred & gt]  = [0.1,  0.1,  0.85]
        rgb[~valid]      = [0.6,  0.6,  0.6]

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        axes[0].imshow(stretch(sar_vv), cmap="gray"); axes[0].axis("off")
        axes[0].set_title("SAR T1-VV", fontsize=7)

        axes[1].imshow(gt.astype(float), cmap="gray", vmin=0, vmax=1)
        axes[1].axis("off")
        axes[1].set_title("Ground truth\n" + str(int(gt.sum())) + " px", fontsize=7)

        axes[2].imshow(prob, cmap="magma", vmin=0, vmax=1)
        axes[2].axis("off")
        axes[2].set_title("Probability\n" + cfg["label"], fontsize=7)

        axes[3].imshow(pred.astype(float), cmap="gray", vmin=0, vmax=1)
        axes[3].axis("off")
        axes[3].set_title("Pred >= 0.5\n+" + str(int(pred.sum())), fontsize=7)

        axes[4].imshow(rgb); axes[4].axis("off")
        patches = [
            mpatches.Patch(color=[0, 0.75, 0],    label="TP"),
            mpatches.Patch(color=[0.85, 0.1, 0.1], label="FP"),
            mpatches.Patch(color=[0.1, 0.1, 0.85], label="FN"),
            mpatches.Patch(color=[1, 1, 1],        label="TN", ec="grey"),
            mpatches.Patch(color=[0.6, 0.6, 0.6],  label="Invalid SAR"),
        ]
        axes[4].legend(handles=patches, fontsize=6, loc="lower right")
        axes[4].set_title("Error map\n(grey=invalid SAR)", fontsize=7)

        fig.suptitle(cfg["label"] + " | " + sid, fontsize=8)
        fig.tight_layout()
        short = sid[-20:]
        fname = OUT_DIR / (short + "_" + exp_key + "_overlay.png")
        fig.savefig(fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: " + fname.name)

        # Validity-only figure
        cmap_v = matplotlib.colors.ListedColormap(["#cc3333", "#33aa33"])
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.imshow(valid.astype(int), cmap=cmap_v, vmin=0, vmax=1)
        ax2.set_title("SAR validity mask\n" + short + "\nvalid=" +
                      str(int(valid.sum())) + "/" + str(valid.size) +
                      " (" + str(round(100 * valid.mean(), 1)) + "%)", fontsize=8)
        ax2.axis("off")
        pv = [mpatches.Patch(color="#33aa33", label="Valid"),
              mpatches.Patch(color="#cc3333", label="Invalid (NaN)")]
        ax2.legend(handles=pv, fontsize=8, loc="lower right")
        fname2 = OUT_DIR / (short + "_" + exp_key + "_validity.png")
        fig2.savefig(fname2, dpi=DPI, bbox_inches="tight")
        plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  Deep-dive: L15-1289E cross-attention high-FP
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("STEP 5: Deep-dive — L15-1289E cross-attention FP source")
print("=" * 72)

TARGET = "L15-1289E-1169N_5156_3514_13"
valid_1289 = validity_cache[TARGET]

models_dd = {
    "async_cnn":  load_model("async_cnn"),
    "sync_cnn":   load_model("sync_cnn"),
    "cross_attn": load_model("cross_attn"),
}

idx_a = ds_async.sites.index(TARGET)
idx_s = ds_sync.sites.index(TARGET)
samp_a = ds_async[idx_a]
samp_s = ds_sync[idx_s]
gt_1289 = (samp_a["label"][0].numpy() >= 0.5)
sar_vv_1289 = samp_a["image"][0].numpy()

print(f"\n  Site: {TARGET}")
print(f"  Total pixels:   {gt_1289.size:,}")
print(f"  Change pixels:  {int(gt_1289.sum()):,}  ({100*gt_1289.mean():.3f}%)")
print(f"  Valid SAR px:   {int(valid_1289.sum()):,}  ({100*valid_1289.mean():.1f}%)")
print(f"  Invalid SAR px: {int((~valid_1289).sum()):,}  ({100*(~valid_1289).mean():.1f}%)")
print()

probs_dd = {}
for name, mdl in models_dd.items():
    samp = samp_a if name != "sync_cnn" else samp_s
    img  = samp["image"].unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = mdl(img)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    probs_dd[name] = prob

    pred = prob >= 0.5
    all_c  = pixel_counts(pred, gt_1289)
    val_c  = pixel_counts(pred, gt_1289, valid_1289)
    all_r  = derived_rates(all_c)
    val_r  = derived_rates(val_c)

    fp_in_invalid = int(pred[~valid_1289].sum())
    fp_in_valid   = val_c["fp"]
    total_fp      = all_c["fp"]
    pct_fp_inv    = 100 * fp_in_invalid / max(1, total_fp)
    pct_fp_val    = 100 * fp_in_valid   / max(1, total_fp)

    print(f"  {name}:")
    print(f"    All  pixels: TP={all_c['tp']:5d} FP={total_fp:6d} FN={all_c['fn']:5d}"
          f"  FAR={fmt(all_r['false_alarm_rate'])}  F1={fmt(all_r['f1'])}")
    print(f"    Valid pixels: TP={val_c['tp']:5d} FP={fp_in_valid:6d} FN={val_c['fn']:5d}"
          f"  FAR={fmt(val_r['false_alarm_rate'])}  F1={fmt(val_r['f1'])}")
    print(f"    FP in invalid SAR region: {fp_in_invalid:6,} ({pct_fp_inv:.1f}% of all FP)")
    print(f"    FP in valid   SAR region: {fp_in_valid:6,} ({pct_fp_val:.1f}% of all FP)")

    prob_inv = prob[~valid_1289]
    prob_val = prob[ valid_1289]
    print(f"    Prob (invalid SAR): mean={prob_inv.mean():.4f} med={np.median(prob_inv):.4f} "
          f"pct>0.5={100*(prob_inv>=0.5).mean():.2f}%")
    print(f"    Prob (valid SAR):   mean={prob_val.mean():.4f} med={np.median(prob_val):.4f} "
          f"pct>0.5={100*(prob_val>=0.5).mean():.2f}%")
    print()

# Deep-dive figure: 3 models x 6 columns
fig, axes = plt.subplots(3, 6, figsize=(22, 12))
model_list = [("async_cnn", samp_a, probs_dd["async_cnn"]),
              ("sync_cnn",  samp_s, probs_dd["sync_cnn"]),
              ("cross_attn", samp_a, probs_dd["cross_attn"])]

for row_i, (name, samp, prob) in enumerate(model_list):
    pred = prob >= 0.5

    err_rgb = np.ones((*gt_1289.shape, 3), dtype=np.float32)
    err_rgb[pred &  gt_1289] = [0.0, 0.75, 0.0]
    err_rgb[pred & ~gt_1289] = [0.85, 0.1, 0.1]
    err_rgb[~pred & gt_1289] = [0.1, 0.1, 0.85]
    err_rgb[~valid_1289]     = [0.6, 0.6, 0.6]

    fp_valid_map   = pred & ~gt_1289 &  valid_1289
    fp_invalid_map = pred & ~gt_1289 & ~valid_1289
    fp_src = np.zeros((*gt_1289.shape, 3), dtype=np.float32)
    fp_src[fp_valid_map]   = [0.9, 0.2, 0.2]
    fp_src[fp_invalid_map] = [0.9, 0.6, 0.0]

    lbl  = EXPERIMENTS[name]["label"]
    n_fp_val = int(fp_valid_map.sum())
    n_fp_inv = int(fp_invalid_map.sum())

    axes[row_i, 0].imshow(stretch(sar_vv_1289), cmap="gray")
    axes[row_i, 0].axis("off")
    axes[row_i, 0].set_title("SAR T1-VV\n" + lbl, fontsize=7)

    axes[row_i, 1].imshow(gt_1289.astype(float), cmap="gray", vmin=0, vmax=1)
    axes[row_i, 1].axis("off")
    axes[row_i, 1].set_title("Ground truth\n" + str(int(gt_1289.sum())) + " px", fontsize=7)

    axes[row_i, 2].imshow(prob, cmap="magma", vmin=0, vmax=1)
    axes[row_i, 2].axis("off")
    axes[row_i, 2].set_title("Probability", fontsize=7)

    axes[row_i, 3].imshow(pred.astype(float), cmap="gray", vmin=0, vmax=1)
    axes[row_i, 3].axis("off")
    axes[row_i, 3].set_title("Pred >=0.5\n+" + str(int(pred.sum())), fontsize=7)

    axes[row_i, 4].imshow(err_rgb)
    axes[row_i, 4].axis("off")
    axes[row_i, 4].set_title("Error map\n(grey=invalid)", fontsize=7)

    axes[row_i, 5].imshow(fp_src)
    axes[row_i, 5].axis("off")
    p1 = mpatches.Patch(color=[0.9,0.2,0.2], label="FP valid SAR: " + str(n_fp_val))
    p2 = mpatches.Patch(color=[0.9,0.6,0.0], label="FP invalid SAR: " + str(n_fp_inv))
    axes[row_i, 5].legend(handles=[p1, p2], fontsize=6, loc="lower right")
    axes[row_i, 5].set_title("FP by SAR validity", fontsize=7)

fig.suptitle("L15-1289E: FP source analysis — grey=invalid SAR / orange=FP-in-invalid / red=FP-in-valid",
             fontsize=9)
fig.tight_layout()
dd_path = OUT_DIR / "l15_1289_cross_attn_deep_dive.png"
fig.savefig(dd_path, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  Saved: l15_1289_cross_attn_deep_dive.png")

print()
print("=" * 72)
print("Inspection complete.")
print("All original metrics.json files are unchanged.")
print("Outputs: results/scenario_2/valid_pixel_inspection/")
print("=" * 72)
