"""Phase B: Generate side-by-side visualizations and error analysis for problematic site.
Site: L15-1289E-1169N_5156_3514_13
Outputs saved in experiments/scenario_2/error_analysis/
"""
import json
import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.scenario2_dataset import (
    Scenario2Dataset,
    _s1_path, _s2_path, _bld_path,
    T1_YEAR, T1_MONTH, T2_YEAR, T2_MONTH,
    SAR_CHANNELS, OPT_CHANNELS,
)
from src.models.dual_stream_unet_clean import DualStreamUNetClean
from src.models.dual_stream_unet import DualStreamUNet
from src.models.transformer_cross_attention import TransformerCrossAttention
from src.utils.device import get_device

device = get_device("auto")

SITE_ID = "L15-1289E-1169N_5156_3514_13"
OUT_DIR = PROJECT_ROOT / "experiments" / "scenario_2" / "error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(f"PHASE B: VISUALIZATION & ERROR ANALYSIS FOR {SITE_ID}")
print("=" * 80)

# 1. Load raw rasters
p_s1_t1 = _s1_path(SITE_ID, T1_YEAR, T1_MONTH)
p_s1_t2 = _s1_path(SITE_ID, T2_YEAR, T2_MONTH)
p_s2_t1 = _s2_path(SITE_ID, T1_YEAR, T1_MONTH)
p_s2_t2 = _s2_path(SITE_ID, T2_YEAR, T2_MONTH)
p_bld_t1 = _bld_path(SITE_ID, T1_YEAR, T1_MONTH)
p_bld_t2 = _bld_path(SITE_ID, T2_YEAR, T2_MONTH)

with rasterio.open(p_s1_t1) as ds:
    s1_t1_raw = ds.read().astype(np.float32)  # (2, H, W) VV, VH
with rasterio.open(p_s1_t2) as ds:
    s1_t2_raw = ds.read().astype(np.float32)  # (2, H, W) VV, VH
with rasterio.open(p_s2_t1) as ds:
    s2_t1_raw = ds.read().astype(np.float32)  # (10, H, W) B2,B3,B4...
with rasterio.open(p_s2_t2) as ds:
    s2_t2_raw = ds.read().astype(np.float32)  # (10, H, W)
with rasterio.open(p_bld_t1) as ds:
    bld_t1 = ds.read(1).astype(np.float32)
with rasterio.open(p_bld_t2) as ds:
    bld_t2 = ds.read(1).astype(np.float32)

gt_label = ((bld_t1 >= 0.5) ^ (bld_t2 >= 0.5)).astype(bool)

# Validity mask
s1_t1_valid = ~np.any(np.isnan(s1_t1_raw), axis=0)
s1_t2_valid = ~np.any(np.isnan(s1_t2_raw), axis=0)
valid_mask = s1_t1_valid & s1_t2_valid

# S1 nan to 0.0 for model inference
s1_t1 = np.nan_to_num(s1_t1_raw, nan=0.0).transpose(1, 2, 0)
s1_t2 = np.nan_to_num(s1_t2_raw, nan=0.0).transpose(1, 2, 0)
s2_t1 = s2_t1_raw.transpose(1, 2, 0)
s2_t2 = s2_t2_raw.transpose(1, 2, 0)

# Build tensors
sar = np.concatenate([s1_t1, s1_t2], axis=2)  # (H, W, 4)
async_input = np.concatenate([sar, s2_t1], axis=2)  # (H, W, 14)
sync_input  = np.concatenate([sar, s2_t1, s2_t2], axis=2)  # (H, W, 24)

t_async = torch.from_numpy(async_input.transpose(2, 0, 1)).unsqueeze(0).to(device)
t_sync  = torch.from_numpy(sync_input.transpose(2, 0, 1)).unsqueeze(0).to(device)

# 2. Run model predictions
m_async = DualStreamUNetClean(base_channels=64, sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS)
m_async.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints/scenario_2/async_dual_stream/best.pt", map_location="cpu", weights_only=False)["model_state_dict"])
m_async = m_async.to(device).eval()

m_sync = DualStreamUNet(base_channels=64, sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS * 2)
m_sync.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints/scenario_2/sync_dual_stream/best.pt", map_location="cpu", weights_only=False)["model_state_dict"])
m_sync = m_sync.to(device).eval()

m_xattn = TransformerCrossAttention(sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS)
m_xattn.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints/scenario_2/cross_attention/best.pt", map_location="cpu", weights_only=False)["model_state_dict"])
m_xattn = m_xattn.to(device).eval()

with torch.inference_mode():
    pred_async_prob = torch.sigmoid(m_async(t_async)).squeeze().cpu().numpy()
    pred_sync_prob  = torch.sigmoid(m_sync(t_sync)).squeeze().cpu().numpy()
    pred_xattn_prob = torch.sigmoid(m_xattn(t_async)).squeeze().cpu().numpy()

pred_async = pred_async_prob >= 0.5
pred_sync  = pred_sync_prob >= 0.5
pred_xattn = pred_xattn_prob >= 0.5

# 3. Quantify overlap between false-positive regions and invalid SAR pixels
def analyze_errors(name, pred_bin):
    tp_mask = pred_bin & gt_label
    fp_mask = pred_bin & (~gt_label)
    fn_mask = (~pred_bin) & gt_label
    tn_mask = (~pred_bin) & (~gt_label)
    
    fp_total = int(fp_mask.sum())
    fp_on_invalid = int((fp_mask & (~valid_mask)).sum())
    invalid_total = int((~valid_mask).sum())
    
    fp_on_invalid_pct = (fp_on_invalid / fp_total * 100) if fp_total > 0 else 0.0
    invalid_as_fp_pct = (fp_on_invalid / invalid_total * 100) if invalid_total > 0 else 0.0
    
    # Analyze raw SAR and optical statistics inside FP vs TP vs TN regions
    vv_t1 = s1_t1[:, :, 0]
    vv_t2 = s1_t2[:, :, 0]
    sar_diff = np.abs(vv_t2 - vv_t1)
    
    mean_fp_diff = float(np.mean(sar_diff[fp_mask])) if fp_mask.sum() > 0 else 0.0
    mean_tp_diff = float(np.mean(sar_diff[tp_mask])) if tp_mask.sum() > 0 else 0.0
    mean_tn_diff = float(np.mean(sar_diff[tn_mask])) if tn_mask.sum() > 0 else 0.0
    
    return {
        "model": name,
        "tp": int(tp_mask.sum()),
        "fp": fp_total,
        "fn": int(fn_mask.sum()),
        "tn": int(tn_mask.sum()),
        "total_invalid_pixels": invalid_total,
        "fp_on_invalid_pixels": fp_on_invalid,
        "pct_of_fp_on_invalid": fp_on_invalid_pct,
        "pct_of_invalid_predicted_fp": invalid_as_fp_pct,
        "sar_vv_diff_mean_tp": mean_tp_diff,
        "sar_vv_diff_mean_fp": mean_fp_diff,
        "sar_vv_diff_mean_tn": mean_tn_diff,
    }

stats_async = analyze_errors("Async Dual-Stream CNN", pred_async)
stats_sync  = analyze_errors("Sync Dual-Stream CNN", pred_sync)
stats_xattn = analyze_errors("Cross-Attention Transformer", pred_xattn)

error_stats = {
    "site": SITE_ID,
    "total_pixels": int(gt_label.size),
    "total_change_pixels": int(gt_label.sum()),
    "total_invalid_pixels": int((~valid_mask).sum()),
    "models": [stats_async, stats_sync, stats_xattn]
}

print(json.dumps(error_stats, indent=2))
with open(OUT_DIR / "error_overlap_stats.json", "w", encoding="utf-8") as f:
    json.dump(error_stats, f, indent=2)

# 4. Generate Error Overlay Maps
# Helper for error overlay: TP=Green, FP=Red, FN=Blue, TN=White/Light Gray
def make_error_rgb(pred_bin, label_bin):
    h, w = label_bin.shape
    rgb = np.ones((h, w, 3), dtype=np.float32) * 0.95  # light gray TN
    # TP = Green
    tp = pred_bin & label_bin
    rgb[tp] = [0.1, 0.8, 0.2]
    # FP = Red
    fp = pred_bin & (~label_bin)
    rgb[fp] = [0.9, 0.1, 0.1]
    # FN = Blue
    fn = (~pred_bin) & label_bin
    rgb[fn] = [0.1, 0.3, 0.9]
    return rgb

# True Color Optical (B4=idx 2, B3=idx 1, B2=idx 0)
def make_rgb(s2_arr):
    # s2: (H, W, 10): 0:B2, 1:B3, 2:B4...
    rgb = np.stack([s2_arr[:, :, 2], s2_arr[:, :, 1], s2_arr[:, :, 0]], axis=2)
    # Clip and enhance for display
    rgb = np.clip(rgb * 3.5, 0.0, 1.0)
    return rgb

opt_t1_rgb = make_rgb(s2_t1)
opt_t2_rgb = make_rgb(s2_t2)

# SAR False-color: R=VV_T2, G=VV_T1, B=|VV_T2 - VV_T1|
sar_change_rgb = np.stack([
    np.clip(s1_t2[:, :, 0] * 2.5, 0, 1),
    np.clip(s1_t1[:, :, 0] * 2.5, 0, 1),
    np.clip(np.abs(s1_t2[:, :, 0] - s1_t1[:, :, 0]) * 5.0, 0, 1)
], axis=2)

fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=150)
fig.suptitle(f"Error Analysis: {SITE_ID} (Test Site with Transformer FP Surge)", fontsize=16, fontweight="bold", y=0.98)

# Row 1: Inputs and Targets
axes[0, 0].imshow(s1_t1[:, :, 0], cmap="gray", vmin=0, vmax=0.5)
axes[0, 0].set_title("SAR T1 (VV Backscatter)")
axes[0, 0].axis("off")

axes[0, 1].imshow(s1_t2[:, :, 0], cmap="gray", vmin=0, vmax=0.5)
axes[0, 1].set_title("SAR T2 (VV Backscatter)")
axes[0, 1].axis("off")

axes[0, 2].imshow(opt_t1_rgb)
axes[0, 2].set_title("Historical Sentinel-2 T1 (RGB)")
axes[0, 2].axis("off")

axes[0, 3].imshow(gt_label, cmap="gray")
axes[0, 3].set_title(f"Change Proxy Ground Truth ({int(gt_label.sum()):,} px)")
axes[0, 3].axis("off")

# Row 2: Optical T2 reference, SAR composite, Validity Mask, Transformer Probability Heatmap
axes[1, 0].imshow(opt_t2_rgb)
axes[1, 0].set_title("Contemporary S2 T2 (RGB, withheld in async)")
axes[1, 0].axis("off")

axes[1, 1].imshow(sar_change_rgb)
axes[1, 1].set_title("SAR Multi-temporal (R:T2, G:T1, B:Diff)")
axes[1, 1].axis("off")

axes[1, 2].imshow(valid_mask, cmap="Blues_r")
axes[1, 2].set_title(f"SAR Validity Mask ({int((~valid_mask).sum())} invalid px)")
axes[1, 2].axis("off")

im_prob = axes[1, 3].imshow(pred_xattn_prob, cmap="magma", vmin=0, vmax=1)
axes[1, 3].set_title(f"Cross-Attn Prob (Mean={pred_xattn_prob.mean():.2f})")
axes[1, 3].axis("off")
fig.colorbar(im_prob, ax=axes[1, 3], fraction=0.046, pad=0.04)

# Row 3: Model Error Overlays (Green=TP, Red=FP, Blue=FN, Gray=TN)
axes[2, 0].imshow(make_error_rgb(pred_async, gt_label))
axes[2, 0].set_title(f"Async CNN Errors\n(TP={stats_async['tp']:,}, FP={stats_async['fp']:,}, FN={stats_async['fn']:,})")
axes[2, 0].axis("off")

axes[2, 1].imshow(make_error_rgb(pred_sync, gt_label))
axes[2, 1].set_title(f"Sync CNN Errors\n(TP={stats_sync['tp']:,}, FP={stats_sync['fp']:,}, FN={stats_sync['fn']:,})")
axes[2, 1].axis("off")

axes[2, 2].imshow(make_error_rgb(pred_xattn, gt_label))
axes[2, 2].set_title(f"Cross-Attn Errors\n(TP={stats_xattn['tp']:,}, FP={stats_xattn['fp']:,}, FN={stats_xattn['fn']:,})")
axes[2, 2].axis("off")

# Legend in the last subplot
axes[2, 3].axis("off")
legend_elements = [
    plt.Line2D([0], [0], marker="s", color="w", label="True Positive (TP)", markerfacecolor=(0.1, 0.8, 0.2), markersize=14),
    plt.Line2D([0], [0], marker="s", color="w", label="False Positive (FP)", markerfacecolor=(0.9, 0.1, 0.1), markersize=14),
    plt.Line2D([0], [0], marker="s", color="w", label="False Negative (FN)", markerfacecolor=(0.1, 0.3, 0.9), markersize=14),
    plt.Line2D([0], [0], marker="s", color="w", label="True Negative (TN)", markerfacecolor=(0.95, 0.95, 0.95), markersize=14),
]
axes[2, 3].legend(handles=legend_elements, loc="center", fontsize=12, frameon=True, title="Error Overlay Legend")

plt.tight_layout()
fig_path = OUT_DIR / "L15_1289E_visual_panel.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved visualization panel to: {fig_path}")

# Individual high-res error overlays
for m_name, pred_map in [("async_cnn", pred_async), ("sync_cnn", pred_sync), ("cross_attn", pred_xattn)]:
    plt.figure(figsize=(8, 8), dpi=150)
    plt.imshow(make_error_rgb(pred_map, gt_label))
    plt.title(f"{m_name.upper()} Error Map — {SITE_ID}", fontsize=12)
    plt.axis("off")
    plt.savefig(OUT_DIR / f"{m_name}_error_map.png", dpi=150, bbox_inches="tight")
    plt.close()

print("Phase B complete successfully.")
