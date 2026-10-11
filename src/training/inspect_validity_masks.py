"""Audit of per-pixel SAR and Optical validity across all 28 Scenario 2 sites.
Computes valid pixel rates, checks optical validity, and calculates test metrics on valid pixels only.
"""
import json
import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import rasterio
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.scenario2_dataset import (
    Scenario2Dataset, load_split,
    _s1_path, _s2_path, _bld_path,
    T1_YEAR, T1_MONTH, T2_YEAR, T2_MONTH,
    LABEL_THRESHOLD, SAR_CHANNELS, OPT_CHANNELS,
)
from src.evaluation.metrics import counts_from_probabilities, rates_from_counts
from src.models.dual_stream_unet_clean import DualStreamUNetClean
from src.models.dual_stream_unet import DualStreamUNet
from src.models.transformer_cross_attention import TransformerCrossAttention
from src.utils.device import get_device

device = get_device("auto")
split = load_split()

print("=" * 80)
print("PHASE A: AUDIT OF S1 AND S2 VALIDITY ACROSS ALL 28 SITES")
print("=" * 80)

site_validity = {}

for split_name, sites in split.items():
    print(f"\n--- {split_name.upper()} ({len(sites)} sites) ---")
    site_validity[split_name] = {}
    for site in sites:
        p1 = _s1_path(site, T1_YEAR, T1_MONTH)
        p2 = _s1_path(site, T2_YEAR, T2_MONTH)
        with rasterio.open(p1) as ds1:
            arr1 = ds1.read().astype(np.float32)  # (2, H, W)
        with rasterio.open(p2) as ds2:
            arr2 = ds2.read().astype(np.float32)  # (2, H, W)
        
        # S1 NaN masks
        s1_t1_nan = np.isnan(arr1)  # (2, H, W)
        s1_t2_nan = np.isnan(arr2)  # (2, H, W)
        
        # A pixel is valid for SAR if neither VV nor VH is NaN
        s1_t1_valid = ~np.any(s1_t1_nan, axis=0)  # (H, W)
        s1_t2_valid = ~np.any(s1_t2_nan, axis=0)  # (H, W)
        both_s1_valid = s1_t1_valid & s1_t2_valid  # (H, W)
        
        # Optical NaN checks
        p_opt1 = _s2_path(site, T1_YEAR, T1_MONTH)
        with rasterio.open(p_opt1) as dso1:
            arr_opt1 = dso1.read().astype(np.float32)
        opt_t1_valid = ~np.any(np.isnan(arr_opt1), axis=0)
        
        p_opt2 = _s2_path(site, T2_YEAR, T2_MONTH)
        with rasterio.open(p_opt2) as dso2:
            arr_opt2 = dso2.read().astype(np.float32)
        opt_t2_valid = ~np.any(np.isnan(arr_opt2), axis=0)
        
        # Building label check
        pbld1 = _bld_path(site, T1_YEAR, T1_MONTH)
        with rasterio.open(pbld1) as dsb1:
            arr_bld1 = dsb1.read(1).astype(np.float32)
        pbld2 = _bld_path(site, T2_YEAR, T2_MONTH)
        with rasterio.open(pbld2) as dsb2:
            arr_bld2 = dsb2.read(1).astype(np.float32)
        bld_valid = ~(np.isnan(arr_bld1) | np.isnan(arr_bld2))
        
        tot = both_s1_valid.size
        s1_t1_inv = 100 * (1 - s1_t1_valid.sum() / tot)
        s1_t2_inv = 100 * (1 - s1_t2_valid.sum() / tot)
        both_s1_val_pct = 100 * (both_s1_valid.sum() / tot)
        opt_t1_inv = 100 * (1 - opt_t1_valid.sum() / tot)
        opt_t2_inv = 100 * (1 - opt_t2_valid.sum() / tot)
        bld_inv = 100 * (1 - bld_valid.sum() / tot)
        
        site_validity[split_name][site] = {
            "total_pixels": int(tot),
            "s1_t1_invalid_pct": float(s1_t1_inv),
            "s1_t2_invalid_pct": float(s1_t2_inv),
            "both_s1_valid_pct": float(both_s1_val_pct),
            "both_s1_valid_pixels": int(both_s1_valid.sum()),
            "opt_t1_invalid_pct": float(opt_t1_inv),
            "opt_t2_invalid_pct": float(opt_t2_inv),
            "bld_invalid_pct": float(bld_inv),
        }
        
        if (100 - both_s1_val_pct > 0.01) or opt_t1_inv > 0.01 or opt_t2_inv > 0.01:
            print(f"  {site}: S1_T1 inv={s1_t1_inv:.3f}%, S1_T2 inv={s1_t2_inv:.3f}%, Both S1 Valid={both_s1_val_pct:.3f}%, Opt_T1 inv={opt_t1_inv:.3f}%, Opt_T2 inv={opt_t2_inv:.3f}%")

print("\n" + "=" * 80)
print("EVALUATION ON VALID PIXELS ONLY VS ALL PIXELS (TEST SET)")
print("=" * 80)

# Load test models
test_sites = split["test"]
ds_async = Scenario2Dataset(split="test", mode="async")
ds_sync  = Scenario2Dataset(split="test", mode="sync")

EXPERIMENTS = {
    "Async Dual-Stream CNN": {
        "ckpt": PROJECT_ROOT / "checkpoints/scenario_2/async_dual_stream/best.pt",
        "mode": "async",
        "model": DualStreamUNetClean(base_channels=64, sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS)
    },
    "Sync Dual-Stream CNN": {
        "ckpt": PROJECT_ROOT / "checkpoints/scenario_2/sync_dual_stream/best.pt",
        "mode": "sync",
        "model": DualStreamUNet(base_channels=64, sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS * 2)
    },
    "Cross-Attention Transformer": {
        "ckpt": PROJECT_ROOT / "checkpoints/scenario_2/cross_attention/best.pt",
        "mode": "async",
        "model": TransformerCrossAttention(sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS)
    }
}

for exp_name, exp_cfg in EXPERIMENTS.items():
    m = exp_cfg["model"]
    saved = torch.load(exp_cfg["ckpt"], map_location="cpu", weights_only=False)
    m.load_state_dict(saved["model_state_dict"])
    m = m.to(device).eval()
    ds = ds_async if exp_cfg["mode"] == "async" else ds_sync
    
    pooled_all = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    pooled_valid = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    per_site_res = {}
    
    with torch.inference_mode():
        for site in test_sites:
            idx = ds.sites.index(site)
            sample = ds[idx]
            img = sample["image"].unsqueeze(0).to(device)
            lbl = sample["label"].squeeze(0).cpu().numpy().astype(bool) # (H, W)
            
            logits = m(img).squeeze(0).squeeze(0).cpu() # (H, W)
            prob = torch.sigmoid(logits).numpy()
            pred = prob >= 0.5
            
            # Load exact SAR validity mask for this site
            p1 = _s1_path(site, T1_YEAR, T1_MONTH)
            p2 = _s1_path(site, T2_YEAR, T2_MONTH)
            with rasterio.open(p1) as ds1:
                a1 = ds1.read().astype(np.float32)
            with rasterio.open(p2) as ds2:
                a2 = ds2.read().astype(np.float32)
            valid_mask = (~np.any(np.isnan(a1), axis=0)) & (~np.any(np.isnan(a2), axis=0))
            
            # 1. All pixels
            tp_all = int(((pred == True) & (lbl == True)).sum())
            fp_all = int(((pred == True) & (lbl == False)).sum())
            fn_all = int(((pred == False) & (lbl == True)).sum())
            tn_all = int(((pred == False) & (lbl == False)).sum())
            
            # 2. Valid pixels only
            pred_v = pred[valid_mask]
            lbl_v  = lbl[valid_mask]
            tp_v = int(((pred_v == True) & (lbl_v == True)).sum())
            fp_v = int(((pred_v == True) & (lbl_v == False)).sum())
            fn_v = int(((pred_v == False) & (lbl_v == True)).sum())
            tn_v = int(((pred_v == False) & (lbl_v == False)).sum())
            
            # Invalid pixels breakdown
            pred_inv = pred[~valid_mask]
            lbl_inv  = lbl[~valid_mask]
            tp_inv = int(((pred_inv == True) & (lbl_inv == True)).sum())
            fp_inv = int(((pred_inv == True) & (lbl_inv == False)).sum())
            fn_inv = int(((pred_inv == False) & (lbl_inv == True)).sum())
            tn_inv = int(((pred_inv == False) & (lbl_inv == False)).sum())
            
            per_site_res[site] = {
                "all": {"tp": tp_all, "fp": fp_all, "fn": fn_all, "tn": tn_all},
                "valid": {"tp": tp_v, "fp": fp_v, "fn": fn_v, "tn": tn_v},
                "invalid": {"tp": tp_inv, "fp": fp_inv, "fn": fn_inv, "tn": tn_inv},
                "invalid_pixel_count": int((~valid_mask).sum())
            }
            
            for k in pooled_all:
                pooled_all[k] += per_site_res[site]["all"][k]
                pooled_valid[k] += per_site_res[site]["valid"][k]
                
    rates_all = rates_from_counts(pooled_all)
    rates_valid = rates_from_counts(pooled_valid)
    
    print(f"\n{exp_name}:")
    print(f"  {'Subset':<14} {'Prec':>9} {'Rec':>9} {'F1':>9} {'IoU':>9} {'FAR':>9} {'TP':>7} {'FP':>8} {'FN':>7} {'TN':>8}")
    print(f"  {'All pixels':<14} {rates_all['precision']:>9.4f} {rates_all['recall']:>9.4f} {rates_all['f1']:>9.4f} {rates_all['iou']:>9.4f} {rates_all['false_alarm_rate']:>9.4f} {pooled_all['tp']:>7} {pooled_all['fp']:>8} {pooled_all['fn']:>7} {pooled_all['tn']:>8}")
    print(f"  {'Valid pixels':<14} {rates_valid['precision']:>9.4f} {rates_valid['recall']:>9.4f} {rates_valid['f1']:>9.4f} {rates_valid['iou']:>9.4f} {rates_valid['false_alarm_rate']:>9.4f} {pooled_valid['tp']:>7} {pooled_valid['fp']:>8} {pooled_valid['fn']:>7} {pooled_valid['tn']:>8}")
    
    print("  Per-site breakdown (Valid vs All):")
    for site, sdata in per_site_res.items():
        r_a = rates_from_counts(sdata["all"])
        r_v = rates_from_counts(sdata["valid"])
        inv_fp = sdata["invalid"]["fp"]
        inv_tp = sdata["invalid"]["tp"]
        inv_cnt = sdata["invalid_pixel_count"]
        print(f"    {site[-20:]}: Valid F1={r_v['f1']:.4f} (All F1={r_a['f1']:.4f}) | Inv Px={inv_cnt:,} | Inv FP={inv_fp:,} | Inv TP={inv_tp:,}")

# Output validity statistics json
out_json = PROJECT_ROOT / "results" / "scenario_2" / "validity_mask_audit.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(site_validity, f, indent=2)
print("\nSaved validity audit data successfully.")
