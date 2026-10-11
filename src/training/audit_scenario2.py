"""Audit script: recompute Scenario 2 test metrics from saved checkpoints.
Verifies reported metrics, checks checkpoint epoch, NaN handling, and label stats.

Run from project root:
    .venv\\Scripts\\python.exe src\\training\\audit_scenario2.py
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.scenario2_dataset import (
    Scenario2Dataset, load_split,
    SAR_CHANNELS, OPT_CHANNELS, ASYNC_CHANNELS, SYNC_CHANNELS,
    LABEL_THRESHOLD, T1_YEAR, T1_MONTH, T2_YEAR, T2_MONTH,
    DATASET_ROOT, _s1_path, preprocess_site,
)
from src.evaluation.metrics import counts_from_logits, rates_from_counts, FAR_DEFINITION
from src.models.dual_stream_unet_clean import DualStreamUNetClean
from src.models.dual_stream_unet import DualStreamUNet, count_parameters
from src.models.transformer_cross_attention import TransformerCrossAttention
from src.utils.device import get_device

device = get_device("auto")

EXPERIMENTS = {
    "async_dual_stream": {
        "ckpt":    PROJECT_ROOT / "checkpoints/scenario_2/async_dual_stream/best.pt",
        "metrics": PROJECT_ROOT / "experiments/scenario_2/async_dual_stream/metrics.json",
        "mode":    "async",
        "model_cls": "DualStreamUNetClean",
    },
    "sync_dual_stream": {
        "ckpt":    PROJECT_ROOT / "checkpoints/scenario_2/sync_dual_stream/best.pt",
        "metrics": PROJECT_ROOT / "experiments/scenario_2/sync_dual_stream/metrics.json",
        "mode":    "sync",
        "model_cls": "DualStreamUNet",
    },
    "cross_attention": {
        "ckpt":    PROJECT_ROOT / "checkpoints/scenario_2/cross_attention/best.pt",
        "metrics": PROJECT_ROOT / "experiments/scenario_2/cross_attention/metrics.json",
        "mode":    "async",
        "model_cls": "TransformerCrossAttention",
    },
}

split = load_split()
test_sites = split["test"]

print(f"\nTest sites: {test_sites}")
print(f"Device: {device}\n")

# ── Check checkpoint epoch metadata ──────────────────────────────────────────
print("=" * 70)
print("CHECKPOINT METADATA")
print("=" * 70)
for exp, cfg in EXPERIMENTS.items():
    saved = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    reported = json.loads(cfg["metrics"].read_text())
    ckpt_epoch = saved.get("epoch", "?")
    best_epoch = reported.get("validation_best_epoch", "?")
    ckpt_f1    = saved.get("validation_f1")
    reported_f1 = reported.get("validation_best_f1")
    epoch_match = ckpt_epoch == best_epoch
    f1_match    = (ckpt_f1 is not None and reported_f1 is not None and
                   abs(ckpt_f1 - reported_f1) < 1e-8)
    print(f"\n{exp}:")
    print(f"  checkpoint epoch:       {ckpt_epoch}")
    print(f"  reported best epoch:    {best_epoch}  {'MATCH' if epoch_match else 'MISMATCH !!!'}")
    print(f"  checkpoint val F1:      {ckpt_f1}")
    print(f"  reported  val F1:       {reported_f1}  {'MATCH' if f1_match else 'MISMATCH !!!'}")


# ── Load datasets ─────────────────────────────────────────────────────────────
ds_async = Scenario2Dataset(split="test", mode="async")
ds_sync  = Scenario2Dataset(split="test", mode="sync")

# ── Recompute test metrics from checkpoint ────────────────────────────────────
print("\n" + "=" * 70)
print("RECOMPUTED VS REPORTED METRICS (from best.pt)")
print("=" * 70)

def build_model(exp_name, cfg, device):
    saved = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    cls = cfg["model_cls"]
    if cls == "DualStreamUNetClean":
        m = DualStreamUNetClean(base_channels=64, sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS)
    elif cls == "DualStreamUNet":
        m = DualStreamUNet(base_channels=64, sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS * 2)
    else:
        m = TransformerCrossAttention(sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS)
    m.load_state_dict(saved["model_state_dict"])
    return m.to(device).eval()


for exp, cfg in EXPERIMENTS.items():
    model = build_model(exp, cfg, device)
    ds = ds_async if cfg["mode"] == "async" else ds_sync
    reported = json.loads(cfg["metrics"].read_text())

    pooled = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    site_rows = []

    with torch.inference_mode():
        for site_id in test_sites:
            idx = ds.sites.index(site_id)
            sample = ds[idx]
            image = sample["image"].unsqueeze(0).to(device)
            label = sample["label"].unsqueeze(0).to(device)
            logits = model(image)
            counts = counts_from_logits(logits, label)
            for k in pooled:
                pooled[k] += counts[k]
            rates = rates_from_counts(counts)
            site_rows.append({"site": site_id, **counts, **rates})

    rates_pooled = rates_from_counts(pooled)
    r_prec = reported["final_test_precision"]
    r_rec  = reported["final_test_recall"]
    r_f1   = reported["final_test_f1"]
    r_tp   = reported["test_counts"]["tp"]
    r_fp   = reported["test_counts"]["fp"]
    r_fn   = reported["test_counts"]["fn"]
    r_tn   = reported["test_counts"]["tn"]

    c_prec = rates_pooled["precision"]
    c_rec  = rates_pooled["recall"]
    c_f1   = rates_pooled["f1"]

    def close(a, b, tol=1e-4):
        if a is None or b is None: return False
        return abs(a - b) < tol

    tp_ok   = pooled["tp"] == r_tp
    fp_ok   = pooled["fp"] == r_fp
    fn_ok   = pooled["fn"] == r_fn
    tn_ok   = pooled["tn"] == r_tn
    prec_ok = close(c_prec, r_prec)
    rec_ok  = close(c_rec,  r_rec)
    f1_ok   = close(c_f1,   r_f1)

    print(f"\n{exp}:")
    print(f"  {'Metric':<12} {'Reported':>14} {'Recomputed':>14} {'Match':>7}")
    print(f"  {'TP':<12} {r_tp:>14,} {pooled['tp']:>14,} {'OK' if tp_ok else 'MISMATCH':>7}")
    print(f"  {'FP':<12} {r_fp:>14,} {pooled['fp']:>14,} {'OK' if fp_ok else 'MISMATCH':>7}")
    print(f"  {'FN':<12} {r_fn:>14,} {pooled['fn']:>14,} {'OK' if fn_ok else 'MISMATCH':>7}")
    print(f"  {'TN':<12} {r_tn:>14,} {pooled['tn']:>14,} {'OK' if tn_ok else 'MISMATCH':>7}")
    print(f"  {'Precision':<12} {r_prec:>14.6f} {(c_prec or 0):>14.6f} {'OK' if prec_ok else 'MISMATCH':>7}")
    print(f"  {'Recall':<12} {r_rec:>14.6f} {(c_rec or 0):>14.6f} {'OK' if rec_ok else 'MISMATCH':>7}")
    print(f"  {'F1':<12} {r_f1:>14.6f} {(c_f1 or 0):>14.6f} {'OK' if f1_ok else 'MISMATCH':>7}")


# ── Label pixel stats for all four test sites ────────────────────────────────
print("\n" + "=" * 70)
print("TEST SITE LABEL STATS")
print("=" * 70)
print(f"{'Site':<40} {'Total px':>10} {'Change px':>10} {'Change%':>9} {'Pred+ (async)':>14}")
ds_async_eval = Scenario2Dataset(split="test", mode="async")

# load async model for predicted positives
async_saved = torch.load(
    PROJECT_ROOT / "checkpoints/scenario_2/async_dual_stream/best.pt",
    map_location="cpu", weights_only=False
)
async_model = DualStreamUNetClean(
    base_channels=64, sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS
)
async_model.load_state_dict(async_saved["model_state_dict"])
async_model = async_model.to(device).eval()

with torch.inference_mode():
    for site_id in test_sites:
        idx = ds_async_eval.sites.index(site_id)
        sample = ds_async_eval[idx]
        lbl = sample["label"]
        image = sample["image"].unsqueeze(0).to(device)
        logits = async_model(image)
        pred = (torch.sigmoid(logits) >= 0.5).cpu().float()
        total = lbl.numel()
        change_px = int(lbl.sum().item())
        pred_pos  = int(pred.sum().item())
        frac = change_px / total * 100
        print(f"  {site_id:<38} {total:>10,} {change_px:>10,} {frac:>8.3f}% {pred_pos:>14,}")


# ── NaN check in raw S1 files for test sites ─────────────────────────────────
print("\n" + "=" * 70)
print("NaN AUDIT IN RAW S1 FILES (test sites)")
print("=" * 70)
import rasterio, warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

for site_id in test_sites:
    for label_t, year, month in [("T1", T1_YEAR, T1_MONTH), ("T2", T2_YEAR, T2_MONTH)]:
        p = _s1_path(site_id, year, month)
        with rasterio.open(p) as ds_r:
            arr = ds_r.read().astype(np.float32)
        nan_count = int(np.isnan(arr).sum())
        total     = arr.size
        print(f"  {site_id[-20:]} {label_t}: NaN={nan_count}/{total} ({100*nan_count/total:.3f}%)")


# ── NaN in tensor after preprocessing ────────────────────────────────────────
print("\n" + "=" * 70)
print("NaN IN PREPROCESSED TENSORS (after NaN→0 fill)")
print("=" * 70)
ds_nan = Scenario2Dataset(split="test", mode="async")
for site_id in test_sites:
    idx = ds_nan.sites.index(site_id)
    sample = ds_nan[idx]
    img = sample["image"]
    nan_in_tensor = torch.isnan(img).any().item()
    print(f"  {site_id[-20:]}: NaN in tensor = {nan_in_tensor}")


# ── pos_weight verification ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("pos_weight VERIFICATION FROM TRAINING SITES")
print("=" * 70)
train_sites = split["train"]
ds_train = Scenario2Dataset(split="train", mode="async")
total_ch = 0; total_nch = 0
for site_id in train_sites:
    idx = ds_train.sites.index(site_id)
    sample = ds_train[idx]
    lbl = sample["label"]
    ch  = int(lbl.sum().item())
    nch = int(lbl.numel()) - ch
    total_ch  += ch
    total_nch += nch

pw = total_nch / total_ch if total_ch > 0 else float("inf")
print(f"  Training change pixels:    {total_ch:,}")
print(f"  Training no-change pixels: {total_nch:,}")
print(f"  Computed pos_weight:       {pw:.2f}")
print(f"  Used pos_weight:           94.0  (reported in config)")
print(f"  Difference:                {abs(pw - 94.0):.2f}  {'OK' if abs(pw-94.0)<1.0 else 'DISCREPANCY'}")


# ── Crop sampling reproducibility ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("CROP SAMPLING CHECK")
print("=" * 70)
from src.datasets.scenario2_dataset import RandomCropScenario2
from torch.utils.data import DataLoader
from src.training.train_sar_baseline import set_seed

set_seed(42)
ds_tr = Scenario2Dataset(split="train", mode="async")
crops = RandomCropScenario2(ds_tr, crop_size=32, crops_per_site=64)
gen = torch.Generator(); gen.manual_seed(42)
loader = DataLoader(crops, batch_size=64, shuffle=True, generator=gen)
batch = next(iter(loader))
sites_in_batch = set(batch["site"])
print(f"  Crop dataset length: {len(crops)} ({len(ds_tr)} sites × 64 crops)")
print(f"  Sites in first batch: {len(sites_in_batch)} unique — {sorted(sites_in_batch)[:4]}...")
# Verify no test site appears in training crops
test_in_train = set(test_sites) & sites_in_batch
print(f"  Test sites in training batch: {test_in_train if test_in_train else 'NONE (correct)'}")


print("\n\nAudit complete.")
