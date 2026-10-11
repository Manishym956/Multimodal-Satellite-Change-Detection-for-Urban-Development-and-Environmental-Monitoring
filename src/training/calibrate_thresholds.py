"""Phase C: Validation-only threshold calibration.
Sweeps decision thresholds exclusively on the 4 validation sites to select tau*,
then evaluates test sites at both tau=0.5 and frozen tau*.
Outputs saved in experiments/scenario_2/threshold_calibration/
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
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.scenario2_dataset import (
    Scenario2Dataset, load_split,
    SAR_CHANNELS, OPT_CHANNELS,
)
from src.evaluation.metrics import counts_from_probabilities, rates_from_counts
from src.models.dual_stream_unet_clean import DualStreamUNetClean
from src.models.dual_stream_unet import DualStreamUNet
from src.models.transformer_cross_attention import TransformerCrossAttention
from src.utils.device import get_device

device = get_device("auto")
split = load_split()
val_sites  = split["validation"]
test_sites = split["test"]

OUT_DIR = PROJECT_ROOT / "experiments" / "scenario_2" / "threshold_calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PHASE C: THRESHOLD CALIBRATION ON VALIDATION SITES")
print("=" * 80)
print(f"Validation sites ({len(val_sites)}): {val_sites}")
print(f"Test sites ({len(test_sites)}): {test_sites}\n")

# Load models
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

ds_val_async = Scenario2Dataset(split="validation", mode="async")
ds_val_sync  = Scenario2Dataset(split="validation", mode="sync")
ds_test_async = Scenario2Dataset(split="test", mode="async")
ds_test_sync  = Scenario2Dataset(split="test", mode="sync")

thresholds = np.linspace(0.01, 0.99, 99)

calibration_results = {}

plt.figure(figsize=(15, 5), dpi=150)

for m_idx, (exp_name, exp_cfg) in enumerate(EXPERIMENTS.items(), 1):
    m = exp_cfg["model"]
    saved = torch.load(exp_cfg["ckpt"], map_location="cpu", weights_only=False)
    m.load_state_dict(saved["model_state_dict"])
    m = m.to(device).eval()
    
    ds_val = ds_val_async if exp_cfg["mode"] == "async" else ds_val_sync
    ds_test = ds_test_async if exp_cfg["mode"] == "async" else ds_test_sync
    
    # 1. Collect validation probabilities and labels
    val_probs = []
    val_labels = []
    with torch.inference_mode():
        for site in val_sites:
            idx = ds_val.sites.index(site)
            sample = ds_val[idx]
            img = sample["image"].unsqueeze(0).to(device)
            lbl = sample["label"].squeeze().cpu().numpy().astype(bool)
            prob = torch.sigmoid(m(img)).squeeze().cpu().numpy()
            val_probs.append(prob)
            val_labels.append(lbl)
            
    val_probs = np.concatenate([p.flatten() for p in val_probs])
    val_labels = np.concatenate([l.flatten() for l in val_labels])
    
    # 2. Sweep thresholds on validation data
    val_precisions = []
    val_recalls = []
    val_f1s = []
    
    best_val_f1 = -1.0
    best_tau = 0.5
    
    for tau in thresholds:
        pred = val_probs >= tau
        tp = int((pred & val_labels).sum())
        fp = int((pred & (~val_labels)).sum())
        fn = int(((~pred) & val_labels).sum())
        tn = int(((~pred) & (~val_labels)).sum())
        
        rates = rates_from_counts({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
        p = rates["precision"] or 0.0
        r = rates["recall"] or 0.0
        f1 = rates["f1"] or 0.0
        
        val_precisions.append(p)
        val_recalls.append(r)
        val_f1s.append(f1)
        
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_tau = float(tau)
            
    print(f"[{exp_name}] Validation Calibration:")
    print(f"  Default tau=0.5: Val F1 = {val_f1s[np.argmin(np.abs(thresholds - 0.5))]:.4f}")
    print(f"  Optimal tau* = {best_tau:.3f}: Max Val F1 = {best_val_f1:.4f}")
    
    # Plot PR and F1 curves for validation
    plt.subplot(1, 3, m_idx)
    plt.plot(val_recalls, val_precisions, label="PR curve (Val)", color="blue", lw=2)
    plt.plot(thresholds, val_f1s, label="F1 vs tau (Val)", color="red", linestyle="--", lw=1.5)
    plt.axvline(0.5, color="gray", linestyle=":", label="Default tau=0.5")
    plt.axvline(best_tau, color="green", linestyle="-.", label=f"Optimal tau*={best_tau:.2f}")
    plt.title(f"{exp_name}\nVal tau*={best_tau:.2f} (F1={best_val_f1:.4f})")
    plt.xlabel("Recall / Threshold")
    plt.ylabel("Precision / F1")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 3. Evaluate TEST set at frozen tau=0.5 and frozen tau*
    def eval_test_at_threshold(tau):
        test_pooled = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        site_metrics = {}
        with torch.inference_mode():
            for site in test_sites:
                idx = ds_test.sites.index(site)
                sample = ds_test[idx]
                img = sample["image"].unsqueeze(0).to(device)
                lbl = sample["label"].squeeze().cpu().numpy().astype(bool)
                prob = torch.sigmoid(m(img)).squeeze().cpu().numpy()
                pred = prob >= tau
                
                tp = int((pred & lbl).sum())
                fp = int((pred & (~lbl)).sum())
                fn = int(((~pred) & lbl).sum())
                tn = int(((~pred) & (~lbl)).sum())
                
                counts = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
                rates = rates_from_counts(counts)
                site_metrics[site] = {**counts, **rates}
                for k in test_pooled:
                    test_pooled[k] += counts[k]
        rates_pooled = rates_from_counts(test_pooled)
        return {**test_pooled, **rates_pooled, "per_site": site_metrics}

    test_res_default = eval_test_at_threshold(0.5)
    test_res_calibrated = eval_test_at_threshold(best_tau)
    
    print(f"  Test Evaluation Comparison:")
    print(f"    tau=0.50: Prec={test_res_default['precision']:.4f}, Rec={test_res_default['recall']:.4f}, F1={test_res_default['f1']:.4f}, IoU={test_res_default['iou']:.4f}, FAR={test_res_default['false_alarm_rate']:.4f}")
    print(f"    tau={best_tau:.2f}: Prec={test_res_calibrated['precision']:.4f}, Rec={test_res_calibrated['recall']:.4f}, F1={test_res_calibrated['f1']:.4f}, IoU={test_res_calibrated['iou']:.4f}, FAR={test_res_calibrated['false_alarm_rate']:.4f}")
    
    calibration_results[exp_name] = {
        "best_validation_tau": best_tau,
        "best_validation_f1": best_val_f1,
        "validation_f1_at_0_5": float(val_f1s[np.argmin(np.abs(thresholds - 0.5))]),
        "test_default_0_5": {
            "threshold": 0.5,
            "precision": test_res_default["precision"],
            "recall": test_res_default["recall"],
            "f1": test_res_default["f1"],
            "iou": test_res_default["iou"],
            "false_alarm_rate": test_res_default["false_alarm_rate"],
            "tp": test_res_default["tp"],
            "fp": test_res_default["fp"],
            "fn": test_res_default["fn"],
            "tn": test_res_default["tn"],
        },
        "test_calibrated": {
            "threshold": best_tau,
            "precision": test_res_calibrated["precision"],
            "recall": test_res_calibrated["recall"],
            "f1": test_res_calibrated["f1"],
            "iou": test_res_calibrated["iou"],
            "false_alarm_rate": test_res_calibrated["false_alarm_rate"],
            "tp": test_res_calibrated["tp"],
            "fp": test_res_calibrated["fp"],
            "fn": test_res_calibrated["fn"],
            "tn": test_res_calibrated["tn"],
        },
        "test_per_site_calibrated": test_res_calibrated["per_site"]
    }

plt.tight_layout()
pr_plot_path = OUT_DIR / "validation_pr_curves.png"
plt.savefig(pr_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved PR and threshold curves to: {pr_plot_path}")

out_json = OUT_DIR / "threshold_calibration_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(calibration_results, f, indent=2)
print(f"Saved calibration results to: {out_json}")
print("Phase C complete successfully.")
