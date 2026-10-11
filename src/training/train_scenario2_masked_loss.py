"""Scenario 2 — Experiment S2-1: Masked-Loss Training.

Trains models on SEN12-MT Urban Mapping (Scenario 2) with pixel-level SAR validity masking,
ensuring invalid sensor pixels (e.g. SAR NaNs) do not contribute to the loss numerator or denominator.

Models supported via --model:
  1. async_cnn       (DualStreamUNetClean, 14 channels, mode='async')
  2. sync_cnn        (DualStreamUNet, 24 channels, mode='sync')
  3. cross_attention (TransformerCrossAttention, 14 channels, mode='async')

Experimental protocol (identical to baseline):
  - Locked 20 train / 4 val / 4 test site split
  - Strategy B temporal pair (T1=2018-03, T2=2019-12)
  - Seed: 42
  - Epochs: 50
  - Batch size: 64
  - Learning rate: 1e-4
  - Optimizer: Adam
  - Crop size: 32x32
  - Crops per site: 64 (1,280 training crops/epoch)
  - Loss: MaskedBCEWithLogitsLoss(pos_weight=94.0)
  - Checkpoint selection: Best validation F1 under the Valid-Pixel Protocol
  - Dual evaluation: Evaluates both All-Pixel and Valid-Pixel protocols

Outputs:
  checkpoints/scenario_2_masked_loss/<model>/best.pt
  checkpoints/scenario_2_masked_loss/<model>/last.pt
  experiments/scenario_2_masked_loss/<model>/
      config.yaml
      training_history.csv
      per_city_metrics.csv
      metrics.json
"""
import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import rasterio
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.scenario2_dataset import (
    ASYNC_CHANNELS,
    OPT_CHANNELS,
    SAR_CHANNELS,
    SYNC_CHANNELS,
    RandomCropScenario2,
    Scenario2Dataset,
    load_split,
)
from src.evaluation.metrics import (
    FAR_DEFINITION,
    counts_from_logits,
    rates_from_counts,
)
from src.models.dual_stream_unet import DualStreamUNet, count_parameters as count_params_sync
from src.models.dual_stream_unet_clean import DualStreamUNetClean
from src.models.masked_loss import MaskedBCEWithLogitsLoss
from src.models.transformer_cross_attention import (
    TransformerCrossAttention,
    count_parameters as count_params_xattn,
)
from src.training.train_sar_baseline import save_checkpoint, set_seed


def save_history(path: Path, history: list) -> None:
    """Write training history CSV, dynamically deriving fieldnames from the data."""
    if not history:
        return
    # Collect superset of all keys (preserving insertion order via dict)
    fieldnames = list(dict.fromkeys(k for row in history for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
from src.utils.device import add_device_argument, get_device

POS_WEIGHT = 94.0


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment S2-1: Masked-Loss Training for Scenario 2."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["async_cnn", "sync_cnn", "cross_attention"],
        help="Model to train: async_cnn, sync_cnn, or cross_attention.",
    )
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch-size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--crop-size",       type=int,   default=32)
    parser.add_argument("--crops-per-site",  type=int,   default=64)
    parser.add_argument("--base-channels",   type=int,   default=64, help="CNN base channels.")
    parser.add_argument("--embed-dim",       type=int,   default=128, help="Transformer embed dim.")
    parser.add_argument("--patch-stride",    type=int,   default=4)
    parser.add_argument("--num-self-blocks", type=int,   default=2)
    parser.add_argument("--num-cross-blocks",type=int,   default=1)
    parser.add_argument("--num-heads",       type=int,   default=4)
    parser.add_argument("--mlp-ratio",       type=int,   default=4)
    parser.add_argument("--num-workers",     type=int,   default=0)
    parser.add_argument("--verify-only",     action="store_true", help="Run verification only; do not train.")
    add_device_argument(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace) -> tuple[nn.Module, str, int]:
    if args.model == "async_cnn":
        model = DualStreamUNetClean(
            base_channels=args.base_channels,
            sar_channels=SAR_CHANNELS,
            opt_channels=OPT_CHANNELS,
        )
        mode = "async"
        params = count_params_sync(model)
    elif args.model == "sync_cnn":
        model = DualStreamUNet(
            base_channels=args.base_channels,
            sar_channels=SAR_CHANNELS,
            opt_channels=OPT_CHANNELS * 2,
        )
        mode = "sync"
        params = count_params_sync(model)
    elif args.model == "cross_attention":
        model = TransformerCrossAttention(
            sar_channels=SAR_CHANNELS,
            opt_channels=OPT_CHANNELS,
            embed_dim=args.embed_dim,
            patch_stride=args.patch_stride,
            num_self_blocks=args.num_self_blocks,
            num_cross_blocks=args.num_cross_blocks,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
        )
        mode = "async"
        params = count_params_xattn(model)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model, mode, params


# ---------------------------------------------------------------------------
# Masked Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch_masked(
    model: nn.Module,
    loader: DataLoader,
    criterion: MaskedBCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    seen = 0
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        mask  = batch["valid_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = criterion(logits, label, mask)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * image.shape[0]
        seen += image.shape[0]
    return total_loss / seen if seen > 0 else 0.0


# ---------------------------------------------------------------------------
# Dual-Protocol Full-Image Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_dual_protocol(
    model: nn.Module,
    dataset: Scenario2Dataset,
    sites: list[str],
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[dict, dict, list[dict], list[str]]:
    """Evaluates under both All-Pixel and Valid-Pixel protocols simultaneously.

    Returns:
        (pooled_all, pooled_valid, per_site_rows, warnings)
    """
    model.eval()
    pooled_all   = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    pooled_valid = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    per_site_rows: list[dict] = []
    warnings: list[str] = []

    for site_id in sites:
        idx    = dataset.sites.index(site_id)
        sample = dataset[idx]
        image  = sample["image"].unsqueeze(0).to(device)
        label  = sample["label"].squeeze().cpu().numpy().astype(bool)
        vmask  = sample["valid_mask"].squeeze().cpu().numpy().astype(bool)

        if not torch.isfinite(image).all():
            warnings.append(f"{site_id}: non-finite input tensor")

        logits = model(image).squeeze().cpu()
        if not torch.isfinite(logits).all():
            warnings.append(f"{site_id}: non-finite output logits")

        probs = torch.sigmoid(logits).numpy()
        pred  = probs >= threshold

        # 1. All-pixel protocol
        tp_a = int(((pred == True) & (label == True)).sum())
        fp_a = int(((pred == True) & (label == False)).sum())
        fn_a = int(((pred == False) & (label == True)).sum())
        tn_a = int(((pred == False) & (label == False)).sum())
        counts_all = {"tp": tp_a, "fp": fp_a, "fn": fn_a, "tn": tn_a}

        # 2. Valid-pixel protocol
        pred_v = pred[vmask]
        lbl_v  = label[vmask]
        tp_v = int(((pred_v == True) & (lbl_v == True)).sum())
        fp_v = int(((pred_v == True) & (lbl_v == False)).sum())
        fn_v = int(((pred_v == False) & (lbl_v == True)).sum())
        tn_v = int(((pred_v == False) & (lbl_v == False)).sum())
        counts_valid = {"tp": tp_v, "fp": fp_v, "fn": fn_v, "tn": tn_v}

        for k in pooled_all:
            pooled_all[k]   += counts_all[k]
            pooled_valid[k] += counts_valid[k]

        rates_all   = rates_from_counts(counts_all)
        rates_valid = rates_from_counts(counts_valid)

        row = {
            "site": site_id,
            "threshold": threshold,
            "invalid_pixels": int((~vmask).sum()),
            "valid_pixels": int(vmask.sum()),
            # All-pixel metrics
            "all_tp": tp_a, "all_fp": fp_a, "all_fn": fn_a, "all_tn": tn_a,
            "all_precision": rates_all["precision"],
            "all_recall":    rates_all["recall"],
            "all_f1":        rates_all["f1"],
            "all_iou":       rates_all["iou"],
            "all_far":       rates_all["false_alarm_rate"],
            # Valid-pixel metrics
            "valid_tp": tp_v, "valid_fp": fp_v, "valid_fn": fn_v, "valid_tn": tn_v,
            "valid_precision": rates_valid["precision"],
            "valid_recall":    rates_valid["recall"],
            "valid_f1":        rates_valid["f1"],
            "valid_iou":       rates_valid["iou"],
            "valid_far":       rates_valid["false_alarm_rate"],
        }
        per_site_rows.append(row)

    return (
        {**pooled_all, **rates_from_counts(pooled_all)},
        {**pooled_valid, **rates_from_counts(pooled_valid)},
        per_site_rows,
        warnings,
    )


# ---------------------------------------------------------------------------
# Validation-Only Threshold Sweep
# ---------------------------------------------------------------------------

@torch.inference_mode()
def calibrate_threshold_on_validation(
    model: nn.Module,
    val_dataset: Scenario2Dataset,
    val_sites: list[str],
    device: torch.device,
) -> float:
    """Finds tau* in [0.01, 0.99] that maximizes validation F1 on valid pixels."""
    model.eval()
    val_probs = []
    val_labels = []
    val_masks = []

    for site in val_sites:
        idx = val_dataset.sites.index(site)
        sample = val_dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        probs = torch.sigmoid(model(image)).squeeze().cpu().numpy()
        val_probs.append(probs)
        val_labels.append(sample["label"].squeeze().numpy().astype(bool))
        val_masks.append(sample["valid_mask"].squeeze().numpy().astype(bool))

    probs_flat = np.concatenate([p.flatten() for p in val_probs])
    lbls_flat  = np.concatenate([l.flatten() for l in val_labels])
    msks_flat  = np.concatenate([m.flatten() for m in val_masks])

    # Filter to valid pixels only for calibration
    v_probs = probs_flat[msks_flat]
    v_lbls  = lbls_flat[msks_flat]

    best_tau = 0.50
    best_f1  = -1.0

    for tau in np.linspace(0.01, 0.99, 99):
        pred = v_probs >= tau
        tp = int((pred & v_lbls).sum())
        fp = int((pred & (~v_lbls)).sum())
        fn = int(((~pred) & v_lbls).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec  = tp / (tp + fn)
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)

    return best_tau


# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    model, mode, param_count = build_model(args)
    model = model.to(device)

    EXPERIMENT_DIR  = PROJECT_ROOT / "experiments" / "scenario_2_masked_loss" / args.model
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints"  / "scenario_2_masked_loss" / args.model / "best.pt"
    LAST_CKPT_PATH  = PROJECT_ROOT / "checkpoints"  / "scenario_2_masked_loss" / args.model / "last.pt"

    split = load_split()
    train_sites = split["train"]
    val_sites   = split["validation"]
    test_sites  = split["test"]

    train_ds = Scenario2Dataset(split="train", mode=mode, return_mask=True)
    val_ds   = Scenario2Dataset(split="validation", mode=mode, return_mask=True)
    test_ds  = Scenario2Dataset(split="test", mode=mode, return_mask=True)

    crops = RandomCropScenario2(train_ds, crop_size=args.crop_size, crops_per_site=args.crops_per_site)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    loader = DataLoader(
        crops,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=gen,
    )

    if args.verify_only:
        print(f"Verified model {args.model}: {param_count:,} parameters. Exiting (--verify-only).")
        return

    pos_weight = torch.tensor([POS_WEIGHT]).to(device)
    criterion  = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)

    gpu_name = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
    config_record = {
        "experiment": f"Scenario 2 Masked-Loss Training — {args.model}",
        "hypothesis": (
            "Masking invalid SAR pixels (weight=0) during training prevents the model "
            "from learning spurious associations on dummy zero-filled pixels, improving feature quality."
        ),
        "model_name": args.model,
        "model_class": model.__class__.__name__,
        "mode": mode,
        "total_input_channels": ASYNC_CHANNELS if mode == "async" else SYNC_CHANNELS,
        "parameter_count": int(param_count),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "crop_size": int(args.crop_size),
        "crops_per_site": int(args.crops_per_site),
        "loss_function": "MaskedBCEWithLogitsLoss(pos_weight=94.0)",
        "pos_weight": POS_WEIGHT,
        "optimizer": "Adam",
        "checkpoint_selection": "Highest validation F1 under the Valid-Pixel Protocol",
        "primary_threshold": 0.5,
        "false_alarm_rate_definition": str(FAR_DEFINITION),
        "train_sites": train_sites,
        "validation_sites": val_sites,
        "test_sites": test_sites,
        "device": str(device),
        "gpu_name": gpu_name,
        "pytorch_version": str(torch.__version__),
    }

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with (EXPERIMENT_DIR / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_record, fh, sort_keys=False)

    print("=" * 80)
    print(f"STARTING EXPERIMENT S2-1: {args.model.upper()} (MASKED LOSS)")
    print("=" * 80)
    print(f"Model:          {model.__class__.__name__} ({param_count:,} params)")
    print(f"Mode:           {mode} ({config_record['total_input_channels']} channels)")
    print(f"Loss:           MaskedBCEWithLogitsLoss (pos_weight={POS_WEIGHT})")
    print(f"Split:          {len(train_sites)} train / {len(val_sites)} val / {len(test_sites)} test")
    print(f"Training crops: {len(crops)} ({len(train_sites)} sites × {args.crops_per_site} crops)")
    print(f"Checkpoints:    {CHECKPOINT_PATH}")
    print(f"Experiment dir: {EXPERIMENT_DIR}\n", flush=True)

    history: list[dict] = []
    best_f1_valid = -1.0
    best_epoch: int | None = None

    t_train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch_masked(model, loader, criterion, optimizer, device)
        epoch_time = time.perf_counter() - t0

        val_all, val_valid, _, val_warn = evaluate_dual_protocol(
            model, val_ds, val_sites, device, threshold=0.5
        )

        f1_valid = val_valid["f1"]
        f1_all   = val_all["f1"]

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_time_s": round(epoch_time, 2),
            "lr": optimizer.param_groups[0]["lr"],
            # Valid-pixel validation metrics
            "val_f1_valid": f1_valid,
            "val_prec_valid": val_valid["precision"],
            "val_rec_valid": val_valid["recall"],
            "val_iou_valid": val_valid["iou"],
            "val_far_valid": val_valid["false_alarm_rate"],
            # All-pixel validation metrics
            "val_f1_all": f1_all,
            "val_prec_all": val_all["precision"],
            "val_rec_all": val_all["recall"],
            "val_iou_all": val_all["iou"],
            "val_far_all": val_all["false_alarm_rate"],
        }
        history.append(history_row)

        print(
            f"epoch {epoch:>3}  loss {train_loss:.4f}  "
            f"val_f1_valid {f1_valid if f1_valid is not None else 'None':>7}  "
            f"val_f1_all {f1_all if f1_all is not None else 'None':>7}  "
            f"({epoch_time:.1f}s)",
            flush=True,
        )
        if val_warn:
            print(f"  warnings: {'; '.join(val_warn)}", flush=True)

        # Save last checkpoint every epoch
        save_checkpoint(LAST_CKPT_PATH, model, epoch, f1_valid or 0.0, config_record)

        # Checkpoint selection: based on highest validation F1 on valid pixels
        if f1_valid is not None and f1_valid > best_f1_valid:
            best_f1_valid = f1_valid
            best_epoch = epoch
            save_checkpoint(CHECKPOINT_PATH, model, epoch, f1_valid, config_record)

    total_training_time = time.perf_counter() - t_train_start

    if best_epoch is None or not CHECKPOINT_PATH.is_file():
        raise RuntimeError("Masked-loss training produced no valid checkpoint.")

    print(f"\nTraining complete. Best checkpoint: Epoch {best_epoch} with Val F1 (valid) = {best_f1_valid:.4f}")

    # ---------------------------------------------------------------------------
    # Evaluation: Load Best Checkpoint & Test Evaluation
    # ---------------------------------------------------------------------------
    saved = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)

    # 1. Validation-only threshold calibration
    calibrated_tau = calibrate_threshold_on_validation(model, val_ds, val_sites, device)
    print(f"Validation-calibrated optimal threshold: tau* = {calibrated_tau:.3f}")

    # 2. Test evaluation at default threshold 0.50
    t_eval_start = time.perf_counter()
    test_all_05, test_valid_05, per_site_05, test_warn = evaluate_dual_protocol(
        model, test_ds, test_sites, device, threshold=0.50
    )
    # 3. Test evaluation at calibrated threshold tau*
    test_all_cal, test_valid_cal, per_site_cal, _ = evaluate_dual_protocol(
        model, test_ds, test_sites, device, threshold=calibrated_tau
    )
    eval_time = time.perf_counter() - t_eval_start

    # Per-site metrics CSV
    csv_path = EXPERIMENT_DIR / "per_city_metrics.csv"
    if per_site_05:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_site_05[0].keys()))
            writer.writeheader()
            writer.writerows(per_site_05)

    # Training history CSV
    save_history(EXPERIMENT_DIR / "training_history.csv", history)

    # Peak memory
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else None

    # Specific site L15-1289E metrics
    l1289_row_05 = next((r for r in per_site_05 if "L15-1289E" in r["site"]), {})
    l1289_row_cal = next((r for r in per_site_cal if "L15-1289E" in r["site"]), {})

    metrics_document = {
        "experiment": config_record["experiment"],
        "model": args.model,
        "validation_best_epoch": best_epoch,
        "validation_best_f1_valid": best_f1_valid,
        "calibrated_threshold_tau_star": calibrated_tau,
        "training_time_seconds": round(total_training_time, 2),
        "evaluation_time_seconds": round(eval_time, 2),
        "peak_gpu_memory_mb": round(peak_mem_mb, 2) if peak_mem_mb is not None else None,
        # Default Threshold 0.50 Results
        "test_threshold_0_5": {
            "valid_pixel_protocol": test_valid_05,
            "all_pixel_protocol": test_all_05,
        },
        # Calibrated Threshold tau* Results
        "test_threshold_calibrated": {
            "threshold": calibrated_tau,
            "valid_pixel_protocol": test_valid_cal,
            "all_pixel_protocol": test_all_cal,
        },
        "problematic_site_L15_1289E": {
            "threshold_0_5": {
                "all_fp": l1289_row_05.get("all_fp"),
                "all_f1": l1289_row_05.get("all_f1"),
                "all_far": l1289_row_05.get("all_far"),
                "valid_fp": l1289_row_05.get("valid_fp"),
                "valid_f1": l1289_row_05.get("valid_f1"),
                "valid_far": l1289_row_05.get("valid_far"),
            },
            "threshold_calibrated": {
                "threshold": calibrated_tau,
                "all_fp": l1289_row_cal.get("all_fp"),
                "all_f1": l1289_row_cal.get("all_f1"),
                "all_far": l1289_row_cal.get("all_far"),
                "valid_fp": l1289_row_cal.get("valid_fp"),
                "valid_f1": l1289_row_cal.get("valid_f1"),
                "valid_far": l1289_row_cal.get("valid_far"),
            },
        },
        "per_site_metrics_tau_0_5": per_site_05,
        "per_site_metrics_tau_calibrated": per_site_cal,
        "warnings": test_warn,
        "config": config_record,
    }

    with (EXPERIMENT_DIR / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_document, fh, indent=2)

    # ---------------------------------------------------------------------------
    # Summary Report
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"EXPERIMENT S2-1 ({args.model.upper()}) FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Best Val Epoch:            {best_epoch}")
    print(f"Best Val F1 (valid px):    {best_f1_valid:.4f}")
    print(f"Optimal Val Threshold tau*: {calibrated_tau:.3f}")
    print("\n[Default Threshold tau = 0.50]")
    print(f"  Valid-Pixel Protocol: F1={test_valid_05['f1']:.4f}, Prec={test_valid_05['precision']:.4f}, Rec={test_valid_05['recall']:.4f}, IoU={test_valid_05['iou']:.4f}, FAR={test_valid_05['false_alarm_rate']:.4f}")
    print(f"  All-Pixel Protocol:   F1={test_all_05['f1']:.4f}, Prec={test_all_05['precision']:.4f}, Rec={test_all_05['recall']:.4f}, IoU={test_all_05['iou']:.4f}, FAR={test_all_05['false_alarm_rate']:.4f}")
    print("\n[Validation-Calibrated Threshold tau = tau*]")
    print(f"  Valid-Pixel Protocol: F1={test_valid_cal['f1']:.4f}, Prec={test_valid_cal['precision']:.4f}, Rec={test_valid_cal['recall']:.4f}, IoU={test_valid_cal['iou']:.4f}, FAR={test_valid_cal['false_alarm_rate']:.4f}")
    print(f"  All-Pixel Protocol:   F1={test_all_cal['f1']:.4f}, Prec={test_all_cal['precision']:.4f}, Rec={test_all_cal['recall']:.4f}, IoU={test_all_cal['iou']:.4f}, FAR={test_all_cal['false_alarm_rate']:.4f}")
    print("\n[Site L15-1289E-1169N]")
    print(f"  tau=0.50: FP (all)={l1289_row_05.get('all_fp')}, F1 (all)={l1289_row_05.get('all_f1')}, FAR (all)={l1289_row_05.get('all_far')}")
    print(f"  tau={calibrated_tau:.2f}: FP (all)={l1289_row_cal.get('all_fp')}, F1 (all)={l1289_row_cal.get('all_f1')}, FAR (all)={l1289_row_cal.get('all_far')}")
    print(f"\nArtifacts saved in: {EXPERIMENT_DIR}")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
