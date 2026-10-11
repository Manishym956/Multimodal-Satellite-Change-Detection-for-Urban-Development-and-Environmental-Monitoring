import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from src.datasets.scenario2_dataset import (
    OPT_CHANNELS,
    SAR_CHANNELS,
    Scenario2Dataset,
    load_split,
)
from src.models.dual_stream_unet import DualStreamUNet
from src.models.masked_loss import MaskedBCEWithLogitsLoss
from src.training.train_sar_baseline import count_parameters, save_checkpoint, set_seed
from src.training.train_scenario2_dual_stream import counts_from_logits, rates_from_counts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POS_WEIGHT = 94.0
FAR_DEFINITION = "FAR = FP / (FP + TN): fraction of unchanged pixels predicted as change."


def generate_crop_schedule(
    sites: list[str],
    site_dims: dict[str, tuple[int, int]],
    crops_per_site: int,
    crop_size: int,
    epochs: int,
    seed: int,
) -> dict[int, list[dict]]:
    rng = np.random.RandomState(seed)
    schedule = {}
    for ep in range(1, epochs + 1):
        epoch_crops = []
        for site_idx, site in enumerate(sites):
            h, w = site_dims[site]
            max_top = h - crop_size
            max_left = w - crop_size
            if max_top < 0 or max_left < 0:
                raise ValueError(f"{site}: dimensions ({h}, {w}) smaller than crop {crop_size}")
            for _ in range(crops_per_site):
                top = int(rng.randint(0, max_top + 1))
                left = int(rng.randint(0, max_left + 1))
                epoch_crops.append({
                    "site": site,
                    "site_idx": site_idx,
                    "top": top,
                    "left": left,
                })
        schedule[ep] = epoch_crops
    return schedule


class ScheduledCropScenario2(Dataset):
    def __init__(self, base: Scenario2Dataset, crop_size: int) -> None:
        self.base = base
        self.crop_size = crop_size
        self.current_schedule: list[dict] = []

    def set_epoch_schedule(self, schedule: list[dict]) -> None:
        self.current_schedule = schedule

    def __len__(self) -> int:
        return len(self.current_schedule)

    def __getitem__(self, index: int) -> dict:
        item = self.current_schedule[index]
        sample = self.base[item["site_idx"]]
        top = item["top"]
        left = item["left"]
        cs = self.crop_size

        image = sample["image"][:, top:top+cs, left:left+cs]
        label = sample["label"][:, top:top+cs, left:left+cs]
        mask  = sample["valid_mask"][:, top:top+cs, left:left+cs]

        return {
            "image": image,
            "label": label,
            "valid_mask": mask,
            "site": item["site"],
            "coord": (top, left),
        }


@torch.inference_mode()
def evaluate_dual_protocol(
    model: nn.Module,
    dataset: Scenario2Dataset,
    sites: list[str],
    device: torch.device,
    threshold: float = 0.50,
) -> tuple[dict, dict, list[dict], list[str]]:
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
            "all_tp": tp_a, "all_fp": fp_a, "all_fn": fn_a, "all_tn": tn_a,
            "all_precision": rates_all["precision"],
            "all_recall":    rates_all["recall"],
            "all_f1":        rates_all["f1"],
            "all_iou":       rates_all["iou"],
            "all_far":       rates_all["false_alarm_rate"],
            "valid_tp": tp_v, "valid_fp": fp_v, "valid_fn": fn_v, "valid_tn": tn_v,
            "valid_precision": rates_valid["precision"],
            "valid_recall":    rates_valid["recall"],
            "valid_f1":        rates_valid["f1"],
            "valid_iou":       rates_valid["iou"],
            "valid_far":       rates_valid["false_alarm_rate"],
        }
        per_site_rows.append(row)

    res_all = {**pooled_all, **rates_from_counts(pooled_all)}
    res_val = {**pooled_valid, **rates_from_counts(pooled_valid)}
    return res_all, res_val, per_site_rows, warnings


@torch.inference_mode()
def calibrate_threshold_on_validation(
    model: nn.Module,
    val_dataset: Scenario2Dataset,
    val_sites: list[str],
    device: torch.device,
) -> float:
    model.eval()
    val_probs, val_labels, val_masks = [], [], []

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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str,
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
        if loss_type == "masked":
            loss = criterion(logits, label, mask)
        else:
            loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * image.shape[0]
        seen += image.shape[0]
    return total_loss / seen if seen > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario 2 Sync Multi-Seed Runner")
    parser.add_argument("--seed", type=int, required=True, choices=[43, 44])
    parser.add_argument("--loss-type", type=str, required=True, choices=["standard", "masked"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--crops-per-site", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    output_dir = PROJECT_ROOT / args.output_dir
    checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir

    assert "scenario_2_multiseed" in str(output_dir), f"Unsafe output directory: {output_dir}"
    assert "scenario_2_multiseed" in str(checkpoint_dir), f"Unsafe checkpoint directory: {checkpoint_dir}"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = checkpoint_dir / "best.pt"
    last_ckpt_path = checkpoint_dir / "last.pt"

    train_ds = Scenario2Dataset(split="train", mode="sync", return_mask=True)
    val_ds   = Scenario2Dataset(split="validation", mode="sync", return_mask=True)
    test_ds  = Scenario2Dataset(split="test", mode="sync", return_mask=True)

    train_sites = train_ds.sites
    val_sites   = val_ds.sites
    test_sites  = test_ds.sites

    site_dims = {s: train_ds._arrays(s)["s1_t1"].shape[:2] for s in train_sites}

    schedule = generate_crop_schedule(
        sites=train_sites,
        site_dims=site_dims,
        crops_per_site=args.crops_per_site,
        crop_size=args.crop_size,
        epochs=args.epochs,
        seed=args.seed,
    )

    crop_dataset = ScheduledCropScenario2(train_ds, crop_size=args.crop_size)

    torch.manual_seed(args.seed)
    model = DualStreamUNet(
        base_channels=args.base_channels,
        sar_channels=SAR_CHANNELS,
        opt_channels=OPT_CHANNELS * 2,
    ).to(device)

    param_count = count_parameters(model)
    pos_weight = torch.tensor([POS_WEIGHT]).to(device)

    if args.loss_type == "masked":
        criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config_record = {
        "experiment": f"Sync CNN ({args.loss_type.upper()}) Seed {args.seed}",
        "model": "DualStreamUNet",
        "loss_type": args.loss_type,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "crop_size": args.crop_size,
        "crops_per_site": args.crops_per_site,
        "pos_weight": POS_WEIGHT,
        "parameter_count": param_count,
        "sar_channels": SAR_CHANNELS,
        "optical_channels": OPT_CHANNELS * 2,
        "total_input_channels": SAR_CHANNELS + OPT_CHANNELS * 2,
        "checkpoint_selection": "Highest validation F1 under Valid-Pixel Protocol (val_f1_valid)",
        "train_sites": train_sites,
        "validation_sites": val_sites,
        "test_sites": test_sites,
        "device": str(device),
        "gpu_name": gpu_name,
        "pytorch_version": str(torch.__version__),
    }

    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_record, f, sort_keys=False)

    print("=" * 80)
    print(f"SCENARIO 2 MULTI-SEED SYNC CNN: {args.loss_type.upper()} LOSS (SEED {args.seed})")
    print("=" * 80)
    print(f"Model:          DualStreamUNet ({param_count:,} params)")
    print(f"Input channels: 24 (4 SAR + 20 Optical)")
    print(f"Loss function:  {criterion.__class__.__name__} (pos_weight={POS_WEIGHT})")
    print(f"Split:          {len(train_sites)} train / {len(val_sites)} val / {len(test_sites)} test")
    print(f"Crops/epoch:    {len(train_sites) * args.crops_per_site:,} (from {len(train_sites)} sites)")
    print("Checkpoints:   ", str(checkpoint_dir).encode("ascii", "replace").decode())
    print("Output dir:    ", str(output_dir).encode("ascii", "replace").decode(), "\n", flush=True)

    if args.verify_only:
        print("Verification check completed successfully. Exiting before training.")
        return

    history = []
    best_f1_valid = -1.0
    best_epoch = None

    t_train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        crop_dataset.set_epoch_schedule(schedule[epoch])

        loader_gen = torch.Generator()
        loader_gen.manual_seed(args.seed * 1000 + epoch)
        loader = DataLoader(
            crop_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=loader_gen,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        train_loss = train_one_epoch(model, loader, criterion, optimizer, device, args.loss_type)
        epoch_time = time.perf_counter() - t0

        val_all, val_valid, _, val_warn = evaluate_dual_protocol(
            model, val_ds, val_sites, device, threshold=0.50
        )

        f1_valid = val_valid["f1"]
        f1_all   = val_all["f1"]

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_time_s": round(epoch_time, 2),
            "lr": optimizer.param_groups[0]["lr"],
            "val_f1_valid": f1_valid,
            "val_prec_valid": val_valid["precision"],
            "val_rec_valid": val_valid["recall"],
            "val_iou_valid": val_valid["iou"],
            "val_far_valid": val_valid["false_alarm_rate"],
            "val_f1_all": f1_all,
            "val_prec_all": val_all["precision"],
            "val_rec_all": val_all["recall"],
            "val_iou_all": val_all["iou"],
            "val_far_all": val_all["false_alarm_rate"],
        }
        history.append(history_row)

        print(
            f"epoch {epoch:>2}/{args.epochs}  loss {train_loss:.4f}  "
            f"val_f1_valid {f1_valid if f1_valid is not None else 'None':>7}  "
            f"val_f1_all {f1_all if f1_all is not None else 'None':>7}  "
            f"({epoch_time:.1f}s)",
            flush=True,
        )

        save_checkpoint(last_ckpt_path, model, epoch, f1_valid or 0.0, config_record)

        if f1_valid is not None and f1_valid > best_f1_valid:
            best_f1_valid = f1_valid
            best_epoch = epoch
            save_checkpoint(best_ckpt_path, model, epoch, f1_valid, config_record)

    total_train_time = time.perf_counter() - t_train_start

    saved = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)

    calibrated_tau = calibrate_threshold_on_validation(model, val_ds, val_sites, device)
    print(f"\nBest checkpoint loaded: Epoch {best_epoch} (Val F1 valid: {best_f1_valid:.4f})")
    print(f"Validation-calibrated tau*: {calibrated_tau:.3f}")

    t_eval_start = time.perf_counter()
    test_all_05, test_valid_05, per_site_05, test_warn = evaluate_dual_protocol(
        model, test_ds, test_sites, device, threshold=0.50
    )
    test_all_cal, test_valid_cal, per_site_cal, _ = evaluate_dual_protocol(
        model, test_ds, test_sites, device, threshold=calibrated_tau
    )
    eval_time = time.perf_counter() - t_eval_start

    if per_site_05:
        with (output_dir / "per_city_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_site_05[0].keys()))
            writer.writeheader()
            writer.writerows(per_site_05)

    if history:
        with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else None

    metrics_doc = {
        "experiment": config_record["experiment"],
        "model": "DualStreamUNet",
        "loss_type": args.loss_type,
        "seed": args.seed,
        "validation_best_epoch": best_epoch,
        "validation_best_f1_valid": best_f1_valid,
        "calibrated_threshold_tau_star": calibrated_tau,
        "training_time_seconds": round(total_train_time, 2),
        "evaluation_time_seconds": round(eval_time, 2),
        "peak_gpu_memory_mb": round(peak_mem_mb, 2) if peak_mem_mb is not None else None,
        "test_threshold_0_5": {
            "valid_pixel_protocol": test_valid_05,
            "all_pixel_protocol": test_all_05,
        },
        "test_threshold_calibrated": {
            "threshold": calibrated_tau,
            "valid_pixel_protocol": test_valid_cal,
            "all_pixel_protocol": test_all_cal,
        },
        "warnings": test_warn,
        "config": config_record,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_doc, f, indent=2)

    print("\nSaved all metrics and checkpoints cleanly to", str(output_dir).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()


