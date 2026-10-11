"""Scenario 2 — Synchronous and Clean Asynchronous Dual-Stream CNN.

This script trains one of two CNN variants on the SEN12-MT Urban Mapping
dataset (Scenario 2) using the locked 20/4/4 site split.

  --mode async  (default)  Clean Async Dual-Stream CNN  — 14-ch input
  --mode sync              Synchronous Dual-Stream CNN  — 24-ch input

Architecture:
  Async: DualStreamUNetClean   SAR encoder (4 ch) + Optical encoder (10 ch)
  Sync:  DualStreamUNet        SAR encoder (4 ch) + Optical encoder (20 ch)

Key differences from Scenario 1:
  SAR:      4 channels (VV+VH × T1+T2) vs 2 ch (VV only)
  Optical:  10 bands vs 13 bands
  S2 norm:  already in [0, 1], no /10000
  pos_weight: 94 (computed from 20 training sites, ~1.06% change fraction)
  Sites:    20 train / 4 val / 4 test (from split_definition.json)

Invariants preserved from Scenario 1:
  seed=42, batch=64, lr=1e-4, crop=32, crops/site=64, Adam, BCE,
  threshold=0.5, best val F1 checkpoint, last.pt every epoch.

Outputs:
  checkpoints/scenario_2/{async_dual_stream|sync_dual_stream}/best.pt
  checkpoints/scenario_2/{async_dual_stream|sync_dual_stream}/last.pt
  experiments/scenario_2/{async_dual_stream|sync_dual_stream}/
      config.yaml  training_history.csv  per_city_metrics.csv  metrics.json
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
from src.evaluation.metrics import FAR_DEFINITION, counts_from_logits, rates_from_counts
from src.models.dual_stream_unet import DualStreamUNet, count_parameters
from src.models.dual_stream_unet_clean import DualStreamUNetClean
from src.training.train_sar_baseline import (
    save_checkpoint,
    save_city_metrics,
    save_history,
    set_seed,
    train_one_epoch,
)
from src.utils.device import add_device_argument, get_device

POS_WEIGHT = 94.0   # from training split: 4,718,348 no-change / 50,387 change = 93.6 → 94


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Dual-Stream CNN on SEN12-MT (Scenario 2)."
    )
    parser.add_argument("--mode",           type=str,   default="async",
                        choices=["async", "sync"],
                        help="async=14ch asynchronous, sync=24ch synchronous")
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--crop-size",      type=int,   default=32)
    parser.add_argument("--crops-per-site", type=int,   default=64)
    parser.add_argument("--base-channels",  type=int,   default=64)
    parser.add_argument("--num-workers",    type=int,   default=0)
    parser.add_argument("--verify-only",    action="store_true",
                        help="Run checks only; do not train.")
    add_device_argument(parser)
    return parser.parse_args()


def _exp_name(mode: str) -> str:
    return "async_dual_stream" if mode == "async" else "sync_dual_stream"


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(model, dataset: Scenario2Dataset, sites: list[str],
             device: torch.device) -> tuple[dict, list[dict], list[str]]:
    model.eval()
    pooled   = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    rows:     list[dict]  = []
    warnings: list[str]   = []

    for site_id in sites:
        idx    = dataset.sites.index(site_id)
        sample = dataset[idx]
        image  = sample["image"].unsqueeze(0).to(device)
        label  = sample["label"].unsqueeze(0).to(device)

        if not torch.isfinite(image).all():
            warnings.append(f"{site_id}: non-finite input")

        logits = model(image)

        if not torch.isfinite(logits).all():
            warnings.append(f"{site_id}: non-finite logits")

        counts = counts_from_logits(logits, label)
        for k in pooled:
            pooled[k] += counts[k]
        rows.append({"site": site_id, **counts, **rates_from_counts(counts)})

    return {**pooled, **rates_from_counts(pooled)}, rows, warnings


def _finite(v) -> bool:
    return v is not None and math.isfinite(v)


def summarize_sites(rows: list[dict]) -> tuple[dict, list[str]]:
    summary: dict = {}
    warnings: list[str] = []
    for metric in ("precision", "recall", "f1", "iou", "false_alarm_rate"):
        values, undefined = [], []
        for row in rows:
            v = row[metric]
            if not _finite(v):
                undefined.append(row["site"])
                if v is not None:
                    warnings.append(f"{row['site']} {metric} not finite")
                continue
            values.append(float(v))
        summary[metric] = {
            "n": len(values),
            "undefined_sites": undefined,
            "mean": statistics.fmean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        }
    return summary, warnings


def save_site_metrics(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── verification ──────────────────────────────────────────────────────────────

def run_verification(loader: DataLoader, model: nn.Module,
                     device: torch.device, mode: str) -> None:
    expected_ch = ASYNC_CHANNELS if mode == "async" else SYNC_CHANNELS
    print("=" * 65)
    print(f"SCENARIO 2 DUAL-STREAM CNN ({mode.upper()}) VERIFICATION")
    print("=" * 65)

    batch = next(iter(loader))
    image, label = batch["image"], batch["label"]

    # 1. Channel count
    assert image.shape[1] == expected_ch, (
        f"Expected {expected_ch} channels, got {image.shape[1]}"
    )
    print(f"1. Input channels:    {image.shape[1]}  "
          f"({SAR_CHANNELS} SAR + {OPT_CHANNELS} S2 T1"
          + (f" + {OPT_CHANNELS} S2 T2" if mode == "sync" else " [T2 withheld]")
          + ")")
    print(f"   SAR range [0-3]:   [{image[:, :4].min():.4f}, {image[:, :4].max():.4f}]")
    print(f"   S2 T1 range [4-13]:[{image[:, 4:14].min():.4f}, {image[:, 4:14].max():.4f}]")
    print(f"   Label values:      {torch.unique(label).tolist()}")

    # 2. S2 T2 status
    if mode == "async":
        print("2. S2 T2:             NOT LOADED (async mode)")
    else:
        print(f"2. S2 T2 range [14-23]:[{image[:, 14:].min():.4f}, {image[:, 14:].max():.4f}]")

    # 3. Forward pass
    model.eval()
    with torch.no_grad():
        out = model(image[:1].to(device))
    assert out.shape == (1, 1, 32, 32), f"Expected (1,1,32,32), got {out.shape}"
    assert torch.isfinite(out).all(), "Forward output contains NaN/Inf"
    print(f"3. 32×32 forward:     {tuple(out.shape)} on {out.device} ✓")

    # 4. Backward pass
    model.train()
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))
    opt   = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt.zero_grad()
    logits = model(image.to(device))
    loss   = crit(logits, label.to(device))
    loss.backward()
    has_grad = all(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters() if p.requires_grad
    )
    opt.zero_grad()
    assert has_grad, "Gradient check failed"
    print(f"4. Backward pass:     loss={loss.item():.4f}, grads present ✓")

    from src.models.dual_stream_unet import count_parameters as cp
    print(f"\nParameter count:      {cp(model):,}")
    print("=" * 65)
    print()


# ── reproducibility record ────────────────────────────────────────────────────

def _record(args, device, param_count, split) -> dict:
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    mode = args.mode
    exp  = _exp_name(mode)
    return {
        "scenario": 2,
        "experiment": exp,
        "dataset": "SEN12-MT Urban Mapping",
        "temporal_pairing": "Strategy B: T1=2018-03, T2=2019-12 (21 months)",
        "mode": mode,
        "model": (
            "DualStreamUNetClean" if mode == "async" else "DualStreamUNet"
        ),
        "modality": (
            "Async: SAR T1+T2 (4ch VV+VH) + historical S2 T1 (10ch)"
            if mode == "async" else
            "Sync: SAR T1+T2 (4ch VV+VH) + S2 T1 (10ch) + S2 T2 (10ch)"
        ),
        "s2_t2_handling": (
            "NOT LOADED — withheld from async model"
            if mode == "async" else
            "LOADED — synchronous baseline"
        ),
        "sar_channels": SAR_CHANNELS,
        "optical_channels": OPT_CHANNELS,
        "total_input_channels": ASYNC_CHANNELS if mode == "async" else SYNC_CHANNELS,
        "label_derivation": (
            "(buildings_T2 >= 0.5) XOR (buildings_T1 >= 0.5)"
        ),
        "label_threshold": 0.5,
        "seed": int(args.seed),
        "pytorch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
        "device": str(device),
        "learning_rate": float(args.lr),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "crop_size": int(args.crop_size),
        "crops_per_site": int(args.crops_per_site),
        "loss_function": f"BCEWithLogitsLoss(pos_weight={POS_WEIGHT})",
        "pos_weight": POS_WEIGHT,
        "pos_weight_rationale": (
            "20 training sites: 50,387 change / 4,718,348 no-change = 93.6, rounded to 94"
        ),
        "optimizer": "Adam",
        "base_channels": int(args.base_channels),
        "parameter_count": int(param_count),
        "threshold": 0.5,
        "false_alarm_rate": str(FAR_DEFINITION),
        "train_sites": split["train"],
        "validation_sites": split["validation"],
        "test_sites": split["test"],
        "checkpoint_selection": "highest validation F1",
        "test_metric_aggregation": "pixel counts pooled across 4 test sites, once after training",
        "city_summary": "unweighted mean and sample std, omitting undefined rates",
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    mode   = args.mode
    exp    = _exp_name(mode)

    EXPERIMENT_DIR  = PROJECT_ROOT / "experiments" / "scenario_2" / exp
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints"  / "scenario_2" / exp / "best.pt"

    split = load_split()
    train_sites = split["train"]
    val_sites   = split["validation"]
    test_sites  = split["test"]

    # Datasets
    train_ds = Scenario2Dataset(split="train",      mode=mode)
    val_ds   = Scenario2Dataset(split="validation", mode=mode)
    test_ds  = Scenario2Dataset(split="test",       mode=mode)

    crops = RandomCropScenario2(train_ds, args.crop_size, args.crops_per_site)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    loader = DataLoader(
        crops,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=loader_gen,
        pin_memory=(device.type == "cuda"),
    )

    # Model — use constructor channel overrides for both modes
    if mode == "async":
        model = DualStreamUNetClean(
            base_channels=args.base_channels,
            sar_channels=SAR_CHANNELS,        # 4 (VV+VH × 2 dates)
            opt_channels=OPT_CHANNELS,        # 10 (S2 T1 10 bands)
        ).to(device)
    else:
        # Sync: 4-ch SAR + 20-ch optical (10 bands × 2 dates)
        model = DualStreamUNet(
            base_channels=args.base_channels,
            sar_channels=SAR_CHANNELS,        # 4
            opt_channels=OPT_CHANNELS * 2,    # 20 (T1 + T2)
        ).to(device)
    param_count = count_parameters(model)

    run_verification(loader, model, device, mode)

    if args.verify_only:
        print("Verify-only flag set. Exiting before training.")
        return

    pos_weight = torch.tensor([POS_WEIGHT]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)

    record = _record(args, device, param_count, split)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with (EXPERIMENT_DIR / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(record, fh, sort_keys=False)

    print(f"Model:          {record['model']}")
    print(f"Mode:           {mode}  ({record['total_input_channels']} input channels)")
    print(f"Parameters:     {param_count:,}")
    print(f"pos_weight:     {POS_WEIGHT}")
    print(f"Split:          {len(train_sites)} train / {len(val_sites)} val / {len(test_sites)} test sites")
    print(f"Training crops: {len(crops)}  ({len(train_sites)} sites × {args.crops_per_site} crops)")
    print(f"S2 T2 status:   {'NOT LOADED' if mode == 'async' else 'LOADED (sync baseline)'}\n")
    print("Starting training...\n", flush=True)

    history: list[dict] = []
    best_f1 = -1.0
    best_epoch: int | None = None

    t_train = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        val_metrics, _, val_warn = evaluate(model, val_ds, val_sites, device)
        val_f1 = val_metrics["f1"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_f1,
            "validation_iou": val_metrics["iou"],
            "validation_false_alarm_rate": val_metrics["false_alarm_rate"],
        })
        print(f"epoch {epoch:>3}  loss {train_loss:.4f}  validation_f1 {val_f1}", flush=True)
        if val_warn:
            print("  warnings: " + "; ".join(val_warn), flush=True)

        last_path = CHECKPOINT_PATH.parent / "last.pt"
        save_checkpoint(last_path, model, epoch, val_f1, record)
        if val_f1 is not None and val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            save_checkpoint(CHECKPOINT_PATH, model, epoch, val_f1, record)

    training_time = time.perf_counter() - t_train

    last_path = CHECKPOINT_PATH.parent / "last.pt"
    ckpt_load = CHECKPOINT_PATH if CHECKPOINT_PATH.is_file() else last_path
    if not ckpt_load.is_file():
        raise RuntimeError("No checkpoint saved (neither best.pt nor last.pt).")
    if best_epoch is None:
        print("WARNING: val_f1 was None every epoch — loading last.pt for test evaluation.")
        ckpt_load = last_path

    saved = torch.load(ckpt_load, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)

    t_eval = time.perf_counter()
    pooled, city_rows, test_warn = evaluate(model, test_ds, test_sites, device)
    eval_time = time.perf_counter() - t_eval

    city_summary, sum_warn = summarize_sites(city_rows)
    all_warn = test_warn + sum_warn

    save_history(EXPERIMENT_DIR / "training_history.csv", history)
    save_site_metrics(EXPERIMENT_DIR / "per_city_metrics.csv", city_rows)

    metrics_doc = {
        "validation_best_epoch": best_epoch,
        "validation_best_f1": best_f1,
        "final_test_precision": pooled["precision"],
        "final_test_recall": pooled["recall"],
        "final_test_f1": pooled["f1"],
        "final_test_iou": pooled["iou"],
        "final_test_far": pooled["false_alarm_rate"],
        "false_alarm_rate_definition": FAR_DEFINITION,
        "threshold": 0.5,
        "test_counts": {k: pooled[k] for k in ("tp", "tn", "fp", "fn")},
        "per_site_mean_std": city_summary,
        "training_time_seconds": round(training_time, 2),
        "evaluation_time_seconds": round(eval_time, 2),
        "warnings": all_warn,
        "reproducibility": record,
    }
    with (EXPERIMENT_DIR / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_doc, fh, indent=2)
        fh.write("\n")

    print(f"\nbest validation epoch: {best_epoch}  validation_f1: {best_f1}")
    print(f"final test f1:         {pooled['f1']}")
    print(f"training time:         {training_time:.2f}s")
    print(f"evaluation time:       {eval_time:.2f}s")
    print(f"checkpoint:            checkpoints/scenario_2/{exp}/best.pt")


if __name__ == "__main__":
    main()
