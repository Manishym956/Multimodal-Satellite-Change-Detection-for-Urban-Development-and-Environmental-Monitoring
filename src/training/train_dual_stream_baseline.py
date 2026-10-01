"""Train the Dual Stream U-Net baseline on the fixed OSCD city-level split.

Architecture: DualStreamUNet from src/models/dual_stream_unet.py.
  - Separate SAR encoder (S1 VV T1 + T2, 2 ch)
  - Separate Optical encoder (S2 13 bands × 2 dates, 26 ch)
  - Shared decoder operating on concatenated skip features
  - 1×1 head for binary change logit

Input tensor layout (28 channels, same as the fusion baseline):
    [0]     S1 VV T1
    [1]     S1 VV T2
    [2:15]  S2 13 bands T1
    [15:28] S2 13 bands T2

Protocol matches all other baselines exactly:
  - same 10/4/10 city split, same VALIDATION_SPLIT_SEED = 42
  - same random seed 42 for dataloader
  - same batch size 64, crop size 32, crops-per-city 64
  - same learning rate 1e-4, Adam, BCEWithLogitsLoss
  - same 50 epoch budget
  - checkpoint on best validation F1 (full-city images)
  - single final test pass on official 10 test cities
  - per-city metrics: pooled Precision, Recall, F1, IoU, FAR
  - unweighted mean and sample std across cities

Output:
  experiments/dual_stream_baseline/config.yaml
  experiments/dual_stream_baseline/training_history.csv
  experiments/dual_stream_baseline/per_city_metrics.csv
  experiments/dual_stream_baseline/metrics.json
  checkpoints/dual_stream_baseline/best.pt

WARNING: Do NOT modify this script to point at existing baseline directories.
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.oscd_dataset import ORBITS, OSCDDataset, SENTINEL2_BANDS
from src.evaluation.metrics import FAR_DEFINITION, counts_from_logits, rates_from_counts
from src.models.dual_stream_unet import DualStreamUNet, count_parameters
from src.training.train_sar_baseline import (
    CitySubset,
    assert_city_separation,
    sample_at_orbit,
    save_checkpoint,
    save_city_metrics,
    save_history,
    set_seed,
    split_train_cities,
    train_one_epoch,
)
# reuse the fusion channel-order helper; it is a pure function with no side effects
from src.training.train_fusion_baseline import fuse_dates

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "dual_stream_baseline"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "dual_stream_baseline" / "best.pt"

# 28 channels: S1 T1, S1 T2, S2 T1 (13 bands), S2 T2 (13 bands)
INPUT_CHANNELS = 28
SAR_CHANNELS = 2       # S1 VV T1 + T2
OPT_CHANNELS = 26      # S2 13 bands × 2 dates

CHANNEL_ORDER = (
    ["sentinel1_t1_vv", "sentinel1_t2_vv"]
    + [f"sentinel2_t1_{band}" for band in SENTINEL2_BANDS]
    + [f"sentinel2_t2_{band}" for band in SENTINEL2_BANDS]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Dual Stream U-Net baseline (Hafner-inspired)."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--crops-per-city", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=64,
                        help="Encoder base channels (Hafner default: 64).")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset wrapper (reuses fuse_dates to get the canonical 28-channel tensor)
# ---------------------------------------------------------------------------

class RandomCropDualStream(Dataset):
    """Aligned 32×32 crops in the 28-channel fusion order.

    Identical to RandomCropFusion in train_fusion_baseline.py.
    The DualStreamUNet.forward splits the tensor internally.
    """

    def __init__(self, base: Dataset, crop_size: int, crops_per_city: int) -> None:
        self.base = base
        self.crop_size = crop_size
        self.crops_per_city = crops_per_city

    def __len__(self) -> int:
        return len(self.base) * self.crops_per_city

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index % len(self.base)]
        t1 = sample["t1_img"]
        t2 = sample["t2_img"]
        label = sample["label"]
        _, height, width = t1.shape
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(
                f"{sample['city']} is smaller than the {self.crop_size} crop"
            )
        top = int(torch.randint(0, height - self.crop_size + 1, (1,)).item())
        left = int(torch.randint(0, width - self.crop_size + 1, (1,)).item())
        bottom = top + self.crop_size
        right = left + self.crop_size
        image = fuse_dates(
            t1[:, top:bottom, left:right],
            t2[:, top:bottom, left:right],
        )
        return {
            "image": image,
            "label": label[:, top:bottom, left:right],
            "city": sample["city"],
            "orbit": sample["orbit"],
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_dual_stream(
    model: DualStreamUNet,
    dataset: OSCDDataset,
    cities: list[str],
    device: torch.device,
) -> tuple[dict, list[dict], list[str]]:
    """Full-image evaluation over a list of cities."""
    model.eval()
    pooled = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    rows: list[dict] = []
    warnings: list[str] = []

    for city in cities:
        index = dataset.cities.index(city)
        orbit = int(ORBITS[city][0])
        sample = sample_at_orbit(dataset, index, orbit)

        # Build the 28-channel tensor (identical to fusion baseline)
        image = fuse_dates(sample["t1_img"], sample["t2_img"]).unsqueeze(0).to(device)
        label = sample["label"].unsqueeze(0).to(device)

        if not torch.isfinite(image).all():
            warnings.append(f"{city}: non-finite input")

        logits = model(image)

        if not torch.isfinite(logits).all():
            warnings.append(f"{city}: non-finite logits")

        counts = counts_from_logits(logits, label)
        for key in pooled:
            pooled[key] += counts[key]
        rows.append({"city": city, "orbit": orbit, **counts, **rates_from_counts(counts)})

    return {**pooled, **rates_from_counts(pooled)}, rows, warnings


# ---------------------------------------------------------------------------
# Per-city summary
# ---------------------------------------------------------------------------

def _finite_number(value) -> bool:
    return value is not None and math.isfinite(value)


def summarize_cities(rows: list[dict]) -> tuple[dict, list[str]]:
    """Unweighted mean and sample std.  Undefined rates are omitted."""
    summary: dict = {}
    warnings: list[str] = []
    for metric in ("precision", "recall", "f1", "iou", "false_alarm_rate"):
        values: list[float] = []
        undefined: list[str] = []
        for row in rows:
            value = row[metric]
            if not _finite_number(value):
                undefined.append(row["city"])
                if value is not None:
                    warnings.append(f"{row['city']} {metric} is not finite")
                continue
            values.append(float(value))
        summary[metric] = {
            "n": len(values),
            "undefined_cities": undefined,
            "mean": statistics.fmean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        }
    return summary, warnings


# ---------------------------------------------------------------------------
# Reproducibility record
# ---------------------------------------------------------------------------

def reproducibility_record(
    args: argparse.Namespace,
    device: torch.device,
    parameter_count: int,
    train_cities: list[str],
    validation_cities: list[str],
    test_cities: list[str],
) -> dict:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "seed": int(args.seed),
        "pytorch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": None if gpu_name is None else str(gpu_name),
        "device": str(device),
        "learning_rate": float(args.lr),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "crop_size": int(args.crop_size),
        "crops_per_city": int(args.crops_per_city),
        "loss_function": "BCEWithLogitsLoss",
        "optimizer": "Adam",
        "model": (
            "DualStreamUNet — controlled re-implementation inspired by "
            "Hafner et al. 2021 (IEEE GRSL doi:10.1109/LGRS.2021.3119856). "
            "NOT verified as an exact reproduction."
        ),
        "modality": "bitemporal Sentinel-1 VV (2 ch) and Sentinel-2 13 bands × 2 dates (26 ch)",
        "sar_input_channels": SAR_CHANNELS,
        "optical_input_channels": OPT_CHANNELS,
        "total_input_channels": INPUT_CHANNELS,
        "base_channels": int(args.base_channels),
        "channel_order": list(CHANNEL_ORDER),
        "output_channels": 1,
        "parameter_count": int(parameter_count),
        "threshold": 0.5,
        "false_alarm_rate": str(FAR_DEFINITION),
        "validation_split_seed": 42,
        "train_cities": [str(c) for c in train_cities],
        "validation_cities": [str(c) for c in validation_cities],
        "test_cities": [str(c) for c in test_cities],
        "training_orbit": (
            "np.random.choice of ORBITS[city] inside OSCDDataset, after the seed is set"
        ),
        "evaluation_orbit": "first orbit listed for each city",
        "checkpoint_selection": "highest validation F1 on full validation-city images",
        "test_metric_aggregation": (
            "pixel counts pooled across the official test cities, once, after training"
        ),
        "city_summary": "unweighted mean and sample standard deviation, omitting undefined rates",
        "hafner_differences": [
            "loss: BCEWithLogitsLoss instead of JaccardLikeLoss",
            "no random flip / rotate augmentation",
            "50 epochs instead of 390",
            "lr 1e-4 instead of 5e-5",
            "importance-crop oversampling not used",
            "shared decoder processes fused skip features at all levels "
            "(not only at the final level as in the reference OutConv)",
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    official_train = OSCDDataset(split="train", mode="fusion")
    official_test = OSCDDataset(split="test", mode="fusion")
    train_cities, validation_cities = split_train_cities(official_train.cities)
    test_cities = list(official_test.cities)
    assert_city_separation(train_cities, validation_cities, test_cities)

    # Training crop loader
    crops = RandomCropDualStream(
        CitySubset(official_train, train_cities),
        args.crop_size,
        args.crops_per_city,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        crops,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=loader_generator,
    )

    # Model
    model = DualStreamUNet(base_channels=args.base_channels).to(device)
    parameter_count = count_parameters(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    record = reproducibility_record(
        args, device, parameter_count, train_cities, validation_cities, test_cities
    )

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with (EXPERIMENT_DIR / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(record, handle, sort_keys=False)

    print("=== PRE-TRAINING VERIFICATION ===")
    print(f"1. Dataset split: {len(train_cities)} train, {len(validation_cities)} validation, {len(test_cities)} test cities")
    print(f"   Train cities: {train_cities}")
    print(f"   Validation cities: {validation_cities}")
    print(f"   Test cities: {test_cities}")
    print(f"2. Number of training/validation/test cities: {len(train_cities)} / {len(validation_cities)} / {len(test_cities)}")
    print(f"3. Model parameter count: {parameter_count:,}")
    print(f"4. Input shape: ({args.batch_size}, {INPUT_CHANNELS}, {args.crop_size}, {args.crop_size}) [SAR: {SAR_CHANNELS} ch, Optical: {OPT_CHANNELS} ch]")
    print(f"5. GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"6. CUDA availability: {torch.cuda.is_available()}")
    print(f"7. Number of training crops: {len(crops)} ({len(train_cities)} cities * {args.crops_per_city} crops/city)")
    print("=================================\n", flush=True)

    # Training loop
    history: list[dict] = []
    best_f1 = -1.0
    best_epoch: int | None = None

    start_train_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        validation_metrics, _, val_warnings = evaluate_dual_stream(
            model, official_train, validation_cities, device
        )
        validation_f1 = validation_metrics["f1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_precision": validation_metrics["precision"],
                "validation_recall": validation_metrics["recall"],
                "validation_f1": validation_f1,
                "validation_iou": validation_metrics["iou"],
                "validation_false_alarm_rate": validation_metrics["false_alarm_rate"],
            }
        )
        print(
            f"epoch {epoch} loss {train_loss:.4f} validation_f1 {validation_f1}",
            flush=True,
        )
        if val_warnings:
            print("validation warnings: " + "; ".join(val_warnings), flush=True)
        if validation_f1 is not None and validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            save_checkpoint(CHECKPOINT_PATH, model, epoch, validation_f1, record)
    training_time = time.perf_counter() - start_train_time

    if best_epoch is None or not CHECKPOINT_PATH.is_file():
        raise RuntimeError("training produced no checkpoint")

    # Load best checkpoint and run single test pass
    saved = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)

    start_eval_time = time.perf_counter()
    pooled, city_rows, test_warnings = evaluate_dual_stream(
        model, official_test, test_cities, device
    )
    eval_time = time.perf_counter() - start_eval_time

    city_summary, summary_warnings = summarize_cities(city_rows)
    all_warnings = test_warnings + summary_warnings

    save_history(EXPERIMENT_DIR / "training_history.csv", history)
    save_city_metrics(EXPERIMENT_DIR / "per_city_metrics.csv", city_rows)

    metrics_document = {
        "validation_best_epoch": best_epoch,
        "validation_best_f1": best_f1,
        "final_test_precision": pooled["precision"],
        "final_test_recall": pooled["recall"],
        "final_test_f1": pooled["f1"],
        "final_test_iou": pooled["iou"],
        "final_test_far": pooled["false_alarm_rate"],
        "false_alarm_rate_definition": FAR_DEFINITION,
        "threshold": 0.5,
        "test_counts": {key: pooled[key] for key in ("tp", "tn", "fp", "fn")},
        "per_city_mean_std": city_summary,
        "training_time_seconds": round(training_time, 2),
        "evaluation_time_seconds": round(eval_time, 2),
        "warnings": all_warnings,
        "reproducibility": record,
    }
    with (EXPERIMENT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_document, handle, indent=2)
        handle.write("\n")

    print(f"best validation epoch {best_epoch} validation_f1 {best_f1}")
    print(f"final test f1 {pooled['f1']}")
    print(f"training time: {training_time:.2f}s")
    print(f"evaluation time: {eval_time:.2f}s")
    print("checkpoint saved to checkpoints/dual_stream_baseline/best.pt")


if __name__ == "__main__":
    main()
