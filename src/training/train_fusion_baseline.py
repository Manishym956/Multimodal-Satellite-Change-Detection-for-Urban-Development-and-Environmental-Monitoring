"""Synchronous SAR and optical fusion U-Net baseline.

The 28-channel input order is fixed:

0. Sentinel-1 T1 VV
1. Sentinel-1 T2 VV
2:15. Sentinel-2 T1, 13 bands
15:28. Sentinel-2 T2, 13 bands

OSCDDataset fusion mode stores each date as VV followed by 13 optical bands.
This script reorders those tensors. It does not change the dataset module or
the SAR and optical training scripts. The network is the existing U-Net with
28 input channels. The official test cities are scored once after training.
"""

import argparse
import json
import math
import statistics
import sys
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
from src.models.unet import UNet, count_parameters
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

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "fusion_baseline"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "fusion_baseline" / "best.pt"
INPUT_CHANNELS = 28
CHANNEL_ORDER = (
    ["sentinel1_t1_vv", "sentinel1_t2_vv"]
    + [f"sentinel2_t1_{band}" for band in SENTINEL2_BANDS]
    + [f"sentinel2_t2_{band}" for band in SENTINEL2_BANDS]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the synchronous SAR-optical fusion U-Net.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--crops-per-city", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def fuse_dates(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Reorder per-date fusion tensors into S1 T1, S1 T2, S2 T1, S2 T2."""
    if t1.ndim != 3 or t2.ndim != 3 or t1.shape[0] != 14 or t2.shape[0] != 14:
        raise ValueError(f"expected fusion dates (14, H, W), got {tuple(t1.shape)} and {tuple(t2.shape)}")
    if t1.shape[1:] != t2.shape[1:]:
        raise ValueError("Sentinel dates do not share spatial size")
    image = torch.cat((t1[0:1], t2[0:1], t1[1:], t2[1:]), dim=0)
    if image.shape[0] != INPUT_CHANNELS:
        raise ValueError(f"expected {INPUT_CHANNELS} channels, got {image.shape[0]}")
    return image


class RandomCropFusion(Dataset):
    """Aligned 32x32 crops, then the required 28-channel order."""

    def __init__(self, base: Dataset, crop_size: int, crops_per_city: int):
        self.base = base
        self.crop_size = crop_size
        self.crops_per_city = crops_per_city

    def __len__(self) -> int:
        return len(self.base) * self.crops_per_city

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.base[index % len(self.base)]
        t1 = sample["t1_img"]
        t2 = sample["t2_img"]
        label = sample["label"]
        _, height, width = t1.shape
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(f"{sample['city']} is smaller than the {self.crop_size} crop")
        top = int(torch.randint(0, height - self.crop_size + 1, (1,)).item())
        left = int(torch.randint(0, width - self.crop_size + 1, (1,)).item())
        bottom = top + self.crop_size
        right = left + self.crop_size
        image = fuse_dates(t1[:, top:bottom, left:right], t2[:, top:bottom, left:right])
        return {
            "image": image,
            "label": label[:, top:bottom, left:right],
            "city": sample["city"],
            "orbit": sample["orbit"],
        }


@torch.inference_mode()
def evaluate_fusion(model, dataset: OSCDDataset, cities: list[str], device) -> tuple[dict, list[dict], list[str]]:
    model.eval()
    pooled = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    rows = []
    warnings: list[str] = []
    for city in cities:
        index = dataset.cities.index(city)
        orbit = int(ORBITS[city][0])
        sample = sample_at_orbit(dataset, index, orbit)
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


def _finite_number(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def summarize_cities(rows: list[dict]) -> tuple[dict, list[str]]:
    """Unweighted mean and sample standard deviation. Undefined rates are omitted."""
    summary = {}
    warnings: list[str] = []
    for metric in ("precision", "recall", "f1", "iou", "false_alarm_rate"):
        values = []
        undefined = []
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
        "model": "standard single-stream U-Net, not DS-UNet",
        "modality": "synchronous Sentinel-1 VV and Sentinel-2",
        "input_channels": INPUT_CHANNELS,
        "channel_order": list(CHANNEL_ORDER),
        "output_channels": 1,
        "parameter_count": int(parameter_count),
        "threshold": 0.5,
        "false_alarm_rate": str(FAR_DEFINITION),
        "validation_split_seed": 42,
        "train_cities": [str(city) for city in train_cities],
        "validation_cities": [str(city) for city in validation_cities],
        "test_cities": [str(city) for city in test_cities],
        "training_orbit": "np.random.choice of ORBITS[city] inside OSCDDataset, after the seed is set",
        "evaluation_orbit": "first orbit listed for each city",
        "checkpoint_selection": "highest validation F1 on full validation-city images",
        "test_metric_aggregation": "pixel counts pooled across the official test cities, once, after training",
        "city_summary": "unweighted mean and sample standard deviation, omitting undefined rates",
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    official_train = OSCDDataset(split="train", mode="fusion")
    official_test = OSCDDataset(split="test", mode="fusion")
    train_cities, validation_cities = split_train_cities(official_train.cities)
    test_cities = list(official_test.cities)
    assert_city_separation(train_cities, validation_cities, test_cities)
    crops = RandomCropFusion(CitySubset(official_train, train_cities), args.crop_size, args.crops_per_city)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        crops,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=loader_generator,
    )
    model = UNet(in_channels=INPUT_CHANNELS, out_channels=1).to(device)
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

    history = []
    best_f1 = -1.0
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        validation_metrics, _, validation_warnings = evaluate_fusion(
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
        if validation_warnings:
            print("validation warnings: " + "; ".join(validation_warnings), flush=True)
        if validation_f1 is not None and validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            save_checkpoint(CHECKPOINT_PATH, model, epoch, validation_f1, record)

    if best_epoch is None or not CHECKPOINT_PATH.is_file():
        raise RuntimeError("training produced no checkpoint")
    saved = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)
    pooled, city_rows, test_warnings = evaluate_fusion(model, official_test, test_cities, device)
    city_summary, summary_warnings = summarize_cities(city_rows)
    warnings = test_warnings + summary_warnings
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
        "warnings": warnings,
        "reproducibility": record,
    }
    with (EXPERIMENT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_document, handle, indent=2)
        handle.write("\n")
    print(f"best validation epoch {best_epoch} validation_f1 {best_f1}")
    print(f"final test f1 {pooled['f1']}")
    print(f"checkpoint {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
