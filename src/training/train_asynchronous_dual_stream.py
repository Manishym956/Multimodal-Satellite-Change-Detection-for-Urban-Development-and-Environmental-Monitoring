"""Asynchronous Dual Stream U-Net baseline for multimodal change detection.

Research Question:
    Can historical Sentinel-2 optical imagery improve Sentinel-1-based urban change
    detection when contemporaneous Sentinel-2 imagery is unavailable?

Experimental Control:
    The synchronous Dual Stream U-Net baseline is the reference.
    Architecture: DualStreamUNet (src/models/dual_stream_unet.py), identical to the
    synchronous baseline (27,284,097 parameters).

Input Structure (28 channels total):
    Channels 0-1:
        S1_T1 (ch 0), S1_T2 (ch 1) -- both SAR observations available
    Channels 2-14:
        S2_T1 (13 bands, ch 2:15)  -- historical optical observation available
    Channels 15-27:
        EXACTLY ZERO (13 bands, ch 15:28) -- representing unavailable S2_T2

CRITICAL CONSTRAINT:
    The actual S2_T2 image (imgs_2_rect) must NEVER be loaded or passed to the model.
    In this script, AsynchronousOSCDDataset only ever reads imgs_1_rect from disk.
    Channels 15-27 are synthesized strictly in-memory as zeros.
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.oscd_dataset import (
    ORBITS,
    OSCD_ROOT,
    S1_ROOT,
    SENTINEL1_BANDS,
    SENTINEL2_BANDS,
    combine_bands,
    load_cities,
    read_change_mask,
    resize_sentinel1,
)
from src.evaluation.metrics import FAR_DEFINITION, counts_from_logits, rates_from_counts
from src.models.dual_stream_unet import DualStreamUNet, count_parameters
from src.training.train_sar_baseline import (
    CitySubset,
    assert_city_separation,
    save_checkpoint,
    save_city_metrics,
    save_history,
    set_seed,
    split_train_cities,
    train_one_epoch,
)

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "asynchronous_dual_stream"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "asynchronous_dual_stream" / "best.pt"

INPUT_CHANNELS = 28
SAR_CHANNELS = 2    # S1 VV T1 (ch 0) + S1 VV T2 (ch 1)
OPT_CHANNELS = 26   # S2 T1 (13 bands, ch 2:15) + S2 T2 placeholder (13 zero bands, ch 15:28)

CHANNEL_ORDER = (
    ["sentinel1_t1_vv", "sentinel1_t2_vv"]
    + [f"sentinel2_t1_{band}" for band in SENTINEL2_BANDS]
    + [f"sentinel2_t2_{band}_EXACTLY_ZERO_UNAVAILABLE" for band in SENTINEL2_BANDS]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Asynchronous Dual Stream U-Net baseline."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--crops-per-city", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true",
                        help="Perform full input/batch/model verification and exit without training.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Asynchronous Dataset (Never loads S2 T2 from disk)
# ---------------------------------------------------------------------------

def preprocess_city_asynchronous(city: str, orbit: int | None = None) -> dict[str, np.ndarray]:
    """Load S1 T1, S1 T2, S2 T1, and label.

    S2 T2 (imgs_2_rect) is NEVER accessed, opened, or read from disk.
    Placeholder zeros are generated in-memory.
    """
    if orbit is None:
        orbit = int(ORBITS[city][0])

    # ONLY load T1 optical (historical observations) from imgs_1_rect
    s2_t1_folder = OSCD_ROOT / "images" / city / "imgs_1_rect"
    sentinel2_t1 = combine_bands(s2_t1_folder)  # (H, W, 13)

    # Change mask ground truth
    label = read_change_mask(city)  # (H, W)
    height, width = label.shape

    # SAR observations for both dates
    sentinel1_t1 = resize_sentinel1(city, orbit, 1, height, width)  # (H, W, 1)
    sentinel1_t2 = resize_sentinel1(city, orbit, 2, height, width)  # (H, W, 1)

    # Synthetic zeros for missing S2 T2 - imgs_2_rect is NOT touched
    sentinel2_t2_zeros = np.zeros_like(sentinel2_t1)  # (H, W, 13)

    return {
        "sentinel1_t1": sentinel1_t1.astype(np.float32, copy=False),
        "sentinel1_t2": sentinel1_t2.astype(np.float32, copy=False),
        "sentinel2_t1": sentinel2_t1.astype(np.float32, copy=False),
        "sentinel2_t2_zeros": sentinel2_t2_zeros.astype(np.float32, copy=False),
        "label": label,
        "orbit": np.int64(orbit),
    }


class AsynchronousOSCDDataset(Dataset):
    """Full-city asynchronous samples.

    Constructs a 28-channel tensor:
        Channels 0-1:   S1_T1, S1_T2
        Channels 2-14:  S2_T1 (13 bands)
        Channels 15-27: EXACTLY ZERO (missing contemporary S2_T2)
    """

    def __init__(self, split: str = "train", orbit: int | None = None) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.split = split
        self.cities = load_cities(split)
        self.fixed_orbit = orbit
        self._cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.cities)

    def _arrays(self, city: str, orbit: int) -> dict[str, np.ndarray]:
        key = (city, orbit)
        if key not in self._cache:
            self._cache[key] = preprocess_city_asynchronous(city, orbit)
        return self._cache[key]

    def __getitem__(self, index: int) -> dict:
        city = self.cities[index]
        orbit = self.fixed_orbit if self.fixed_orbit is not None else int(np.random.choice(ORBITS[city]))
        arrays = self._arrays(city, orbit)

        # Convert to channel-first PyTorch tensors
        s1_t1 = TF.to_tensor(arrays["sentinel1_t1"])        # (1, H, W)
        s1_t2 = TF.to_tensor(arrays["sentinel1_t2"])        # (1, H, W)
        s2_t1 = TF.to_tensor(arrays["sentinel2_t1"])        # (13, H, W)
        s2_t2_zeros = torch.zeros_like(s2_t1)               # (13, H, W) EXACTLY ZERO
        label_float = np.asarray(arrays["label"], dtype=np.float32)[:, :, np.newaxis]
        label = TF.to_tensor(label_float)                   # (1, H, W) in {0.0, 1.0}

        # Assemble canonical 28-channel tensor
        # [0]: S1_T1, [1]: S1_T2, [2:15]: S2_T1, [15:28]: S2_T2_ZEROS
        image = torch.cat((s1_t1, s1_t2, s2_t1, s2_t2_zeros), dim=0)

        # Verification invariant:
        assert image.shape[0] == INPUT_CHANNELS, f"Expected {INPUT_CHANNELS} channels, got {image.shape[0]}"
        assert torch.all(image[15:28] == 0), "Channels 15:28 are not exactly zero!"

        return {
            "image": image,
            "label": label,
            "city": city,
            "orbit": orbit,
        }


# ---------------------------------------------------------------------------
# Training Crop Dataset
# ---------------------------------------------------------------------------

class RandomCropAsynchronous(Dataset):
    """Draw aligned 32x32 crops from full-city asynchronous rasters."""

    def __init__(self, base: Dataset, crop_size: int, crops_per_city: int) -> None:
        self.base = base
        self.crop_size = crop_size
        self.crops_per_city = crops_per_city

    def __len__(self) -> int:
        return len(self.base) * self.crops_per_city

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index % len(self.base)]
        image = sample["image"]
        label = sample["label"]
        _, height, width = image.shape
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(f"{sample['city']} is smaller than the {self.crop_size} crop")

        top = int(torch.randint(0, height - self.crop_size + 1, (1,)).item())
        left = int(torch.randint(0, width - self.crop_size + 1, (1,)).item())
        bottom = top + self.crop_size
        right = left + self.crop_size

        crop_image = image[:, top:bottom, left:right]
        crop_label = label[:, top:bottom, left:right]

        # Verify crop maintains zero invariant
        assert torch.all(crop_image[15:28] == 0), "Crop missing-optical channels are not zero!"

        return {
            "image": crop_image,
            "label": crop_label,
            "city": sample["city"],
            "orbit": sample["orbit"],
        }


# ---------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------

def sample_at_orbit_async(dataset: AsynchronousOSCDDataset, index: int, orbit: int) -> dict:
    previous = dataset.fixed_orbit
    dataset.fixed_orbit = orbit
    try:
        return dataset[index]
    finally:
        dataset.fixed_orbit = previous


@torch.inference_mode()
def evaluate_asynchronous(
    model: DualStreamUNet,
    dataset: AsynchronousOSCDDataset,
    cities: list[str],
    device: torch.device,
) -> tuple[dict, list[dict], list[str]]:
    """Full-image asynchronous evaluation over a list of cities."""
    model.eval()
    pooled = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    rows: list[dict] = []
    warnings: list[str] = []

    for city in cities:
        index = dataset.cities.index(city)
        orbit = int(ORBITS[city][0])
        sample = sample_at_orbit_async(dataset, index, orbit)

        image = sample["image"].unsqueeze(0).to(device)  # (1, 28, H, W)
        label = sample["label"].unsqueeze(0).to(device)  # (1, 1, H, W)

        # Runtime assertion on evaluation input:
        optical_missing_channels = image[:, 15:28]
        assert torch.all(optical_missing_channels == 0), f"{city}: S2 T2 missing channels are not zero!"

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
# Metric Aggregation
# ---------------------------------------------------------------------------

def _finite_number(value) -> bool:
    return value is not None and math.isfinite(value)


def summarize_cities(rows: list[dict]) -> tuple[dict, list[str]]:
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
        "experiment": "Asynchronous Dual Stream U-Net baseline",
        "research_question": (
            "Can historical Sentinel-2 optical imagery improve Sentinel-1-based urban "
            "change detection when contemporaneous Sentinel-2 imagery is unavailable?"
        ),
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
        "model": "DualStreamUNet (src/models/dual_stream_unet.py)",
        "modality": "Asynchronous multimodal: S1 VV (T1+T2) and historical S2 (T1 only; T2 zero-filled)",
        "sar_input_channels": SAR_CHANNELS,
        "optical_input_channels": OPT_CHANNELS,
        "total_input_channels": INPUT_CHANNELS,
        "base_channels": int(args.base_channels),
        "channel_order": list(CHANNEL_ORDER),
        "optical_t2_handling": "zero-filled placeholder channels (missing contemporary optical data, never loaded from disk)",
        "output_channels": 1,
        "parameter_count": int(parameter_count),
        "threshold": 0.5,
        "false_alarm_rate": str(FAR_DEFINITION),
        "validation_split_seed": 42,
        "train_cities": [str(c) for c in train_cities],
        "validation_cities": [str(c) for c in validation_cities],
        "test_cities": [str(c) for c in test_cities],
        "training_orbit": "np.random.choice of ORBITS[city] inside OSCDDataset, after the seed is set",
        "evaluation_orbit": "first orbit listed for each city",
        "checkpoint_selection": "highest validation F1 on full validation-city images",
        "test_metric_aggregation": "pixel counts pooled across the official test cities, once, after training",
        "city_summary": "unweighted mean and sample standard deviation, omitting undefined rates",
    }


# ---------------------------------------------------------------------------
# Detailed Verification Routine
# ---------------------------------------------------------------------------

def run_detailed_verification(loader: DataLoader, model: nn.Module, device: torch.device) -> None:
    print("=================================================================")
    print("ASYNCHRONOUS DUAL STREAM IMPLEMENTATION & INPUT VERIFICATION")
    print("=================================================================\n")

    # 1. Fetch sample training batch
    batch = next(iter(loader))
    image = batch["image"]  # (B, 28, 32, 32)
    label = batch["label"]  # (B, 1, 32, 32)
    batch_size = image.shape[0]

    # 2. Extract channel slices
    s1_t1 = image[:, 0:1]                    # Channels 0 (SAR T1)
    s1_t2 = image[:, 1:2]                    # Channels 1 (SAR T2)
    s2_t1 = image[:, 2:15]                   # Channels 2-14 (13 optical bands T1)
    optical_missing_channels = image[:, 15:28]  # Channels 15-27 (13 missing optical bands T2)

    # 3. Explicit runtime assertion on the sample batch
    assert torch.all(optical_missing_channels == 0), "FAIL: optical_missing_channels contains non-zero elements!"
    print("RUNTIME ASSERTION PASSED: assert torch.all(optical_missing_channels == 0)")
    print(f"Batch tensor shape: {tuple(image.shape)}")
    print(f"Labels tensor shape: {tuple(label.shape)}\n")

    # 4. Print detailed min / max statistics
    print("CHANNEL VALUE RANGES:")
    print(f"  SAR T1 range:                 [{s1_t1.min().item():.6f}, {s1_t1.max().item():.6f}]")
    print(f"  SAR T2 range:                 [{s1_t2.min().item():.6f}, {s1_t2.max().item():.6f}]")
    print(f"  historical S2 T1 range:       [{s2_t1.min().item():.6f}, {s2_t1.max().item():.6f}]")
    print(f"  missing S2 T2 channels min:   {optical_missing_channels.min().item():.1f}")
    print(f"  missing S2 T2 channels max:   {optical_missing_channels.max().item():.1f}")
    print(f"  missing S2 T2 min/max ratio:  {optical_missing_channels.min().item():.1f} / {optical_missing_channels.max().item():.1f} (EXPECTED: 0.0 / 0.0)\n")

    assert optical_missing_channels.min().item() == 0.0
    assert optical_missing_channels.max().item() == 0.0

    # 5. Model forward call inspection
    model.eval()
    with torch.no_grad():
        x_in = image.to(device)
        out = model(x_in)
        assert out.shape == (batch_size, 1, 32, 32)
    print("MODEL FORWARD CALL VERIFICATION:")
    print(f"  Model input:  {tuple(x_in.shape)} on {x_in.device}")
    print(f"  Model output: {tuple(out.shape)} on {out.device}")
    print(f"  DualStreamUNet splits input internally:")
    print(f"    - x_sar (channels 0:2)   -> SAR Encoder:     shape {tuple(x_in[:, :2].shape)}")
    print(f"    - x_opt (channels 2:28)  -> Optical Encoder: shape {tuple(x_in[:, 2:28].shape)}")
    print(f"      where x_opt[:, :13] is historical S2 T1 and x_opt[:, 13:] is EXACTLY ZERO.")
    print("\nALL VERIFICATION CHECKS PASSED PERFECTLY.")
    print("=================================================================\n", flush=True)


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    official_train = AsynchronousOSCDDataset(split="train")
    official_test = AsynchronousOSCDDataset(split="test")
    train_cities, validation_cities = split_train_cities(official_train.cities)
    test_cities = list(official_test.cities)
    assert_city_separation(train_cities, validation_cities, test_cities)

    # Crop loader
    crops = RandomCropAsynchronous(
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

    # Run detailed verification
    run_detailed_verification(loader, model, device)

    if args.verify_only:
        print("Verify-only flag is set. Stopping before starting 50-epoch training as requested.")
        return

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    record = reproducibility_record(
        args, device, parameter_count, train_cities, validation_cities, test_cities
    )

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with (EXPERIMENT_DIR / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(record, handle, sort_keys=False)

    print("=== FINAL CHANNEL STATUS CONFIRMATION ===")
    print("S1 T1: active")
    print("S1 T2: active")
    print("S2 T1: active")
    print("S2 T2: zero-filled")
    print("=========================================\n")
    print("Starting Asynchronous Dual Stream U-Net training (50 epochs)...\n", flush=True)

    # Training loop
    history: list[dict] = []
    best_f1 = -1.0
    best_epoch: int | None = None

    start_train_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        validation_metrics, _, val_warnings = evaluate_asynchronous(
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

    # Final evaluation on official test cities
    saved = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)

    start_eval_time = time.perf_counter()
    pooled, city_rows, test_warnings = evaluate_asynchronous(
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
    print("checkpoint saved to checkpoints/asynchronous_dual_stream/best.pt")


if __name__ == "__main__":
    main()
