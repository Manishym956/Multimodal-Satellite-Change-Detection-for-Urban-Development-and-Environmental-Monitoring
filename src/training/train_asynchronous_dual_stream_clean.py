"""Train the Clean Asynchronous Dual Stream U-Net baseline on the fixed OSCD split.

Research Question:
    Isolating whether the performance degradation in the naive asynchronous setup
    was caused by the zero-filled placeholder representation of missing T2 optical channels.

Architecture:
    SAR Encoder:      2 channels (S1_T1 VV + S1_T2 VV)
    Optical Encoder:  13 channels (historical S2_T1 only; NO zero-filled channels)
    Fusion:           Concatenation at bottleneck and all skip levels (1024, 512, 256, 128)
    Shared Decoder:   Identical to the synchronous baseline
    Head:             Conv2d(128, 1, 1)

Input:
    Total channels: 15 (2 SAR + 13 Optical)
    Channels 0-1:   S1_T1, S1_T2
    Channels 2-14:  S2_T1 (13 bands)

Outputs:
    checkpoints/asynchronous_dual_stream_clean/best.pt
    experiments/asynchronous_dual_stream_clean/
        config.yaml
        training_history.csv
        per_city_metrics.csv
        metrics.json
"""

import argparse
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
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.oscd_dataset import (
    ORBITS,
    OSCD_ROOT,
    SENTINEL1_BANDS,
    SENTINEL2_BANDS,
    combine_bands,
    load_cities,
    read_change_mask,
    resize_sentinel1,
)
from src.evaluation.metrics import FAR_DEFINITION, counts_from_logits, rates_from_counts
from src.models.dual_stream_unet import count_parameters
from src.models.dual_stream_unet_clean import DualStreamUNetClean
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

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "asynchronous_dual_stream_clean"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "asynchronous_dual_stream_clean" / "best.pt"

INPUT_CHANNELS = 15
SAR_CHANNELS = 2    # S1 VV T1 (ch 0) + S1 VV T2 (ch 1)
OPT_CHANNELS = 13   # S2 T1 (13 bands, ch 2:15)

CHANNEL_ORDER = (
    ["sentinel1_t1_vv", "sentinel1_t2_vv"]
    + [f"sentinel2_t1_{band}" for band in SENTINEL2_BANDS]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Clean Asynchronous Dual Stream U-Net baseline."
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
                        help="Perform verification checks and exit without training.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Clean Asynchronous Dataset
# ---------------------------------------------------------------------------

def preprocess_city_clean_async(city: str, orbit: int | None = None) -> dict[str, np.ndarray]:
    """Load ONLY S1 T1, S1 T2, S2 T1, and label.

    S2 T2 (imgs_2_rect) is NEVER accessed, opened, or read from disk.
    NO zero-filled channels are generated.
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

    return {
        "sentinel1_t1": sentinel1_t1.astype(np.float32, copy=False),
        "sentinel1_t2": sentinel1_t2.astype(np.float32, copy=False),
        "sentinel2_t1": sentinel2_t1.astype(np.float32, copy=False),
        "label": label,
        "orbit": np.int64(orbit),
    }


class CleanAsynchronousOSCDDataset(Dataset):
    """Full-city clean asynchronous samples.

    Constructs a 15-channel tensor:
        Channels 0-1:  S1_T1, S1_T2 (2 channels)
        Channels 2-14: S2_T1 13 bands (13 channels)
    NO zero-filled placeholder channels.
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
            self._cache[key] = preprocess_city_clean_async(city, orbit)
        return self._cache[key]

    def __getitem__(self, index: int) -> dict:
        city = self.cities[index]
        orbit = self.fixed_orbit if self.fixed_orbit is not None else int(np.random.choice(ORBITS[city]))
        arrays = self._arrays(city, orbit)

        # Convert to channel-first PyTorch tensors
        s1_t1 = TF.to_tensor(arrays["sentinel1_t1"])        # (1, H, W)
        s1_t2 = TF.to_tensor(arrays["sentinel1_t2"])        # (1, H, W)
        s2_t1 = TF.to_tensor(arrays["sentinel2_t1"])        # (13, H, W)
        label_float = np.asarray(arrays["label"], dtype=np.float32)[:, :, np.newaxis]
        label = TF.to_tensor(label_float)                   # (1, H, W) in {0.0, 1.0}

        # Assemble clean 15-channel tensor
        # [0]: S1_T1, [1]: S1_T2, [2:15]: S2_T1
        image = torch.cat((s1_t1, s1_t2, s2_t1), dim=0)

        # Invariant checks:
        assert image.shape[0] == INPUT_CHANNELS, f"Expected {INPUT_CHANNELS} channels, got {image.shape[0]}"

        return {
            "image": image,
            "label": label,
            "city": city,
            "orbit": orbit,
        }


# ---------------------------------------------------------------------------
# Training Crop Dataset
# ---------------------------------------------------------------------------

class RandomCropCleanAsynchronous(Dataset):
    """Draw aligned 32x32 crops from full-city 15-channel rasters."""

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

        assert crop_image.shape == (INPUT_CHANNELS, self.crop_size, self.crop_size)

        return {
            "image": crop_image,
            "label": crop_label,
            "city": sample["city"],
            "orbit": sample["orbit"],
        }


# ---------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------

def sample_at_orbit_clean_async(dataset: CleanAsynchronousOSCDDataset, index: int, orbit: int) -> dict:
    previous = dataset.fixed_orbit
    dataset.fixed_orbit = orbit
    try:
        return dataset[index]
    finally:
        dataset.fixed_orbit = previous


@torch.inference_mode()
def evaluate_clean_asynchronous(
    model: DualStreamUNetClean,
    dataset: CleanAsynchronousOSCDDataset,
    cities: list[str],
    device: torch.device,
) -> tuple[dict, list[dict], list[str]]:
    """Full-image clean asynchronous evaluation over a list of cities."""
    model.eval()
    pooled = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    rows: list[dict] = []
    warnings: list[str] = []

    for city in cities:
        index = dataset.cities.index(city)
        orbit = int(ORBITS[city][0])
        sample = sample_at_orbit_clean_async(dataset, index, orbit)

        image = sample["image"].unsqueeze(0).to(device)  # (1, 15, H, W)
        label = sample["label"].unsqueeze(0).to(device)  # (1, 1, H, W)

        assert image.shape[1] == INPUT_CHANNELS, f"{city}: Expected {INPUT_CHANNELS} channels, got {image.shape[1]}"

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
# City Summary Helpers
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
        "experiment": "Clean Asynchronous Dual Stream U-Net baseline",
        "research_question": (
            "Isolating whether the performance degradation in the naive asynchronous setup "
            "was caused by the zero-filled placeholder representation of missing T2 optical channels."
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
        "model": "DualStreamUNetClean (src/models/dual_stream_unet_clean.py)",
        "modality": "Clean Asynchronous Multimodal: S1 VV (T1+T2, 2 ch) and historical S2 (T1 only, 13 ch)",
        "sar_input_channels": SAR_CHANNELS,
        "optical_input_channels": OPT_CHANNELS,
        "total_input_channels": INPUT_CHANNELS,
        "base_channels": int(args.base_channels),
        "channel_order": list(CHANNEL_ORDER),
        "optical_t2_handling": "NO S2 T2 loaded or used; optical encoder directly takes 13 historical bands with NO zero-filled channels",
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
# Pre-training Verification Function
# ---------------------------------------------------------------------------

def run_pre_training_verification(loader: DataLoader, model: nn.Module, device: torch.device) -> None:
    print("=================================================================")
    print("CLEAN ASYNCHRONOUS DUAL STREAM PRE-TRAINING VERIFICATION")
    print("=================================================================\n")

    # 1. Fetch sample batch
    batch = next(iter(loader))
    image = batch["image"]  # (B, 15, 32, 32)
    label = batch["label"]  # (B, 1, 32, 32)

    # 1. SAR input = 2 channels
    s1_channels = image[:, :2]
    assert s1_channels.shape[1] == 2, f"Expected 2 SAR channels, got {s1_channels.shape[1]}"
    print(f"1. SAR input channels:     {s1_channels.shape[1]} (S1_T1, S1_T2) [shape: {tuple(s1_channels.shape)}]")

    # 2. Optical input = 13 channels
    s2_channels = image[:, 2:]
    assert s2_channels.shape[1] == 13, f"Expected 13 Optical channels, got {s2_channels.shape[1]}"
    print(f"2. Optical input channels: {s2_channels.shape[1]} (S2_T1 13 bands) [shape: {tuple(s2_channels.shape)}]")

    # 3. No S2 T2 data is loaded
    print("3. No S2 T2 data loaded:   imgs_2_rect directory is NEVER accessed or read.")

    # 4. No zero-filled optical channels are used
    total_ch = image.shape[1]
    assert total_ch == 15, f"Expected exactly 15 channels, got {total_ch}"
    print(f"4. Zero-filled channels:   NONE (total input channels is strictly 15: 2 SAR + 13 Optical)")

    # Channel ranges
    print(f"\nChannel Value Ranges:")
    print(f"  SAR T1 range:           [{s1_channels[:, 0].min().item():.6f}, {s1_channels[:, 0].max().item():.6f}]")
    print(f"  SAR T2 range:           [{s1_channels[:, 1].min().item():.6f}, {s1_channels[:, 1].max().item():.6f}]")
    print(f"  historical S2 T1 range: [{s2_channels.min().item():.6f}, {s2_channels.max().item():.6f}]")
    print(f"  Label values:           {torch.unique(label).tolist()}")

    # 5. Forward output shape = 1 x 32 x 32
    model.eval()
    with torch.no_grad():
        x_single = image[0:1].to(device)
        out_single = model(x_single)
        assert out_single.shape == (1, 1, 32, 32), f"Expected (1, 1, 32, 32), got {out_single.shape}"
    print(f"\n5. Forward output shape:   {tuple(out_single.shape)} matches (1, 1, 32, 32)")

    # 6. GPU forward / backward works
    model.train()
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt.zero_grad()
    x_gpu = image.to(device)
    y_gpu = label.to(device)
    pred_gpu = model(x_gpu)
    loss_gpu = crit(pred_gpu, y_gpu)
    loss_gpu.backward()
    has_grad = all(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters() if p.requires_grad)
    opt.zero_grad()
    print(f"6. GPU forward/backward:   loss = {loss_gpu.item():.4f}, all params received gradients = {has_grad}")
    assert has_grad, "Gradient check failed!"

    param_count = count_parameters(model)
    print(f"\nModel Parameter Count:     {param_count:,}")
    print("=================================================================\n", flush=True)


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    official_train = CleanAsynchronousOSCDDataset(split="train")
    official_test = CleanAsynchronousOSCDDataset(split="test")
    train_cities, validation_cities = split_train_cities(official_train.cities)
    test_cities = list(official_test.cities)
    assert_city_separation(train_cities, validation_cities, test_cities)

    # Crop loader
    crops = RandomCropCleanAsynchronous(
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
    model = DualStreamUNetClean(base_channels=args.base_channels).to(device)
    parameter_count = count_parameters(model)

    # Run verification checks
    run_pre_training_verification(loader, model, device)

    if args.verify_only:
        print("Verify-only flag is set. Stopping before starting training as requested.")
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
    print("S1 T1: active (1 ch)")
    print("S1 T2: active (1 ch)")
    print("S2 T1: active (13 ch)")
    print("S2 T2: completely absent (no placeholder channels, optical encoder receives 13 ch)")
    print("=========================================\n")
    print(f"Dataset split: {len(train_cities)} train, {len(validation_cities)} validation, {len(test_cities)} test cities")
    print(f"Training crops: {len(crops)} total ({len(train_cities)} cities * {args.crops_per_city} crops/city)")
    print("Starting Clean Asynchronous Dual Stream U-Net training (50 epochs)...\n", flush=True)

    # Training loop
    history: list[dict] = []
    best_f1 = -1.0
    best_epoch: int | None = None

    start_train_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        validation_metrics, _, val_warnings = evaluate_clean_asynchronous(
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
    pooled, city_rows, test_warnings = evaluate_clean_asynchronous(
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
    print("checkpoint saved to checkpoints/asynchronous_dual_stream_clean/best.pt")


if __name__ == "__main__":
    main()
