"""Train a SAR-only U-Net baseline on a fixed city-level OSCD split.

The network is the standard U-Net in src/models/unet.py. It is not DS-UNet.
Inputs are the two Sentinel-1 dates already prepared by OSCDDataset, concatenated
on the channel axis to shape (2, H, W). Training samples random 32x32 crops
because OSCDDataset returns the full city. Validation and the single final test
pass use full city images. The official test cities are not used to select the
checkpoint.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.oscd_dataset import ORBITS, OSCDDataset
from src.evaluation.metrics import (
    FAR_DEFINITION,
    counts_from_logits,
    rates_from_counts,
)
from src.models.unet import UNet, count_parameters

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "sar_baseline"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "sar_baseline" / "best.pt"
FIGURE_DIR = PROJECT_ROOT / "figures" / "sar_baseline"
VALIDATION_SPLIT_SEED = 42
OFFICIAL_TRAIN_CITIES = [
    "aguasclaras",
    "bercy",
    "bordeaux",
    "nantes",
    "paris",
    "rennes",
    "saclay_e",
    "abudhabi",
    "cupertino",
    "pisa",
    "beihai",
    "hongkong",
    "beirut",
    "mumbai",
]
OFFICIAL_TEST_CITIES = [
    "brasilia",
    "montpellier",
    "norcia",
    "rio",
    "saclay_w",
    "valencia",
    "dubai",
    "lasvegas",
    "milano",
    "chongqing",
]
# random.Random(42).sample(OFFICIAL_TRAIN_CITIES, 4) on Python 3.11.
EXPECTED_VALIDATION_CITIES = ["beihai", "bercy", "aguasclaras", "paris"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SAR-only U-Net baseline.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--crops-per-city", type=int, default=64)
    parser.add_argument("--figure-city", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_train_cities(official_train: list[str]) -> tuple[list[str], list[str]]:
    """Hold out 4 training cities with a private Random(42) draw.

    The draw is checked against the cities selected on Python 3.11 so a later
    random-module change cannot silently move a city between train and validation.
    """
    if list(official_train) != OFFICIAL_TRAIN_CITIES:
        raise RuntimeError("the official 14-city OSCD training list changed")
    validation = random.Random(VALIDATION_SPLIT_SEED).sample(list(official_train), 4)
    if validation != EXPECTED_VALIDATION_CITIES:
        raise RuntimeError("the seed-42 validation draw changed")
    held_out = set(validation)
    train = [city for city in official_train if city not in held_out]
    return train, list(validation)


def assert_city_separation(train: list[str], validation: list[str], test: list[str]) -> None:
    train_set, validation_set, test_set = set(train), set(validation), set(test)
    if train_set & validation_set or train_set & test_set or validation_set & test_set:
        raise RuntimeError("train, validation, and test cities overlap")
    if (len(train), len(validation), len(test)) != (10, 4, 10):
        raise RuntimeError("expected 10 train, 4 validation, and 10 test cities")
    if list(test) != OFFICIAL_TEST_CITIES:
        raise RuntimeError("the official OSCD test split changed")


class CitySubset(Dataset):
    """Crops only from the requested cities. The parent dataset still owns the rasters."""

    def __init__(self, base: OSCDDataset, cities: list[str]):
        missing = [city for city in cities if city not in base.cities]
        if missing:
            raise ValueError(f"cities are not in this OSCD split: {missing}")
        self.base = base
        self.cities = list(cities)
        self.indices = [base.cities.index(city) for city in self.cities]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        return self.base[self.indices[index]]


class RandomCropSAR(Dataset):
    """Draw aligned 32x32 crops from full-city SAR samples."""

    def __init__(self, base: OSCDDataset, crop_size: int, crops_per_city: int):
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
        image = torch.cat((t1[:, top:bottom, left:right], t2[:, top:bottom, left:right]), dim=0)
        return {
            "image": image,
            "label": label[:, top:bottom, left:right],
            "city": sample["city"],
            "orbit": sample["orbit"],
        }


def sample_at_orbit(dataset: OSCDDataset, index: int, orbit: int) -> dict:
    previous = dataset.fixed_orbit
    dataset.fixed_orbit = orbit
    try:
        return dataset[index]
    finally:
        dataset.fixed_orbit = previous


def concatenate_dates(sample: dict) -> torch.Tensor:
    return torch.cat((sample["t1_img"], sample["t2_img"]), dim=0)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    seen = 0
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * image.shape[0]
        seen += image.shape[0]
    return total / seen


@torch.inference_mode()
def evaluate_cities(model, dataset: OSCDDataset, cities: list[str], device) -> tuple[dict, list[dict]]:
    model.eval()
    pooled = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    rows = []
    for city in cities:
        index = dataset.cities.index(city)
        orbit = int(ORBITS[city][0])
        sample = sample_at_orbit(dataset, index, orbit)
        image = concatenate_dates(sample).unsqueeze(0).to(device)
        label = sample["label"].unsqueeze(0).to(device)
        logits = model(image)
        counts = counts_from_logits(logits, label)
        for key in pooled:
            pooled[key] += counts[key]
        rows.append({"city": city, "orbit": orbit, **counts, **rates_from_counts(counts)})
    return {**pooled, **rates_from_counts(pooled)}, rows


def save_checkpoint(path: Path, model: nn.Module, epoch: int, validation_f1: float, config: dict) -> None:
    """Write best.pt immediately. The stored weights are a detached CPU copy."""
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state,
            "epoch": epoch,
            "validation_f1": validation_f1,
            "config": config,
        },
        path,
    )


def save_history(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_iou",
        "validation_false_alarm_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_city_metrics(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "city",
        "orbit",
        "tp",
        "tn",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "iou",
        "false_alarm_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_image(path: Path, array: np.ndarray, title: str, cmap: str, vmin: float, vmax: float) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    image = axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_figures(model, dataset: OSCDDataset, city: str, device) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    index = dataset.cities.index(city)
    orbit = int(ORBITS[city][0])
    sample = sample_at_orbit(dataset, index, orbit)
    image = concatenate_dates(sample).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(image))[0, 0].cpu().numpy()
    binary = (probability >= 0.5).astype(np.float32)
    target = sample["label"][0].numpy()
    t1 = sample["t1_img"][0].numpy()
    t2 = sample["t2_img"][0].numpy()
    predicted = binary >= 0.5
    changed = target >= 0.5
    error = np.zeros((*target.shape, 3), dtype=np.float32)
    error[(predicted & ~changed)] = (1.0, 0.0, 0.0)
    error[(~predicted & changed)] = (0.0, 0.0, 1.0)
    error[(predicted & changed)] = (0.0, 1.0, 0.0)

    def stretched(channel: np.ndarray) -> tuple[np.ndarray, float, float]:
        low, high = np.percentile(channel, (2, 98))
        if high <= low:
            high = low + 1e-6
        return channel, float(low), float(high)

    t1_show, t1_low, t1_high = stretched(t1)
    t2_show, t2_low, t2_high = stretched(t2)
    _save_image(
        FIGURE_DIR / f"{city}_sar_t1.png",
        t1_show,
        f"{city} SAR T1 orbit {orbit} (display 2-98 percentile)",
        "gray",
        t1_low,
        t1_high,
    )
    _save_image(
        FIGURE_DIR / f"{city}_sar_t2.png",
        t2_show,
        f"{city} SAR T2 orbit {orbit} (display 2-98 percentile)",
        "gray",
        t2_low,
        t2_high,
    )
    _save_image(FIGURE_DIR / f"{city}_ground_truth.png", target, f"{city} ground truth", "gray", 0, 1)
    _save_image(FIGURE_DIR / f"{city}_probability.png", probability, f"{city} predicted probability", "magma", 0, 1)
    _save_image(FIGURE_DIR / f"{city}_binary.png", binary, f"{city} binary prediction at 0.5", "gray", 0, 1)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(error)
    axis.set_title(f"{city} errors: red FP, blue FN, green TP")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / f"{city}_error_map.png", dpi=150)
    plt.close(figure)


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
        "input_channels": 2,
        "output_channels": 1,
        "parameter_count": int(parameter_count),
        "threshold": 0.5,
        "false_alarm_rate": str(FAR_DEFINITION),
        "validation_split_seed": VALIDATION_SPLIT_SEED,
        "train_cities": [str(city) for city in train_cities],
        "validation_cities": [str(city) for city in validation_cities],
        "test_cities": [str(city) for city in test_cities],
        "training_orbit": "np.random.choice of ORBITS[city] inside OSCDDataset, after the seed is set",
        "evaluation_orbit": "first orbit listed for each city",
        "checkpoint_selection": "highest validation F1 on full validation-city images",
        "test_metric_aggregation": "pixel counts pooled across the official test cities, once, after training",
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    official_train = OSCDDataset(split="train", mode="sar")
    official_test = OSCDDataset(split="test", mode="sar")
    train_cities, validation_cities = split_train_cities(official_train.cities)
    test_cities = list(official_test.cities)
    assert_city_separation(train_cities, validation_cities, test_cities)
    crops = RandomCropSAR(CitySubset(official_train, train_cities), args.crop_size, args.crops_per_city)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        crops,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=loader_generator,
    )
    model = UNet(in_channels=2, out_channels=1).to(device)
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
        validation_metrics, _ = evaluate_cities(model, official_train, validation_cities, device)
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
        if validation_f1 is not None and validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            save_checkpoint(CHECKPOINT_PATH, model, epoch, validation_f1, record)

    if best_epoch is None or not CHECKPOINT_PATH.is_file():
        raise RuntimeError("training produced no checkpoint")
    saved = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)
    pooled, city_rows = evaluate_cities(model, official_test, test_cities, device)
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
        "reproducibility": record,
    }
    with (EXPERIMENT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_document, handle, indent=2)
        handle.write("\n")

    figure_city = args.figure_city or test_cities[0]
    save_figures(model, official_test, figure_city, device)
    print(f"best validation epoch {best_epoch} validation_f1 {best_f1}")
    print(f"final test f1 {pooled['f1']}")
    print(f"checkpoint {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
