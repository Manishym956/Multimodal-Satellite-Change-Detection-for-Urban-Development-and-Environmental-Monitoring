"""Optical-only U-Net baseline using the SAR baseline city split and evaluation protocol.

Sentinel-2 T1 and T2 are concatenated to 26 channels. The network is the same
U-Net class as the SAR baseline, with in_channels=26. This is not DS-UNet.
The official test cities are scored once after training, from best.pt.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.oscd_dataset import ORBITS, OSCDDataset
from src.evaluation.metrics import FAR_DEFINITION
from src.models.unet import UNet, count_parameters
from src.training.train_sar_baseline import (
    CitySubset,
    RandomCropSAR,
    assert_city_separation,
    evaluate_cities,
    sample_at_orbit,
    save_checkpoint,
    save_city_metrics,
    save_history,
    set_seed,
    split_train_cities,
    train_one_epoch,
)

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "optical_baseline"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "optical_baseline" / "best.pt"
FIGURE_DIR = PROJECT_ROOT / "figures" / "optical_baseline"
INPUT_CHANNELS = 26
# Display-only RGB from the rectified Sentinel-2 order B04, B03, B02.
RGB_BANDS = (3, 2, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the optical-only U-Net baseline.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--crops-per-city", type=int, default=64)
    parser.add_argument("--figure-city", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


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
        "modality": "Sentinel-2 optical, T1 and T2 concatenated",
        "input_channels": INPUT_CHANNELS,
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
    }


def _display_rgb(image: np.ndarray) -> np.ndarray:
    """Percentile-stretch B04, B03, B02 for a figure. The model input is unchanged."""
    channels = []
    for band in RGB_BANDS:
        channel = image[band]
        low, high = np.percentile(channel, (2, 98))
        if high <= low:
            high = low + 1e-6
        channels.append(np.clip((channel - low) / (high - low), 0, 1))
    return np.stack(channels, axis=-1)


def save_figures(model, dataset: OSCDDataset, city: str, device) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    index = dataset.cities.index(city)
    orbit = int(ORBITS[city][0])
    sample = sample_at_orbit(dataset, index, orbit)
    image = torch.cat((sample["t1_img"], sample["t2_img"]), dim=0).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(image))[0, 0].cpu().numpy()
    binary = (probability >= 0.5).astype(np.float32)
    target = sample["label"][0].numpy()
    predicted = binary >= 0.5
    changed = target >= 0.5
    error = np.zeros((*target.shape, 3), dtype=np.float32)
    error[(predicted & ~changed)] = (1.0, 0.0, 0.0)
    error[(~predicted & changed)] = (0.0, 0.0, 1.0)
    error[(predicted & changed)] = (0.0, 1.0, 0.0)

    def save_rgb(path: Path, array: np.ndarray, title: str) -> None:
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(array)
        axis.set_title(title)
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)

    save_rgb(
        FIGURE_DIR / f"{city}_optical_t1.png",
        _display_rgb(sample["t1_img"].numpy()),
        f"{city} optical T1 orbit {orbit} (display B04-B03-B02)",
    )
    save_rgb(
        FIGURE_DIR / f"{city}_optical_t2.png",
        _display_rgb(sample["t2_img"].numpy()),
        f"{city} optical T2 orbit {orbit} (display B04-B03-B02)",
    )
    figure, axis = plt.subplots(figsize=(6, 6))
    image_axis = axis.imshow(target, cmap="gray", vmin=0, vmax=1)
    axis.set_title(f"{city} ground truth")
    axis.axis("off")
    figure.colorbar(image_axis, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / f"{city}_ground_truth.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 6))
    image_axis = axis.imshow(probability, cmap="magma", vmin=0, vmax=1)
    axis.set_title(f"{city} predicted probability")
    axis.axis("off")
    figure.colorbar(image_axis, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / f"{city}_probability.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 6))
    image_axis = axis.imshow(binary, cmap="gray", vmin=0, vmax=1)
    axis.set_title(f"{city} binary prediction at 0.5")
    axis.axis("off")
    figure.colorbar(image_axis, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / f"{city}_binary.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(error)
    axis.set_title(f"{city} errors: red FP, blue FN, green TP")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / f"{city}_error_map.png", dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    official_train = OSCDDataset(split="train", mode="optical")
    official_test = OSCDDataset(split="test", mode="optical")
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
