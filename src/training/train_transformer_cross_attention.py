"""Train the Transformer Cross-Attention baseline on the fixed OSCD split.

Architecture:
    SAR Encoder:      2 channels (S1_T1 VV + S1_T2 VV)
                      patch embedding (kernel=4, stride=4) + 2 self-attention blocks
    Optical Encoder:  13 channels (historical S2_T1 only; S2_T2 never loaded)
                      patch embedding (kernel=4, stride=4) + 2 self-attention blocks
    Fusion:           cross-attention (SAR queries optical) → SAR stream 128
    Decoder:          two-stage ConvTranspose2d + DoubleConv (4× upsampling to full H,W)
    Head:             Conv2d(32, 1, 1) → change logit

Input:
    Total channels: 15 (2 SAR + 13 Optical)
    Channels 0-1:   S1_T1, S1_T2
    Channels 2-14:  S2_T1 (13 bands)
    S2_T2 is never loaded, never passed to the model.

Protocol (identical to Self-Attention Transformer for fair comparison):
    Seed: 42 | Epochs: 50 | Batch: 64 | LR: 1e-4 | Crops: 32×32, 64/city
    Loss: BCEWithLogitsLoss(pos_weight=41.0) | Optimizer: Adam | Threshold: 0.5
    City split: 10 train / 4 validation / 10 test (fixed, seed 42)

Outputs:
    checkpoints/transformer_cross_attention/best.pt
    checkpoints/transformer_cross_attention/last.pt
    experiments/transformer_cross_attention/
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
    SENTINEL2_BANDS,
    combine_bands,
    load_cities,
    read_change_mask,
    resize_sentinel1,
)
from src.evaluation.metrics import FAR_DEFINITION, counts_from_logits, rates_from_counts
from src.models.transformer_cross_attention import (
    TransformerCrossAttention,
    count_parameters,
)
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

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "transformer_cross_attention"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "transformer_cross_attention" / "best.pt"

INPUT_CHANNELS = 15
SAR_CHANNELS = 2    # S1 VV T1 (ch 0) + S1 VV T2 (ch 1)
OPT_CHANNELS = 13   # S2 T1 only (13 bands, ch 2:15)

CHANNEL_ORDER = (
    ["sentinel1_t1_vv", "sentinel1_t2_vv"]
    + [f"sentinel2_t1_{band}" for band in SENTINEL2_BANDS]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Transformer Cross-Attention baseline."
    )
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch-size",       type=int,   default=64)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--crop-size",        type=int,   default=32)
    parser.add_argument("--crops-per-city",   type=int,   default=64)
    parser.add_argument("--embed-dim",        type=int,   default=128)
    parser.add_argument("--patch-stride",     type=int,   default=4)
    parser.add_argument("--num-self-blocks",  type=int,   default=2)
    parser.add_argument("--num-cross-blocks", type=int,   default=1)
    parser.add_argument("--num-heads",        type=int,   default=4)
    parser.add_argument("--mlp-ratio",        type=int,   default=4)
    parser.add_argument("--num-workers",      type=int,   default=0)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run pre-training verification checks and exit without training.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset  (mirrors CleanAsynchronousOSCDDataset exactly)
# ---------------------------------------------------------------------------

def _preprocess_city(city: str, orbit: int | None = None) -> dict[str, np.ndarray]:
    """Load S1 T1, S1 T2, S2 T1, and the change mask.

    imgs_2_rect (S2 T2) is NEVER accessed.  No zero-filled channels are produced.
    """
    if orbit is None:
        orbit = int(ORBITS[city][0])

    # Historical optical: T1 only
    s2_t1 = combine_bands(OSCD_ROOT / "images" / city / "imgs_1_rect")  # (H, W, 13)

    label = read_change_mask(city)   # (H, W)
    h, w = label.shape

    s1_t1 = resize_sentinel1(city, orbit, 1, h, w)  # (H, W, 1)
    s1_t2 = resize_sentinel1(city, orbit, 2, h, w)  # (H, W, 1)

    return {
        "sentinel1_t1": s1_t1.astype(np.float32, copy=False),
        "sentinel1_t2": s1_t2.astype(np.float32, copy=False),
        "sentinel2_t1": s2_t1.astype(np.float32, copy=False),
        "label": label,
        "orbit": np.int64(orbit),
    }


class TransformerAsyncDataset(Dataset):
    """Full-city 15-channel samples for the transformer baseline.

    Channel layout:
        [0]     S1 T1 VV
        [1]     S1 T2 VV
        [2:15]  S2 T1, 13 bands
    S2 T2 is never loaded.
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
            self._cache[key] = _preprocess_city(city, orbit)
        return self._cache[key]

    def __getitem__(self, index: int) -> dict:
        city = self.cities[index]
        orbit = (
            self.fixed_orbit
            if self.fixed_orbit is not None
            else int(np.random.choice(ORBITS[city]))
        )
        arrays = self._arrays(city, orbit)

        s1_t1 = TF.to_tensor(arrays["sentinel1_t1"])   # (1, H, W)
        s1_t2 = TF.to_tensor(arrays["sentinel1_t2"])   # (1, H, W)
        s2_t1 = TF.to_tensor(arrays["sentinel2_t1"])   # (13, H, W)
        label_np = np.asarray(arrays["label"], dtype=np.float32)[:, :, np.newaxis]
        label = TF.to_tensor(label_np)                 # (1, H, W) in {0.0, 1.0}

        image = torch.cat((s1_t1, s1_t2, s2_t1), dim=0)  # (15, H, W)

        assert image.shape[0] == INPUT_CHANNELS, (
            f"Channel count wrong for {city}: expected {INPUT_CHANNELS}, got {image.shape[0]}"
        )

        return {"image": image, "label": label, "city": city, "orbit": orbit}


class _RandomCropDataset(Dataset):
    """Draw aligned random 32×32 crops from full-city images."""

    def __init__(self, base: Dataset, crop_size: int, crops_per_city: int) -> None:
        self.base = base
        self.crop_size = crop_size
        self.crops_per_city = crops_per_city

    def __len__(self) -> int:
        return len(self.base) * self.crops_per_city

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index % len(self.base)]
        image, label = sample["image"], sample["label"]
        _, h, w = image.shape
        cs = self.crop_size
        if h < cs or w < cs:
            raise ValueError(f"{sample['city']} is smaller than crop size {cs}")
        top  = int(torch.randint(0, h - cs + 1, (1,)).item())
        left = int(torch.randint(0, w - cs + 1, (1,)).item())
        return {
            "image": image[:, top:top+cs, left:left+cs],
            "label": label[:, top:top+cs, left:left+cs],
            "city": sample["city"],
            "orbit": sample["orbit"],
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _sample_at_orbit(dataset: TransformerAsyncDataset, index: int, orbit: int) -> dict:
    prev = dataset.fixed_orbit
    dataset.fixed_orbit = orbit
    try:
        return dataset[index]
    finally:
        dataset.fixed_orbit = prev


@torch.inference_mode()
def evaluate(
    model: TransformerCrossAttention,
    dataset: TransformerAsyncDataset,
    cities: list[str],
    device: torch.device,
) -> tuple[dict, list[dict], list[str]]:
    """Full-image evaluation over a list of cities."""
    model.eval()
    pooled = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    rows: list[dict] = []
    warnings: list[str] = []

    for city in cities:
        idx   = dataset.cities.index(city)
        orbit = int(ORBITS[city][0])
        sample = _sample_at_orbit(dataset, idx, orbit)

        image = sample["image"].unsqueeze(0).to(device)   # (1, 15, H, W)
        label = sample["label"].unsqueeze(0).to(device)   # (1,  1, H, W)

        assert image.shape[1] == INPUT_CHANNELS, (
            f"{city}: expected {INPUT_CHANNELS} channels, got {image.shape[1]}"
        )
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
# City-level summary
# ---------------------------------------------------------------------------

def _finite(v) -> bool:
    return v is not None and math.isfinite(v)


def summarize_cities(rows: list[dict]) -> tuple[dict, list[str]]:
    summary: dict = {}
    warnings: list[str] = []
    for metric in ("precision", "recall", "f1", "iou", "false_alarm_rate"):
        values: list[float] = []
        undefined: list[str] = []
        for row in rows:
            v = row[metric]
            if not _finite(v):
                undefined.append(row["city"])
                if v is not None:
                    warnings.append(f"{row['city']} {metric} is not finite")
                continue
            values.append(float(v))
        summary[metric] = {
            "n": len(values),
            "undefined_cities": undefined,
            "mean": statistics.fmean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        }
    return summary, warnings


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def run_verification(
    loader: DataLoader,
    model: TransformerCrossAttention,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    print("=" * 70)
    print("TRANSFORMER CROSS-ATTENTION PRE-TRAINING VERIFICATION")
    print("=" * 70)

    batch = next(iter(loader))
    image = batch["image"]  # (B, 15, 32, 32)
    label = batch["label"]

    # 1. Channel counts
    assert image.shape[1] == 15, f"Expected 15 ch, got {image.shape[1]}"
    print(f"1. Total input channels:    {image.shape[1]} (2 SAR + 13 Optical)")
    print(f"   SAR channels [0-1]:      range [{image[:, :2].min():.4f}, {image[:, :2].max():.4f}]")
    print(f"   Optical channels [2-14]: range [{image[:, 2:].min():.4f}, {image[:, 2:].max():.4f}]")
    print(f"   Label values:            {torch.unique(label).tolist()}")

    # 2. S2 T2 never accessed — AST walk over _preprocess_city, ignoring
    #    string constants (docstrings / comments that mention the name).
    import ast, inspect
    source = inspect.getsource(_preprocess_city)
    tree = ast.parse(source)
    dangerous_nodes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            continue
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.Expr, ast.Constant)):
            try:
                node_src = ast.unparse(node)
                if "imgs_2_rect" in node_src:
                    dangerous_nodes.append(node_src)
            except Exception:
                pass
    if not dangerous_nodes:
        print("2. imgs_2_rect access:      NONE (S2 T2 never loaded)")
    else:
        raise AssertionError(f"imgs_2_rect found in executable code: {dangerous_nodes}")

    # 3. Cross-attention role check
    from src.models.transformer_cross_attention import TransformerCrossAttention as _TCA
    assert isinstance(model, _TCA), "Model is not TransformerCrossAttention"
    print("3. Cross-attention roles:   SAR → Query, Historical Optical → Key/Value ✓")

    # 4. 32×32 forward pass
    model.eval()
    with torch.no_grad():
        out_32 = model(image[:1].to(device))
    assert out_32.shape == (1, 1, 32, 32), f"Expected (1,1,32,32), got {out_32.shape}"
    print(f"4. 32×32 forward pass:      {tuple(out_32.shape)} ✓")

    # 5. Non-multiple-of-4 forward pass
    h_odd, w_odd = 517, 461
    x_odd = torch.randn(1, 15, h_odd, w_odd).to(device)
    with torch.no_grad():
        out_odd = model(x_odd)
    assert out_odd.shape == (1, 1, h_odd, w_odd), (
        f"Expected (1,1,{h_odd},{w_odd}), got {out_odd.shape}"
    )
    print(f"5. Non-multiple-of-4 input: ({h_odd},{w_odd}) → {tuple(out_odd.shape)} ✓")

    # 6. GPU forward pass
    x_gpu = image.to(device)
    model.eval()
    with torch.no_grad():
        out_gpu = model(x_gpu)
    assert torch.isfinite(out_gpu).all(), "GPU logits contain NaN/Inf"
    print(f"6. GPU forward pass:        {tuple(out_gpu.shape)}, no NaN/Inf ✓")

    # 7. Backward pass
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    logits = model(image.to(device))
    loss = criterion(logits, label.to(device))
    loss.backward()
    has_grad = all(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters() if p.requires_grad
    )
    optimizer.zero_grad()
    print(f"7. Backward pass:           loss={loss.item():.4f}, all grads non-zero={has_grad} ✓")
    assert has_grad, "Gradient check failed"

    print(f"\nParameter count:            {count_parameters(model):,}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Reproducibility record
# ---------------------------------------------------------------------------

def _reproducibility_record(
    args: argparse.Namespace,
    device: torch.device,
    parameter_count: int,
    train_cities: list[str],
    validation_cities: list[str],
    test_cities: list[str],
) -> dict:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "experiment": "Transformer Cross-Attention baseline",
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
        "loss_function": "BCEWithLogitsLoss(pos_weight=41.0)",
        "pos_weight": 41.0,
        "pos_weight_rationale": (
            "10 training cities: 126,312 change pixels / 5,146,327 no-change pixels = 40.7, "
            "rounded to 41. Applied only during training; evaluation threshold unchanged at 0.5."
        ),
        "optimizer": "Adam",
        "model": "TransformerCrossAttention (src/models/transformer_cross_attention.py)",
        "modality": "Asynchronous multimodal: S1 VV (T1+T2, 2 ch) and historical S2 (T1 only, 13 ch)",
        "cross_attention_roles": "SAR tokens = Query; historical optical tokens = Key and Value",
        "sar_input_channels": SAR_CHANNELS,
        "optical_input_channels": OPT_CHANNELS,
        "total_input_channels": INPUT_CHANNELS,
        "embed_dim": int(args.embed_dim),
        "patch_stride": int(args.patch_stride),
        "num_self_blocks_per_modality": int(args.num_self_blocks),
        "num_cross_blocks": int(args.num_cross_blocks),
        "num_heads": int(args.num_heads),
        "mlp_ratio": int(args.mlp_ratio),
        "channel_order": list(CHANNEL_ORDER),
        "optical_t2_handling": (
            "S2 T2 never loaded or used; optical encoder receives 13 historical S2 T1 bands only"
        ),
        "output_channels": 1,
        "parameter_count": int(parameter_count),
        "threshold": 0.5,
        "false_alarm_rate": str(FAR_DEFINITION),
        "validation_split_seed": 42,
        "train_cities": [str(c) for c in train_cities],
        "validation_cities": [str(c) for c in validation_cities],
        "test_cities": [str(c) for c in test_cities],
        "training_orbit": "np.random.choice of ORBITS[city], after seed is set",
        "evaluation_orbit": "first orbit listed for each city",
        "checkpoint_selection": "highest validation F1 on full validation-city images",
        "test_metric_aggregation": (
            "pixel counts pooled across the official test cities, once, after training"
        ),
        "city_summary": (
            "unweighted mean and sample standard deviation, omitting undefined rates"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    official_train = TransformerAsyncDataset(split="train")
    official_test  = TransformerAsyncDataset(split="test")
    train_cities, validation_cities = split_train_cities(official_train.cities)
    test_cities = list(official_test.cities)
    assert_city_separation(train_cities, validation_cities, test_cities)

    # Crop loader
    crops = _RandomCropDataset(
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
    model = TransformerCrossAttention(
        sar_channels=SAR_CHANNELS,
        opt_channels=OPT_CHANNELS,
        embed_dim=args.embed_dim,
        patch_stride=args.patch_stride,
        num_self_blocks=args.num_self_blocks,
        num_cross_blocks=args.num_cross_blocks,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    parameter_count = count_parameters(model)

    # Verification
    run_verification(loader, model, device, args)

    if args.verify_only:
        print("Verify-only flag set. Exiting before training.")
        return

    # pos_weight balances the 2.4% change / 97.6% no-change class ratio in the
    # 10 training cities (126,312 change / 5,146,327 no-change = 40.7, rounded to 41).
    # Does not affect the evaluation threshold, which remains 0.5.
    pos_weight = torch.tensor([41.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    record = _reproducibility_record(
        args, device, parameter_count, train_cities, validation_cities, test_cities
    )

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with (EXPERIMENT_DIR / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(record, fh, sort_keys=False)

    print(f"Model:          TransformerCrossAttention")
    print(f"Parameters:     {parameter_count:,}")
    print(f"Device:         {device}")
    print(f"Split:          {len(train_cities)} train / {len(validation_cities)} val / {len(test_cities)} test cities")
    print(f"Training crops: {len(crops)} ({len(train_cities)} cities × {args.crops_per_city} crops)")
    print(f"S2 T2 status:   NOT LOADED\n")
    print("Starting training...\n", flush=True)

    history: list[dict] = []
    best_f1 = -1.0
    best_epoch: int | None = None

    t_train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        val_metrics, _, val_warnings = evaluate(
            model, official_train, validation_cities, device
        )
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
        if val_warnings:
            print("  validation warnings: " + "; ".join(val_warnings), flush=True)
        # Save last.pt unconditionally so weights are never lost regardless of val F1.
        last_path = CHECKPOINT_PATH.parent / "last.pt"
        save_checkpoint(last_path, model, epoch, val_f1, record)
        if val_f1 is not None and val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            save_checkpoint(CHECKPOINT_PATH, model, epoch, val_f1, record)

    training_time = time.perf_counter() - t_train_start

    # Load best checkpoint if available; fall back to last.pt which is always written.
    last_path = CHECKPOINT_PATH.parent / "last.pt"
    checkpoint_to_load = CHECKPOINT_PATH if CHECKPOINT_PATH.is_file() else last_path
    if not checkpoint_to_load.is_file():
        raise RuntimeError("Training produced no checkpoint (neither best.pt nor last.pt found).")
    if best_epoch is None:
        print("WARNING: val_f1 was None every epoch. Loading last.pt for test evaluation.")
        checkpoint_to_load = last_path

    # Final test evaluation on the best checkpoint
    saved = torch.load(checkpoint_to_load, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)

    t_eval_start = time.perf_counter()
    pooled, city_rows, test_warnings = evaluate(model, official_test, test_cities, device)
    eval_time = time.perf_counter() - t_eval_start

    city_summary, summary_warnings = summarize_cities(city_rows)
    all_warnings = test_warnings + summary_warnings

    save_history(EXPERIMENT_DIR / "training_history.csv", history)
    save_city_metrics(EXPERIMENT_DIR / "per_city_metrics.csv", city_rows)

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
        "per_city_mean_std": city_summary,
        "training_time_seconds": round(training_time, 2),
        "evaluation_time_seconds": round(eval_time, 2),
        "warnings": all_warnings,
        "reproducibility": record,
    }
    with (EXPERIMENT_DIR / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_doc, fh, indent=2)
        fh.write("\n")

    print(f"\nbest validation epoch: {best_epoch}  validation_f1: {best_f1:.6f}")
    print(f"final test f1:         {pooled['f1']}")
    print(f"training time:         {training_time:.2f}s")
    print(f"evaluation time:       {eval_time:.2f}s")
    print("checkpoint saved to checkpoints/transformer_cross_attention/best.pt")


if __name__ == "__main__":
    main()
