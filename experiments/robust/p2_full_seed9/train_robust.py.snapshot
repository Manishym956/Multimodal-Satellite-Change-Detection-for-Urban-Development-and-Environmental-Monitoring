"""Train Hafner's DS-UNet on OSCD under a mix of optical-availability conditions.

One script for the robustness study. The network and Jaccard-like loss are
loaded at run time from the local DS-UNet clone (repos/2/DS_UNet, commit
8bfa4be, GPL-2), so the architecture is Hafner's code, not a re-implementation.

Input conditions (missing optical bands are zero-filled; SAR is always present):
    full    S1 t1, S1 t2, S2 t1, S2 t2
    no_o2   post-change optical missing (S2 t2 = 0)
    no_opt  all optical missing (S2 t1 = S2 t2 = 0)
    o2_copy post-change optical missing, filled with the pre-change optical (S2 t2 = S2 t1)

--train-conditions gives the per-sample probability of each condition during
training, e.g. "full=1" (ordinary model), "no_o2=1" (specialist), or
"full=0.5,no_o2=0.25,no_opt=0.25" (modality dropout). Every run is evaluated
under all three conditions.

Protocols:
    holdout  train on 10 cities, select the step and a per-condition threshold on
             the 4 validation cities, report the official 10 test cities once
    hafner   train on all 14 cities, keep the final step, choose the threshold
             that maximises F1 on the training cities (as in DS_UNet/train_network.py)

Usage:
    .venv/Scripts/python.exe -m src.training.train_robust --name dsunet_full --seed 1
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.datasets.oscd_dataset import ORBITS, OSCDDataset
from src.evaluation.metrics import rates_from_counts
from src.training.train_sar_baseline import (
    assert_city_separation,
    sample_at_orbit,
    set_seed,
    split_train_cities,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DSUNET_ROOT = PROJECT_ROOT.parents[1] / "repos" / "2" / "DS_UNet"
# appended conditions keep earlier draws identical: rng.choice maps one uniform through the cdf
CONDITIONS = ("full", "no_o2", "no_opt", "o2_copy")
THRESHOLDS = torch.linspace(0, 1, 101)
N_S1 = 1


def load_dsunet_module(name: str):
    spec = importlib.util.spec_from_file_location(f"dsunet_{name}", DSUNET_ROOT / "utils" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model() -> torch.nn.Module:
    networks = load_dsunet_module("networks")
    cfg = SimpleNamespace(
        MODEL=SimpleNamespace(TYPE="dualstreamunet", TOPOLOGY=[64, 128, 256, 512], OUT_CHANNELS=1, IN_CHANNELS=28),
        DATASET=SimpleNamespace(MODE="fusion", SENTINEL1_BANDS=["VV"], SENTINEL2_BANDS=[None] * 13),
    )
    with contextlib.redirect_stdout(io.StringIO()):  # the reference UNet prints its layers
        return networks.create_network(cfg)


def parse_conditions(text: str) -> dict[str, float]:
    probs = {k: 0.0 for k in CONDITIONS}
    for part in text.split(","):
        key, value = part.split("=")
        if key not in probs:
            raise ValueError(f"unknown condition {key}")
        probs[key] = float(value)
    if abs(sum(probs.values()) - 1) > 1e-6:
        raise ValueError("condition probabilities must sum to 1")
    return probs


def apply_condition(t1: torch.Tensor, t2: torch.Tensor, condition: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero the optical channels that are unavailable. Tensors are (B, 14, H, W)."""
    t1, t2 = t1.clone(), t2.clone()
    if condition == "o2_copy":
        t2[:, N_S1:] = t1[:, N_S1:]
    if condition in ("no_o2", "no_opt"):
        t2[:, N_S1:] = 0
    if condition == "no_opt":
        t1[:, N_S1:] = 0
    return t1, t2


def random_crop(t1, t2, label, size, importance: bool, rng: np.random.Generator):
    """Uniform or DS-UNet importance crop (20 candidates weighted by changed pixels + 5)."""
    _, height, width = label.shape
    candidates = 20 if importance else 1
    ys = rng.integers(0, height - size, candidates)
    xs = rng.integers(0, width - size, candidates)
    if importance:
        weights = np.array([label[0, y:y + size, x:x + size].sum().item() for y, x in zip(ys, xs)]) + 5
        pick = rng.choice(candidates, p=weights / weights.sum())
    else:
        pick = 0
    y, x = ys[pick], xs[pick]
    return t1[:, y:y + size, x:x + size], t2[:, y:y + size, x:x + size], label[:, y:y + size, x:x + size]


def augment(t1, t2, label, rng: np.random.Generator):
    """DS-UNet RandomFlip then RandomRotate (k in 1..3), applied identically to all three."""
    stack = torch.cat((t1, t2, label), 0)
    if rng.random() < 0.5:
        stack = stack.flip(2)
    if rng.random() < 0.5:
        stack = stack.flip(1)
    stack = torch.rot90(stack, int(rng.integers(1, 4)), (1, 2))
    c = t1.shape[0]
    return stack[:c], stack[c:2 * c], stack[2 * c:]


def threshold_counts(prob: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """(101, 4) TP, FP, FN, TN for prediction prob >= k/100, k = 0..100."""
    bins = torch.clamp((prob.flatten() * 100).floor().long(), 0, 100)
    pos = label.flatten() >= 0.5
    pos_hist = torch.bincount(bins[pos], minlength=101)
    neg_hist = torch.bincount(bins[~pos], minlength=101)
    tp = pos_hist.flip(0).cumsum(0).flip(0)  # positives with bin >= k
    fp = neg_hist.flip(0).cumsum(0).flip(0)
    fn = pos.sum() - tp
    tn = (~pos).sum() - fp
    return torch.stack((tp, fp, fn, tn), 1).cpu()


@torch.inference_mode()
def evaluate(model, dataset: OSCDDataset, cities: list[str], device) -> dict[str, torch.Tensor]:
    """Pooled threshold counts per condition, first listed orbit per city."""
    model.eval()
    totals = {c: torch.zeros(101, 4, dtype=torch.long) for c in CONDITIONS}
    for city in cities:
        sample = sample_at_orbit(dataset, dataset.cities.index(city), int(ORBITS[city][0]))
        t1, t2 = sample["t1_img"][None].to(device), sample["t2_img"][None].to(device)
        label = sample["label"].to(device)
        for condition in CONDITIONS:
            prob = torch.sigmoid(model(*apply_condition(t1, t2, condition)))[0]
            totals[condition] += threshold_counts(prob, label)
    return totals


def rates_at(counts: torch.Tensor, k: int) -> dict:
    tp, fp, fn, tn = (int(v) for v in counts[k])
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, **rates_from_counts({"tp": tp, "fp": fp, "fn": fn, "tn": tn})}


def best_threshold(counts: torch.Tensor) -> tuple[int, float]:
    f1s = [rates_at(counts, k)["f1"] or 0.0 for k in range(101)]
    k = int(np.argmax(f1s))
    return k, f1s[k]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--protocol", choices=("holdout", "hafner"), default="holdout")
    p.add_argument("--train-conditions", default="full=1")
    p.add_argument("--steps", type=int, default=390, help="optimizer steps (DS-UNet: 390 epochs x 1 batch)")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--crop-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--loss", choices=("jaccard", "bce"), default="jaccard")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--uniform-crop", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs = parse_conditions(args.train_conditions)
    trained = [c for c in CONDITIONS if probs[c] > 0]

    train_ds = OSCDDataset(split="train", mode="fusion")
    test_ds = OSCDDataset(split="test", mode="fusion")
    test_cities = list(test_ds.cities)
    if args.protocol == "holdout":
        train_cities, val_cities = split_train_cities(train_ds.cities)
        assert_city_separation(train_cities, val_cities, test_cities)
    else:
        train_cities, val_cities = list(train_ds.cities), list(train_ds.cities)  # threshold tuned on train, as DS-UNet

    model = build_model().to(device)
    criterion = load_dsunet_module("loss_functions").jaccard_like_loss if args.loss == "jaccard" else torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = PROJECT_ROOT / "experiments" / "robust" / f"{args.name}_seed{args.seed}"
    ckpt = PROJECT_ROOT / "checkpoints" / "robust" / f"{args.name}_seed{args.seed}.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    # provenance: exact training source of this run, hashed and copied next to its metrics
    sources = [Path(__file__), PROJECT_ROOT / "src" / "datasets" / "oscd_dataset.py",
               DSUNET_ROOT / "utils" / "networks.py", DSUNET_ROOT / "utils" / "loss_functions.py"]
    source_sha256 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (out_dir / "train_robust.py.snapshot").write_bytes(Path(__file__).read_bytes())

    history, best = [], (-1.0, None)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        t1s, t2s, labels = [], [], []
        for city in rng.choice(train_cities, args.batch_size):
            s = train_ds[train_ds.cities.index(city)]  # random orbit, as DS-UNet
            t1, t2, lab = random_crop(s["t1_img"], s["t2_img"], s["label"], args.crop_size, not args.uniform_crop, rng)
            if not args.no_augment:
                t1, t2, lab = augment(t1, t2, lab, rng)
            t1, t2 = apply_condition(t1[None], t2[None], rng.choice(CONDITIONS, p=[probs[c] for c in CONDITIONS]))
            t1s.append(t1[0]), t2s.append(t2[0]), labels.append(lab)
        t1, t2, label = (torch.stack(x).to(device) for x in (t1s, t2s, labels))
        optimizer.zero_grad()
        loss = criterion(model(t1, t2), label)
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            counts = evaluate(model, train_ds, val_cities, device)
            score = float(np.mean([best_threshold(counts[c])[1] for c in trained]))
            history.append({"step": step, "loss": loss.item(), "val_score": score,
                            **{f"val_f1_{c}": best_threshold(counts[c])[1] for c in CONDITIONS}})
            print(f"step {step} loss {loss.item():.4f} val {score:.4f}", flush=True)
            keep = step == args.steps if args.protocol == "hafner" else score > best[0]
            if keep:
                best = (score, step)
                thresholds = {c: best_threshold(counts[c])[0] for c in CONDITIONS}
                torch.save({"model_state_dict": model.state_dict(), "step": step, "thresholds": thresholds,
                            "args": vars(args)}, ckpt)

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    test_counts = evaluate(model, test_ds, test_cities, device)
    test = {c: {"selected_threshold": saved["thresholds"][c] / 100,
                "at_selected": rates_at(test_counts[c], saved["thresholds"][c]),
                "at_0.5": rates_at(test_counts[c], 50)} for c in CONDITIONS}
    doc = {"args": vars(args), "train_cities": train_cities, "val_cities": val_cities, "test_cities": test_cities,
           "selected_step": saved["step"], "selection_score": best[0], "test": test, "history": history,
           "train_seconds": round(time.perf_counter() - start, 1), "torch": torch.__version__,
           "dsunet_source": "repos/2/DS_UNet@8bfa4be utils/networks.py, utils/loss_functions.py",
           "source_sha256": source_sha256}
    (out_dir / "metrics.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(" | ".join(f"{c} F1 {test[c]['at_selected']['f1']:.4f} (tau {test[c]['selected_threshold']:.2f})" for c in CONDITIONS))


if __name__ == "__main__":
    main()
