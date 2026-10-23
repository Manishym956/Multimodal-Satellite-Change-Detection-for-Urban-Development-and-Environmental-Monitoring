"""Steps 6-7 (RRD RQ3, RQ4): train_robust.py's recipe with a choice of architecture and SAR input.

Everything except the model and the SAR channels is train_robust.py / train_spatial.py, reused unchanged (frozen
Phase 2 code): Jaccard-like loss, Adam, importance crops, flips and rotations, training-condition draw, validation
every --eval-every steps, checkpoint selection (mean best-threshold validation F1 over the trained conditions),
per-condition thresholds and one test evaluation. With --arch dsunet --sar hafner_vv this reproduces train_robust.py.

--arch (all from the official DS-UNet code, repos/2/DS_UNet@8bfa4be):
    dsunet       DualStreamUNet, SAR and optical streams fused by a 1x1 convolution (late fusion)
    unet_sar     UNet on the SAR channels of both dates
    unet_opt     UNet on the 13 optical bands of both dates
    unet_fusion  UNet on SAR and optical stacked at the input, as DS-UNet's 'fusion' mode with a single UNet
--sar:
    hafner_vv    the VV rasters released with DS-UNet (datasets/oscd_sentinel1), as in every earlier step
    vv, vh, vvvh bands of datasets/oscd_s1_gee/sentinel1_{city}_{orbit}_t{1,2}.tif (band descriptions VV, VH), resized
                 to the label grid with cubic interpolation exactly as the Hafner VV rasters are

Optical-availability conditions zero the optical channels, which follow the SAR channels in each date's tensor.

Usage:
    .venv/Scripts/python.exe -m src.training.train_rrd --name p6_unet_sar --arch unet_sar --seed 1 --train-conditions full=1
"""

import argparse
import contextlib
import hashlib
import io
import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import rasterio
import torch
import torchvision.transforms.functional as TF

from src.datasets.oscd_dataset import ORBITS, OSCDDataset
from src.training import train_robust as tr
from src.training.train_sar_baseline import assert_city_separation, sample_at_orbit, set_seed, split_train_cities

GEE_S1_ROOT = tr.PROJECT_ROOT / "datasets" / "oscd_s1_gee"
SAR_BANDS = {"hafner_vv": ["VV"], "vv": ["VV"], "vh": ["VH"], "vvvh": ["VV", "VH"]}
N_S2 = 13


class RRDDataset(OSCDDataset):
    """OSCDDataset in fusion mode with the SAR channels taken from --sar; optical bands and labels unchanged."""

    def __init__(self, split: str, sar: str):
        super().__init__(split=split, mode="fusion")
        self.sar = sar

    def _tensors(self, city: str, orbit: int):
        t1, t2, label = super()._tensors(city, orbit)
        if self.sar == "hafner_vv":
            return t1, t2, label
        h, w = label.shape[1:]
        sar = []
        for date in (1, 2):
            with rasterio.open(GEE_S1_ROOT / f"sentinel1_{city}_{orbit}_t{date}.tif") as src:
                idx = [src.descriptions.index(b) + 1 for b in SAR_BANDS[self.sar]]
                bands = [cv2.resize(src.read(i).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) for i in idx]
            sar.append(TF.to_tensor(np.stack(bands, axis=2)))
        return torch.cat((sar[0], t1[1:]), 0), torch.cat((sar[1], t2[1:]), 0), label


class ChannelUNet(torch.nn.Module):
    """The official UNet applied to a channel subset of both dates (UNet.forward concatenates t1 and t2)."""

    def __init__(self, unet: torch.nn.Module, channels: list[int]):
        super().__init__()
        self.unet, self.channels = unet, channels

    def forward(self, t1, t2):
        return self.unet(t1[:, self.channels], t2[:, self.channels])


def build_model(arch: str, n_s1: int) -> torch.nn.Module:
    networks = tr.load_dsunet_module("networks")
    channels = {"dsunet": None, "unet_sar": list(range(n_s1)), "unet_opt": list(range(n_s1, n_s1 + N_S2)),
                "unet_fusion": list(range(n_s1 + N_S2))}[arch]
    cfg = SimpleNamespace(
        MODEL=SimpleNamespace(TYPE="dualstreamunet" if arch == "dsunet" else "unet", TOPOLOGY=[64, 128, 256, 512],
                              OUT_CHANNELS=1, IN_CHANNELS=2 * (n_s1 + N_S2) if channels is None else 2 * len(channels)),
        DATASET=SimpleNamespace(MODE="fusion", SENTINEL1_BANDS=[None] * n_s1, SENTINEL2_BANDS=[None] * N_S2),
    )
    with contextlib.redirect_stdout(io.StringIO()):  # the reference UNet prints its layers
        net = networks.create_network(cfg)
    return net if channels is None else ChannelUNet(net, channels)


def apply_condition(t1, t2, condition: str, n_s1: int):
    """tr.apply_condition with the optical channels starting after n_s1 SAR channels."""
    t1, t2 = t1.clone(), t2.clone()
    if condition == "o2_copy":
        t2[:, n_s1:] = t1[:, n_s1:]
    if condition in ("no_o2", "no_opt"):
        t2[:, n_s1:] = 0
    if condition == "no_opt":
        t1[:, n_s1:] = 0
    return t1, t2


@torch.inference_mode()
def evaluate(model, dataset, cities, device, n_s1: int) -> dict[str, torch.Tensor]:
    """tr.evaluate with n_s1 SAR channels: pooled threshold counts per condition, first listed orbit per city."""
    model.eval()
    totals = {c: torch.zeros(101, 4, dtype=torch.long) for c in tr.CONDITIONS}
    for city in cities:
        sample = sample_at_orbit(dataset, dataset.cities.index(city), int(ORBITS[city][0]))
        t1, t2 = sample["t1_img"][None].to(device), sample["t2_img"][None].to(device)
        label = sample["label"].to(device)
        for condition in tr.CONDITIONS:
            prob = torch.sigmoid(model(*apply_condition(t1, t2, condition, n_s1)))[0]
            totals[condition] += tr.threshold_counts(prob, label)
    return totals


def main() -> None:
    p = argparse.ArgumentParser()
    for a, kw in (("--name", {"required": True}), ("--seed", {"type": int, "default": 1}), ("--train-conditions", {"required": True}),
                  ("--arch", {"choices": ("dsunet", "unet_sar", "unet_opt", "unet_fusion"), "default": "dsunet"}),
                  ("--sar", {"choices": tuple(SAR_BANDS), "default": "hafner_vv"}),
                  ("--steps", {"type": int, "default": 2500}), ("--eval-every", {"type": int, "default": 25}),
                  ("--batch-size", {"type": int, "default": 64}), ("--crop-size", {"type": int, "default": 32}),
                  ("--lr", {"type": float, "default": 5e-5}), ("--weight-decay", {"type": float, "default": 5e-4})):
        p.add_argument(a, **kw)
    args = p.parse_args()
    args.protocol, args.loss, args.no_augment, args.uniform_crop = "holdout", "jaccard", False, False
    n_s1 = len(SAR_BANDS[args.sar])
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs = tr.parse_conditions(args.train_conditions)
    selected_on = [c for c in tr.CONDITIONS if probs[c] > 0]

    train_ds, test_ds = RRDDataset("train", args.sar), RRDDataset("test", args.sar)
    test_cities = list(test_ds.cities)
    train_cities, val_cities = split_train_cities(train_ds.cities)
    assert_city_separation(train_cities, val_cities, test_cities)

    model = build_model(args.arch, n_s1).to(device)
    criterion = tr.load_dsunet_module("loss_functions").jaccard_like_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = tr.PROJECT_ROOT / "experiments" / "robust" / f"{args.name}_seed{args.seed}"
    ckpt = tr.PROJECT_ROOT / "checkpoints" / "robust" / f"{args.name}_seed{args.seed}.pt"
    if ckpt.exists():
        raise SystemExit(f"{ckpt} exists; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [tr.PROJECT_ROOT / "src" / "training" / "train_rrd.py", tr.PROJECT_ROOT / "src" / "training" / "train_robust.py",
               tr.PROJECT_ROOT / "src" / "datasets" / "oscd_dataset.py", tr.DSUNET_ROOT / "utils" / "networks.py",
               tr.DSUNET_ROOT / "utils" / "loss_functions.py"]
    source_sha256 = {s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
    (out_dir / "train_rrd.py.snapshot").write_bytes(sources[0].read_bytes())

    history, best = [], (-1.0, None)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        t1s, t2s, labels = [], [], []
        for city in rng.choice(train_cities, args.batch_size):
            s = train_ds[train_ds.cities.index(city)]  # random orbit, as DS-UNet
            t1, t2, lab = tr.random_crop(s["t1_img"], s["t2_img"], s["label"], args.crop_size, True, rng)
            t1, t2, lab = tr.augment(t1, t2, lab, rng)
            t1, t2 = apply_condition(t1[None], t2[None], rng.choice(tr.CONDITIONS, p=[probs[c] for c in tr.CONDITIONS]), n_s1)
            t1s.append(t1[0]), t2s.append(t2[0]), labels.append(lab)
        t1, t2, label = (torch.stack(x).to(device) for x in (t1s, t2s, labels))
        optimizer.zero_grad()
        loss = criterion(model(t1, t2), label)
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            counts = evaluate(model, train_ds, val_cities, device, n_s1)
            score = float(np.mean([tr.best_threshold(counts[c])[1] for c in selected_on]))
            history.append({"step": step, "loss": loss.item(), "val_score": score,
                            **{f"val_f1_{c}": tr.best_threshold(counts[c])[1] for c in tr.CONDITIONS}})
            print(f"step {step} loss {loss.item():.4f} val {score:.4f}", flush=True)
            if score > best[0]:
                best = (score, step)
                torch.save({"model_state_dict": model.state_dict(), "step": step,
                            "thresholds": {c: tr.best_threshold(counts[c])[0] for c in tr.CONDITIONS}, "args": vars(args)}, ckpt)

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    test_counts = evaluate(model, test_ds, test_cities, device, n_s1)
    test = {c: {"selected_threshold": saved["thresholds"][c] / 100, "at_selected": tr.rates_at(test_counts[c], saved["thresholds"][c]),
                "at_0.5": tr.rates_at(test_counts[c], 50)} for c in tr.CONDITIONS}
    doc = {"args": vars(args), "train_cities": train_cities, "val_cities": val_cities, "test_cities": test_cities,
           "selected_step": saved["step"], "selection_score": best[0], "selected_on": selected_on, "test": test, "history": history,
           "train_seconds": round(time.perf_counter() - start, 1), "torch": torch.__version__, "source_sha256": source_sha256}
    (out_dir / "metrics.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(" | ".join(f"{c} F1 {test[c]['at_selected']['f1']:.4f} (tau {test[c]['selected_threshold']:.2f})" for c in tr.CONDITIONS))


if __name__ == "__main__":
    main()
