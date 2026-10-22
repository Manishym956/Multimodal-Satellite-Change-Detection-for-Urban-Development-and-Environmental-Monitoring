"""Step 4B: train_robust.py plus one extra training condition, `spatial` (cloud-shaped partial zero-fill of S2 t2).

Everything except the training-condition draw is train_robust.py, imported unchanged (frozen Phase 2 code):
model, loss, optimizer, crops, augmentation, validation evaluation (full / no_o2 / no_opt / o2_copy), checkpoint
selection (mean best-threshold validation F1 over the trained conditions OTHER than `spatial`, i.e. the same rule as
p2_dropout), per-condition thresholds and the single test evaluation.

`spatial`: S2 t2 is set to 0 under a mask drawn uniformly from phase4b/spatial_mask_bank.npy (cloud shapes from catalogue
scenes not used by any evaluation mask), applied after crop and augmentation.

Usage:
    .venv/Scripts/python.exe -m src.training.train_spatial --name p4b_spatial --seed 1 --steps 2500 --eval-every 25 \
        --train-conditions full=0.25,spatial=0.25,no_o2=0.25,no_opt=0.25
"""

import argparse
import hashlib
import json
import time

import numpy as np
import torch

from src.datasets.oscd_dataset import OSCDDataset
from src.training import train_robust as tr
from src.training.train_sar_baseline import assert_city_separation, set_seed, split_train_cities

BANK = tr.PROJECT_ROOT / "experiments" / "robust" / "phase4b" / "spatial_mask_bank.npy"
TRAIN_CONDITIONS = tr.CONDITIONS + ("spatial",)  # appended: the cdf draw of earlier conditions is unchanged


def parse_conditions(text: str) -> dict[str, float]:
    probs = {k: 0.0 for k in TRAIN_CONDITIONS}
    for part in text.split(","):
        key, value = part.split("=")
        if key not in probs:
            raise ValueError(f"unknown condition {key}")
        probs[key] = float(value)
    if abs(sum(probs.values()) - 1) > 1e-6:
        raise ValueError("condition probabilities must sum to 1")
    return probs


def apply_train_condition(t1, t2, condition, bank, rng):
    if condition != "spatial":
        return tr.apply_condition(t1, t2, condition)
    t2 = t2.clone()
    mask = torch.from_numpy(bank[rng.integers(len(bank))])
    t2[:, tr.N_S1:] = t2[:, tr.N_S1:].masked_fill(mask[None, None], 0.0)
    return t1, t2


def main() -> None:
    p = argparse.ArgumentParser()
    for a, kw in (("--name", {"required": True}), ("--seed", {"type": int, "default": 1}), ("--train-conditions", {"required": True}),
                  ("--steps", {"type": int, "default": 2500}), ("--eval-every", {"type": int, "default": 25}),
                  ("--batch-size", {"type": int, "default": 64}), ("--crop-size", {"type": int, "default": 32}),
                  ("--lr", {"type": float, "default": 5e-5}), ("--weight-decay", {"type": float, "default": 5e-4})):
        p.add_argument(a, **kw)
    args = p.parse_args()
    args.protocol, args.loss, args.no_augment, args.uniform_crop = "holdout", "jaccard", False, False
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs = parse_conditions(args.train_conditions)
    selected_on = [c for c in tr.CONDITIONS if probs[c] > 0]
    bank = np.load(BANK)
    assert bank.shape[1:] == (args.crop_size, args.crop_size)

    train_ds, test_ds = OSCDDataset(split="train", mode="fusion"), OSCDDataset(split="test", mode="fusion")
    test_cities = list(test_ds.cities)
    train_cities, val_cities = split_train_cities(train_ds.cities)
    assert_city_separation(train_cities, val_cities, test_cities)

    model = tr.build_model().to(device)
    criterion = tr.load_dsunet_module("loss_functions").jaccard_like_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = tr.PROJECT_ROOT / "experiments" / "robust" / f"{args.name}_seed{args.seed}"
    ckpt = tr.PROJECT_ROOT / "checkpoints" / "robust" / f"{args.name}_seed{args.seed}.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [tr.PROJECT_ROOT / "src" / "training" / "train_spatial.py", tr.PROJECT_ROOT / "src" / "training" / "train_robust.py",
               tr.PROJECT_ROOT / "src" / "datasets" / "oscd_dataset.py", tr.DSUNET_ROOT / "utils" / "networks.py",
               tr.DSUNET_ROOT / "utils" / "loss_functions.py", BANK]
    source_sha256 = {s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
    (out_dir / "train_spatial.py.snapshot").write_bytes(sources[0].read_bytes())

    history, best = [], (-1.0, None)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        t1s, t2s, labels = [], [], []
        for city in rng.choice(train_cities, args.batch_size):
            s = train_ds[train_ds.cities.index(city)]
            t1, t2, lab = tr.random_crop(s["t1_img"], s["t2_img"], s["label"], args.crop_size, True, rng)
            t1, t2, lab = tr.augment(t1, t2, lab, rng)
            condition = rng.choice(TRAIN_CONDITIONS, p=[probs[c] for c in TRAIN_CONDITIONS])
            t1, t2 = apply_train_condition(t1[None], t2[None], condition, bank, rng)
            t1s.append(t1[0]), t2s.append(t2[0]), labels.append(lab)
        t1, t2, label = (torch.stack(x).to(device) for x in (t1s, t2s, labels))
        optimizer.zero_grad()
        loss = criterion(model(t1, t2), label)
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            counts = tr.evaluate(model, train_ds, val_cities, device)
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
    test_counts = tr.evaluate(model, test_ds, test_cities, device)
    test = {c: {"selected_threshold": saved["thresholds"][c] / 100, "at_selected": tr.rates_at(test_counts[c], saved["thresholds"][c]),
                "at_0.5": tr.rates_at(test_counts[c], 50)} for c in tr.CONDITIONS}
    doc = {"args": vars(args), "train_cities": train_cities, "val_cities": val_cities, "test_cities": test_cities,
           "selected_step": saved["step"], "selection_score": best[0], "selected_on": selected_on, "test": test, "history": history,
           "train_seconds": round(time.perf_counter() - start, 1), "torch": torch.__version__, "source_sha256": source_sha256}
    (out_dir / "metrics.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(" | ".join(f"{c} F1 {test[c]['at_selected']['f1']:.4f} (tau {test[c]['selected_threshold']:.2f})" for c in tr.CONDITIONS))


if __name__ == "__main__":
    main()
