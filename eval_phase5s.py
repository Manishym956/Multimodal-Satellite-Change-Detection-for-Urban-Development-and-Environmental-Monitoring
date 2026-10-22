"""Step 5S (synthetic): does dilating an under-masking detector mask recover F1? Evaluation only, 30 frozen checkpoints.

Image: Phase 3 corruption (true mask M, Phase 3 cloud spectrum) at c10/c30/c60. Detector mask P: the Step 4A-bis
under-masking masks (phase4abis/masks: erode2/4/8, miss10/25/50). Recall-biased mask: P dilated by a disk of radius r
(10 m px), r in RADII (r = 0 is P itself). Conditions: clean, cL (no fill), zL (oracle fill), L__v__r for every level,
under-masking variant v and radius r. Pooled (101, 4) counts on validation and test cities.
Anchors (failure: run not saved): clean, cL, zL and L__v__0 counts equal the stored Step 4A-bis (A, B) or Step 4B (C)
counts on validation and test; the clean validation argmax equals the stored threshold. Prints no test metric.

Usage:
    .venv/Scripts/python.exe eval_phase5s.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from eval_phase4a import zero_post
from src.datasets.oscd_dataset import ORBITS, OSCDDataset
from src.phase3.corruption import LEVELS, PHASE3, corrupt_post, load_spectrum
from src.phase3.mask_errors import disk
from src.training.train_robust import best_threshold, build_model, threshold_counts
from src.training.train_sar_baseline import sample_at_orbit, split_train_cities

ROOT = Path(__file__).resolve().parent
R = ROOT / "experiments" / "robust"
OUT = R / "phase5s"
ARMS = {"p2_full": "phase4abis", "p2_dropout": "phase4abis", "p4b_spatial": "phase4b"}
UNDER = ("erode2", "erode4", "erode8", "miss10", "miss25", "miss50")
RADII = (0, 1, 2, 3, 4, 6, 8, 12, 16)
REFS = ("clean", *LEVELS, *(f"z{lv[1:]}" for lv in LEVELS))
CONDITIONS = REFS + tuple(f"{lv}__{v}__{r}" for lv in LEVELS for v in UNDER for r in RADII)


def city_masks(city):
    m = {lv: np.load(PHASE3 / "masks" / f"{city}_{lv}.npy") for lv in LEVELS}
    p = {}
    for lv in LEVELS:
        for v in UNDER:
            base = np.load(R / "phase4abis" / "masks" / f"{city}_{lv}_{v}.npy")
            for r in RADII:
                p[(lv, v, r)] = torch.from_numpy(ndimage.binary_dilation(base, disk(r)) if r else base)
    return {lv: torch.from_numpy(x) for lv, x in m.items()}, p


@torch.inference_mode()
def evaluate(model, ds, cities, masks, spectrum, device):
    model.eval()
    out = {c: torch.zeros(101, 4, dtype=torch.long) for c in CONDITIONS}
    for city in cities:
        sample = sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))
        t1, t2 = sample["t1_img"][None].to(device), sample["t2_img"][None].to(device)
        label = sample["label"].to(device)
        oracle, det = masks[city]
        run = lambda x2: threshold_counts(torch.sigmoid(model(t1, x2))[0], label)
        out["clean"] += run(t2)
        for lv in LEVELS:
            m = oracle[lv].to(device)
            clouded = corrupt_post(t2, m, spectrum)
            out[lv] += run(clouded)
            out[f"z{lv[1:]}"] += run(zero_post(clouded, m))
            for v in UNDER:
                for r in RADII:
                    out[f"{lv}__{v}__{r}"] += run(zero_post(clouded, det[(lv, v, r)].to(device)))
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    spectrum = torch.from_numpy(load_spectrum()).to(device)
    train_ds, test_ds = OSCDDataset(split="train", mode="fusion"), OSCDDataset(split="test", mode="fusion")
    _, val_cities = split_train_cities(train_ds.cities)
    test_cities = list(test_ds.cities)
    masks = {c: city_masks(c) for c in val_cities + test_cities}
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
               for p in ("eval_phase5s.py", "eval_phase4a.py", "src/phase3/corruption.py", "src/phase3/mask_errors.py",
                         "src/training/train_robust.py", "src/datasets/oscd_dataset.py")}
    failures = []
    for arm, ref in ARMS.items():
        for seed in range(1, 11):
            name = f"{arm}_seed{seed}"
            if (OUT / "runs" / f"{name}.json").exists():
                print(name, "exists, skipped")
                continue
            ckpt = torch.load(ROOT / "checkpoints" / "robust" / f"{name}.pt", map_location="cpu", weights_only=False)
            model = build_model().to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            stored = np.load(R / ref / "runs" / f"{name}.npz")
            val = evaluate(model, train_ds, val_cities, masks, spectrum, device)
            test = evaluate(model, test_ds, test_cities, masks, spectrum, device)
            anchor_keys = {**{c: c for c in REFS}, **{f"{lv}__{v}__0": f"{lv}__{v}" for lv in LEVELS for v in UNDER}}
            anchors = {"refs_and_r0_val_equal_stored": all(np.array_equal(val[k].numpy(), stored[f"val__{s}"]) for k, s in anchor_keys.items()),
                       "refs_and_r0_test_equal_stored": all(np.array_equal(test[k].numpy(), stored[f"test__{s}"]) for k, s in anchor_keys.items()),
                       "clean_val_argmax": best_threshold(val["clean"])[0] == ckpt["thresholds"]["full"]}
            if not all(anchors.values()):
                failures.append(name)
                print(name, "ANCHOR MISMATCH", anchors, flush=True)
                continue
            np.savez_compressed(OUT / "runs" / f"{name}.npz", **{f"val__{c}": v.numpy() for c, v in val.items()},
                                **{f"test__{c}": v.numpy() for c, v in test.items()})
            doc = {"run": name, "arm": arm, "seed": seed, "tau_clean": ckpt["thresholds"]["full"] / 100, "anchors": anchors,
                   "source_sha256": sources}
            (OUT / "runs" / f"{name}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
            print(name, "done", flush=True)
    if failures:
        sys.exit(1)
    print("P5S EVAL DONE")


if __name__ == "__main__":
    main()
