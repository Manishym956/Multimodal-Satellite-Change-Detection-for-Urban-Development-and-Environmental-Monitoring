"""Step 4A-bis: zero-fill under imperfect cloud masks, evaluation only, on the 20 frozen Phase 2 checkpoints.

For each level L in c10/c30/c60 the post-change optical image is corrupted with the true Phase 3 mask M (cloud spectrum),
then zero-filled under a detector mask P (phase4abis/masks, see src/phase3/mask_errors.py):
    L__erode{2,4,8}  L__dilate{2,4,8}  L__miss{10,25,50}
References re-evaluated as anchors: clean, cL (cloud, no fill), zL (oracle fill P = M), which must equal the stored Step 4A counts.
Stores pooled (101, 4) counts on validation and test cities, per test city, and on test pixels that are
missed cloud (M \\ P), over-masked clear (P \\ M) and the rest, for the corrupted input and for clean input.

Usage:
    .venv/Scripts/python.exe eval_phase4abis.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from eval_phase4a import zero_post
from src.datasets.oscd_dataset import ORBITS, OSCDDataset
from src.phase3.corruption import LEVELS, PHASE3, corrupt_post, load_spectrum
from src.phase3.mask_errors import VARIANTS
from src.training.train_robust import best_threshold, build_model, threshold_counts
from src.training.train_sar_baseline import sample_at_orbit, split_train_cities

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "robust" / "phase4abis"
P4A = ROOT / "experiments" / "robust" / "phase4a"
REFS = ("clean", *LEVELS, *(f"z{lv[1:]}" for lv in LEVELS))
IMPERFECT = tuple(f"{lv}__{v}" for lv in LEVELS for v in VARIANTS)
CONDITIONS = REFS + IMPERFECT
REGIONS = ("missed", "over", "rest")


@torch.inference_mode()
def evaluate(model, ds, cities, spectrum, device, regions: bool):
    model.eval()
    zeros = lambda keys: {k: torch.zeros(101, 4, dtype=torch.long) for k in keys}
    out = {"pooled": zeros(CONDITIONS), "city": {},
           **{f"{r}{s}": zeros(IMPERFECT) for r in REGIONS for s in ("", "_clean")}}
    for city in cities:
        sample = sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))
        t1, t2 = sample["t1_img"][None].to(device), sample["t2_img"][None].to(device)
        label = sample["label"].to(device)
        oracle = {lv: torch.from_numpy(np.load(PHASE3 / "masks" / f"{city}_{lv}.npy")).to(device) for lv in LEVELS}
        clouded = {lv: corrupt_post(t2, oracle[lv], spectrum) for lv in LEVELS}
        clean_prob = None
        for c in CONDITIONS:
            if c == "clean":
                x2 = t2
            elif c in LEVELS:
                x2 = clouded[c]
            elif c.startswith("z"):
                lv = f"c{c[1:]}"
                x2 = zero_post(clouded[lv], oracle[lv])
            else:
                lv, v = c.split("__")
                det = torch.from_numpy(np.load(OUT / "masks" / f"{city}_{lv}_{v}.npy")).to(device)
                x2 = zero_post(clouded[lv], det)
            prob = torch.sigmoid(model(t1, x2))[0]
            if c == "clean":
                clean_prob = prob
            out["pooled"][c] += threshold_counts(prob, label)
            out["city"][(city, c)] = threshold_counts(prob, label)
            if regions and c in IMPERFECT:
                m = oracle[lv][None]
                p = det[None]
                for r, sel in (("missed", m & ~p), ("over", p & ~m), ("rest", ~(m ^ p))):
                    out[r][c] += threshold_counts(prob[sel], label[sel])
                    out[f"{r}_clean"][c] += threshold_counts(clean_prob[sel], label[sel])
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    spectrum = torch.from_numpy(load_spectrum()).to(device)
    train_ds, test_ds = OSCDDataset(split="train", mode="fusion"), OSCDDataset(split="test", mode="fusion")
    _, val_cities = split_train_cities(train_ds.cities)
    test_cities = list(test_ds.cities)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
               for p in ("eval_phase4abis.py", "eval_phase4a.py", "src/phase3/mask_errors.py", "src/phase3/corruption.py",
                         "src/training/train_robust.py")}
    failures = []
    for arm in ("p2_full", "p2_dropout"):
        for seed in range(1, 11):
            name = f"{arm}_seed{seed}"
            if (OUT / "runs" / f"{name}.json").exists():
                print(name, "exists, skipped")
                continue
            ckpt = torch.load(ROOT / "checkpoints" / "robust" / f"{name}.pt", map_location="cpu", weights_only=False)
            model = build_model().to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            p4a = np.load(P4A / "runs" / f"{name}.npz")
            val = evaluate(model, train_ds, val_cities, spectrum, device, regions=False)
            test = evaluate(model, test_ds, test_cities, spectrum, device, regions=True)
            tau = ckpt["thresholds"]["full"]
            anchors = {"clean_val_argmax": best_threshold(val["pooled"]["clean"])[0] == tau,
                       **{f"{c}_test": np.array_equal(test["pooled"][c].numpy(), p4a[f"test__{c}"]) for c in REFS},
                       **{f"{c}_val": np.array_equal(val["pooled"][c].numpy(), p4a[f"val__{c}"]) for c in REFS}}
            if not all(anchors.values()):
                failures.append(name)
                print(name, "ANCHOR MISMATCH", {k: v for k, v in anchors.items() if not v}, flush=True)
                continue
            np.savez_compressed(OUT / "runs" / f"{name}.npz",
                                **{f"val__{c}": v.numpy() for c, v in val["pooled"].items()},
                                **{f"test__{c}": v.numpy() for c, v in test["pooled"].items()},
                                **{f"{key}__{c}": v.numpy() for key in (f"{r}{s}" for r in REGIONS for s in ("", "_clean"))
                                   for c, v in test[key].items()},
                                **{f"city__{city}__{c}": v.numpy() for (city, c), v in test["city"].items()})
            doc = {"run": name, "arm": arm, "seed": seed, "tau_clean": tau / 100,
                   "tau_level_val": {c: best_threshold(val["pooled"][c])[0] / 100 for c in CONDITIONS},
                   "anchors": anchors, "source_sha256": sources}
            (OUT / "runs" / f"{name}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
            f1 = {}
            for c in ("clean", "c30", "z30") + tuple(f"c30__{v}" for v in VARIANTS):
                tp, fp, fn, _ = (int(v) for v in test["pooled"][c][tau])
                f1[c] = 2 * tp / (2 * tp + fp + fn)
            print(name, " | ".join(f"{c} {v:.3f}" for c, v in f1.items()), flush=True)
    if failures:
        sys.exit(1)
    print("P4ABIS EVAL DONE")


if __name__ == "__main__":
    main()
