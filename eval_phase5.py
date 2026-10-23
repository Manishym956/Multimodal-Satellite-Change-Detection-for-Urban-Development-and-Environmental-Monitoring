"""Step 5: evaluate the 30 frozen checkpoints (A p2_full, B p2_dropout, C p4b_spatial; seeds 1-10) on composite clouds
with real-detector masks.

Per composite ({city}_{level}_k{k}, src/phase5/composite.py), with M the true cloud mask and P an s2cloudless detector mask:
    L__cloud    composite post-change image, no fill                       (real cloud pixels reach the model)
    L__oracle   composite, zero-filled under M
    L__<op>     composite, zero-filled under P at operating point op in OPS (operating_points.json: rb95, rb90, rb99, bal)
    L__default  composite, zero-filled under the library's default s2cloudless mask
Per city, on the clean post-change image: clean, and clean__<op> / clean__default (zero-filled under the detector mask run on
the clean image: the cost of false detections alone).
Counts (101 thresholds x TP/FP/FN/TN) are pooled per level over cities and replicates, on validation and test; test counts are
also stored per city, per composite and inside / outside M. Zero-fill is eval_phase4a.zero_post.
Anchors (any failure: the run is not saved): clean validation and test counts equal the stored Step 4A-bis (A, B) or Step 4B (C)
counts; the clean validation argmax equals the stored clean threshold; inside + outside M equals pooled for every composite
condition; outside M every composite equals the clean image.

Usage:
    .venv/Scripts/python.exe eval_phase5.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from eval_phase4a import zero_post
from src.datasets.oscd_dataset import ORBITS, OSCDDataset
from src.phase5.composite import LEVELS, N_S1, PHASE5, REPLICATES, composite_post, detector_mask, upsample
from src.training.train_robust import best_threshold, build_model, rates_at, threshold_counts
from src.training.train_sar_baseline import sample_at_orbit, split_train_cities

ROOT = Path(__file__).resolve().parent
R = ROOT / "experiments" / "robust"
ARMS = {"p2_full": "phase4abis", "p2_dropout": "phase4abis", "p4b_spatial": "phase4b"}
OPS = ("rb95", "rb90", "rb99", "bal")
DETS = OPS + ("default",)
LEVEL_CONDS = ("cloud", "oracle") + DETS
CONDITIONS = ("clean",) + tuple(f"clean__{d}" for d in DETS) + tuple(f"{lv}__{c}" for lv in LEVELS for c in LEVEL_CONDS)
COMPOSITE_CONDS = tuple(c for c in CONDITIONS if not c.startswith("clean"))


def load_masks(split, cities, ops):
    """{name: {"m": M or None, det: P}} for every clean image and composite of the split (computed once)."""
    out = {}
    for city in cities:
        for name in [f"{city}_clean"] + [f"{city}_{lv}_k{k}" for lv in LEVELS for k in range(REPLICATES)]:
            inp, det = np.load(PHASE5 / "detector_inputs" / split / f"{name}.npz"), np.load(PHASE5 / "detector_outputs" / split / f"{name}.npz")
            masks = {o: torch.from_numpy(detector_mask(det["prob"], ops[o]["t"], ops[o]["r"])) for o in OPS}
            masks["default"] = torch.from_numpy(det["default"])
            masks["m"] = torch.from_numpy(inp["m"]) if "m" in inp.files else None
            out[name] = masks
    return out


@torch.inference_mode()
def evaluate(model, ds, cities, masks, device, inside_outside: bool):
    model.eval()
    zeros = lambda keys: {k: torch.zeros(101, 4, dtype=torch.long) for k in keys}
    out = {"pooled": zeros(CONDITIONS), "city": {}, "comp": {}, "inside": zeros(COMPOSITE_CONDS), "outside": zeros(COMPOSITE_CONDS),
           "outside_identical": True}
    for city in cities:
        sample = sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))
        t1, t2 = sample["t1_img"][None].to(device), sample["t2_img"][None].to(device)
        label = sample["label"].to(device)
        shape = tuple(t2.shape[-2:])

        def add(c, x2, m=None, comp=None):
            prob = torch.sigmoid(model(t1, x2))[0]
            cnt = threshold_counts(prob, label)
            out["pooled"][c] += cnt
            out["city"][(city, c)] = out["city"].get((city, c), torch.zeros(101, 4, dtype=torch.long)) + cnt
            if comp is not None:
                out["comp"][(comp, c)] = cnt
            if inside_outside and m is not None:
                out["inside"][c] += threshold_counts(prob[m[None]], label[m[None]])
                out["outside"][c] += threshold_counts(prob[~m[None]], label[~m[None]])

        cm = masks[f"{city}_clean"]
        add("clean", t2)
        for d in DETS:
            add(f"clean__{d}", zero_post(t2, cm[d].to(device)))
        for lv in LEVELS:
            for k in range(REPLICATES):
                z = np.load(PHASE5 / "composites" / f"{city}_{lv}_k{k}.npz")
                mk = masks[f"{city}_{lv}_k{k}"]
                m = mk["m"].to(device)
                assert torch.equal(m.cpu(), torch.from_numpy(upsample(z["cloud20"], shape)))
                bands = torch.from_numpy(np.ascontiguousarray(upsample(z["bands20"], shape).transpose(2, 0, 1))).to(device)
                x2 = composite_post(t2, m, bands)
                out["outside_identical"] &= bool(torch.equal(x2[0, :, ~m], t2[0, :, ~m]))
                comp = f"{city}_{lv}_k{k}"
                add(f"{lv}__cloud", x2, m, comp)
                add(f"{lv}__oracle", zero_post(x2, m), m, comp)
                for d in DETS:
                    add(f"{lv}__{d}", zero_post(x2, mk[d].to(device)), m, comp)
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    ops = json.loads((PHASE5 / "operating_points.json").read_text(encoding="utf-8"))["points"]
    train_ds, test_ds = OSCDDataset(split="train", mode="fusion"), OSCDDataset(split="test", mode="fusion")
    _, val_cities = split_train_cities(train_ds.cities)
    test_cities = list(test_ds.cities)
    masks = {"val": load_masks("val", val_cities, ops), "test": load_masks("test", test_cities, ops)}
    (PHASE5 / "runs").mkdir(parents=True, exist_ok=True)
    sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
               for p in ("eval_phase5.py", "src/phase5/composite.py", "eval_phase4a.py", "src/training/train_robust.py",
                         "src/datasets/oscd_dataset.py", "experiments/robust/phase5/operating_points.json",
                         "experiments/robust/phase5/composite_manifest.json")}
    for split in ("val", "test"):
        sources[f"detector_info_{split}.json"] = hashlib.sha256(
            (PHASE5 / "detector_outputs" / split / "detector_info.json").read_bytes()).hexdigest()
    failures = []
    for arm, ref in ARMS.items():
        for seed in range(1, 11):
            name = f"{arm}_seed{seed}"
            if (PHASE5 / "runs" / f"{name}.json").exists():
                print(name, "exists, skipped")
                continue
            ckpt = torch.load(ROOT / "checkpoints" / "robust" / f"{name}.pt", map_location="cpu", weights_only=False)
            model = build_model().to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            stored = np.load(R / ref / "runs" / f"{name}.npz")
            val = evaluate(model, train_ds, val_cities, masks["val"], device, inside_outside=False)
            test = evaluate(model, test_ds, test_cities, masks["test"], device, inside_outside=True)
            tau = ckpt["thresholds"]["full"]
            anchors = {"clean_val_equals_stored": np.array_equal(val["pooled"]["clean"].numpy(), stored["val__clean"]),
                       "clean_test_equals_stored": np.array_equal(test["pooled"]["clean"].numpy(), stored["test__clean"]),
                       "clean_val_argmax": best_threshold(val["pooled"]["clean"])[0] == tau,
                       "inside_plus_outside_equals_pooled": all(torch.equal(test["inside"][c] + test["outside"][c], test["pooled"][c])
                                                                for c in COMPOSITE_CONDS),
                       "outside_mask_identical": val["outside_identical"] and test["outside_identical"]}
            if not all(anchors.values()):
                failures.append(name)
                print(name, "ANCHOR MISMATCH", anchors, flush=True)
                continue
            np.savez_compressed(PHASE5 / "runs" / f"{name}.npz",
                                **{f"val__{c}": v.numpy() for c, v in val["pooled"].items()},
                                **{f"test__{c}": v.numpy() for c, v in test["pooled"].items()},
                                **{f"{part}__{c}": v.numpy() for part in ("inside", "outside") for c, v in test[part].items()},
                                **{f"city__{city}__{c}": v.numpy() for (city, c), v in test["city"].items()},
                                **{f"comp__{comp}__{c}": v.numpy() for (comp, c), v in test["comp"].items()})
            doc = {"run": name, "arm": arm, "seed": seed, "tau_clean": tau / 100,
                   "tau_level_val": {c: best_threshold(val["pooled"][c])[0] / 100 for c in CONDITIONS},
                   "anchors": anchors, "source_sha256": sources}
            (PHASE5 / "runs" / f"{name}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
            print(name, "done", flush=True)
    if failures:
        sys.exit(1)
    print("P5 EVAL DONE")


if __name__ == "__main__":
    main()
