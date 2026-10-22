"""Step 4A: oracle-mask zero-fill, evaluation only, on the 20 frozen Phase 2 checkpoints.

Post-change optical (S2 t2) inputs, with SAR and S2 t1 untouched:
    clean          reference (= Phase 2 `full`)
    c10/c30/c60    Phase 3 cloud corruption (same masks, same spectrum, same code)
    z10/z30/z60    S2 t2 set to 0 only inside the same Phase 3 mask (oracle cloud mask), unchanged outside
Stores pooled (101, 4) TP/FP/FN/TN tables on validation and test cities, per test city, and on test pixels inside /
outside the mask for both the corrupted input and clean input. One test evaluation per checkpoint and condition.

Anchors (a mismatch skips the checkpoint and exits 1): clean and cloud test counts (pooled, inside, outside) equal the
stored Phase 3 counts; the clean validation argmax equals the stored clean threshold.

Usage:
    .venv/Scripts/python.exe eval_phase4a.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.datasets.oscd_dataset import ORBITS, OSCDDataset
from src.phase3.corruption import N_S1, PHASE3, corrupt_post, load_spectrum
from src.training.train_robust import best_threshold, build_model, threshold_counts
from src.training.train_sar_baseline import sample_at_orbit, split_train_cities

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "robust" / "phase4a"
PAIRS = {"c10": "z10", "c30": "z30", "c60": "z60"}
CONDITIONS = ("clean", *PAIRS, *PAIRS.values())
MASK_OF = {**{c: c for c in PAIRS}, **{z: c for c, z in PAIRS.items()}}


def zero_post(t2, mask):
    """S2 t2 = 0 where mask, unchanged elsewhere (the Phase 2 `no_o2` fill, restricted to the mask)."""
    t2 = t2.clone()
    t2[:, N_S1:] = t2[:, N_S1:].masked_fill(mask[None, None], 0.0)
    return t2


def condition_input(t1, t2, c, masks, spectrum):
    if c == "clean":
        return t1, t2
    if c in PAIRS:
        return t1, corrupt_post(t2, masks[c], spectrum)
    return t1, zero_post(t2, masks[MASK_OF[c]])


@torch.inference_mode()
def evaluate(model, ds, cities, spectrum, device):
    model.eval()
    zeros = lambda keys: {k: torch.zeros(101, 4, dtype=torch.long) for k in keys}
    out = {"pooled": zeros(CONDITIONS), "city": {}, "inside": zeros(CONDITIONS[1:]), "outside": zeros(CONDITIONS[1:]),
           "inside_clean": zeros(PAIRS), "outside_clean": zeros(PAIRS)}
    for city in cities:
        sample = sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))
        t1, t2 = sample["t1_img"][None].to(device), sample["t2_img"][None].to(device)
        label = sample["label"].to(device)
        masks = {lv: torch.from_numpy(np.load(PHASE3 / "masks" / f"{city}_{lv}.npy")).to(device) for lv in PAIRS}
        for c in CONDITIONS:
            prob = torch.sigmoid(model(*condition_input(t1, t2, c, masks, spectrum)))[0]
            out["pooled"][c] += threshold_counts(prob, label)
            out["city"][(city, c)] = threshold_counts(prob, label)
            if c == "clean":
                clean_prob = prob
                continue
            m = masks[MASK_OF[c]][None]
            out["inside"][c] += threshold_counts(prob[m], label[m])
            out["outside"][c] += threshold_counts(prob[~m], label[~m])
            if c in PAIRS:
                out["inside_clean"][c] += threshold_counts(clean_prob[m], label[m])
                out["outside_clean"][c] += threshold_counts(clean_prob[~m], label[~m])
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
               for p in ("eval_phase4a.py", "src/phase3/corruption.py", "src/training/train_robust.py")}
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
            p3 = np.load(PHASE3 / "runs" / f"{name}.npz")
            val = evaluate(model, train_ds, val_cities, spectrum, device)
            test = evaluate(model, test_ds, test_cities, spectrum, device)
            tau = ckpt["thresholds"]["full"]
            anchors = {"clean_val_argmax": best_threshold(val["pooled"]["clean"])[0] == tau,
                       "clean_test": np.array_equal(test["pooled"]["clean"].numpy(), p3["test__clean"])}
            for c in PAIRS:
                for key, p3key in (("pooled", "test"), ("inside", "inside"), ("outside", "outside"),
                                   ("inside_clean", "inside_clean"), ("outside_clean", "outside_clean")):
                    anchors[f"{c}_{key}"] = np.array_equal(test[key][c].numpy(), p3[f"{p3key}__{c}"])
            if not all(anchors.values()):
                failures.append(name)
                print(name, "ANCHOR MISMATCH", {k: v for k, v in anchors.items() if not v}, flush=True)
                continue
            np.savez_compressed(OUT / "runs" / f"{name}.npz",
                                **{f"val__{c}": v.numpy() for c, v in val["pooled"].items()},
                                **{f"{key}__{c}": v.numpy() for key in ("inside", "outside", "inside_clean", "outside_clean")
                                   for c, v in test[key].items()},
                                **{f"test__{c}": v.numpy() for c, v in test["pooled"].items()},
                                **{f"city__{city}__{c}": v.numpy() for (city, c), v in test["city"].items()})
            doc = {"run": name, "arm": arm, "seed": seed, "checkpoint_step": ckpt["step"], "tau_clean": tau / 100,
                   "tau_level_val": {c: best_threshold(val["pooled"][c])[0] / 100 for c in CONDITIONS},
                   "anchors": anchors, "val_cities": val_cities, "test_cities": test_cities, "source_sha256": sources}
            (OUT / "runs" / f"{name}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
            k = tau
            f1 = {}
            for c in CONDITIONS:
                tp, fp, fn, _ = (int(v) for v in test["pooled"][c][k])
                f1[c] = 2 * tp / (2 * tp + fp + fn)
            print(name, " | ".join(f"{c} {v:.3f}" for c, v in f1.items()), flush=True)
    if failures:
        sys.exit(1)
    print("P4A EVAL DONE")


if __name__ == "__main__":
    main()
