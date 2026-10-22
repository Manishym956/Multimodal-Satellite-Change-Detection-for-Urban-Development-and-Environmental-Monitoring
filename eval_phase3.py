"""Phase 3: evaluate the 20 frozen Phase 2 checkpoints under post-change optical corruption.

Evaluation only: no training, no change to any checkpoint. For each checkpoint and condition it stores
pooled (101, 4) TP/FP/FN/TN tables (threshold k/100) on the validation and test cities, per city, and on
test pixels inside / outside the cloud mask. One test evaluation per checkpoint and condition.

Conditions (S2 t2 only; SAR and S2 t1 untouched):
    clean    reference, identical to Phase 2 `full`
    c10/c30/c60  real-shape cloud mask at exactly 10/30/60 % of pixels, opaque cloud spectrum
    missing  S2 t2 zero-filled, identical to Phase 2 `no_o2`
    c100     whole post image replaced by the cloud spectrum   (secondary)
    haze     whole post image blended with the spectrum, alpha 0.3 (secondary)

Anchors checked before anything is written (exit 1 on mismatch):
    - clean test counts at the stored clean threshold and at 0.5 equal Phase 2 `full` counts
    - missing test counts at the stored no_o2 threshold and at 0.5 equal Phase 2 `no_o2` counts
    - clean validation argmax equals the stored clean threshold

Usage:
    .venv/Scripts/python.exe eval_phase3.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.datasets.oscd_dataset import ORBITS, OSCDDataset
from src.phase3.corruption import HAZE_ALPHA, LEVELS, PHASE3, corrupt_post, load_spectrum
from src.training.train_robust import apply_condition, best_threshold, build_model, rates_at, threshold_counts
from src.training.train_sar_baseline import sample_at_orbit, split_train_cities

ROOT = Path(__file__).resolve().parent
CONDITIONS = ("clean", *LEVELS, "missing", "c100", "haze")
ARMS = ("p2_full", "p2_dropout")


def condition_input(t1, t2, condition, masks, spectrum):
    if condition == "clean":
        return t1, t2
    if condition == "missing":
        return apply_condition(t1, t2, "no_o2")
    if condition == "c100":
        return t1, corrupt_post(t2, None, spectrum)
    if condition == "haze":
        return t1, corrupt_post(t2, None, spectrum, HAZE_ALPHA)
    return t1, corrupt_post(t2, masks[condition], spectrum)


@torch.inference_mode()
def evaluate(model, ds, cities, spectrum, device, regions: bool):
    model.eval()
    out = {"pooled": {c: torch.zeros(101, 4, dtype=torch.long) for c in CONDITIONS}, "city": {},
           "inside": {c: torch.zeros(101, 4, dtype=torch.long) for c in LEVELS},
           "outside": {c: torch.zeros(101, 4, dtype=torch.long) for c in LEVELS},
           # clean-input counts on the same pixel sets, the baseline for inside/outside comparisons
           "inside_clean": {c: torch.zeros(101, 4, dtype=torch.long) for c in LEVELS},
           "outside_clean": {c: torch.zeros(101, 4, dtype=torch.long) for c in LEVELS}}
    for city in cities:
        sample = sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))
        t1, t2 = sample["t1_img"][None].to(device), sample["t2_img"][None].to(device)
        label = sample["label"].to(device)
        masks = {lv: torch.from_numpy(np.load(PHASE3 / "masks" / f"{city}_{lv}.npy")).to(device) for lv in LEVELS}
        for c in CONDITIONS:
            prob = torch.sigmoid(model(*condition_input(t1, t2, c, masks, spectrum)))[0]
            counts = threshold_counts(prob, label)
            out["pooled"][c] += counts
            out["city"][(city, c)] = counts
            if c == "clean":
                clean_prob = prob
            if regions and c in LEVELS:
                m = masks[c][None]
                out["inside"][c] += threshold_counts(prob[m], label[m])
                out["outside"][c] += threshold_counts(prob[~m], label[~m])
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
    run_dir = PHASE3 / "runs"
    run_dir.mkdir(exist_ok=True)
    sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
               for p in ("eval_phase3.py", "src/phase3/corruption.py", "src/training/train_robust.py")}
    failures = []
    for arm in ARMS:
        for seed in range(1, 11):
            name = f"{arm}_seed{seed}"
            if (run_dir / f"{name}.json").exists():
                print(name, "exists, skipped")
                continue
            ckpt = torch.load(ROOT / "checkpoints" / "robust" / f"{name}.pt", map_location="cpu", weights_only=False)
            model = build_model().to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            p2 = json.loads((ROOT / "experiments" / "robust" / name / "metrics.json").read_text(encoding="utf-8"))["test"]
            val = evaluate(model, train_ds, val_cities, spectrum, device, regions=False)
            test = evaluate(model, test_ds, test_cities, spectrum, device, regions=True)
            tau_clean, tau_missing = ckpt["thresholds"]["full"], ckpt["thresholds"]["no_o2"]
            anchors = {
                "clean_val_argmax": best_threshold(val["pooled"]["clean"])[0] == tau_clean,
                **{f"clean_test_{k}": rates_at(test["pooled"]["clean"], kk) == {m: p2["full"][k][m] for m in rates_at(test["pooled"]["clean"], kk)}
                   for k, kk in (("at_selected", tau_clean), ("at_0.5", 50))},
                **{f"missing_test_{k}": rates_at(test["pooled"]["missing"], kk) == {m: p2["no_o2"][k][m] for m in rates_at(test["pooled"]["missing"], kk)}
                   for k, kk in (("at_selected", tau_missing), ("at_0.5", 50))},
            }
            if not all(anchors.values()):
                failures.append((name, anchors))
                print(name, "ANCHOR MISMATCH", anchors, flush=True)
                continue
            np.savez_compressed(run_dir / f"{name}.npz",
                                **{f"val__{c}": v.numpy() for c, v in val["pooled"].items()},
                                **{f"test__{c}": v.numpy() for c, v in test["pooled"].items()},
                                **{f"inside__{c}": v.numpy() for c, v in test["inside"].items()},
                                **{f"outside__{c}": v.numpy() for c, v in test["outside"].items()},
                                **{f"inside_clean__{c}": v.numpy() for c, v in test["inside_clean"].items()},
                                **{f"outside_clean__{c}": v.numpy() for c, v in test["outside_clean"].items()},
                                **{f"city__{city}__{c}": v.numpy() for (city, c), v in test["city"].items()})
            doc = {"run": name, "arm": arm, "seed": seed, "checkpoint_step": ckpt["step"],
                   "tau_clean": tau_clean / 100,
                   "tau_level_val": {c: best_threshold(val["pooled"][c])[0] / 100 for c in CONDITIONS},
                   "anchors": anchors, "val_cities": val_cities, "test_cities": test_cities, "source_sha256": sources}
            (run_dir / f"{name}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
            f1 = {c: rates_at(test["pooled"][c], tau_clean)["f1"] for c in CONDITIONS}
            print(name, " | ".join(f"{c} {v:.3f}" for c, v in f1.items()), flush=True)
    if failures:
        sys.exit(1)
    print("P3 EVAL DONE")


if __name__ == "__main__":
    main()
