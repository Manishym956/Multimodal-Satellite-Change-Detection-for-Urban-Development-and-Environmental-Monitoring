"""Step 4A-bis EXPLORATORY ADDENDUM (not pre-registered): test counts inside / outside the TRUE cloud mask M.

Written 2026-10-22, after the registered Step 4A-bis analysis had been run and read. It selects nothing: same 20 frozen
Phase 2 checkpoints, same detector masks, same corruption, same (101, 4) threshold counts; only the pixel partition
differs. The registered regions (missed M\\P, over-masked P\\M, rest) cannot be regrouped into M / not-M because "rest"
mixes M∩P with clear pixels outside P.

Per test condition c (clean, cL, zL, L__variant) and its level's true mask M, stores test counts on M and on not-M.
Clean input is partitioned by each level's M (keys clean@c10 ...). Anchor: inside + outside == the stored pooled
phase4abis test counts, for every condition, so the predictions are identical to the registered evaluation.

Writes experiments/robust/phase4abis/addendum_inside_outside/runs/{run}.npz/.json; never touches phase4abis/runs.

Usage:
    .venv/Scripts/python.exe eval_phase4abis_addendum.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from eval_phase4a import zero_post
from eval_phase4abis import CONDITIONS, OUT as P4ABIS
from src.datasets.oscd_dataset import ORBITS, OSCDDataset
from src.phase3.corruption import LEVELS, PHASE3, corrupt_post, load_spectrum
from src.training.train_robust import build_model, threshold_counts
from src.training.train_sar_baseline import sample_at_orbit

ROOT = Path(__file__).resolve().parent
OUT = P4ABIS / "addendum_inside_outside"
KEYS = tuple(f"clean@{lv}" for lv in LEVELS) + CONDITIONS[1:]


def level_of(key: str) -> str:
    if key.startswith("clean@"):
        return key[6:]
    if key.startswith("z"):
        return f"c{key[1:]}"
    return key.split("__")[0]


@torch.inference_mode()
def evaluate(model, ds, spectrum, device):
    model.eval()
    out = {f"{part}__{k}": torch.zeros(101, 4, dtype=torch.long) for part in ("inside", "outside") for k in KEYS}
    pooled = {c: torch.zeros(101, 4, dtype=torch.long) for c in CONDITIONS}
    for city in ds.cities:
        sample = sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))
        t1, t2 = sample["t1_img"][None].to(device), sample["t2_img"][None].to(device)
        label = sample["label"].to(device)
        oracle = {lv: torch.from_numpy(np.load(PHASE3 / "masks" / f"{city}_{lv}.npy")).to(device) for lv in LEVELS}
        clouded = {lv: corrupt_post(t2, oracle[lv], spectrum) for lv in LEVELS}
        for c in CONDITIONS:
            if c == "clean":
                x2 = t2
            elif c in LEVELS:
                x2 = clouded[c]
            elif c.startswith("z"):
                x2 = zero_post(clouded[f"c{c[1:]}"], oracle[f"c{c[1:]}"])
            else:
                lv, v = c.split("__")
                x2 = zero_post(clouded[lv], torch.from_numpy(np.load(P4ABIS / "masks" / f"{city}_{lv}_{v}.npy")).to(device))
            prob = torch.sigmoid(model(t1, x2))[0]
            pooled[c] += threshold_counts(prob, label)
            for k in ([f"clean@{lv}" for lv in LEVELS] if c == "clean" else [c]):
                m = oracle[level_of(k)][None]
                out[f"inside__{k}"] += threshold_counts(prob[m], label[m])
                out[f"outside__{k}"] += threshold_counts(prob[~m], label[~m])
    return out, pooled


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    spectrum = torch.from_numpy(load_spectrum()).to(device)
    test_ds = OSCDDataset(split="test", mode="fusion")
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
               for p in ("eval_phase4abis_addendum.py", "eval_phase4abis.py", "eval_phase4a.py", "src/phase3/corruption.py",
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
            stored = np.load(P4ABIS / "runs" / f"{name}.npz")
            out, pooled = evaluate(model, test_ds, spectrum, device)
            anchors = {c: np.array_equal(pooled[c].numpy(), stored[f"test__{c}"]) for c in CONDITIONS}
            anchors |= {f"partition_{k}": np.array_equal((out[f"inside__{k}"] + out[f"outside__{k}"]).numpy(),
                                                         stored[f"test__{'clean' if k.startswith('clean@') else k}"]) for k in KEYS}
            if not all(anchors.values()):
                failures.append(name)
                print(name, "ANCHOR MISMATCH", [k for k, v in anchors.items() if not v][:5], flush=True)
                continue
            np.savez_compressed(OUT / "runs" / f"{name}.npz", **{k: v.numpy() for k, v in out.items()})
            (OUT / "runs" / f"{name}.json").write_text(json.dumps(
                {"run": name, "arm": arm, "seed": seed, "tau_clean": ckpt["thresholds"]["full"] / 100,
                 "anchors": anchors, "source_sha256": sources}, indent=2), encoding="utf-8")
            print(name, "ok", flush=True)
    if failures:
        sys.exit(1)
    print("P4ABIS ADDENDUM EVAL DONE")


if __name__ == "__main__":
    main()
