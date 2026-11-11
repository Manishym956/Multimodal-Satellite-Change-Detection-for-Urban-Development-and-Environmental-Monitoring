"""Step 10 per-city evaluation (pre-registered in STEP10_PREREGISTRATION.md): per-city test counts of trained checkpoints.

For each seed, loads checkpoints/robust/{arm}_seed{s}.pt read-only and saves (101, 4) TP, FP, FN, TN per test city and
condition (first listed orbit, cuDNN deterministic, as training's own evaluation). Anchor (failure: nothing saved, run
stops): the city counts summed over the 10 test cities equal the counts train_rrd.py stored in metrics.json, at the
selected threshold and at 0.5, for every condition. Checkpoint hashes are checked again at the end. Refuses to overwrite.
Prints no test metric.

Usage (repository root):
    .venv/Scripts/python.exe experiments/robust/step10/eval_step10_cities.py --arm p10_fusion_dropout --seeds 11-20
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.datasets.oscd_dataset import ORBITS  # noqa: E402
from src.training import train_robust as tr  # noqa: E402
from src.training import train_rrd as rrd  # noqa: E402
from src.training.train_sar_baseline import sample_at_orbit  # noqa: E402

CONDS = {"unet_fusion": ("full", "no_o2", "no_opt"), "unet_sar": ("full",)}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@torch.inference_mode()
def city_counts(model, ds, conds, device) -> dict[str, np.ndarray]:
    out = {}
    for city in ds.cities:
        s = sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))
        t1, t2, label = s["t1_img"][None].to(device), s["t2_img"][None].to(device), s["label"].to(device)
        for c in conds:
            prob = torch.sigmoid(model(*rrd.apply_condition(t1, t2, c, 1)))[0]
            out[f"{city}__{c}"] = tr.threshold_counts(prob, label).numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--seeds", required=True, help="e.g. 11-20")
    p.add_argument("--runs", type=Path, default=ROOT / "experiments" / "robust")
    p.add_argument("--ckpts", type=Path, default=ROOT / "checkpoints" / "robust")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "cities")
    a = p.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    ds = rrd.RRDDataset("test", "hafner_vv")
    a.out.mkdir(parents=True, exist_ok=True)
    ckpts = {s: a.ckpts / f"{a.arm}_seed{s}.pt" for s in range(lo, hi + 1)}
    ck_sha = {s: sha(f) for s, f in ckpts.items()}
    for s, f in ckpts.items():
        dest = a.out / f"{a.arm}_seed{s}.npz"
        if dest.exists():
            raise SystemExit(f"{dest} exists; refusing to overwrite")
        ck = torch.load(f, map_location="cpu", weights_only=False)
        arch = ck["args"]["arch"]
        model = rrd.build_model(arch, 1)
        model.load_state_dict(ck["model_state_dict"])
        model.to(device).eval()
        counts = city_counts(model, ds, CONDS[arch], device)
        stored = json.loads((a.runs / f"{a.arm}_seed{s}" / "metrics.json").read_text(encoding="utf-8"))["test"]
        anchors = {}
        for c in CONDS[arch]:
            pooled = sum(counts[f"{city}__{c}"] for city in ds.cities)
            for thr, k in (("at_selected", ck["thresholds"][c]), ("at_0.5", 50)):
                e = stored[c][thr]
                anchors[f"{c} {thr}"] = [int(x) for x in pooled[k]] == [e["tp"], e["fp"], e["fn"], e["tn"]]
        if not all(anchors.values()):
            raise SystemExit(f"ANCHOR FAILED {a.arm} seed {s}: {[k for k, v in anchors.items() if not v]}")
        np.savez_compressed(dest, **counts)
        dest.with_suffix(".json").write_text(json.dumps(
            {"arm": a.arm, "seed": s, "arch": arch, "thresholds": ck["thresholds"], "test_cities": list(ds.cities),
             "anchors": anchors, "checkpoint_sha256": ck_sha[s]}, indent=2), encoding="utf-8")
        print(a.arm, s, "done", flush=True)
    changed = [s for s, f in ckpts.items() if sha(f) != ck_sha[s]]
    if changed:
        raise SystemExit(f"CHECKPOINT CHANGED during evaluation: seeds {changed}")
    print("CITY EVAL DONE")


if __name__ == "__main__":
    main()
