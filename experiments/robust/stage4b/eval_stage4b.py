"""Stage 4B evaluation (pre-registered in STAGE4B_PREREGISTRATION.md): model B with and without the historical image.

Frozen B checkpoints (p2_dropout, seeds 1-10) on the 10 OSCD test cities, Step 8's pipeline (eval_phase8.py): DS-UNet VV
rasters, first listed orbit per city, cuDNN deterministic. Two conditions, post-change optical image absent in both:
    present  `no_o2`: OSCD's own date-1 image in the pre-change optical slot (Step 8 tag `oscd`)
    absent   `no_opt`: both optical slots zero (SAR t1 and t2 only)
Saved per seed and city: (101, 4) TP, FP, FN, TN for each condition, and the same (101, 4) counts per WorldCover class.
No training; checkpoints are hashed before loading and again at the end.

Anchors (failure: nothing saved): `present` equals Step 8's stored `<city>__oscd` counts at all 101 thresholds and its
class counts at the frozen no_o2 threshold; `absent` pooled over the 10 cities equals the stored Phase 2 `no_opt` test
counts at the stored threshold and at 0.5; class counts sum to the city counts at every threshold. Refuses to
overwrite. Prints no test metric.

Usage (from the repository root):
    .venv/Scripts/python.exe experiments/robust/stage4b/eval_stage4b.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eval_phase7 import WC_CLASSES, worldcover  # noqa: E402
from src.datasets.oscd_dataset import ORBITS  # noqa: E402
from src.training import train_robust as tr  # noqa: E402
from src.training import train_rrd as rrd  # noqa: E402
from src.training.train_sar_baseline import sample_at_orbit  # noqa: E402

R = ROOT / "experiments" / "robust"
OUT = Path(__file__).resolve().parent / "runs"
ARM, SEEDS = "p2_dropout", range(1, 11)
CONDS = {"present": "no_o2", "absent": "no_opt"}
SOURCES = ["experiments/robust/stage4b/eval_stage4b.py", "eval_phase8.py", "eval_phase7.py", "src/training/train_rrd.py",
           "src/training/train_robust.py", "src/datasets/oscd_dataset.py"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@torch.inference_mode()
def run(model, ds, device):
    out = {}
    for city in ds.cities:
        s = sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))
        t1, t2, label = s["t1_img"][None].to(device), s["t2_img"][None].to(device), s["label"].to(device)
        wc = torch.from_numpy(worldcover(city, label.shape[1:]).astype(np.int64)).to(device)
        for name, cond in CONDS.items():
            prob = torch.sigmoid(model(*rrd.apply_condition(t1, t2, cond, 1)))[0]
            out[f"{city}__{name}"] = tr.threshold_counts(prob, label).numpy()
            for cls in WC_CLASSES:
                m = wc == cls
                out[f"{city}__{name}__wc{cls}"] = tr.threshold_counts(prob[0][m], label[0][m]).numpy()
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False  # as training and every earlier eval
    OUT.mkdir(parents=True, exist_ok=True)
    source_sha256 = {f: sha(ROOT / f) for f in SOURCES}
    ckpts = {s: ROOT / "checkpoints" / "robust" / f"{ARM}_seed{s}.pt" for s in SEEDS}
    ck_sha = {s: sha(p) for s, p in ckpts.items()}
    ds = rrd.RRDDataset("test", "hafner_vv")
    for s in SEEDS:
        f = OUT / f"{ARM}_seed{s}.npz"
        if f.exists():
            raise SystemExit(f"{f} exists; refusing to overwrite")
        ck = torch.load(ckpts[s], map_location="cpu", weights_only=False)
        model = rrd.build_model("dsunet", 1)
        model.load_state_dict(ck["model_state_dict"])
        model.to(device)
        model.eval()
        counts = run(model, ds, device)

        p8 = np.load(R / "phase8" / "runs" / f"{ARM}_seed{s}.npz")
        stored = json.loads((R / f"{ARM}_seed{s}" / "metrics.json").read_text(encoding="utf-8"))["test"]["no_opt"]
        k8, ka = ck["thresholds"]["no_o2"], ck["thresholds"]["no_opt"]
        anchors = {}
        for c in ds.cities:
            anchors[f"{c} present = Step 8 oscd"] = np.array_equal(counts[f"{c}__present"], p8[f"{c}__oscd"])
            anchors[f"{c} present classes = Step 8"] = all(
                np.array_equal(counts[f"{c}__present__wc{cls}"][k8], p8[f"{c}__oscd__wc{cls}"]) for cls in WC_CLASSES)
            for name in CONDS:
                anchors[f"{c} {name} classes sum"] = np.array_equal(
                    sum(counts[f"{c}__{name}__wc{cls}"] for cls in WC_CLASSES), counts[f"{c}__{name}"])
        pooled = sum(counts[f"{c}__absent"] for c in ds.cities)
        for thr, k in (("at_selected", ka), ("at_0.5", 50)):
            e = stored[thr]
            anchors[f"absent pooled {thr} = Phase 2 no_opt"] = [int(x) for x in pooled[k]] == [e["tp"], e["fp"], e["fn"], e["tn"]]
        if not all(anchors.values()):
            raise SystemExit(f"ANCHOR FAILED seed {s}: {[k for k, v in anchors.items() if not v]}")
        np.savez_compressed(f, **counts)
        (OUT / f"{ARM}_seed{s}.json").write_text(json.dumps(
            {"arm": ARM, "seed": s, "conditions": CONDS, "thresholds": ck["thresholds"], "selected_step": ck["step"],
             "test_cities": list(ds.cities), "anchors": anchors, "checkpoint_sha256": ck_sha[s],
             "source_sha256": source_sha256}, indent=2), encoding="utf-8")
        print(ARM, s, "done", flush=True)
    changed = [s for s, p in ckpts.items() if sha(p) != ck_sha[s]]
    if changed:
        raise SystemExit(f"CHECKPOINT CHANGED during evaluation: seeds {changed}")
    print("STAGE 4B EVAL DONE")


if __name__ == "__main__":
    main()
