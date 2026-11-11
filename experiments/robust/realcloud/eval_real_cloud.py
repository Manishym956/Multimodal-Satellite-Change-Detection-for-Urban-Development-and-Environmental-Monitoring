"""Step 11 evaluation (pre-registered in STEP11_PREREGISTRATION.md): frozen A, B and SAR-only U-Net on real cloudy scenes.

Models (seeds 1-10, loaded read-only, hashed before and after): A p2_full, B p2_dropout (DS-UNet), S p6_unet_sar. Pre-change
inputs and SAR are OSCD's/DS-UNet's as in every step; only the post-change optical image changes. Conditions per test city:
    oscd       OSCD's own post-change image (anchor)
    control    the same-day Earth Engine scene (processing-chain control)
    real       the real cloudy/overcast scene, no mask
    real_<op>  the real scene with s2cloudless mask (Step 5's frozen operating point op in bal, rb95, rb99) zero-filled
    control_rb95  the control scene with its rb95 mask zero-filled (cost of the mask on a clear scene)
    missing    post-change optical image zero (no_o2)
Saved per seed and city: (101, 4) TP, FP, FN, TN per condition. S sees only SAR, so it is evaluated once per city and its
invariance to the optical input is checked.
Anchors (any failure: nothing saved, run stops): A and B `oscd` pooled over the 10 cities equal Step 5's stored
test__clean counts at all 101 thresholds; S pooled equals its stored Step 6 counts at the selected threshold and at 0.5;
S gives identical counts with the OSCD and the real post-change image. Refuses to overwrite. Prints no test metric.

Usage (repository root):  .venv/Scripts/python.exe experiments/robust/realcloud/eval_real_cloud.py [--seeds 1-10 --out DIR]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from eval_phase4a import zero_post  # noqa: E402
from export_real_cloud_inputs import manifest, scene  # noqa: E402
from src.datasets.oscd_dataset import ORBITS  # noqa: E402
from src.phase5.composite import N_S1, detector_mask  # noqa: E402
from src.training import train_robust as tr  # noqa: E402
from src.training import train_rrd as rrd  # noqa: E402
from src.training.train_sar_baseline import sample_at_orbit  # noqa: E402

R = ROOT / "experiments" / "robust"
OPS = ("bal", "rb95", "rb99")
CONDS = ("oscd", "control", "real") + tuple(f"real_{o}" for o in OPS) + ("control_rb95", "missing")
ARMS = {"p2_full": "dsunet", "p2_dropout": "dsunet", "p6_unet_sar": "unet_sar"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def with_post(t2: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
    x = t2.clone()
    x[:, N_S1:] = optical[None].to(x.device)
    return x


@torch.inference_mode()
def run(model, ds, scenes, masks, device, sar_only: bool) -> dict:
    out = {}
    for city in ds.cities:
        s = sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))
        t1, t2, label = s["t1_img"][None].to(device), s["t2_img"][None].to(device), s["label"].to(device)
        real, ctrl = with_post(t2, scenes[city, "real"]), with_post(t2, scenes[city, "control"])
        inputs = {"oscd": t2, "real": real} if sar_only else {
            "oscd": t2, "control": ctrl, "real": real, "missing": tr.apply_condition(t1, t2, "no_o2")[1],
            **{f"real_{o}": zero_post(real, masks[city, "real", o].to(device)) for o in OPS},
            "control_rb95": zero_post(ctrl, masks[city, "control", "rb95"].to(device))}
        for c, x2 in inputs.items():
            out[f"{city}__{c}"] = tr.threshold_counts(torch.sigmoid(model(t1, x2))[0], label).numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1-10")
    p.add_argument("--out", type=Path, default=HERE / "runs")
    a = p.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    ops = json.loads((R / "phase5" / "operating_points.json").read_text(encoding="utf-8"))["points"]
    ds = rrd.RRDDataset("test", "hafner_vv")
    real_tag = {r["city"]: r["tag"] for r in manifest() if r["tag"] != "control"}  # cloudy or overcast
    scenes, masks = {}, {}
    for city in ds.cities:
        shape = tuple(sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))["label"].shape[1:])
        for kind, tag in (("real", real_tag[city]), ("control", "control")):
            scenes[city, kind] = scene(city, tag, shape)
            prob = np.load(HERE / "detector_outputs" / f"{city}_{tag}.npz")["prob"]
            for o in OPS:
                masks[city, kind, o] = torch.from_numpy(detector_mask(prob, ops[o]["t"], ops[o]["r"]))
    a.out.mkdir(parents=True, exist_ok=True)
    ckpts = {(arm, s): ROOT / "checkpoints" / "robust" / f"{arm}_seed{s}.pt" for arm in ARMS for s in range(lo, hi + 1)}
    ck_sha = {k: sha(f) for k, f in ckpts.items()}
    for (arm, s), f in ckpts.items():
        dest = a.out / f"{arm}_seed{s}.npz"
        if dest.exists():
            raise SystemExit(f"{dest} exists; refusing to overwrite")
        ck = torch.load(f, map_location="cpu", weights_only=False)
        model = tr.build_model() if ARMS[arm] == "dsunet" else rrd.build_model("unet_sar", 1)
        model.load_state_dict(ck["model_state_dict"])
        model.to(device).eval()
        counts = run(model, ds, scenes, masks, device, ARMS[arm] == "unet_sar")
        pooled = sum(counts[f"{c}__oscd"] for c in ds.cities)
        anchors = {}
        if ARMS[arm] == "dsunet":
            anchors["oscd pooled = Step 5 test__clean"] = np.array_equal(pooled, np.load(R / "phase5" / "runs" / f"{arm}_seed{s}.npz")["test__clean"])
        else:
            stored = json.loads((R / f"{arm}_seed{s}" / "metrics.json").read_text(encoding="utf-8"))["test"]["full"]
            for thr, k in (("at_selected", ck["thresholds"]["full"]), ("at_0.5", 50)):
                e = stored[thr]
                anchors[f"oscd pooled = Step 6 {thr}"] = [int(x) for x in pooled[k]] == [e["tp"], e["fp"], e["fn"], e["tn"]]
            anchors["SAR-only invariant to the optical input"] = all(
                np.array_equal(counts[f"{c}__oscd"], counts[f"{c}__real"]) for c in ds.cities)
        if not all(anchors.values()):
            raise SystemExit(f"ANCHOR FAILED {arm} seed {s}: {[k for k, v in anchors.items() if not v]}")
        np.savez_compressed(dest, **counts)
        dest.with_suffix(".json").write_text(json.dumps(
            {"arm": arm, "seed": s, "tau_clean": ck["thresholds"]["full"], "test_cities": list(ds.cities), "anchors": anchors,
             "checkpoint_sha256": ck_sha[arm, s], "operating_points": {o: ops[o] for o in OPS}}, indent=2), encoding="utf-8")
        print(arm, s, "done", flush=True)
    changed = [k for k, f in ckpts.items() if sha(f) != ck_sha[k]]
    if changed:
        raise SystemExit(f"CHECKPOINT CHANGED during evaluation: {changed}")
    print("REAL-CLOUD EVAL DONE")


if __name__ == "__main__":
    main()
