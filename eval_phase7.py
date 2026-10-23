"""Step 7 (RRD RQ3) evaluation: per-city threshold counts of every polarization arm on the test cities, once per checkpoint.

For each arm and seed: load the frozen checkpoint, run it on each test city (first listed orbit) under every
optical-availability condition, and save the (101, 4) TP/FP/FN/TN counts per city and condition. The analysis pools
the cities it registers (9 test cities with VH at both dates) at the frozen validation threshold.
Anchors (failure: run not saved): per-city counts summed over the 10 test cities equal, at the stored threshold and at
0.5, the counts train_robust.py / train_rrd.py stored at training time, for every condition.
Writes experiments/robust/phase7/runs/{arm}_seed{s}.npz and .json; refuses to overwrite. Prints no test metric.

Usage:
    .venv/Scripts/python.exe eval_phase7.py
"""

import hashlib
import json

import cv2
import numpy as np
import rasterio
import torch

from src.datasets.oscd_dataset import ORBITS
from src.training import train_robust as tr
from src.training import train_rrd as rrd
from src.training.train_sar_baseline import sample_at_orbit

R = tr.PROJECT_ROOT / "experiments" / "robust"
OUT = R / "phase7"
SEEDS = range(1, 11)
ARMS = {  # arm: (arch, sar)
    "p6_unet_sar": ("unet_sar", "hafner_vv"), "p7_unet_vh": ("unet_sar", "vh"), "p7_unet_vvvh": ("unet_sar", "vvvh"),
    "p2_full": ("dsunet", "hafner_vv"), "p7_dsunet_vh": ("dsunet", "vh"), "p7_dsunet_vvvh": ("dsunet", "vvvh")}
SOURCES = ["eval_phase7.py", "src/training/train_rrd.py", "src/training/train_robust.py", "src/datasets/oscd_dataset.py"]
WC_ROOT = tr.PROJECT_ROOT / "datasets" / "oscd_wc_gee"
WC_CLASSES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)  # ESA WorldCover v100 codes; 0 = no data


def worldcover(city, shape):
    """ESA WorldCover 2020 class per label pixel (nearest-neighbour resize from the DS-UNet SAR grid)."""
    with rasterio.open(WC_ROOT / f"{city}_worldcover.tif") as src:
        return cv2.resize(src.read(1), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


@torch.inference_mode()
def city_counts(model, ds, n_s1, device, k_full):
    """Per city: (101, 4) counts per condition, and full-input counts per WorldCover class at threshold k_full."""
    out = {}
    for city in ds.cities:
        s = sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))
        t1, t2, label = s["t1_img"][None].to(device), s["t2_img"][None].to(device), s["label"].to(device)
        for c in tr.CONDITIONS:
            prob = torch.sigmoid(model(*rrd.apply_condition(t1, t2, c, n_s1)))[0]
            out[f"{city}__{c}"] = tr.threshold_counts(prob, label).numpy()
            if c == "full":
                pred = np.floor(np.clip(prob[0].cpu().numpy() * 100, 0, 100)) >= k_full  # threshold_counts' rule: bin >= k
                pos, wc = label[0].cpu().numpy() >= 0.5, worldcover(city, pred.shape)
                for cls in WC_CLASSES:
                    m = wc == cls
                    out[f"{city}__wc{cls}"] = np.array([(pred & pos & m).sum(), (pred & ~pos & m).sum(),
                                                        (~pred & pos & m).sum(), (~pred & ~pos & m).sum()], dtype=np.int64)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    source_sha256 = {f: hashlib.sha256((tr.PROJECT_ROOT / f).read_bytes()).hexdigest() for f in SOURCES}
    for arm, (arch, sar) in ARMS.items():
        n_s1 = len(rrd.SAR_BANDS[sar])
        ds = rrd.RRDDataset("test", sar)
        for s in SEEDS:
            f = OUT / "runs" / f"{arm}_seed{s}.npz"
            if f.exists():
                raise SystemExit(f"{f} exists; refusing to overwrite")
            ck = torch.load(tr.PROJECT_ROOT / "checkpoints" / "robust" / f"{arm}_seed{s}.pt", map_location="cpu", weights_only=False)
            model = rrd.build_model(arch, n_s1)
            model.load_state_dict(ck["model_state_dict"])
            model.to(device)
            k_full = ck["thresholds"]["full"]
            counts = city_counts(model, ds, n_s1, device, k_full)
            stored = json.loads((R / f"{arm}_seed{s}" / "metrics.json").read_text(encoding="utf-8"))["test"]
            anchors = {f"{city} land-cover counts sum to full counts": np.array_equal(
                sum(counts[f"{city}__wc{cls}"] for cls in WC_CLASSES), counts[f"{city}__full"][k_full]) for city in ds.cities}
            for c in tr.CONDITIONS:
                pooled = sum(counts[f"{city}__{c}"] for city in ds.cities)
                for thr, k in (("at_selected", ck["thresholds"][c]), ("at_0.5", 50)):
                    e = stored[c][thr]
                    anchors[f"{c} {thr}"] = [int(x) for x in pooled[k]] == [e["tp"], e["fp"], e["fn"], e["tn"]]
            if not all(anchors.values()):
                raise SystemExit(f"ANCHOR FAILED {arm} seed {s}: {[k for k, v in anchors.items() if not v]}")
            np.savez_compressed(f, **counts)
            (OUT / "runs" / f"{arm}_seed{s}.json").write_text(json.dumps(
                {"arm": arm, "seed": s, "arch": arch, "sar": sar, "thresholds": ck["thresholds"], "selected_step": ck["step"],
                 "test_cities": list(ds.cities), "anchors": anchors, "source_sha256": source_sha256}, indent=2), encoding="utf-8")
            print(arm, s, "done", flush=True)
    print("P7 EVAL DONE")


if __name__ == "__main__":
    main()
