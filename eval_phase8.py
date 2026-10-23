"""Step 8 (RRD RQ1, RQ2, RQ5) evaluation: frozen checkpoints with the historical optical image taken at different dates.

Setting `no_o2`: SAR t1 and t2 as always (DS-UNet VV rasters), the pre-change optical slot holds a historical Sentinel-2
image, the post-change optical image is absent (zero). Historical image tags:
    oscd     OSCD's own date-1 image (the image every model was trained with; anchor)
    gee_t1   Earth Engine re-export of the date-1 acquisition (source control: same date, different processing chain)
    gapNN    Earth Engine L1C acquisition NN months before OSCD date 2 (01, 03, 06, 12, 18), download_gee.py's rule
GEE images (13 bands, DN) are resized from the DS-UNet SAR grid to the label grid with bilinear interpolation, divided by
10000 and clipped to [0, 1], as OSCD's own bands.
Arms: B p2_dropout, async p6_async, A p2_full (all DS-UNet, at each run's frozen no_o2 threshold); SAR-only U-Net
p6_unet_sar (reference; optical-independent, evaluated once on full input at its full threshold).
Saved per run: (101, 4) counts per city and tag; per WorldCover class counts at the frozen threshold for every tag, and
for B and A also on full input (OSCD t1 and t2) at their full threshold.
Anchors (failure: run not saved): tag oscd pooled over the 10 test cities equals the stored no_o2 counts (stored full
counts for the SAR-only arm and for the full-input pass) at the stored threshold and at 0.5; land-cover counts sum to the
counts at the frozen threshold. Refuses to overwrite. Prints no test metric.

Usage:
    .venv/Scripts/python.exe eval_phase8.py
"""

import hashlib
import json

import cv2
import numpy as np
import rasterio
import torch

from eval_phase7 import WC_CLASSES, worldcover
from src.datasets.oscd_dataset import ORBITS
from src.training import train_robust as tr
from src.training import train_rrd as rrd
from src.training.train_sar_baseline import sample_at_orbit

R = tr.PROJECT_ROOT / "experiments" / "robust"
OUT = R / "phase8"
S2_ROOT = tr.PROJECT_ROOT / "datasets" / "oscd_s2_gee"
SEEDS = range(1, 11)
ARMS = {"p2_dropout": "dsunet", "p6_async": "dsunet", "p2_full": "dsunet", "p6_unet_sar": "unet_sar"}
TAGS = ("oscd", "gee_t1", "gap01", "gap03", "gap06", "gap12", "gap18")
SOURCES = ["eval_phase8.py", "eval_phase7.py", "src/training/train_rrd.py", "src/training/train_robust.py",
           "src/datasets/oscd_dataset.py", "datasets/oscd_s2_gee/manifest.json"]


def historical(city, tag, shape):
    """GEE Sentinel-2 L1C image as a (13, H, W) tensor in OSCD's scaling, or None if no scene was selected."""
    f = S2_ROOT / f"{city}_{'oscd_t1' if tag == 'gee_t1' else tag}.tif"
    if not f.exists():
        return None
    with rasterio.open(f) as src:
        bands = [cv2.resize(src.read(i).astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
                 for i in range(1, 14)]
    return torch.from_numpy(np.clip(np.stack(bands) / 10000.0, 0, 1).astype(np.float32))


def class_counts(prob, label, wc, k):
    pred = np.floor(np.clip(prob * 100, 0, 100)) >= k  # threshold_counts' rule: bin >= k
    pos = label >= 0.5
    return {cls: np.array([(pred & pos & (wc == cls)).sum(), (pred & ~pos & (wc == cls)).sum(),
                           (~pred & pos & (wc == cls)).sum(), (~pred & ~pos & (wc == cls)).sum()], dtype=np.int64) for cls in WC_CLASSES}


@torch.inference_mode()
def run(model, arch, ck, ds, device):
    out, missing = {}, []
    for city in ds.cities:
        s = sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))
        t1, t2, label = s["t1_img"][None].to(device), s["t2_img"][None].to(device), s["label"].to(device)
        lab_np, wc = label[0].cpu().numpy(), worldcover(city, label.shape[1:])
        if arch == "unet_sar":
            tags, cond = ("oscd",), "full"
        else:
            tags, cond = TAGS, "no_o2"
        k = ck["thresholds"][cond]
        for tag in tags:
            x1 = t1.clone()
            if tag != "oscd":
                img = historical(city, tag, label.shape[1:])
                if img is None:
                    missing.append(f"{city}__{tag}")
                    continue
                x1[:, 1:] = img.to(device)
            prob = torch.sigmoid(model(*rrd.apply_condition(x1, t2, cond, 1)))[0]
            out[f"{city}__{tag}"] = tr.threshold_counts(prob, label).numpy()
            for cls, v in class_counts(prob[0].cpu().numpy(), lab_np, wc, k).items():
                out[f"{city}__{tag}__wc{cls}"] = v
        if arch == "dsunet":  # full input (OSCD t1 and t2), for the land-cover comparison with full-input models
            kf = ck["thresholds"]["full"]
            prob = torch.sigmoid(model(t1, t2))[0]
            out[f"{city}__full"] = tr.threshold_counts(prob, label).numpy()
            for cls, v in class_counts(prob[0].cpu().numpy(), lab_np, wc, kf).items():
                out[f"{city}__full__wc{cls}"] = v
    return out, missing


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    source_sha256 = {f: hashlib.sha256((tr.PROJECT_ROOT / f).read_bytes()).hexdigest() for f in SOURCES}
    ds = rrd.RRDDataset("test", "hafner_vv")
    for arm, arch in ARMS.items():
        for s in SEEDS:
            f = OUT / "runs" / f"{arm}_seed{s}.npz"
            if f.exists():
                raise SystemExit(f"{f} exists; refusing to overwrite")
            ck = torch.load(tr.PROJECT_ROOT / "checkpoints" / "robust" / f"{arm}_seed{s}.pt", map_location="cpu", weights_only=False)
            model = rrd.build_model(arch, 1)
            model.load_state_dict(ck["model_state_dict"])
            model.to(device)
            counts, missing = run(model, arch, ck, ds, device)
            stored = json.loads((R / f"{arm}_seed{s}" / "metrics.json").read_text(encoding="utf-8"))["test"]
            cond, tag = ("full", "oscd") if arch == "unet_sar" else ("no_o2", "oscd")
            anchors = {}
            checks = [(cond, tag)] + ([("full", "full")] if arch == "dsunet" else [])
            for c, key in checks:
                pooled = sum(counts[f"{city}__{key}"] for city in ds.cities)
                for thr, k in (("at_selected", ck["thresholds"][c]), ("at_0.5", 50)):
                    e = stored[c][thr]
                    anchors[f"{key} {thr}"] = [int(x) for x in pooled[k]] == [e["tp"], e["fp"], e["fn"], e["tn"]]
            for key in [k for k in counts if "__wc" not in k]:
                city, t = key.split("__")
                kk = ck["thresholds"]["full" if t == "full" or arch == "unet_sar" else "no_o2"]
                anchors[f"{key} land cover sums"] = np.array_equal(sum(counts[f"{key}__wc{cls}"] for cls in WC_CLASSES), counts[key][kk])
            if not all(anchors.values()):
                raise SystemExit(f"ANCHOR FAILED {arm} seed {s}: {[k for k, v in anchors.items() if not v]}")
            np.savez_compressed(f, **counts)
            (OUT / "runs" / f"{arm}_seed{s}.json").write_text(json.dumps(
                {"arm": arm, "seed": s, "thresholds": ck["thresholds"], "selected_step": ck["step"], "test_cities": list(ds.cities),
                 "missing": missing, "anchors": anchors, "source_sha256": source_sha256}, indent=2), encoding="utf-8")
            print(arm, s, "done", flush=True)
    print("P8 EVAL DONE")


if __name__ == "__main__":
    main()
