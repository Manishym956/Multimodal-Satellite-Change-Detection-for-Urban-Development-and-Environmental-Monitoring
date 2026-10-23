"""Step 7 smoke test (validation only; no Step 7 test output is produced or read).

1. Data: every Earth Engine VV band (datasets/oscd_s1_gee) equals the DS-UNet VV raster for the same city, orbit and
   date, pixel for pixel; list every file whose VH band is empty (all zero).
2. Each Step 7 arm (SAR-only U-Net and DS-UNet with VH and VV+VH) trains 20 steps on the training cities and is
   evaluated on the validation cities with train_rrd.evaluate; losses finite, counts complete.
3. eval_phase7.city_counts on the validation cities: land-cover counts sum to the full-input counts.
4. analyze_phase7 and audit_p7 run end to end on synthetic counts in a scratch folder; the audit's statistical checks
   agree with the analysis (only anchor, settings and source checks, which need real runs, may fail there).
Writes experiments/robust/phase7/smoke/smoke_results.json.

Usage:
    .venv/Scripts/python.exe smoke_phase7.py
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import torch

import analyze_phase7
import audit_p7
import eval_phase7
from src.training import train_robust as tr
from src.training import train_rrd as rrd
from src.training.train_sar_baseline import set_seed, split_train_cities

OUT = tr.PROJECT_ROOT / "experiments" / "robust" / "phase7" / "smoke"
res = []


def check(n, ok, d=""):
    res.append({"check": n, "ok": bool(ok), "detail": str(d)[:300]})


def train_steps(arch, conditions, device, ds, train_cities, n_s1):
    """smoke_phase6.train_steps with n_s1 SAR channels."""
    set_seed(1)
    rng = np.random.default_rng(1)
    probs = tr.parse_conditions(conditions)
    model = rrd.build_model(arch, n_s1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-4)
    loss_fn = tr.load_dsunet_module("loss_functions").jaccard_like_loss
    losses = []
    for _ in range(20):
        model.train()
        b = []
        for city in rng.choice(train_cities, 8):
            s = ds[ds.cities.index(city)]
            t1, t2, lab = tr.random_crop(s["t1_img"], s["t2_img"], s["label"], 32, True, rng)
            t1, t2, lab = tr.augment(t1, t2, lab, rng)
            t1, t2 = rrd.apply_condition(t1[None], t2[None], rng.choice(tr.CONDITIONS, p=[probs[c] for c in tr.CONDITIONS]), n_s1)
            b.append((t1[0], t2[0], lab))
        t1, t2, lab = (torch.stack(x).to(device) for x in zip(*b))
        opt.zero_grad()
        loss = loss_fn(model(t1, t2), lab)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses


def main():
    # 1. data
    hafner, gee = tr.PROJECT_ROOT / "datasets" / "oscd_sentinel1", rrd.GEE_S1_ROOT
    files = sorted(gee.glob("sentinel1_*.tif"))
    check("102 GEE files (one per city, orbit, date)", len(files) == 102, len(files))
    same, empty_vh = [], []
    for f in files:
        with rasterio.open(f) as g, rasterio.open(hafner / f.name) as h:
            same.append(np.array_equal(g.read(list(g.descriptions).index("VV") + 1), h.read(1)))
            if not g.read(list(g.descriptions).index("VH") + 1).any():
                empty_vh.append(f.stem)
    check("GEE VV equals DS-UNet VV in every file", all(same), f"{sum(same)}/{len(same)} equal")
    check("empty VH files are the inventory's VH-less windows", set(empty_vh) <= {"sentinel1_lasvegas_166_t1", "sentinel1_lasvegas_173_t1",
                                                                                 "sentinel1_dubai_166_t1"}, empty_vh)
    # 2. training on VH inputs, validation only
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for sar in ("vh", "vvvh"):
        ds = rrd.RRDDataset("train", sar)
        train_cities, val_cities = split_train_cities(ds.cities)
        n_val = sum(int(ds[ds.cities.index(c)]["label"].numel()) for c in val_cities)
        n_s1 = len(rrd.SAR_BANDS[sar])
        for arch in ("unet_sar", "dsunet"):
            model, losses = train_steps(arch, "full=1", device, ds, train_cities, n_s1)
            check(f"{arch} {sar} losses finite", all(math.isfinite(x) for x in losses), losses[-1])
            counts = rrd.evaluate(model, ds, val_cities, device, n_s1)
            check(f"{arch} {sar} val counts complete", all(int(counts[c][0].sum()) == n_val for c in tr.CONDITIONS))
            # 3. land-cover counts on the validation cities
            val_ds = rrd.RRDDataset("train", sar)
            val_ds.cities = list(val_cities)
            cc = eval_phase7.city_counts(model, val_ds, n_s1, device, 50)
            check(f"{arch} {sar} land cover sums (validation)", all(np.array_equal(
                sum(cc[f"{c}__wc{k}"] for k in eval_phase7.WC_CLASSES), cc[f"{c}__full"][50]) for c in val_cities))

    # 4. analysis + audit on synthetic counts
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "phase7"
        (out / "runs").mkdir(parents=True)
        cities = list(analyze_phase7.ALL10)
        for arm in eval_phase7.ARMS:
            for s in eval_phase7.SEEDS:
                z = {}
                for c in cities:
                    for cond in tr.CONDITIONS:
                        tp, fp = rng.integers(1000, 5000), rng.integers(1000, 9000)
                        z[f"{c}__{cond}"] = np.tile(np.array([tp, fp, 4000, 90000], dtype=np.int64), (101, 1))
                    for k in eval_phase7.WC_CLASSES:
                        z[f"{c}__wc{k}"] = np.array(rng.integers(0, 500, 4), dtype=np.int64)
                np.savez(out / "runs" / f"{arm}_seed{s}.npz", **z)
                (out / "runs" / f"{arm}_seed{s}.json").write_text(json.dumps(
                    {"arm": arm, "seed": s, "arch": eval_phase7.ARMS[arm][0], "sar": eval_phase7.ARMS[arm][1],
                     "thresholds": {c: 50 for c in tr.CONDITIONS}, "test_cities": cities, "anchors": {"x": True},
                     "source_sha256": {}}), encoding="utf-8")
        analyze_phase7.OUT, audit_p7.OUT = out, out
        analyze_phase7.main()
        S = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        check("summary families", all(k in S for k in ("primary_rq3", "secondary_dsunet", "secondary_far")))
        try:
            audit_p7.main()
        except SystemExit:
            pass
        A = json.loads((out / "p7_audit.json").read_text(encoding="utf-8"))
        allowed = ("registered training", "pooled = stored", "land cover sums", "source", "metrics.json exists")
        stat_fail = [f["check"] for f in A["failed"] if not any(x in f["check"] for x in allowed)]
        check("audit statistics agree with analysis on synthetic data", not stat_fail, stat_fail[:5])

    bad = [r for r in res if not r["ok"]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke_results.json").write_text(json.dumps({"n_checks": len(res), "n_failed": len(bad), "checks": res}, indent=2), encoding="utf-8")
    print(f"{len(res)} checks, {len(bad)} failed")
    for r in bad:
        print("FAIL", r["check"], r["detail"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
