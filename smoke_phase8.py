"""Step 8 smoke test (no Step 8 test output is produced or read: models run on validation cities only).

1. Scene manifest: a row for every city x tag; the primary test-city set (scene cloud <= 0.10 at every gap) has >= 8
   cities (the no-go threshold in docs/final/DATA_FEASIBILITY_REPORT.md).
2. Alignment control (data only, all 24 cities): the Earth Engine re-export of OSCD's date-1 acquisition, processed as
   eval_phase8.historical, against OSCD's own date-1 bands: Pearson correlation of B02, B03, B04, B08 and the
   phase-correlation shift of B04. Registered pass rule: median correlation >= 0.90 and median |shift| <= 1 px.
3. eval_phase8.run with p2_dropout_seed1 on the 4 validation cities: every tag evaluated where a scene exists,
   land-cover counts sum to the counts at the frozen threshold.
4. analyze_phase8 and audit_p8 run end to end on synthetic counts in a scratch folder; the audit's statistics agree.
Writes experiments/robust/phase8/smoke/smoke_results.json and phase8/smoke/alignment.json.

Usage:
    .venv/Scripts/python.exe smoke_phase8.py
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

import analyze_phase8
import audit_p8
import eval_phase8
from eval_phase7 import WC_CLASSES
from src.datasets.oscd_dataset import OSCD_ROOT, combine_bands, read_change_mask
from src.training import train_robust as tr
from src.training import train_rrd as rrd
from src.training.train_sar_baseline import split_train_cities

OUT = tr.PROJECT_ROOT / "experiments" / "robust" / "phase8" / "smoke"
res = []


def check(n, ok, d=""):
    res.append({"check": n, "ok": bool(ok), "detail": str(d)[:300]})


def main():
    # 1. manifest
    man = json.loads((eval_phase8.S2_ROOT / "manifest.json").read_text(encoding="utf-8"))
    tags = ("oscd_t1", "gap01", "gap03", "gap06", "gap12", "gap18")
    cities = sorted({r["city"] for r in man})
    check("manifest covers 24 cities x 6 tags", len(cities) == 24 and len(man) == 144 and
          {(r["city"], r["tag"]) for r in man} == {(c, t) for c in cities for t in tags}, len(man))
    test = list(rrd.RRDDataset("test", "hafner_vv").cities)
    primary, available, _, _ = analyze_phase8.city_sets(test)
    check("primary test cities >= 8", len(primary) >= 8, primary)
    no_scene = [f"{r['city']} {r['tag']}" for r in man if not r.get("day")]

    # 2. alignment control
    align = {}
    for city in cities:
        label = read_change_mask(city)
        ref = combine_bands(OSCD_ROOT / "images" / city / "imgs_1_rect")  # (H, W, 13), OSCD scaling
        img = eval_phase8.historical(city, "gee_t1", label.shape)
        if img is None:
            align[city] = None
            continue
        g = img.numpy()
        corr = {b: float(np.corrcoef(ref[:, :, i].ravel(), g[i].ravel())[0, 1]) for b, i in (("B02", 1), ("B03", 2), ("B04", 3), ("B08", 7))}
        (dx, dy), _ = cv2.phaseCorrelate(ref[:, :, 3].astype(np.float64), g[3].astype(np.float64))
        align[city] = {"corr": corr, "shift_px": [float(dx), float(dy)], "abs_shift": float(np.hypot(dx, dy))}
    ok = [a for a in align.values() if a]
    med_corr = float(np.median([np.mean(list(a["corr"].values())) for a in ok]))
    med_shift = float(np.median([a["abs_shift"] for a in ok]))
    check("alignment: median band correlation >= 0.90", med_corr >= 0.90, med_corr)
    check("alignment: median |shift| <= 1 px", med_shift <= 1.0, med_shift)

    # 3. evaluator on validation cities
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = rrd.RRDDataset("train", "hafner_vv")
    ds.cities = list(split_train_cities(ds.cities)[1])
    ck = torch.load(tr.PROJECT_ROOT / "checkpoints" / "robust" / "p2_dropout_seed1.pt", map_location="cpu", weights_only=False)
    model = rrd.build_model("dsunet", 1)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    counts, missing = eval_phase8.run(model, "dsunet", ck, ds, device)
    want = {f"{c}__{t}" for c in ds.cities for t in eval_phase8.TAGS} | {f"{c}__full" for c in ds.cities}
    have = {k for k in counts if "__wc" not in k}
    check("validation: every tag with a scene evaluated", have | {m for m in missing} == want, sorted(want - have - set(missing)))
    sums = []
    for key in have:
        k = ck["thresholds"]["full" if key.endswith("__full") else "no_o2"]
        sums.append(np.array_equal(sum(counts[f"{key}__wc{c}"] for c in WC_CLASSES), counts[key][k]))
    check("validation: land-cover counts sum", all(sums))

    # 4. analysis + audit on synthetic counts
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "phase8"
        (out / "runs").mkdir(parents=True)
        for arm, arch in eval_phase8.ARMS.items():
            for s in eval_phase8.SEEDS:
                z = {}
                keys = ["oscd"] if arch == "unet_sar" else list(eval_phase8.TAGS) + ["full"]
                for c in test:
                    for t in keys:
                        z[f"{c}__{t}"] = np.tile(np.array([rng.integers(1000, 5000), rng.integers(1000, 9000), 4000, 90000], dtype=np.int64), (101, 1))
                        for w in WC_CLASSES:
                            z[f"{c}__{t}__wc{w}"] = np.array(rng.integers(0, 500, 4), dtype=np.int64)
                np.savez(out / "runs" / f"{arm}_seed{s}.npz", **z)
                (out / "runs" / f"{arm}_seed{s}.json").write_text(json.dumps(
                    {"arm": arm, "seed": s, "thresholds": {c: 50 for c in tr.CONDITIONS}, "test_cities": test,
                     "missing": [], "anchors": {"x": True}, "source_sha256": {}}), encoding="utf-8")
        analyze_phase8.OUT, audit_p8.OUT = out, out
        analyze_phase8.main()
        S = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        check("summary families", all(k in S for k in ("primary_rq2", "secondary_B_vs_sar", "secondary_async_vs_sar",
                                                       "secondary_source", "rq5_landcover")))
        try:
            audit_p8.main()
        except SystemExit:
            pass
        A = json.loads((out / "p8_audit.json").read_text(encoding="utf-8"))
        stat_fail = [f["check"] for f in A["failed"] if "source" not in f["check"]]
        check("audit statistics agree with analysis on synthetic data", not stat_fail, stat_fail[:5])

    bad = [r for r in res if not r["ok"]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "alignment.json").write_text(json.dumps({"median_band_corr": med_corr, "median_abs_shift_px": med_shift,
                                                    "per_city": align}, indent=2), encoding="utf-8")
    (OUT / "smoke_results.json").write_text(json.dumps({"n_checks": len(res), "n_failed": len(bad), "checks": res,
                                                        "primary_cities": primary, "available_cities": available,
                                                        "no_scene": no_scene}, indent=2), encoding="utf-8")
    print(f"{len(res)} checks, {len(bad)} failed")
    for r in bad:
        print("FAIL", r["check"], r["detail"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
