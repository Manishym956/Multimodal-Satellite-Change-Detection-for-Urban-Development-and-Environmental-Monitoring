"""Step 9 smoke test (no new test output). Prints one line per check and 'n_failed N'; exits 1 on any failure.

1-2  analyze_step9.py end to end on stand-in data: Step 6's already-reported input-fusion runs relabelled as E (in a
     temporary folder); B's within-model contrast must reproduce Step 6's audited B@no_o2 - B@no_opt.
3-6  E's code path on the 4 validation cities only (untrained network, nothing saved): channel layout, the no_o2 and
     no_opt conditions, that the historical image reaches the early-fusion output, and validation evaluation.

Usage (repository root):  .venv/Scripts/python.exe experiments/robust/step9/smoke_step9.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.datasets.oscd_dataset import ORBITS  # noqa: E402
from src.training import train_rrd as rrd  # noqa: E402
from src.training.train_sar_baseline import sample_at_orbit, split_train_cities  # noqa: E402

R = ROOT / "experiments" / "robust"
checks = []


def check(name: str, ok: bool) -> None:
    checks.append(ok)
    print(("ok   " if ok else "FAIL ") + name)


with tempfile.TemporaryDirectory() as tmp:
    S = Path(tmp)
    for arm in ("p2_full", "p2_dropout", "p6_unet_sar", "p6_unet_fusion"):
        for s in range(1, 11):
            (S / "runs" / f"{arm}_seed{s}").mkdir(parents=True)
            shutil.copy(R / f"{arm}_seed{s}" / "metrics.json", S / "runs" / f"{arm}_seed{s}")
    for s in range(1, 11):
        m = json.loads((R / f"p6_unet_fusion_seed{s}" / "metrics.json").read_text(encoding="utf-8"))
        m["args"]["train_conditions"] = "full=0.5,no_o2=0.25,no_opt=0.25"
        d = S / "runs" / f"p9_fusion_dropout_seed{s}"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("analyze_step9.py")), "--runs", str(S / "runs"),
                          "--out", str(S / "out")], capture_output=True, text=True)
    check("analysis runs on stand-in data", run.returncode == 0)
    sm = json.loads((S / "out" / "summary.json").read_text(encoding="utf-8"))
    b6 = json.loads((R / "phase6" / "summary.json").read_text(encoding="utf-8"))["secondary_async"]["B@no_o2 - B@no_opt"]["diffs"]
    e = sm["primary"]["E: image present - absent"]["diffs"]
    inter = sm["primary"]["interaction: (E present - absent) - (B present - absent)"]["diffs"]
    check("B within-model contrast reproduces Step 6", bool(np.allclose(np.array(e) - np.array(inter), b6)))

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = rrd.RRDDataset("train", "hafner_vv")
_, val = split_train_cities(ds.cities)
model = rrd.build_model("unet_fusion", 1).to(dev).eval()
x = sample_at_orbit(ds, ds.cities.index(val[0]), int(ORBITS[val[0]][0]))
t1, t2 = x["t1_img"][None].to(dev), x["t2_img"][None].to(dev)
a1, a2 = rrd.apply_condition(t1, t2, "no_opt", 1)
check("14 channels per date; no_opt zeroes both optical dates and keeps SAR",
      t1.shape[1] == 14 and bool((a1[:, 1:] == 0).all() and (a2[:, 1:] == 0).all() and torch.equal(a1[:, :1], t1[:, :1])))
b1, b2 = rrd.apply_condition(t1, t2, "no_o2", 1)
check("no_o2 keeps the historical image and zeroes the post-change optical", bool(torch.equal(b1, t1) and (b2[:, 1:] == 0).all()))
with torch.inference_mode():
    p = {c: torch.sigmoid(model(*rrd.apply_condition(t1, t2, c, 1))) for c in ("no_o2", "no_opt")}
check("historical image changes the early-fusion output", not torch.allclose(p["no_o2"], p["no_opt"]))
counts = rrd.evaluate(model, ds, val, dev, 1)
check("validation evaluation returns all four conditions", sorted(counts) == sorted(("full", "no_o2", "no_opt", "o2_copy")))

print(f"n_failed {checks.count(False)}")
sys.exit(1 if False in checks else 0)
