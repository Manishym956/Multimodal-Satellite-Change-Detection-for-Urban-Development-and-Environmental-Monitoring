"""Step 6 smoke test (validation only; no Step 6 test output is produced or read).

1. Each Step 6 architecture (train_rrd.py) trains for 20 steps on the 10 training cities and is evaluated on the 4
   validation cities with train_rrd.evaluate; losses are finite and the four optical-availability conditions give counts
   that sum to the validation pixel total.
2. The registered analysis and audit run end to end on stand-in data: copies of the already-reported Phase 2 metrics
   (p2_full / p2_dropout, seeds 1-10) under the Step 6 arm names, in a scratch folder. The audit's independent
   statistics must agree with the analysis (only the checks that need real Step 6 runs, i.e. settings, checkpoints and
   source hashes, may fail there).
Writes experiments/robust/phase6/smoke/smoke_results.json.

Usage:
    .venv/Scripts/python.exe smoke_phase6.py
"""

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import analyze_phase6
import audit_p6
from src.training import train_robust as tr
from src.training import train_rrd as rrd
from src.training.train_sar_baseline import set_seed, split_train_cities

OUT = tr.PROJECT_ROOT / "experiments" / "robust" / "phase6" / "smoke"
res = []


def check(n, ok, d=""):
    res.append({"check": n, "ok": bool(ok), "detail": str(d)[:200]})


def train_steps(arch, conditions, device, ds, train_cities, val_cities):
    set_seed(1)
    rng = np.random.default_rng(1)
    probs = tr.parse_conditions(conditions)
    model = rrd.build_model(arch, 1).to(device)
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
            t1, t2 = rrd.apply_condition(t1[None], t2[None], rng.choice(tr.CONDITIONS, p=[probs[c] for c in tr.CONDITIONS]), 1)
            b.append((t1[0], t2[0], lab))
        t1, t2, lab = (torch.stack(x).to(device) for x in zip(*b))
        opt.zero_grad()
        loss = loss_fn(model(t1, t2), lab)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = rrd.RRDDataset("train", "hafner_vv")
    train_cities, val_cities = split_train_cities(ds.cities)
    n_val = sum(int(ds[ds.cities.index(c)]["label"].numel()) for c in val_cities)
    for arm, (arch, cond) in {"p6_unet_sar": ("unet_sar", "full=1"), "p6_unet_opt": ("unet_opt", "full=1"),
                              "p6_unet_fusion": ("unet_fusion", "full=1"), "p6_async": ("dsunet", "no_o2=1")}.items():
        model, losses = train_steps(arch, cond, device, ds, train_cities, val_cities)
        check(f"{arm} losses finite", all(math.isfinite(x) for x in losses), losses[-1])
        counts = rrd.evaluate(model, ds, val_cities, device, 1)
        check(f"{arm} val counts complete", all(int(counts[c][0].sum()) == n_val for c in tr.CONDITIONS))
        if arch == "unet_sar":  # SAR-only ignores optical channels: all conditions identical
            check(f"{arm} conditions identical", all(torch.equal(counts["full"], counts[c]) for c in tr.CONDITIONS))
        check(f"{arm} model eval mode after evaluate", not model.training)

    # analysis + audit end to end on stand-in data in a scratch folder
    with tempfile.TemporaryDirectory() as tmp:
        R = Path(tmp)
        src = tr.PROJECT_ROOT / "experiments" / "robust"
        stand_in = {"p2_full": "p2_full", "p2_dropout": "p2_dropout", "p6_unet_sar": "p2_full", "p6_unet_opt": "p2_full",
                    "p6_unet_fusion": "p2_dropout", "p6_async": "p2_dropout"}
        for arm, real in stand_in.items():
            for s in range(1, 11):
                (R / f"{arm}_seed{s}").mkdir()
                shutil.copy(src / f"{real}_seed{s}" / "metrics.json", R / f"{arm}_seed{s}" / "metrics.json")
        analyze_phase6.R, analyze_phase6.OUT = R, R / "phase6"
        audit_p6.R, audit_p6.OUT = R, R / "phase6"
        analyze_phase6.main()
        S = json.loads((R / "phase6" / "summary.json").read_text(encoding="utf-8"))
        check("summary families", all(k in S for k in ("primary_rq4", "primary_rq1", "secondary_fusion", "secondary_async")))
        check("primary_rq4 has 3, primary_rq1 has 2", len(S["primary_rq4"]) == 3 and len(S["primary_rq1"]) == 2)
        try:
            audit_p6.main()
        except SystemExit:
            pass
        A = json.loads((R / "phase6" / "p6_audit.json").read_text(encoding="utf-8"))
        allowed = ("registered settings", "checkpoint exists", "one source set", "source unchanged")
        stats_fail = [f["check"] for f in A["failed"] if not any(x in f["check"] for x in allowed)]
        check("audit statistics agree with analysis on stand-in data", not stats_fail, stats_fail[:5])
        check("audit ran its statistical checks", A["n_checks"] > 500, A["n_checks"])
        check("stand-in analysis refuses to overwrite", _refuses(analyze_phase6.main))

    bad = [r for r in res if not r["ok"]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke_results.json").write_text(json.dumps({"n_checks": len(res), "n_failed": len(bad), "checks": res}, indent=2), encoding="utf-8")
    print(f"{len(res)} checks, {len(bad)} failed")
    for r in bad:
        print("FAIL", r["check"], r["detail"])
    sys.exit(1 if bad else 0)


def _refuses(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


if __name__ == "__main__":
    main()
