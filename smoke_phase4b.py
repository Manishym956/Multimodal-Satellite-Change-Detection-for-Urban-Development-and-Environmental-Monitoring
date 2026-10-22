"""Step 4B validation-only smoke test (no test city is read). Run after build_spatial_mask_bank.py, before any training.

Checks:
  bank      shape (N, 32, 32) bool, every mask fraction in [0.05, 0.95], file hash = bank json, the excluded scenes are
            exactly the scenes of both Phase 3 manifests
  sampler   the 5-condition draw with spatial = 0 gives the same condition sequence as train_robust's 4-condition draw
            with the B mixture; the C mixture's empirical frequencies match its probabilities
  semantics on a real training crop: spatial zero-fills S2 t2 exactly under the mask, in all 13 bands, and leaves S1 t2,
            all of t1, the label, the input tensors and the dataset's cached tensors unchanged; full / no_o2 / no_opt equal
            train_robust.apply_condition
  alignment crop + augmentation keep t1, t2 and label aligned (coordinate-coded synthetic sample)
  pipeline  3 GPU training steps on spatial batches (finite loss and gradients); checkpoint save/load round trip gives
            identical validation counts; model is in eval mode after evaluate()
  eval path eval_phase4abis.evaluate on the validation cities with p2_dropout_seed1 reproduces the stored 4A-bis
            validation counts for every condition
Writes experiments/robust/phase4b/smoke/smoke_results.json; exits 1 on any failure.

Usage:
    .venv/Scripts/python.exe smoke_phase4b.py
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.datasets.oscd_dataset import OSCDDataset
from src.training import train_robust as tr
from src.training import train_spatial as ts
from src.training.train_sar_baseline import set_seed, split_train_cities

ROOT = Path(__file__).resolve().parent
P4B = ROOT / "experiments" / "robust" / "phase4b"
PHASE3 = ROOT / "experiments" / "robust" / "phase3"
B_MIX = "full=0.5,no_o2=0.25,no_opt=0.25"
C_MIX = "full=0.5,spatial=0.25,no_o2=0.125,no_opt=0.125"
checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "pass": bool(ok), "detail": str(detail)[:300]})
    print(("ok   " if ok else "FAIL ") + name, flush=True)


def main() -> None:
    bank = np.load(ts.BANK)
    doc = json.loads((P4B / "spatial_mask_bank.json").read_text(encoding="utf-8"))
    frac = bank.mean((1, 2))
    check("bank: shape (N, 32, 32) bool", bank.shape == (doc["n"], 32, 32) and bank.dtype == bool, bank.shape)
    check("bank: every fraction in [0.05, 0.95]", frac.min() >= 0.05 and frac.max() <= 0.95, (frac.min(), frac.max()))
    check("bank: file hash = bank json", hashlib.sha256(ts.BANK.read_bytes()).hexdigest() == doc["sha256"])
    scenes = {e["scene"] for m in (PHASE3 / "mask_manifest.json", PHASE3 / "masks_superseded_tol005" / "mask_manifest.json")
              for e in json.loads(m.read_text(encoding="utf-8"))}
    check("bank: excluded scenes = all Phase 3 mask scenes (val + test, current + superseded)", set(doc["excluded_eval_scenes"]) == scenes)

    pb, pc = tr.parse_conditions(B_MIX), ts.parse_conditions(C_MIX)
    pb5 = ts.parse_conditions(B_MIX)
    r1, r2 = np.random.default_rng(1), np.random.default_rng(1)
    seq_b = [r1.choice(tr.CONDITIONS, p=[pb[c] for c in tr.CONDITIONS]) for _ in range(20000)]
    seq_b5 = [r2.choice(ts.TRAIN_CONDITIONS, p=[pb5[c] for c in ts.TRAIN_CONDITIONS]) for _ in range(20000)]
    check("sampler: spatial=0 draws the same sequence as train_robust (B mixture)", seq_b == seq_b5 and r1.random() == r2.random())
    r = np.random.default_rng(7)
    draws = r.choice(ts.TRAIN_CONDITIONS, size=2500 * 64, p=[pc[c] for c in ts.TRAIN_CONDITIONS])
    freq = {c: float(np.mean(draws == c)) for c in ts.TRAIN_CONDITIONS}
    check("sampler: C mixture frequencies within 0.005", all(abs(freq[c] - pc[c]) < 0.005 for c in ts.TRAIN_CONDITIONS), freq)

    set_seed(1)
    train_ds = OSCDDataset(split="train", mode="fusion")
    train_cities, val_cities = split_train_cities(train_ds.cities)
    check("split: validation cities are the registered four", sorted(val_cities) == ["aguasclaras", "beihai", "bercy", "paris"], val_cities)
    rng = np.random.default_rng(3)
    s = train_ds[train_ds.cities.index(train_cities[0])]
    cached = {k: s[k].clone() for k in ("t1_img", "t2_img", "label")}
    t1, t2, lab = tr.random_crop(s["t1_img"], s["t2_img"], s["label"], 32, True, rng)
    t1, t2, lab = tr.augment(t1, t2, lab, rng)
    t1b, t2b, labb = t1[None].clone(), t2[None].clone(), lab.clone()
    mrng = np.random.default_rng(5)
    idx = int(np.random.default_rng(5).integers(len(bank)))
    o1, o2 = ts.apply_train_condition(t1[None], t2[None], "spatial", bank, mrng)
    m = torch.from_numpy(bank[idx])
    check("spatial: S2 t2 is exactly 0 under the mask in all 13 bands", bool((o2[0, tr.N_S1:, m] == 0).all()) and o2.shape == t2b.shape)
    check("spatial: S2 t2 unchanged outside the mask", torch.equal(o2[0, tr.N_S1:, ~m], t2b[0, tr.N_S1:, ~m]))
    check("spatial: S1 t2 and all of t1 unchanged", torch.equal(o2[:, :tr.N_S1], t2b[:, :tr.N_S1]) and torch.equal(o1, t1b))
    check("spatial: mask non-trivial (some pixels zeroed, some kept)", 0 < int(m.sum()) < m.numel(), int(m.sum()))
    check("spatial: input tensors not mutated", torch.equal(t1[None], t1b) and torch.equal(t2[None], t2b) and torch.equal(lab, labb))
    s2 = train_ds[train_ds.cities.index(train_cities[0])]
    check("dataset: cached tensors not mutated by crop/augment/spatial",
          all(torch.equal(cached[k], s[k]) for k in cached) and all(torch.equal(cached[k], s2[k]) for k in cached if s2["orbit"] == s["orbit"]))
    for c in ("full", "no_o2", "no_opt"):
        a, b = ts.apply_train_condition(t1b, t2b, c, bank, mrng), tr.apply_condition(t1b, t2b, c)
        check(f"condition {c}: identical to train_robust.apply_condition", torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))

    h, w = 64, 48
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    code = (yy * 1000 + xx)[None]
    ok = True
    for i in range(50):
        rr = np.random.default_rng(100 + i)
        a, b, l = tr.random_crop(code.repeat(14, 1, 1), code.repeat(14, 1, 1) + 0, code.clone(), 32, False, rr)
        a, b, l = tr.augment(a, b, l, rr)
        ok &= torch.equal(a, b) and torch.equal(a[:1], l) and torch.equal(a[0], a[13])
    check("alignment: crop + augmentation keep t1, t2, label and all channels aligned (50 draws)", ok)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(11)
    model = tr.build_model().to(device)
    crit = tr.load_dsunet_module("loss_functions").jaccard_like_loss
    opt = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-4)
    rng = np.random.default_rng(11)
    finite = True
    for _ in range(3):
        model.train()
        bt1, bt2, bl = [], [], []
        for city in rng.choice(train_cities, 8):
            s = train_ds[train_ds.cities.index(city)]
            a, b, l = tr.random_crop(s["t1_img"], s["t2_img"], s["label"], 32, True, rng)
            a, b, l = tr.augment(a, b, l, rng)
            a, b = ts.apply_train_condition(a[None], b[None], "spatial", bank, rng)
            bt1.append(a[0]), bt2.append(b[0]), bl.append(l)
        x1, x2, y = (torch.stack(v).to(device) for v in (bt1, bt2, bl))
        opt.zero_grad()
        loss = crit(model(x1, x2), y)
        loss.backward()
        finite &= bool(torch.isfinite(loss)) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()
    check("pipeline: 3 spatial training steps, finite loss and gradients", finite, float(loss))
    counts = tr.evaluate(model, train_ds, val_cities, device)
    check("pipeline: evaluate() leaves the model in eval mode", not model.training)
    with tempfile.TemporaryDirectory() as d:
        torch.save({"model_state_dict": model.state_dict()}, Path(d) / "c.pt")
        m2 = tr.build_model().to(device)
        m2.load_state_dict(torch.load(Path(d) / "c.pt", map_location="cpu", weights_only=False)["model_state_dict"])
    counts2 = tr.evaluate(m2, train_ds, val_cities, device)
    check("pipeline: checkpoint round trip gives identical validation counts", all(torch.equal(counts[c], counts2[c]) for c in tr.CONDITIONS))

    from eval_phase4abis import CONDITIONS, evaluate
    from src.phase3.corruption import load_spectrum
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    ck = torch.load(ROOT / "checkpoints" / "robust" / "p2_dropout_seed1.pt", map_location="cpu", weights_only=False)
    mb = tr.build_model().to(device)
    mb.load_state_dict(ck["model_state_dict"])
    val = evaluate(mb, train_ds, val_cities, torch.from_numpy(load_spectrum()).to(device), device, regions=False)
    stored = np.load(ROOT / "experiments" / "robust" / "phase4abis" / "runs" / "p2_dropout_seed1.npz")
    check("eval path: p2_dropout_seed1 validation counts = stored 4A-bis counts, all 33 conditions",
          all(np.array_equal(val["pooled"][c].numpy(), stored[f"val__{c}"]) for c in CONDITIONS))

    (P4B / "smoke").mkdir(parents=True, exist_ok=True)
    out = {"n_checks": len(checks), "n_failed": sum(not c["pass"] for c in checks), "checks": checks,
           "smoke_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "torch": torch.__version__,
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (P4B / "smoke" / "smoke_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
    failed = [c for c in checks if not c["pass"]]
    print(f"{len(checks)} checks, {len(failed)} failed")
    sys.exit(1 if failed else 0)
