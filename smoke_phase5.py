"""Step 5 smoke test: VALIDATION cities only, before any Step 5 test evaluation. No test model output is produced.

Checks: scene pools and exclusions; composite files against the manifest; composite construction (SAR, outside-M and
inside-M pixels); detector-input band order; detector outputs; detector-mask rule; operating points obey the rule; and
eval_phase5.evaluate on the validation cities for p2_dropout_seed1 and p4b_spatial_seed1 (clean counts equal the stored
counts, clean argmax equals the stored threshold, inside + outside M equals pooled, outside M unchanged, deterministic).
Writes experiments/robust/phase5/smoke/smoke_results.json.

Usage:
    .venv/Scripts/python.exe smoke_phase5.py
"""

import hashlib
import json

import numpy as np
import torch
from scipy import ndimage

from eval_phase5 import COMPOSITE_CONDS, CONDITIONS, evaluate, load_masks
from export_phase5_detector_inputs import city_inputs, detector_input
from src.datasets.oscd_dataset import OSCDDataset
from src.phase3.corruption import CATALOGUE
from src.phase5.composite import DETECTOR_BANDS, LEVELS, N_S1, PHASE5, RADII, THRESHOLDS, detector_mask, scene_pools, upsample
from src.training.train_robust import best_threshold, build_model
from src.training.train_sar_baseline import split_train_cities

results = []


def check(name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})
    print("ok  " if ok else "FAIL", name, detail if not ok else "", flush=True)


def main() -> None:
    man = json.loads((PHASE5 / "composite_manifest.json").read_text(encoding="utf-8"))
    val_pool, test_pool, excl = scene_pools()
    check("pools reproduce the manifest", val_pool == man["val_pool"] and test_pool == man["test_pool"])
    check("pools disjoint", not (set(val_pool) & set(test_pool)))
    check("no difficulty-5 scene in a pool", not ((set(val_pool) | set(test_pool)) & set(excl["excluded_difficulty5"])))
    check("bank flags recorded on every composite", all("in_bank" in e and e["in_bank"] == (e["scene"] in set(excl["flag_bank"]))
                                                        for e in man["entries"]))
    bank = json.loads((PHASE5.parent / "phase4b" / "spatial_mask_bank_sources.json").read_text(encoding="utf-8"))
    check("bank sources verified by replay (sha256 equals the recorded bank)",
          bank["bank_sha256"] == json.loads((PHASE5.parent / "phase4b" / "spatial_mask_bank.json").read_text(encoding="utf-8"))["sha256"]
          and len(bank["scenes"]) == 153)
    ents = man["entries"]
    check("126 composites (36 val, 90 test)", len(ents) == 126 and sum(e["split"] == "val" for e in ents) == 36)
    check("composite files match manifest sha256",
          all(hashlib.sha256((PHASE5 / "composites" / e["file"]).read_bytes()).hexdigest() == e["sha256"] for e in ents))
    check("every composite fraction within ±10% of its level", all(abs(e["fraction"] - LEVELS[e["level"]]) <= 0.1 * LEVELS[e["level"]] + 1e-12 for e in ents))
    check("val and test composites share no scene",
          not ({e["scene"] for e in ents if e["split"] == "val"} & {e["scene"] for e in ents if e["split"] == "test"}))
    check("composite scenes come from their split's pool",
          all(e["scene"] in (val_pool if e["split"] == "val" else test_pool) for e in ents))

    ds = OSCDDataset(split="train", mode="fusion")
    _, val_cities = split_train_cities(ds.cities)
    city = val_cities[0]
    t2, comps = city_inputs(ds, city)
    name = f"{city}_c30_k0"
    x2, m = comps[name]
    z = np.load(PHASE5 / "composites" / f"{name}.npz")
    bands = torch.from_numpy(np.ascontiguousarray(upsample(z["bands20"], tuple(t2.shape[-2:])).transpose(2, 0, 1)))
    check("composite: SAR unchanged", torch.equal(x2[:, :N_S1], t2[:, :N_S1]))
    check("composite: outside M unchanged", torch.equal(x2[0, N_S1:, ~m], t2[0, N_S1:, ~m]))
    check("composite: inside M equals catalogue bands", torch.equal(x2[0, N_S1:, m], bands[:, m]))
    check("composite: values in [0, 1]", float(x2[0, N_S1:].min()) >= 0 and float(x2[0, N_S1:].max()) <= 1)
    inp = np.load(PHASE5 / "detector_inputs" / "val" / f"{name}.npz")
    check("detector input equals composite bands in s2cloudless order",
          np.array_equal(inp["x"], detector_input(x2)) and np.array_equal(inp["x"][..., 4], x2[0, N_S1 + 7].numpy()) and DETECTOR_BANDS[4] == 7)
    check("detector input stores M", np.array_equal(inp["m"], m.numpy()))
    info = json.loads((PHASE5 / "detector_outputs" / "val" / "detector_info.json").read_text(encoding="utf-8"))
    check("detector outputs for every validation input", len(info["files"]) == len(list((PHASE5 / "detector_inputs" / "val").glob("*.npz"))) == 40)
    out = np.load(PHASE5 / "detector_outputs" / "val" / f"{name}.npz")
    check("probability map in [0, 1], image shape", out["prob"].shape == m.shape and 0 <= out["prob"].min() and out["prob"].max() <= 1)
    check("detector info: s2cloudless 1.7.3, library defaults", info["s2cloudless"] == "1.7.3" and info["threshold"] == 0.4
          and info["average_over"] == 1 and info["dilation_size"] == 1 and not info["all_bands"])
    p = out["prob"]
    check("detector mask r = 0 is prob >= t", np.array_equal(detector_mask(p, 0.4, 0), p >= 0.4))
    edt = ndimage.distance_transform_edt(~(p >= 0.4))
    check("detector mask dilation equals Euclidean disk", all(np.array_equal(detector_mask(p, 0.4, r), edt <= r) for r in (1, 4, 16)))
    ops = json.loads((PHASE5 / "operating_points.json").read_text(encoding="utf-8"))
    grid = {(g["t"], g["r"]): g for g in ops["grid"]}
    check("grid complete", len(grid) == len(THRESHOLDS) * len(RADII))
    rb = ops["points"]["rb95"]
    ok = [g for g in grid.values() if g["recall"] >= 0.95]
    check("rb95 follows the rule", (not ok and not rb["target_reached"]) or
          (rb["recall"] >= 0.95 and rb["precision"] == max(g["precision"] for g in ok)))
    check("bal is the max mask F1", ops["points"]["bal"]["f1"] == max(g["f1"] for g in grid.values()))
    check("operating points used validation composites only", all(n.split("_c")[0] in val_cities for n in ops["validation_composites"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    masks = load_masks("val", val_cities, ops["points"])
    for arm, ref in (("p2_dropout", "phase4abis"), ("p4b_spatial", "phase4b")):
        ckpt = torch.load(PHASE5.parents[2] / "checkpoints" / "robust" / f"{arm}_seed1.pt", map_location="cpu", weights_only=False)
        model = build_model().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        v1 = evaluate(model, ds, val_cities, masks, device, inside_outside=True)
        v2 = evaluate(model, ds, val_cities, masks, device, inside_outside=True)
        stored = np.load(PHASE5.parent / ref / "runs" / f"{arm}_seed1.npz")
        check(f"{arm}: clean val counts equal stored", np.array_equal(v1["pooled"]["clean"].numpy(), stored["val__clean"]))
        check(f"{arm}: clean val argmax equals stored threshold", best_threshold(v1["pooled"]["clean"])[0] == ckpt["thresholds"]["full"])
        check(f"{arm}: inside + outside equals pooled", all(torch.equal(v1["inside"][c] + v1["outside"][c], v1["pooled"][c]) for c in COMPOSITE_CONDS))
        check(f"{arm}: outside M unchanged in every composite", v1["outside_identical"])
        check(f"{arm}: deterministic", all(torch.equal(v1["pooled"][c], v2["pooled"][c]) for c in CONDITIONS))
        check(f"{arm}: model in eval mode after evaluate", not model.training)
    check("catalogue masks present", (CATALOGUE / "masks_raw" / "masks").exists())
    doc = {"n_checks": len(results), "n_failed": sum(not r["ok"] for r in results), "results": results}
    (PHASE5 / "smoke").mkdir(exist_ok=True)
    (PHASE5 / "smoke" / "smoke_results.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"{doc['n_checks']} checks, {doc['n_failed']} failed")


if __name__ == "__main__":
    main()
