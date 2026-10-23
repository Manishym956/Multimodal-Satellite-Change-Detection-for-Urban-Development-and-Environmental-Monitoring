"""Step 5 analysis (fixed in PHASE5_PREREGISTRATION.md before any Step 5 test evaluation).

Arms: A p2_full, B p2_dropout, C p4b_spatial, seeds 1-10, frozen checkpoints. Test F1 at each checkpoint's frozen clean-
validation threshold, counts pooled per level over the 10 test cities and 3 composite replicates.
PRIMARY (B, Holm over 9; paired t-test, two-sided): at each level L in c10, c30, c60
    L__rb95 - L__bal      does the recall-biased detector mask beat the mask-accuracy-optimal one?
    L__rb95 - L__oracle   what does the recall-biased real-detector mask cost relative to a perfect mask?
    L__rb95 - L__cloud    does zero-fill under the recall-biased mask help relative to no fill?
  Labels: "rb95 better" if Holm p < 0.05 and Δ > 0; "rb95 worse" if Holm p < 0.05 and Δ < 0; else "no detectable difference".
SECONDARY (descriptive, each its own Holm family): the same 9 contrasts for A and for C; for B, L__rb90, L__rb99 and
L__default each minus L__bal; C - B at clean and at L__rb95; for B, clean__d - clean for every detector mask d (false
detections on a clean image).
DESCRIPTIVE (no tests): F1 / precision / recall / IoU / false-alarm rate for every condition; retained gain
(F1_mask - F1_cloud) / (F1_oracle - F1_cloud) per seed; F1 inside / outside M; the detector masks' recall, precision and
over-masked clear fraction on the TEST composites against M (no model); C - B pooled over the test composites whose scene is
not in the Step 4B mask bank (arm C trained on cloud shapes from bank scenes).

Writes experiments/robust/phase5/{summary.json, tables.md}.

Usage:
    .venv/Scripts/python.exe analyze_phase5.py
"""

import json

import numpy as np
import torch

from analyze_phase3 import describe, fmt_p, holm, ms, paired
from eval_phase5 import CONDITIONS, COMPOSITE_CONDS, DETS, OPS
from select_phase5_operating_points import rates as mask_rates
from src.phase5.composite import LEVELS, PHASE5, detector_mask, mask_counts
from src.training.train_robust import rates_at

SEEDS = list(range(1, 11))
ARMS = {"p2_full": "A full-only", "p2_dropout": "B dropout", "p4b_spatial": "C spatial dropout"}
METRICS = ("f1", "precision", "recall", "iou", "false_alarm_rate")
CONTRASTS = (("bal", "rb95"), ("oracle", "rb95"), ("cloud", "rb95"))  # (reference, mask): Δ = mask - reference


def label(r):
    return ("rb95 better" if r["holm_p"] < 0.05 and r["mean_diff"] > 0 else
            "rb95 worse" if r["holm_p"] < 0.05 and r["mean_diff"] < 0 else "no detectable difference")


def main() -> None:
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            doc = json.loads((PHASE5 / "runs" / f"{arm}_seed{s}.json").read_text(encoding="utf-8"))
            assert all(doc["anchors"].values()), doc["run"]
            npz = np.load(PHASE5 / "runs" / f"{arm}_seed{s}.npz")
            runs[(arm, s)] = (doc, {k: torch.from_numpy(npz[k]) for k in npz.files})
    hashes = {json.dumps(d["source_sha256"], sort_keys=True) for d, _ in runs.values()}
    assert len(hashes) == 1, "runs used different source versions"

    def m(arm, s, key, metric="f1"):
        doc, arr = runs[(arm, s)]
        return rates_at(arr[key], round(doc["tau_clean"] * 100))[metric] or 0.0

    f1 = lambda arm, c: [m(arm, s, f"test__{c}") for s in SEEDS]

    def family(pairs):
        """pairs: {name: (a values, b values)} -> rows with Holm over the family (Δ = b - a)."""
        rows = {k: paired(a, b) for k, (a, b) in pairs.items()}
        for k, h in zip(rows, holm([r["t_p"] for r in rows.values()])):
            rows[k]["holm_p"] = h
        for k, h in zip(rows, holm([r["wilcoxon_p"] for r in rows.values()])):
            rows[k]["holm_wilcoxon_p"] = h
        return rows

    contrasts = lambda arm: family({f"{lv}: {b} - {a}": (f1(arm, f"{lv}__{a}"), f1(arm, f"{lv}__{b}"))
                                    for lv in LEVELS for a, b in CONTRASTS})
    out = {"source_sha256": json.loads(hashes.pop()),
           "operating_points": json.loads((PHASE5 / "operating_points.json").read_text(encoding="utf-8"))["points"]}
    out["arms"] = {arm: {c: {mt: describe([m(arm, s, f"test__{c}", mt) for s in SEEDS]) for mt in METRICS} for c in CONDITIONS}
                   for arm in ARMS}
    out["primary"] = contrasts("p2_dropout")
    for r in out["primary"].values():
        r["label"] = label(r)
    out["secondary_A"] = contrasts("p2_full")
    out["secondary_C"] = contrasts("p4b_spatial")
    out["secondary_B_other_masks"] = family({f"{lv}: {d} - bal": (f1("p2_dropout", f"{lv}__bal"), f1("p2_dropout", f"{lv}__{d}"))
                                             for lv in LEVELS for d in ("rb90", "rb99", "default")})
    out["secondary_C_minus_B"] = family({c: (f1("p2_dropout", c), f1("p4b_spatial", c)) for c in ("clean", *(f"{lv}__rb95" for lv in LEVELS))})
    out["secondary_B_false_detections"] = family({f"clean__{d} - clean": (f1("p2_dropout", "clean"), f1("p2_dropout", f"clean__{d}")) for d in DETS})

    def retained(arm, lv, d):
        v = []
        for s in SEEDS:
            c, o, x = (m(arm, s, f"test__{lv}__{k}") for k in ("cloud", "oracle", d))
            v.append((x - c) / (o - c) if o != c else float("nan"))
        return describe(v)

    out["retained_gain"] = {arm: {lv: {d: retained(arm, lv, d) for d in DETS} for lv in LEVELS} for arm in ARMS}
    out["inside_outside"] = {arm: {c: {f"{part}_{mt}": describe([m(arm, s, f"{part}__{c}", mt) for s in SEEDS])
                                       for part in ("inside", "outside") for mt in ("f1", "recall", "false_alarm_rate")}
                                   for c in COMPOSITE_CONDS} for arm in ARMS}
    # C - B pooled over composites whose scene is NOT in the Step 4B mask bank (C never saw their cloud shapes); descriptive
    man = json.loads((PHASE5 / "composite_manifest.json").read_text(encoding="utf-8"))["entries"]
    clean_comps = {lv: [e["file"][:-4] for e in man if e["split"] == "test" and e["level"] == lv and not e["in_bank"]] for lv in LEVELS}

    def f1_subset(arm, s, lv, cond):
        doc, arr = runs[(arm, s)]
        cnt = sum(arr[f"comp__{n}__{lv}__{cond}"] for n in clean_comps[lv])
        return rates_at(cnt, round(doc["tau_clean"] * 100))["f1"] or 0.0

    out["C_minus_B_non_bank_composites"] = {lv: {"n_composites": len(clean_comps[lv]), **{
        cond: describe([f1_subset("p4b_spatial", s, lv, cond) - f1_subset("p2_dropout", s, lv, cond) for s in SEEDS])
        for cond in ("cloud", "oracle", "rb95", "bal")}} if clean_comps[lv] else {"n_composites": 0} for lv in LEVELS}
    # detector masks against M on the test composites (no model)
    ops = out["operating_points"]
    det = {lv: {d: np.zeros(4, np.int64) for d in DETS} for lv in LEVELS}
    for f in sorted((PHASE5 / "detector_inputs" / "test").glob("*_k*.npz")):
        lv = next(lv for lv in LEVELS if f"_{lv}_" in f.stem)
        mm = np.load(f)["m"]
        o = np.load(PHASE5 / "detector_outputs" / "test" / f.name)
        for d in DETS:
            p = o["default"] if d == "default" else detector_mask(o["prob"], ops[d]["t"], ops[d]["r"])
            det[lv][d] += mask_counts(mm, p)
    assert all(int(c.sum()) > 0 for v in det.values() for c in v.values())
    out["detector_test"] = {lv: {d: mask_rates(c) for d, c in v.items()} for lv, v in det.items()}
    (PHASE5 / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (PHASE5 / "tables.md").write_text(tables(out), encoding="utf-8")
    print((PHASE5 / "tables.md").read_text(encoding="utf-8"))


def family_table(rows, title):
    L = [title, "", "| contrast | mean Δ | 95% CI | d_z | Δ > 0 | t p | Holm p (t) | Wilcoxon p | Holm p (W) |", "|---|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | {r['b_gt_a']}/10 | "
                 f"{fmt_p(r['t_p'])} | {fmt_p(r['holm_p'])} | {fmt_p(r['wilcoxon_p'])} | {fmt_p(r['holm_wilcoxon_p'])} |"
                 + (f" **{r['label']}**" if "label" in r else ""))
    return L + [""]


def tables(out):
    L = ["### Operating points (chosen on validation composites from mask agreement only)", "",
         "| point | t | r | val recall | val precision | val mask F1 | val over-masked clear |", "|---|---|---|---|---|---|---|"]
    for n, p in out["operating_points"].items():
        L.append(f"| {n} | {p['t']} | {p['r']} | {p['recall']:.3f} | {p['precision']:.3f} | {p['f1']:.3f} | {p['overmasked_clear']:.3f} |")
    L += ["", "### Detector masks on the TEST composites against M (no model)", "",
          "| level | mask | recall | precision | mask F1 | over-masked clear |", "|---|---|---|---|---|---|"]
    for lv, v in out["detector_test"].items():
        for d, r in v.items():
            L.append(f"| {lv} | {d} | {r['recall']:.3f} | {r['precision']:.3f} | {r['f1']:.3f} | {r['overmasked_clear']:.3f} |")
    L.append("")
    L += family_table(out["primary"], "### PRIMARY: B dropout, test F1 contrasts at the frozen clean threshold, Holm over 9")
    L += ["### Test F1 at the frozen clean threshold (mean ± SD, 10 seeds)", "",
          "| condition | " + " | ".join(ARMS.values()) + " |", "|---|" + "---|" * len(ARMS)]
    for c in CONDITIONS:
        L.append(f"| {c} | " + " | ".join(ms(out["arms"][a][c]["f1"]) for a in ARMS) + " |")
    L += ["", "### Retained gain (F1_mask − F1_cloud) / (F1_oracle − F1_cloud), mean ± SD", "",
          "| arm | level | " + " | ".join(DETS) + " |", "|---|---|" + "---|" * len(DETS)]
    for a in ARMS:
        for lv in LEVELS:
            L.append(f"| {ARMS[a]} | {lv} | " + " | ".join(ms(out["retained_gain"][a][lv][d]) for d in DETS) + " |")
    L.append("")
    for key, title in (("secondary_A", "A full-only, same contrasts"), ("secondary_C", "C spatial dropout, same contrasts"),
                       ("secondary_B_other_masks", "B, other detector masks minus bal"), ("secondary_C_minus_B", "C − B"),
                       ("secondary_B_false_detections", "B, zero-fill under the detector mask on a clean image, minus clean")):
        L += family_table(out[key], f"### Secondary: {title} (own Holm family)")
    L += ["### Descriptive: C − B on test composites from scenes NOT in the Step 4B mask bank (mean ± SD of per-seed Δ; no tests)", "",
          "| level | composites | cloud | oracle | rb95 | bal |", "|---|---|---|---|---|---|"]
    for lv, e in out["C_minus_B_non_bank_composites"].items():
        L.append(f"| {lv} | {e['n_composites']} | " + (" | ".join(ms(e[c]) for c in ("cloud", "oracle", "rb95", "bal")) if e["n_composites"] else "– | – | – | –") + " |")
    L += ["", "### Inside / outside M, B dropout, frozen clean threshold (mean ± SD)", "",
          "| condition | F1 in M | recall in M | FAR in M | F1 outside M | recall outside M | FAR outside M |", "|---|---|---|---|---|---|---|"]
    for c, e in out["inside_outside"]["p2_dropout"].items():
        L.append(f"| {c} | " + " | ".join(ms(e[f"{p}_{mt}"]) for p in ("inside", "outside") for mt in ("f1", "recall", "false_alarm_rate")) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
