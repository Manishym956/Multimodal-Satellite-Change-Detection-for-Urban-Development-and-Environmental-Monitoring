"""Summarise experiments/robust/<config>_seed<N>/ runs into experiments/robust/summary_<args>.json (+ tables_<args>.md).

Per configuration: protocol and training arguments (runs whose arguments other than
the seed differ are refused, so protocols are never pooled), and for every input
condition and both thresholds (validation-selected, 0.5): per-seed F1, precision,
recall, IoU and confusion counts, plus mean, sample SD and n. Conditions added later
by reeval_robust.py are taken from reeval.json and marked source="reeval".
Optionally prints paired comparisons over shared seeds against --ref.

Usage:
    .venv/Scripts/python.exe summarize_robust.py [--prefix holdout_] [--ref holdout_full_s1000]
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent / "experiments" / "robust"
CONDITIONS = ("full", "no_o2", "o2_copy", "no_opt")
METRICS = ("f1", "precision", "recall", "iou", "tp", "fp", "fn", "tn")


def fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def load(prefix: str) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in ROOT.glob(f"{prefix}*_seed*/metrics.json"):
        name, seed = re.fullmatch(r"(.+)_seed(\d+)", path.parent.name).groups()
        doc = json.loads(path.read_text(encoding="utf-8"))
        for c in doc["test"]:
            doc["test"][c]["source"] = "train_run"
        reeval = path.parent / "reeval.json"
        if reeval.exists():
            for c, entry in json.loads(reeval.read_text(encoding="utf-8"))["added"].items():
                doc["test"][c] = {**entry, "source": "reeval"}
        runs[name][int(seed)] = doc
    return dict(sorted(runs.items()))


def stat(values: list) -> dict:
    vals = [v for v in values if v is not None]
    return {"values": values, "n": len(vals), "mean": statistics.mean(vals) if vals else None,
            "sd": statistics.stdev(vals) if len(vals) > 1 else None}


def summarise(seeds: dict[int, dict]) -> dict:
    order = sorted(seeds)
    args = {k: v for k, v in seeds[order[0]]["args"].items() if k != "seed"}
    for s in order:
        other = {k: v for k, v in seeds[s]["args"].items() if k != "seed"}
        if other != args:
            raise SystemExit(f"seed {s} has different arguments; refusing to pool: {other} vs {args}")
    out = {"protocol": args["protocol"], "args": args, "seeds": order,
           "val_cities": seeds[order[0]]["val_cities"], "selected_steps": [seeds[s]["selected_step"] for s in order],
           "conditions": {}}
    for c in CONDITIONS:
        present = [s for s in order if c in seeds[s]["test"]]
        if not present:
            continue
        entry = {"seeds": present, "source": sorted({seeds[s]["test"][c]["source"] for s in present}),
                 "selected_threshold": [seeds[s]["test"][c]["selected_threshold"] for s in present]}
        for at in ("at_selected", "at_0.5"):
            entry[at] = {m: stat([seeds[s]["test"][c][at][m] for s in present]) for m in METRICS}
        out["conditions"][c] = entry
    return out


def cell(summary: dict, c: str, at: str) -> str:
    if c not in summary["conditions"]:
        return "—"
    st = summary["conditions"][c][at]["f1"]
    return f"{st['mean']:.3f} ± {st['sd'] or 0:.3f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="")
    p.add_argument("--ref")
    args = p.parse_args()
    runs = load(args.prefix)
    summary = {name: summarise(seeds) for name, seeds in runs.items()}
    for at in ("at_selected", "at_0.5"):
        print(f"\n### test F1 ({at}), mean ± sample SD\n| config | protocol | steps | n | " + " | ".join(CONDITIONS) + " |")
        print("|---|---|---|---|" + "---|" * len(CONDITIONS))
        for name, s in summary.items():
            print(f"| {name} | {s['protocol']} | {s['args']['steps']} | {len(s['seeds'])} | "
                  + " | ".join(cell(s, c, at) for c in CONDITIONS) + " |")
    if args.ref:
        ref = runs[args.ref]
        print(f"\n### paired vs {args.ref} (F1 at validation-selected threshold)")
        print("| config | condition | n | Δ mean | paired t p | Holm p (t, over conditions) | Wilcoxon p |\n|---|---|---|---|---|---|---|")
        for name, seeds in runs.items():
            if name == args.ref or summary[name]["protocol"] != summary[args.ref]["protocol"]:
                continue
            rows = []
            for c in CONDITIONS:
                shared = sorted(s for s in set(seeds) & set(ref) if c in seeds[s]["test"] and c in ref[s]["test"])
                if len(shared) < 3:
                    continue
                a = [seeds[s]["test"][c]["at_selected"]["f1"] or 0.0 for s in shared]
                b = [ref[s]["test"][c]["at_selected"]["f1"] or 0.0 for s in shared]
                d = [x - y for x, y in zip(a, b)]
                rows.append({"condition": c, "n": len(shared), "seeds": shared, "diffs": d, "mean_diff": statistics.mean(d),
                             "t_p": float(stats.ttest_rel(a, b).pvalue), "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue) if any(d) else 1.0})
            running = 0.0  # Holm step-down over this config's conditions
            for rank, r in enumerate(sorted(rows, key=lambda r: r["t_p"])):
                running = max(running, min(1.0, (len(rows) - rank) * r["t_p"]))
                r["holm_p"] = running
            for r in rows:
                print(f"| {name} | {r['condition']} | {r['n']} | {r['mean_diff']:+.3f} | {fmt_p(r['t_p'])} | {fmt_p(r['holm_p'])} | {fmt_p(r['wilcoxon_p'])} |")
            summary[name]["paired_vs_" + args.ref] = rows
    tag = "_".join(f"{k}-{v}" for k, v in vars(args).items() if v) or "all"
    (ROOT / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import contextlib
    import io
    import sys

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main()
    print(buffer.getvalue())
    tag = "_".join(a.lstrip("-") for a in sys.argv[1:]).replace("=", "-") or "all"
    (ROOT / f"tables_{tag}.md").write_text(buffer.getvalue(), encoding="utf-8")
