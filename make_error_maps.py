"""Qualitative error maps (RRD section 7.5) from frozen checkpoints, with a selection rule fixed before any map was drawn.

Rule (committed before this script first ran):
- Models, seed 1 each: SAR-only U-Net `p6_unet_sar` (VV, full input), DS-UNet B `p2_dropout` without the post-change
  optical image (`no_o2`: SAR t1+t2 and OSCD's own date-1 optical), DS-UNet A `p2_full` (full input). Each at its frozen
  validation threshold for that input.
- Cities: the 10 OSCD test cities ranked by A's per-city test F1 (full input, frozen threshold) averaged over seeds 1-10,
  from the registered Step 7 counts (`phase7/runs/p2_full_seed*.npz`); best, median (6th lowest of 10) and worst.
  Ties are broken by city name.
- Anchor (failure: no figure written): each map's TP/FP/FN/TN equal that city's saved counts at the same threshold
  (`phase7/runs` for SAR-only and A, `phase8/runs` tag `oscd` for B).

Writes figures/paper/fig13_error_maps.png and experiments/robust/error_maps/selection.json; refuses to overwrite.

Usage:
    .venv/Scripts/python.exe make_error_maps.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.datasets.oscd_dataset import ORBITS
from src.training import train_robust as tr
from src.training import train_rrd as rrd
from src.training.train_sar_baseline import sample_at_orbit

ROOT = tr.PROJECT_ROOT
R = ROOT / "experiments" / "robust"
FIG = ROOT / "figures" / "paper" / "fig13_error_maps.png"
SEL = R / "error_maps" / "selection.json"
SEED = 1
MODELS = [  # (panel title, arm, arch, condition, saved counts file, key suffix)
    ("SAR-only U-Net (VV)", "p6_unet_sar", "unet_sar", "full", R / "phase7" / "runs" / "p6_unet_sar_seed1.npz", "full"),
    ("DS-UNet B, no post-change optical", "p2_dropout", "dsunet", "no_o2", R / "phase8" / "runs" / "p2_dropout_seed1.npz", "oscd"),
    ("DS-UNet A, full input", "p2_full", "dsunet", "full", R / "phase7" / "runs" / "p2_full_seed1.npz", "full"),
]
# TN, TP, FP, FN; categorical slots 1-3 of the dataviz reference palette (validated: CVD ΔE ≥ 9.2 on #fcfcfb)
COLORS = ["#fcfcfb", "#2a78d6", "#eb6834", "#1baf7a"]
NAMES = ["true negative", "true positive (change found)", "false positive (false alarm)", "false negative (missed)"]


def f1(c) -> float:
    tp, fp, fn = c[0], c[1], c[2]
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def select_cities(cities):
    k = {s: json.loads((R / "phase7" / "runs" / f"p2_full_seed{s}.json").read_text(encoding="utf-8"))["thresholds"]["full"]
         for s in range(1, 11)}
    z = {s: np.load(R / "phase7" / "runs" / f"p2_full_seed{s}.npz") for s in range(1, 11)}
    mean_f1 = {c: float(np.mean([f1(z[s][f"{c}__full"][k[s]]) for s in range(1, 11)])) for c in cities}
    ranked = sorted(cities, key=lambda c: (mean_f1[c], c))  # ascending
    return {"worst": ranked[0], "median": ranked[len(ranked) // 2], "best": ranked[-1]}, mean_f1


@torch.inference_mode()
def main():
    if FIG.exists() or SEL.exists():
        raise SystemExit(f"{FIG.name} or {SEL.name} exists; refusing to overwrite")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False  # as every registered evaluator
    ds = rrd.RRDDataset("test", "hafner_vv")
    chosen, mean_f1 = select_cities(list(ds.cities))
    rows = [(role, chosen[role]) for role in ("best", "median", "worst")]

    maps, record = {}, {"rule": __doc__.split("Rule")[1].split("Writes")[0].strip(), "seed": SEED,
                        "mean_f1_A_by_city": mean_f1, "chosen": chosen, "panels": {}}
    for title, arm, arch, cond, counts_file, key in MODELS:
        ck = torch.load(ROOT / "checkpoints" / "robust" / f"{arm}_seed{SEED}.pt", map_location="cpu", weights_only=False)
        model = rrd.build_model(arch, 1)
        model.load_state_dict(ck["model_state_dict"])
        model.to(device).eval()
        k = ck["thresholds"][cond]
        saved = np.load(counts_file)
        for _, city in rows:
            s = sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))
            t1, t2, label = s["t1_img"][None].to(device), s["t2_img"][None].to(device), s["label"]
            prob = torch.sigmoid(model(*rrd.apply_condition(t1, t2, cond, 1)))[0, 0].cpu().numpy()
            pred = np.floor(np.clip(prob * 100, 0, 100)) >= k  # threshold_counts' rule: bin >= k
            pos = label[0].numpy() >= 0.5
            c = [int(x) for x in ((pred & pos).sum(), (pred & ~pos).sum(), (~pred & pos).sum(), (~pred & ~pos).sum())]
            expected = [int(x) for x in saved[f"{city}__{key}"][k]]
            if c != expected:
                raise SystemExit(f"ANCHOR FAILED {arm} {city}: {c} != saved {expected}")
            maps[(arm, city)] = np.select([pred & pos, pred & ~pos, ~pred & pos], [1, 2, 3], 0)
            record["panels"][f"{arm} {city}"] = {"threshold_bin": k, "tp_fp_fn_tn": c, "f1": f1(c)}

    fig, axes = plt.subplots(len(rows), 2 + len(MODELS), figsize=(3.1 * (2 + len(MODELS)), 3.3 * len(rows)))
    cmap = ListedColormap(COLORS)
    for r, (role, city) in enumerate(rows):
        s = sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))
        rgb = s["t2_img"][[4, 3, 2]].permute(1, 2, 0).numpy()  # B04, B03, B02 after the one SAR channel
        lo, hi = np.percentile(rgb, [2, 98])
        axes[r, 0].imshow(np.clip((rgb - lo) / (hi - lo), 0, 1))
        axes[r, 0].set_title(f"{city} ({role}): optical date 2", fontsize=9)
        axes[r, 1].imshow(s["label"][0].numpy() >= 0.5, cmap=ListedColormap([COLORS[0], "#1a1a19"]))
        axes[r, 1].set_title("ground truth (change in black)", fontsize=9)
        for j, (title, arm, *_rest) in enumerate(MODELS):
            axes[r, 2 + j].imshow(maps[(arm, city)], cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
            axes[r, 2 + j].set_title(f"{title}\nF1 {record['panels'][f'{arm} {city}']['f1']:.2f}", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#6b6b66", label=n) for c, n in zip(COLORS, NAMES)],
               loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    SEL.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, facecolor="white")
    SEL.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"written {FIG.name}; cities {chosen}")


if __name__ == "__main__":
    main()
