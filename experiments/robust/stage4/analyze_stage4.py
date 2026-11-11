"""Stage 4 exploratory re-analysis of stored test counts. NOT pre-registered.

Written after the Stage 3 simulated peer review (2026-11-02) and after every test result below had been seen. It reads
only stored outputs (run metrics.json, per-run count arrays, step summaries); it trains nothing, evaluates no model and
modifies no existing file. Every test here is exploratory and is labelled so in the paper; the registered analyses in
the step reports stay primary and unchanged.

Outputs: experiments/robust/stage4/stage4_summary.json and STAGE4_EXPLORATORY_REPORT.md.

Usage:
    .venv/Scripts/python.exe experiments/robust/stage4/analyze_stage4.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "experiments" / "robust"
OUT = Path(__file__).resolve().parent
SEEDS = range(1, 11)
GAPS = ["gap01", "gap03", "gap06", "gap12", "gap18"]
GAP_MONTHS = np.array([1, 3, 6, 12, 18])
BOOT = 10000
rng = np.random.default_rng(20261102)


def f1(c):
    tp, fp, fn, _ = (float(x) for x in c)
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def metrics(run):
    return json.loads((R / run / "metrics.json").read_text())["test"]


def pooled(run, cond, at="at_selected"):
    return metrics(run)[cond][at]


def p8(arm, seed):
    j = json.loads((R / "phase8" / "runs" / f"{arm}_seed{seed}.json").read_text())
    return j, np.load(R / "phase8" / "runs" / f"{arm}_seed{seed}.npz")


def paired(x, y, label, holm_k=1):
    """Paired t on per-seed values x - y (+ exact Wilcoxon)."""
    d = np.asarray(x) - np.asarray(y)
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    t = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    p = stats.ttest_1samp(d, 0).pvalue if sd > 0 else 1.0
    w = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return {"contrast": label, "mean_diff": m, "ci95": [m - t, m + t], "d_z": m / sd if sd > 0 else 0.0,
            "n_pos": int((d > 0).sum()), "n": n, "p_t": p, "p_wilcoxon": w, "diffs": d.tolist()}


def holm(rows, key="p_t"):
    order = sorted(range(len(rows)), key=lambda i: rows[i][key])
    k, run = len(rows), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, rows[i][key] * (k - rank)))
        rows[i]["holm_p"] = run
    return rows


# ---------- E1: RQ2 timing, exploratory omnibus and pairwise tests (Step 8, 9 primary cities) ----------
s8 = json.loads((R / "phase8" / "summary.json").read_text())
PRIMARY = s8["primary_cities"]
ALL10 = s8["available_cities"]


def p8_f1(arm, tag, cities, thr_key="no_o2", thr=None):
    out = []
    for s in SEEDS:
        j, z = p8(arm, s)
        t = thr if thr is not None else j["thresholds"][thr_key]
        out.append(f1(sum(z[f"{c}__{tag}"][t] for c in cities)))
    return np.array(out)


e1 = {}
for arm, name in (("p2_dropout", "B"), ("p6_async", "async")):
    Y = np.stack([p8_f1(arm, g, PRIMARY) for g in GAPS], axis=1)  # seeds x gaps
    for gi, g in enumerate(GAPS):  # self-check against the audited Step 8 summary
        assert np.allclose(Y[:, gi], s8["arms"][arm][g]["f1"]["values"]), (arm, g)
    n, k = Y.shape
    gm = Y.mean()
    ss_c = n * ((Y.mean(0) - gm) ** 2).sum()
    ss_s = k * ((Y.mean(1) - gm) ** 2).sum()
    ss_e = ((Y - gm) ** 2).sum() - ss_c - ss_s
    F = (ss_c / (k - 1)) / (ss_e / ((k - 1) * (n - 1)))
    slopes = np.array([np.polyfit(GAP_MONTHS / 12, Y[i], 1)[0] for i in range(n)])
    pw = holm([paired(Y[:, 0], Y[:, j], f"{name}: 1 month - {GAP_MONTHS[j]} months") for j in range(1, k)])
    e1[name] = {"means": Y.mean(0).tolist(), "friedman_p": stats.friedmanchisquare(*Y.T).pvalue,
                "rm_anova_F": F, "rm_anova_df": [k - 1, (k - 1) * (n - 1)],
                "rm_anova_p": stats.f.sf(F, k - 1, (k - 1) * (n - 1)), "slopes_per_year": slopes.tolist(),
                "slope_mean": slopes.mean(), "slope_p_t": stats.ttest_1samp(slopes, 0).pvalue,
                "slope_mean_without_max": np.delete(slopes, slopes.argmax()).mean(),
                "slope_p_t_without_max": stats.ttest_1samp(np.delete(slopes, slopes.argmax()), 0).pvalue,
                "max_slope_seed": int(slopes.argmax()) + 1, "pairwise_1m": pw}

scene_rows = []
for g in GAPS:
    doy_off, day_off_t1, clouds = [], [], []
    for c in PRIMARY:
        sc = s8["scenes"][f"{c}__{g}"]
        d, t1 = np.datetime64(sc["day"]), np.datetime64(sc["oscd_t1"])
        x = abs(int((d - np.datetime64(sc["day"][:4] + "-01-01")).astype(int))
                - int((t1 - np.datetime64(sc["oscd_t1"][:4] + "-01-01")).astype(int)))
        doy_off.append(min(x, 365 - x))
        day_off_t1.append(int((d - t1).astype(int)))
        clouds.append(sc["cloud"])
    scene_rows.append({"gap": g, "season_offset_days_mean": float(np.mean(doy_off)),
                       "days_after_oscd_t1_min": min(day_off_t1), "days_after_oscd_t1_max": max(day_off_t1),
                       "cloud_max": max(clouds),
                       "n_before_oscd_t1": len([c for c in s8["image_before_oscd_t1"][g] if c in PRIMARY])})
e1["scenes"] = scene_rows


# ---------- E2: test-city vs training-seed variation (per-city counts) ----------
def city_counts(getter, cities):
    """getter(seed, city) -> 4 counts; returns array seeds x cities x 4."""
    return np.array([[getter(s, c) for c in cities] for s in SEEDS], dtype=float)


def p8_getter(arm, tag, thr_key):
    cache = {s: p8(arm, s) for s in SEEDS}
    return lambda s, c: cache[s][1][f"{c}__{tag}"][cache[s][0]["thresholds"][thr_key]]


def npz_getter(phase, arm, cond, thr_of_seed):
    cache = {s: np.load(R / phase / "runs" / f"{arm}_seed{s}.npz") for s in SEEDS}
    return lambda s, c: cache[s][f"city__{c}__{cond}"][thr_of_seed(s)]


def pooled_f1(C):  # C: seeds x cities x 4 -> per-seed F1
    return np.array([f1(C[i].sum(0)) for i in range(C.shape[0])])


def city_analysis(label, Cx, Cy, cities):
    seed_d = pooled_f1(Cx) - pooled_f1(Cy)
    per_city = np.array([[f1(Cx[i, j]) - f1(Cy[i, j]) for j in range(len(cities))] for i in range(len(SEEDS))])
    city_mean = per_city.mean(0)
    boot, boot_city = np.empty(BOOT), np.empty(BOOT)
    ns, nc = Cx.shape[0], Cx.shape[1]
    for b in range(BOOT):  # same city and seed indices for both arms, so the seed pairing is kept
        ci, si = rng.integers(0, nc, nc), rng.integers(0, ns, ns)
        dd = np.array([f1(Cx[s, ci].sum(0)) - f1(Cy[s, ci].sum(0)) for s in range(ns)])
        boot[b], boot_city[b] = dd[si].mean(), dd.mean()  # cities and seeds resampled; cities only
    # standard errors of the same pooled estimate under each resampling scheme (separate check by a fresh session of the same AI assistant, 2026-11-02)
    jk = np.array([(pooled_f1(np.delete(Cx, j, 1)) - pooled_f1(np.delete(Cy, j, 1))).mean() for j in range(nc)])
    se_jk = float(np.sqrt((nc - 1) / nc * ((jk - jk.mean()) ** 2).sum()))
    se_seed = float(seed_d.std(ddof=1) / np.sqrt(ns))
    tcrit = stats.t.ppf(0.975, ns - 1)
    pos = Cx[0, :, 0] + Cx[0, :, 2]  # change pixels per city (labels only, same for every seed and arm)
    loco = []
    for j in range(nc):
        keep = [i for i in range(nc) if i != j]
        d = pooled_f1(Cx[:, keep]) - pooled_f1(Cy[:, keep])
        loco.append({"dropped": cities[j], "mean_diff": d.mean(),
                     "p_t": stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else 1.0})
    sign = np.sign(seed_d.mean())
    return {"contrast": label, "n_cities": nc, "seed_mean_diff": seed_d.mean(), "sd_across_seeds": seed_d.std(ddof=1),
            "seed_ci95": paired(pooled_f1(Cx), pooled_f1(Cy), label)["ci95"],
            "sd_across_cities": city_mean.std(ddof=1), "per_city_mean_diff": dict(zip(cities, city_mean.tolist())),
            "cities_same_sign": int((np.sign(city_mean) == sign).sum()),
            "boot_ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
            "boot_excludes_0": bool(np.percentile(boot, 2.5) > 0 or np.percentile(boot, 97.5) < 0),
            "se_seeds": se_seed, "se_cities_boot": float(boot_city.std()), "se_cities_jackknife": se_jk,
            "jackknife_t_ci95": [seed_d.mean() - tcrit * np.hypot(se_jk, se_seed), seed_d.mean() + tcrit * np.hypot(se_jk, se_seed)],
            "unweighted_city_mean_diff": float(city_mean.mean()),
            "change_pixel_share": dict(zip(cities, (pos / pos.sum()).tolist())),
            "loco_drop_mean_diff": dict(zip(cities, jk.tolist())),
            "loco_min": min(r["mean_diff"] for r in loco), "loco_max": max(r["mean_diff"] for r in loco),
            "loco_sign_kept": sum(np.sign(r["mean_diff"]) == sign for r in loco),
            "loco_p_below_05": sum(r["p_t"] < 0.05 for r in loco), "loco": loco}


tau_clean = {s: int(round(json.loads((R / "phase3" / "runs" / f"p2_dropout_seed{s}.json").read_text())["tau_clean"]
                          * 100)) for s in SEEDS}
sar10 = city_counts(p8_getter("p6_unet_sar", "oscd", "full"), ALL10)
assert np.isclose(pooled_f1(sar10).mean(), np.mean([pooled("p6_unet_sar_seed%d" % s, "full")["f1"] for s in SEEDS]))
e2 = [
    city_analysis("A (full input) - SAR-only U-Net", city_counts(p8_getter("p2_full", "full", "full"), ALL10), sar10,
                  ALL10),
    city_analysis("B (no post-change optical, own threshold) - SAR-only U-Net",
                  city_counts(p8_getter("p2_dropout", "oscd", "no_o2"), ALL10), sar10, ALL10),
    city_analysis("async (no post-change optical) - SAR-only U-Net",
                  city_counts(p8_getter("p6_async", "oscd", "no_o2"), ALL10), sar10, ALL10),
    city_analysis("B (no post-change optical, frozen clean threshold) - SAR-only U-Net",
                  city_counts(npz_getter("phase3", "p2_dropout", "missing", lambda s: tau_clean[s]), ALL10), sar10,
                  ALL10),
    city_analysis("B: 1-month image - OSCD date-1 image (9 primary cities)",
                  city_counts(p8_getter("p2_dropout", "gap01", "no_o2"), PRIMARY),
                  city_counts(p8_getter("p2_dropout", "oscd", "no_o2"), PRIMARY), PRIMARY),
]
# self-checks against audited values: B@no_o2 (Phase 2 runs) and B missing at the clean threshold (Phase 3)
b_no_o2 = np.array([pooled(f"p2_dropout_seed{s}", "no_o2")["f1"] for s in SEEDS])
assert np.isclose(e2[1]["seed_mean_diff"], b_no_o2.mean() - pooled_f1(sar10).mean())
s3 = json.loads((R / "phase3" / "summary.json").read_text())


# ---------- E3: RQ1 within-model contrasts (historical image present - absent), pooled run metrics ----------
def seedvals(arm, cond, at="at_selected", key="f1"):
    return np.array([pooled(f"{arm}_seed{s}", cond, at)[key] for s in SEEDS])


e3 = []
for arm, name in (("p2_dropout", "B"), ("p6_async", "async")):
    for at, lab in (("at_selected", "each input's own validation-selected threshold"), ("at_0.5", "fixed 0.5")):
        e3.append(paired(seedvals(arm, "no_o2", at), seedvals(arm, "no_opt", at),
                         f"{name}: historical image present - absent, {lab}"))
sar = seedvals("p6_unet_sar", "full")
b_missing_clean = pooled_f1(city_counts(npz_getter("phase3", "p2_dropout", "missing", lambda s: tau_clean[s]), ALL10))
e3_thr = [paired(b_no_o2, sar, "B - SAR-only, B at its no_o2 validation threshold (registered)"),
          paired(b_missing_clean, sar, "B - SAR-only, B at its frozen clean threshold"),
          paired(seedvals("p2_dropout", "no_o2", "at_0.5"), seedvals("p6_unet_sar", "full", "at_0.5"),
                 "B - SAR-only, both at 0.5")]

# ---------- E4: RQ4 for the asynchronous arms (no post-change optical) against the baselines ----------
base = {"SAR-only U-Net (full input)": sar, "optical-only U-Net (full input)": seedvals("p6_unet_opt", "full"),
        "input-fusion U-Net (full input)": seedvals("p6_unet_fusion", "full"), "A (full input)": seedvals("p2_full",
                                                                                                         "full")}
asyn = {"B (no post-change optical)": b_no_o2, "async (no post-change optical)": seedvals("p6_async", "no_o2")}
e4 = holm([paired(av, bv, f"{an} - {bn}") for an, av in asyn.items() for bn, bv in base.items()])

# ---------- E5: SAR-only fallback rule against the recommended dropout model B ----------
s5 = json.loads((R / "phase5" / "summary.json").read_text())


def p5_vals(arm, cond):
    v = []
    for s in SEEDS:
        j = json.loads((R / "phase5" / "runs" / f"{arm}_seed{s}.json").read_text())
        z = np.load(R / "phase5" / "runs" / f"{arm}_seed{s}.npz")
        v.append(f1(z[f"test__{cond}"][int(round(j['tau_clean'] * 100))]))
    v = np.array(v)
    assert np.allclose(v, s5["arms"][arm][cond]["f1"]["values"]), (arm, cond)
    return v


e5 = [paired(seedvals("p2_dropout", "full"), seedvals("p2_full", "full"),
             "image present: B - A (the rule uses A)"),
      paired(b_no_o2, sar, "image missing: B - SAR-only (the rule uses SAR-only)")]
for lev in ("c10", "c30", "c60"):
    for m in ("cloud", "bal", "rb95", "rb99", "oracle"):
        e5.append(paired(p5_vals("p2_dropout", f"{lev}__{m}"), sar,
                         f"composite clouds {lev}: B with mask {m} - SAR-only"))
e5_pr = [paired(seedvals("p2_dropout", "full", key=k), seedvals("p2_full", "full", key=k), f"full input: B - A, {k}")
         for k in ("precision", "recall", "false_alarm_rate")]

# ---------- E6: RQ5 per-class recall next to the false-alarm rates ----------
e6 = {}
for cls, v in s8["worldcover"]["p2_dropout"]["oscd"].items():
    sv = s8["worldcover"]["p6_unet_sar"]["oscd"][cls]
    e6[cls] = {"B_recall": float(np.mean(v["recall"]["values"])), "SAR_recall": float(np.mean(sv["recall"]["values"])),
               "B_far": float(np.mean(v["false_alarm_rate"]["values"])),
               "SAR_far": float(np.mean(sv["false_alarm_rate"]["values"]))}

# ---------- E7: selected thresholds at the edge of the grid ----------
e7 = {}
for arm in ("p2_full", "p2_dropout", "p6_async", "p6_unet_sar", "p6_unet_opt", "p6_unet_fusion"):
    th = [metrics(f"{arm}_seed{s}")[c]["selected_threshold"] for s in SEEDS for c in ("full", "no_o2", "no_opt")]
    e7[arm] = {"n": len(th), "at_0.99_or_above": sum(t >= 0.99 for t in th), "max": max(th)}

# ---------- E8: Sentinel-1 scenes per 3-month window (feasibility inventory, same windows as the exports) ----------
inv = json.loads((R / "feasibility" / "inventory.json").read_text())["s1"]
e8 = []
for c in ALL10:
    for r1 in (r for r in inv if r["city"] == c and r["t"] == 1):
        r2 = next(r for r in inv if r["city"] == c and r["t"] == 2 and r["orbit"] == r1["orbit"])
        e8.append({"city": c, "orbit": r1["orbit"], "vv_t1": r1["n_vv"], "vv_t2": r2["n_vv"], "vh_t1": r1["n_vh"],
                   "vh_t2": r2["n_vh"]})
e8_stats = {"windows": len(e8), "median_vv_t1": float(np.median([r["vv_t1"] for r in e8])),
            "median_vv_t2": float(np.median([r["vv_t2"] for r in e8])),
            "windows_t1_fewer": sum(r["vv_t1"] < r["vv_t2"] for r in e8)}

summary = {"note": "Exploratory, not pre-registered; written after all results were seen (Stage 4, 2026-11-02).",
           "bootstrap": {"replicates": BOOT, "seed": 20261102, "scheme": "cities then seeds, with replacement"},
           "E1_rq2_timing": e1, "E2_city_vs_seed": e2, "E3_rq1_within_model": e3, "E3_rq1_threshold_rules": e3_thr,
           "E4_rq4_async": e4, "E5_fallback_rule": e5, "E5_full_input_B_vs_A": e5_pr, "E6_rq5_recall": e6,
           "E7_threshold_edges": e7, "E8_s1_scene_counts": {"windows": e8, "stats": e8_stats}}


def conv(o):
    if isinstance(o, dict):
        return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


(OUT / "stage4_summary.json").write_text(json.dumps(conv(summary), indent=1))


# ---------- report ----------
def f(x, d=3):
    return f"{x:+.{d}f}".replace("-", "−")


def p(x):
    return f"{x:.2g}" if x < 0.001 else f"{x:.3f}"


def row(r):
    return (f"| {r['contrast']} | {f(r['mean_diff'])} | [{f(r['ci95'][0])}, {f(r['ci95'][1])}] | {f(r['d_z'], 2)} | "
            f"{r['n_pos']}/{r['n']} | {p(r['p_t'])} | {p(r.get('holm_p', r['p_t']))} | {p(r['p_wilcoxon'])} |")


H = "| contrast | mean Δ | 95% CI (seeds) | d_z | Δ > 0 | p (t) | Holm p | p (Wilcoxon) |\n|---|---|---|---|---|---|---|---|"
L = ["# Stage 4 exploratory re-analysis (not pre-registered)", "",
     "Written on 2026-11-02 after the Stage 3 simulated peer review, after every result below had been seen. It reads only",
     "stored outputs and evaluates no model. **Every test here is exploratory.** The registered analyses in the step",
     "reports are unchanged and remain primary. Generated by `analyze_stage4.py`; numbers in `stage4_summary.json`.", "",
     "## E1. RQ2: timing as a factor (Step 8, B and async, 9 primary cities, frozen `no_o2` threshold)", ""]
for name, e in e1.items():
    if name == "scenes":
        continue
    L += [f"**{name}.** F1 by gap (1, 3, 6, 12, 18 months): " + ", ".join(f"{v:.3f}" for v in e["means"]) +
          f". Friedman p {p(e['friedman_p'])}; repeated-measures ANOVA F({e['rm_anova_df'][0]}, {e['rm_anova_df'][1]}) "
          f"= {e['rm_anova_F']:.2f}, p {p(e['rm_anova_p'])} (no sphericity correction). Per-seed slopes: mean "
          f"{f(e['slope_mean'], 4)} per year (t-test p {p(e['slope_p_t'])}); the largest slope is seed "
          f"{e['max_slope_seed']} ({f(max(e['slopes_per_year']), 4)}); without it the mean is "
          f"{f(e['slope_mean_without_max'], 4)} (p {p(e['slope_p_t_without_max'])}).", "", H]
    L += [row(r) for r in e["pairwise_1m"]] + [""]
L += ["Scenes (9 primary cities): season offset = days between the image's day of year and OSCD date 1's.", "",
      "| gap | mean season offset (days) | days after OSCD date 1 (min–max) | max cloud | images before date 1 |",
      "|---|---|---|---|---|"]
L += [f"| {r['gap']} | {r['season_offset_days_mean']:.0f} | {r['days_after_oscd_t1_min']}–{r['days_after_oscd_t1_max']}"
      f" | {r['cloud_max']:.3f} | {r['n_before_oscd_t1']} of 9 |" for r in scene_rows]
L += ["", "## E2. Variation across test cities against variation across training seeds", "",
      f"Per-city confusion counts at the same frozen thresholds. Bootstrap: {BOOT} replicates, cities resampled, then seeds,",
      "pooled F1 over the resampled cities. Leave-one-city-out (LOCO): seed-level paired t-test with one city removed.", "",
      "| contrast | cities | mean Δ | 95% CI (seeds only) | SD across seeds | SD across cities | cities with same sign "
      "| 95% CI (cities and seeds) | LOCO mean Δ range | LOCO p < 0.05 |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in e2:
    L.append(f"| {r['contrast']} | {r['n_cities']} | {f(r['seed_mean_diff'])} | [{f(r['seed_ci95'][0])}, "
             f"{f(r['seed_ci95'][1])}] | {r['sd_across_seeds']:.3f} | {r['sd_across_cities']:.3f} | "
             f"{r['cities_same_sign']}/{r['n_cities']} | [{f(r['boot_ci95'][0])}, {f(r['boot_ci95'][1])}] | "
             f"{f(r['loco_min'])} to {f(r['loco_max'])} | {r['loco_p_below_05']}/{r['n_cities']} |")
L += ["", "The SD columns are not on a common scale: SD across seeds is of the pixel-pooled Δ, SD across cities is of",
      "unweighted per-city Δ. The comparable quantities are the standard errors of the same pooled estimate under each",
      "resampling scheme (added after a same-assistant check, 2026-11-02). The jackknife-t interval combines the city",
      "jackknife SE and the seed SE; it is a rough sensitivity check, not a validated method.", "",
      "| contrast | SE, seeds resampled | SE, cities resampled (bootstrap) | SE, cities (jackknife) | jackknife-t 95% CI "
      "| unweighted mean of per-city Δ |", "|---|---|---|---|---|---|"]
for r in e2:
    L.append(f"| {r['contrast']} | {r['se_seeds']:.3f} | {r['se_cities_boot']:.3f} | {r['se_cities_jackknife']:.3f} | "
             f"[{f(r['jackknife_t_ci95'][0])}, {f(r['jackknife_t_ci95'][1])}] | {f(r['unweighted_city_mean_diff'])} |")
L += ["", "Per-city mean Δ (mean over seeds), pooled mean Δ with that city left out, and the city's share of change pixels:", ""]
for r in e2:
    L.append(f"- {r['contrast']}: " + ", ".join(
        f"{c} {f(v)} (without: {f(r['loco_drop_mean_diff'][c])}; {100 * r['change_pixel_share'][c]:.0f}%)"
        for c, v in r["per_city_mean_diff"].items()))
L += ["", "## E3. RQ1: the same model with and without the historical image (pooled over 10 test cities)", "",
      "The own-threshold contrast for B is the Step 6 registered secondary contrast; the rest are exploratory.", "", H]
L += [row(r) for r in e3] + ["", "Threshold rule for the registered B − SAR-only contrast:", "", H]
L += [row(r) for r in e3_thr]
L += ["", "No stored output gives B without the historical image at the frozen clean threshold, so the within-B "
      "contrast cannot be computed at that threshold without a new evaluation.", "",
      "## E4. RQ4 for the asynchronous arms (Holm over 8)", "",
      "The asynchronous arms have no post-change optical image; the baselines have full input, which the asynchronous",
      "setting by definition lacks. The comparison therefore shows the cost of losing that image.", "", H]
L += [row(r) for r in e4]
L += ["", "## E5. SAR-only fallback rule against B", "",
      "Rule: A when the post-change optical image is available and clear, the SAR-only U-Net otherwise. The SAR-only "
      "score does not depend on the optical image, so it is the same for every cloud level. Composite clouds (Step 5, "
      "B at its frozen clean threshold).", "", H]
L += [row(r) for r in e5] + ["", "Full input, B against A on other measures:", "", H] + [row(r) for r in e5_pr]
L += ["", "## E6. RQ5: recall next to false-alarm rate (B at `no_o2` and SAR-only, Step 8, 9 primary cities)", "",
      "| class | B recall | SAR-only recall | B false-alarm rate | SAR-only false-alarm rate |", "|---|---|---|---|---|"]
L += [f"| {c} | {v['B_recall']:.3f} | {v['SAR_recall']:.3f} | {v['B_far']:.4f} | {v['SAR_far']:.4f} |"
      for c, v in e6.items()]
L += ["", "## E7. Selected thresholds at the top of the grid (seeds × {full, no_o2, no_opt})", "",
      "| arm | thresholds | at 0.99 or above | maximum |", "|---|---|---|---|"]
L += [f"| {a} | {v['n']} | {v['at_0.99_or_above']} | {v['max']:.2f} |" for a, v in e7.items()]
L += ["", "## E8. Sentinel-1 scenes per 3-month window, test cities (`feasibility/inventory.json`)", "",
      "Scene counts from the Earth Engine query made for the feasibility check, over the same windows as the exports; the",
      f"export logs themselves record no counts. Over {e8_stats['windows']} city-orbit windows, the median VV count is "
      f"{e8_stats['median_vv_t1']:.0f} at date 1 and {e8_stats['median_vv_t2']:.0f} at date 2; date 1 has fewer scenes in "
      f"{e8_stats['windows_t1_fewer']} of {e8_stats['windows']}.", "",
      "| city | orbit | VV date 1 | VV date 2 | VH date 1 | VH date 2 |", "|---|---|---|---|---|---|"]
L += [f"| {r['city']} | {r['orbit']} | {r['vv_t1']} | {r['vv_t2']} | {r['vh_t1']} | {r['vh_t2']} |" for r in e8]
(OUT / "STAGE4_EXPLORATORY_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote experiments/robust/stage4/STAGE4_EXPLORATORY_REPORT.md and stage4_summary.json")
