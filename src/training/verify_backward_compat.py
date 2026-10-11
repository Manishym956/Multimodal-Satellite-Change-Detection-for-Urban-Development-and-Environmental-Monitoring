"""Backward-compatibility and full verification test.

Checks:
  1. Scenario 1 model defaults unchanged (param counts, forward shapes)
  2. Scenario 1 checkpoints still load cleanly
  3. Scenario 2 channel overrides work
  4. Dataset loader: async withholds S2 T2, sync includes it
  5. Dataset loader: split matches split_definition.json
  6. Label is binary and non-empty
  7. NaN handling in S1
  8. pos_weight sanity from training sites
  9. Scenario 2 output dirs do not collide with Scenario 1

Run from project root:
    .venv\\Scripts\\python.exe src\\training\\verify_backward_compat.py
"""

import sys
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

failures = []

def ok(msg):  print(f"  {PASS} {msg}")
def fail(msg): print(f"  {FAIL} {msg}"); failures.append(msg)
def info(msg): print(f"  {INFO} {msg}")
def header(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")


# ─────────────────────────────────────────────────────────────────
# 1. Scenario 1 model defaults
# ─────────────────────────────────────────────────────────────────
header("1. Scenario 1 model defaults — param counts and forward shapes")

from src.models.dual_stream_unet import DualStreamUNet, count_parameters
from src.models.dual_stream_unet_clean import DualStreamUNetClean

S1_CLEAN_PARAMS  = 27_276_609
S1_DUAL_PARAMS   = 27_284_097

m_clean = DualStreamUNetClean()
p_clean = count_parameters(m_clean)
info(f"DualStreamUNetClean default params: {p_clean:,}")
if p_clean == S1_CLEAN_PARAMS:
    ok("DualStreamUNetClean param count unchanged")
else:
    fail(f"DualStreamUNetClean params changed: expected {S1_CLEAN_PARAMS:,}, got {p_clean:,}")

with torch.no_grad():
    out = m_clean(torch.randn(1, 15, 32, 32))
if out.shape == (1, 1, 32, 32):
    ok("DualStreamUNetClean 15-ch forward: (1,1,32,32)")
else:
    fail(f"DualStreamUNetClean 15-ch forward shape wrong: {out.shape}")

m_dual = DualStreamUNet()
p_dual = count_parameters(m_dual)
info(f"DualStreamUNet default params: {p_dual:,}")
if p_dual == S1_DUAL_PARAMS:
    ok("DualStreamUNet param count unchanged")
else:
    fail(f"DualStreamUNet params changed: expected {S1_DUAL_PARAMS:,}, got {p_dual:,}")

with torch.no_grad():
    out = m_dual(torch.randn(1, 28, 32, 32))
if out.shape == (1, 1, 32, 32):
    ok("DualStreamUNet 28-ch forward: (1,1,32,32)")
else:
    fail(f"DualStreamUNet 28-ch forward shape wrong: {out.shape}")


# ─────────────────────────────────────────────────────────────────
# 2. Scenario 1 checkpoints reload
# ─────────────────────────────────────────────────────────────────
header("2. Scenario 1 checkpoints reload")

s1_ckpts = {
    "asynchronous_dual_stream_clean": (DualStreamUNetClean, {}),
    "dual_stream_baseline":           (DualStreamUNet, {}),
    "transformer_cross_attention":    (None, None),   # checked separately below
}

for name, (cls, kwargs) in list(s1_ckpts.items()):
    ckpt_path = PROJECT_ROOT / "checkpoints" / name / "best.pt"
    if not ckpt_path.exists():
        info(f"{name}: checkpoint not found (skip)")
        continue
    if cls is None:
        continue
    try:
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = cls(**kwargs)
        model.load_state_dict(saved["model_state_dict"])
        epoch = saved.get("epoch", "?")
        vf1   = saved.get("validation_f1", "?")
        ok(f"{name}: loaded epoch={epoch}, val_f1={vf1:.4f}")
    except Exception as e:
        fail(f"{name}: reload failed — {e}")

# Transformer cross-attention S1 checkpoint
from src.models.transformer_cross_attention import TransformerCrossAttention
xattn_ckpt = PROJECT_ROOT / "checkpoints" / "transformer_cross_attention" / "best.pt"
if xattn_ckpt.exists():
    try:
        saved = torch.load(xattn_ckpt, map_location="cpu", weights_only=False)
        m = TransformerCrossAttention(sar_channels=2, opt_channels=13)
        m.load_state_dict(saved["model_state_dict"])
        ok(f"transformer_cross_attention: loaded epoch={saved.get('epoch','?')}")
    except Exception as e:
        fail(f"transformer_cross_attention reload failed: {e}")


# ─────────────────────────────────────────────────────────────────
# 3. Scenario 2 channel overrides
# ─────────────────────────────────────────────────────────────────
header("3. Scenario 2 channel overrides (4 SAR + 10/20 optical)")

m_async = DualStreamUNetClean(sar_channels=4, opt_channels=10)
with torch.no_grad():
    o = m_async(torch.randn(1, 14, 32, 32))
if o.shape == (1, 1, 32, 32):
    ok(f"DualStreamUNetClean S2 async (14ch): {o.shape}  params={count_parameters(m_async):,}")
else:
    fail(f"DualStreamUNetClean S2 async shape wrong: {o.shape}")

m_sync = DualStreamUNet(sar_channels=4, opt_channels=20)
with torch.no_grad():
    o = m_sync(torch.randn(1, 24, 32, 32))
if o.shape == (1, 1, 32, 32):
    ok(f"DualStreamUNet S2 sync (24ch): {o.shape}  params={count_parameters(m_sync):,}")
else:
    fail(f"DualStreamUNet S2 sync shape wrong: {o.shape}")

from src.models.transformer_cross_attention import count_parameters as cp_xattn
m_xattn2 = TransformerCrossAttention(sar_channels=4, opt_channels=10)
with torch.no_grad():
    o = m_xattn2(torch.randn(1, 14, 32, 32))
if o.shape == (1, 1, 32, 32):
    ok(f"TransformerCrossAttention S2 (14ch): {o.shape}  params={cp_xattn(m_xattn2):,}")
else:
    fail(f"TransformerCrossAttention S2 shape wrong: {o.shape}")


# ─────────────────────────────────────────────────────────────────
# 4-8. Dataset loader checks
# ─────────────────────────────────────────────────────────────────
header("4–8. Scenario 2 dataset loader checks")

from src.datasets.scenario2_dataset import (
    Scenario2Dataset, load_split,
    ASYNC_CHANNELS, SYNC_CHANNELS, SAR_CHANNELS, OPT_CHANNELS,
    T1_YEAR, T1_MONTH, T2_YEAR, T2_MONTH, LABEL_THRESHOLD,
)
import json

split = load_split()
info(f"Split: {len(split['train'])} train / {len(split['validation'])} val / {len(split['test'])} test")

# Compare against locked split_definition.json
split_file = PROJECT_ROOT / "experiments" / "scenario_2" / "configs" / "split_definition.json"
locked = json.loads(split_file.read_text())
if (sorted(split["train"]) == sorted(locked["train"]) and
        sorted(split["validation"]) == sorted(locked["validation"]) and
        sorted(split["test"]) == sorted(locked["test"])):
    ok("Dataset split matches split_definition.json exactly")
else:
    fail("Dataset split does NOT match split_definition.json")

# Load one training sample in async mode
ds_async = Scenario2Dataset(split="train", mode="async")
sample_a = ds_async[0]
img_a, lbl_a = sample_a["image"], sample_a["label"]

if img_a.shape[0] == ASYNC_CHANNELS:
    ok(f"Async sample channels: {img_a.shape[0]} (expected {ASYNC_CHANNELS})")
else:
    fail(f"Async sample channels: got {img_a.shape[0]}, expected {ASYNC_CHANNELS}")

# S2 T2 NOT present in async (only 14 ch leaves no room)
sar_range_ok = (img_a[:SAR_CHANNELS].min() >= 0.0 and img_a[:SAR_CHANNELS].max() <= 1.0)
s2_range_ok  = (img_a[SAR_CHANNELS:].min() >= 0.0 and img_a[SAR_CHANNELS:].max() <= 1.0)
if sar_range_ok:
    ok(f"SAR channels in [0,1]: [{img_a[:4].min():.4f}, {img_a[:4].max():.4f}]")
else:
    fail(f"SAR channels out of range: [{img_a[:4].min():.4f}, {img_a[:4].max():.4f}]")
if s2_range_ok:
    ok(f"S2 T1 channels in [0,1]: [{img_a[4:].min():.4f}, {img_a[4:].max():.4f}]")
else:
    fail(f"S2 T1 channels out of range: [{img_a[4:].min():.4f}, {img_a[4:].max():.4f}]")

# NaN check (S1 NaN→0 fill should eliminate all NaN)
if not torch.isnan(img_a).any():
    ok("No NaN in async sample after fill")
else:
    fail("NaN found in async sample — NaN fill not working")

# Label is binary
unique_vals = torch.unique(lbl_a).tolist()
if set(unique_vals).issubset({0.0, 1.0}):
    ok(f"Label is binary: unique values = {unique_vals}")
else:
    fail(f"Label not binary: unique values = {unique_vals}")

change_frac = lbl_a.mean().item() * 100
info(f"Label change fraction: {change_frac:.3f}% in site '{sample_a['site']}'")
if change_frac > 0:
    ok("Label has positive-class pixels (non-zero change)")
else:
    fail("Label has zero positive-class pixels")

# Spatial alignment: label shape matches image H,W
if lbl_a.shape[1:] == img_a.shape[1:]:
    ok(f"Label spatially aligned with image: {tuple(lbl_a.shape)}")
else:
    fail(f"Label shape {lbl_a.shape[1:]} mismatches image {img_a.shape[1:]}")

# Load sync sample — must have SYNC_CHANNELS
ds_sync = Scenario2Dataset(split="train", mode="sync")
sample_s = ds_sync[0]
img_s = sample_s["image"]
if img_s.shape[0] == SYNC_CHANNELS:
    ok(f"Sync sample channels: {img_s.shape[0]} (expected {SYNC_CHANNELS})")
else:
    fail(f"Sync sample channels: got {img_s.shape[0]}, expected {SYNC_CHANNELS}")

# S2 T2 range in sync (channels 14-23)
s2t2 = img_s[14:]
if s2t2.shape[0] == OPT_CHANNELS and s2t2.min() >= 0.0 and s2t2.max() <= 1.0:
    ok(f"Sync S2 T2 channels present and in [0,1]: [{s2t2.min():.4f}, {s2t2.max():.4f}]")
else:
    fail(f"Sync S2 T2 channels problem: shape={s2t2.shape} range=[{s2t2.min():.4f},{s2t2.max():.4f}]")

# Same label for async and sync (same site, same T1/T2 endpoints)
if torch.equal(sample_a["label"], sample_s["label"]):
    ok("Async and sync labels identical for same site")
else:
    fail("Async and sync labels differ — should be identical")


# ─────────────────────────────────────────────────────────────────
# 9. pos_weight sanity
# ─────────────────────────────────────────────────────────────────
header("9. pos_weight sanity from locked split")

train_sites = split["train"]
total_change = 0; total_pixels = 0
for site_id in train_sites:
    sample = Scenario2Dataset(split="train", mode="async")[
        Scenario2Dataset(split="train", mode="async").sites.index(site_id)
    ]
    lbl = sample["label"]
    total_change  += int(lbl.sum().item())
    total_pixels  += int(lbl.numel())

no_change = total_pixels - total_change
pw = no_change / total_change if total_change > 0 else float("inf")
info(f"Training pixels: {total_pixels:,}  change={total_change:,}  no-change={no_change:,}")
info(f"Computed pos_weight: {pw:.1f}")

if abs(pw - 93.6) < 5.0:
    ok(f"pos_weight {pw:.1f} close to expected 93.6 (±5)")
else:
    fail(f"pos_weight {pw:.1f} deviates unexpectedly from 93.6")


# ─────────────────────────────────────────────────────────────────
# 10. Output dirs don't collide with Scenario 1
# ─────────────────────────────────────────────────────────────────
header("10. Scenario 2 output paths do not collide with Scenario 1")

s1_ckpt_dirs = {
    "checkpoints/sar_baseline",
    "checkpoints/optical_baseline",
    "checkpoints/fusion_baseline",
    "checkpoints/dual_stream_baseline",
    "checkpoints/asynchronous_dual_stream",
    "checkpoints/asynchronous_dual_stream_clean",
    "checkpoints/transformer_self_attention",
    "checkpoints/transformer_cross_attention",
}
s2_ckpt_dirs = {
    "checkpoints/scenario_2/async_dual_stream",
    "checkpoints/scenario_2/sync_dual_stream",
    "checkpoints/scenario_2/cross_attention",
}
collision = s1_ckpt_dirs & s2_ckpt_dirs
if not collision:
    ok("No checkpoint path collision between Scenario 1 and Scenario 2")
else:
    fail(f"Checkpoint path collision: {collision}")

s1_exp_dirs = {p.name for p in (PROJECT_ROOT / "experiments").iterdir()
               if p.is_dir() and p.name != "scenario_2"}
s2_exp_dirs = {"async_dual_stream", "sync_dual_stream", "cross_attention"}
collision_e = s1_exp_dirs & s2_exp_dirs
if not collision_e:
    ok("No experiment directory collision between Scenario 1 and Scenario 2")
else:
    fail(f"Experiment directory collision: {collision_e}")


# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
header("SUMMARY")
if failures:
    print(f"\n  {FAIL} {len(failures)} test(s) FAILED:")
    for m in failures:
        print(f"    • {m}")
    sys.exit(1)
else:
    print(f"  {PASS} All backward-compatibility and loader tests PASSED")
    sys.exit(0)
