import sys
import numpy as np
import torch
import torch.nn as nn
from src.datasets.scenario2_dataset import Scenario2Dataset, SAR_CHANNELS, OPT_CHANNELS
from src.models.dual_stream_unet import DualStreamUNet
from src.models.masked_loss import MaskedBCEWithLogitsLoss
from src.training.train_scenario2_sync_multiseed import generate_crop_schedule, ScheduledCropScenario2
from torch.utils.data import DataLoader

print("=== PRE-FLIGHT TEST SUITE ===")

# Test 1: Model Initialization Identity
for seed in [43, 44]:
    torch.manual_seed(seed)
    m1 = DualStreamUNet(base_channels=64, sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS*2)
    torch.manual_seed(seed)
    m2 = DualStreamUNet(base_channels=64, sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS*2)
    
    diffs = []
    for k in m1.state_dict():
        d = (m1.state_dict()[k] - m2.state_dict()[k]).abs().max().item()
        diffs.append(d)
    max_d = max(diffs)
    assert max_d == 0.0, f"Seed {seed}: models do not have identical weights!"
    print(f"Test 1 [Seed {seed}]: Identical initialization across pair: PASS (max diff = {max_d})")

# Test 2: Seeds 43 vs 44 Different Initializations
torch.manual_seed(43)
m_s43 = DualStreamUNet(base_channels=64, sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS*2)
torch.manual_seed(44)
m_s44 = DualStreamUNet(base_channels=64, sar_channels=SAR_CHANNELS, opt_channels=OPT_CHANNELS*2)
diff_s43_s44 = (m_s43.sar_encoder.inc.block[0].weight - m_s44.sar_encoder.inc.block[0].weight).abs().max().item()
assert diff_s43_s44 > 0.0, "Seeds 43 and 44 unexpectedly produced identical weights!"
print(f"Test 2: Distinct initializations between Seed 43 and Seed 44: PASS (diff = {diff_s43_s44:.4f})")

# Test 3: Real Dataset Crop Schedule Validity & Coordinate Bounds
train_ds = Scenario2Dataset(split="train", mode="sync", return_mask=True)
site_dims = {s: train_ds._arrays(s)["s1_t1"].shape[:2] for s in train_ds.sites}

for seed in [43, 44]:
    sch = generate_crop_schedule(train_ds.sites, site_dims, crops_per_site=64, crop_size=32, epochs=50, seed=seed)
    for ep, crops in sch.items():
        assert len(crops) == 20 * 64
        for c in crops:
            h, w = site_dims[c["site"]]
            assert 0 <= c["top"] <= h - 32, f"Invalid top coord {c['top']} on site {c['site']} ({h},{w})"
            assert 0 <= c["left"] <= w - 32, f"Invalid left coord {c['left']} on site {c['site']} ({h},{w})"
print("Test 3: Crop coordinates valid for all 20 real training site geometries: PASS")

# Test 4: Identical Crop Schedule within Pair, Distinct Across Seeds
sch_43_a = generate_crop_schedule(train_ds.sites, site_dims, crops_per_site=64, crop_size=32, epochs=50, seed=43)
sch_43_b = generate_crop_schedule(train_ds.sites, site_dims, crops_per_site=64, crop_size=32, epochs=50, seed=43)
sch_44   = generate_crop_schedule(train_ds.sites, site_dims, crops_per_site=64, crop_size=32, epochs=50, seed=44)

assert sch_43_a == sch_43_b, "Seed 43 schedules do not match!"
assert sch_43_a != sch_44, "Seed 43 and 44 schedules are not distinct!"
print("Test 4: Schedule identical for paired conditions, distinct for seeds: PASS")

# Test 5: End-to-end DataLoader batch consumption & tensor identity
c_ds1 = ScheduledCropScenario2(train_ds, crop_size=32)
c_ds2 = ScheduledCropScenario2(train_ds, crop_size=32)

c_ds1.set_epoch_schedule(sch_43_a[1])
c_ds2.set_epoch_schedule(sch_43_b[1])

g1 = torch.Generator().manual_seed(43 * 1000 + 1)
g2 = torch.Generator().manual_seed(43 * 1000 + 1)

loader1 = DataLoader(c_ds1, batch_size=64, shuffle=True, generator=g1)
loader2 = DataLoader(c_ds2, batch_size=64, shuffle=True, generator=g2)

b1 = next(iter(loader1))
b2 = next(iter(loader2))

diff_img = (b1["image"] - b2["image"]).abs().max().item()
diff_lbl = (b1["label"] - b2["label"]).abs().max().item()
diff_msk = (b1["valid_mask"] - b2["valid_mask"]).abs().max().item()
assert diff_img == 0.0 and diff_lbl == 0.0 and diff_msk == 0.0, "DataLoader batches differ between paired conditions!"
print(f"Test 5: DataLoader batches identical between paired conditions: PASS (diffs: img={diff_img}, lbl={diff_lbl}, msk={diff_msk})")

# Test 6: Mask numerical property check
pos_weight = torch.tensor([94.0])
crit_masked = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
crit_std = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

logits = torch.randn(4, 1, 32, 32)
targets = torch.randint(0, 2, (4, 1, 32, 32)).float()
ones_mask = torch.ones(4, 1, 32, 32)

l_mask = crit_masked(logits, targets, ones_mask)
l_std  = crit_std(logits, targets)
diff_loss = abs(l_mask.item() - l_std.item())
assert diff_loss < 1e-6, f"Masked loss on all-ones does not match standard BCE (diff={diff_loss})"
print(f"Test 6: Masked BCE numerical equivalence to standard BCE: PASS (diff={diff_loss:.2e})")

print("=== ALL 6 PRE-FLIGHT TESTS PASSED CLEANLY ===")
