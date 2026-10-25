"""Re-create the Brasilia qualitative images (Figure 8 inputs) from the seed-42 checkpoints.

The SAR-only and optical-only training scripts overwrite figures/<model>/brasilia_*.png on
every run, so after multi-seed training those files belong to the last seed. This restores
the seed-42 versions that the paper's Figure 8 caption refers to.

Usage:
    .venv/Scripts/python.exe regenerate_qualitative_seed42.py
"""

import torch

from src.datasets.oscd_dataset import OSCDDataset
from src.models.unet import UNet
from src.training import train_optical_baseline, train_sar_baseline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for module, mode, channels, name in (
    (train_sar_baseline, "sar", 2, "sar_baseline"),
    (train_optical_baseline, "optical", train_optical_baseline.INPUT_CHANNELS, "optical_baseline"),
):
    saved = torch.load(f"checkpoints/{name}_seed42/best.pt", map_location="cpu", weights_only=False)
    assert saved["config"]["seed"] == 42, f"{name} checkpoint is not seed 42"
    model = UNet(in_channels=channels, out_channels=1)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device).eval()
    module.save_figures(model, OSCDDataset(split="test", mode=mode), "brasilia", device)
    print(f"{name}: seed-42 Brasilia figures written")
