"""Validation-only threshold calibration for the two primary OSCD CNN baselines.

Loads existing checkpoints, performs one inference pass per city in validation
and test, and writes only into this comparison experiment directory.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.oscd_dataset import ORBITS, OSCDDataset
from src.evaluation.metrics import rates_from_counts
from src.models.dual_stream_unet import DualStreamUNet
from src.models.dual_stream_unet_clean import DualStreamUNetClean
from src.training.train_asynchronous_dual_stream_clean import (
    CleanAsynchronousOSCDDataset, sample_at_orbit_clean_async,
)
from src.training.train_fusion_baseline import fuse_dates
from src.training.train_sar_baseline import sample_at_orbit, split_train_cities

OUT = Path(__file__).resolve().parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_GRID = np.arange(0.05, 0.951, 0.05)
FINE_GRID = np.arange(0.30, 0.70001, 0.0125)
THRESHOLDS = sorted(set(np.round(np.r_[BASE_GRID, FINE_GRID, 0.5], 6).tolist()))


def pooled_counts(probability, label, threshold):
    pred = probability >= threshold
    target = label >= 0.5
    return {
        'tp': int(np.logical_and(pred, target).sum()),
        'fp': int(np.logical_and(pred, ~target).sum()),
        'tn': int(np.logical_and(~pred, ~target).sum()),
        'fn': int(np.logical_and(~pred, target).sum()),
    }


def collect_cnn(model, kind, split, cities):
    if kind == 'clean_async':
        ds = CleanAsynchronousOSCDDataset(split=split)
    else:
        ds = OSCDDataset(split=split, mode='fusion')
    prob_list, label_list = [], []
    model.eval()
    with torch.inference_mode():
        for city in cities:
            idx = ds.cities.index(city)
            orbit = int(ORBITS[city][0])
            sample = (sample_at_orbit_clean_async(ds, idx, orbit)
                      if kind == 'clean_async' else sample_at_orbit(ds, idx, orbit))
            image = (sample['image'] if kind == 'clean_async'
                     else fuse_dates(sample['t1_img'], sample['t2_img']))
            logits = model(image.unsqueeze(0).to(DEVICE))
            if logits.shape[-2:] != sample['label'].shape[-2:]:
                raise RuntimeError(f'{city}: prediction/label shape mismatch')
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            labels = sample['label'].squeeze().cpu().numpy()
            if not np.isfinite(probs).all():
                raise RuntimeError(f'{city}: non-finite probabilities')
            prob_list.append(probs.reshape(-1))
            label_list.append(labels.reshape(-1) >= 0.5)
            del logits, image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return np.concatenate(prob_list), np.concatenate(label_list)


def main():
    official_train = OSCDDataset(split='train', mode='fusion').cities
    train_cities, val_cities = split_train_cities(official_train)
    test_cities = OSCDDataset(split='test', mode='fusion').cities
    if len(train_cities) != 10 or len(val_cities) != 4 or len(test_cities) != 10:
        raise RuntimeError('Unexpected OSCD split cardinality')

    models = {
        'clean_async_cnn': (DualStreamUNetClean(base_channels=64),
                            ROOT/'checkpoints/asynchronous_dual_stream_clean/best.pt', 'clean_async'),
        'sync_dual_stream_cnn': (DualStreamUNet(base_channels=64),
                                 ROOT/'checkpoints/dual_stream_baseline/best.pt', 'sync'),
    }
    sweep_rows, frozen_test_rows, selected = [], [], {}
    for name, (model, checkpoint, kind) in models.items():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model_state_dict'])
        model = model.to(DEVICE)
        vp, vy = collect_cnn(model, kind, 'train', val_cities)
        val_scores = []
        for tau in THRESHOLDS:
            counts = pooled_counts(vp, vy, tau)
            metrics = rates_from_counts(counts)
            row = {'model': name, 'split': 'validation', 'threshold': tau,
                   **counts, **metrics, 'pixels': int(vy.size),
                   'selection_role': 'validation_only'}
            sweep_rows.append(row)
            val_scores.append(row)
        # Stable tie-break: choose the higher threshold among equal maximum F1s.
        best = max(val_scores, key=lambda r: (r['f1'] if r['f1'] is not None else -1, r['threshold']))
        selected[name] = {'threshold': best['threshold'], 'validation_metrics': best,
                          'checkpoint': str(checkpoint.relative_to(ROOT)),
                          'checkpoint_epoch': saved.get('epoch'),
                          'checkpoint_validation_f1': saved.get('validation_f1')}
        # Freeze validation choice before any test inference.
        tp, ty = collect_cnn(model, kind, 'test', test_cities)
        for tau, role in [(0.5, 'common_threshold_0.5'),
                          (float(best['threshold']), 'validation_selected_threshold')]:
            counts = pooled_counts(tp, ty, tau)
            frozen_test_rows.append({'model': name, 'split': 'test', 'threshold': tau,
                                     'threshold_role': role, **counts,
                                     **rates_from_counts(counts), 'pixels': int(ty.size),
                                     'test_used_for_selection': False,
                                     'test_cities': ';'.join(test_cities)})
        model.to('cpu')
        del model

    for filename, rows in [('cnn_validation_threshold_sweeps.csv', sweep_rows),
                           ('cnn_frozen_test_thresholds.csv', frozen_test_rows)]:
        with (OUT/filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    payload = {'selection_metric': 'pooled validation F1',
               'grid': THRESHOLDS,
               'tie_break': 'higher threshold when F1 ties',
               'split': {'train': train_cities, 'validation': val_cities, 'test': test_cities},
               'test_used_for_selection': False, 'selected': selected}
    (OUT/'cnn_validation_selected_thresholds.json').write_text(
        json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'device': str(DEVICE), 'validation_sites': val_cities,
                      'test_sites': test_cities, 'selected': selected}, indent=2))


if __name__ == '__main__':
    main()
