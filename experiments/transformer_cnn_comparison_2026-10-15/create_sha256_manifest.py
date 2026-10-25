"""Hash this comparison package and the original artifacts it references."""
import csv, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def resolve(raw):
    if not raw:return None
    p=Path(raw)
    if not p.is_absolute():p=ROOT/p
    return p.resolve()

generated={}
for p in sorted(OUT.rglob('*')):
    if not p.is_file() or p.name=='sha256_manifest.json' or '__pycache__' in p.parts: continue
    generated[str(p.relative_to(OUT))]=digest(p)
references={'checkpoint_references':set(),'metric_and_run_references':set(),'configuration_references':set()}
for filename in ['BEFORE_AFTER_RESULTS.csv','CNN_TRANSFORMER_COMPARISON.csv','TRAINING_RUNS.csv']:
    with (OUT/filename).open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            for field,group in [('checkpoint_path','checkpoint_references'),
                                ('metrics_source','metric_and_run_references'),
                                ('metrics_path','metric_and_run_references'),
                                ('artifact_path','metric_and_run_references'),
                                ('config_path','configuration_references')]:
                if row.get(field): references[group].add(row[field])
resolved={}
for group,values in references.items():
    resolved[group]={}
    for raw in sorted(values):
        p=resolve(raw)
        if p and p.is_file():
            key=str(p.relative_to(ROOT)) if ROOT in p.parents else str(p)
            resolved[group][key]=digest(p)
source_files=[
 'src/models/dual_stream_unet.py','src/models/dual_stream_unet_clean.py',
 'src/models/transformer_self_attention.py','src/models/transformer_cross_attention.py','src/models/unet.py',
 'src/datasets/oscd_dataset.py','src/datasets/scenario2_dataset.py','src/evaluation/metrics.py',
 'src/training/train_sar_baseline.py','src/training/train_fusion_baseline.py',
 'src/training/train_dual_stream_baseline.py','src/training/train_asynchronous_dual_stream_clean.py',
 'src/training/train_scenario2_cross_attention.py',
 'experiments/transformer_tuning_2026-10-15/tuning_pipeline.py',
 'experiments/transformer_tuning_2026-10-15/selected_tuning_configs.json',
 'experiments/transformer_tuning_2026-10-15/final_test_comparison.csv',
 'experiments/research_status/artifact_integrity.json',
 'experiments/scenario_2/configs/split_definition.json',
 'experiments/scenario_2/threshold_calibration/threshold_calibration_results.json']
sources={name:digest(ROOT/name) for name in source_files if (ROOT/name).is_file()}
manifest={'algorithm':'SHA-256','experiment_directory':'experiments/transformer_cnn_comparison_2026-10-15',
          'generated_files':generated,**resolved,'source_and_split_references':sources}
(OUT/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:len(v) for k,v in manifest.items() if isinstance(v,dict)},indent=2))
