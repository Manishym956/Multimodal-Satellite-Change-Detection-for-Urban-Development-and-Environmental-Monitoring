"""Machine checks for this additive Transformer/CNN comparison package."""
import csv, hashlib, json, math, statistics, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
EXPECTED_SPLIT='5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27'
def loadj(p): return json.loads(p.read_text(encoding='utf-8'))
def loadc(p):
    with p.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def near(a,b,tol=1e-9): return abs(float(a)-float(b))<=tol

checks={}; failures=[]
def check(name,passed,detail=None):
    checks[name]={'passed':bool(passed),'detail':detail}
    if not passed: failures.append(name)

# Locked S2 split and all key original baseline artifact hashes.
inventory=loadj(ROOT/'experiments/research_status/artifact_integrity.json')
split=inventory['locked_split']; actual_split=sha(ROOT/split['path'])
check('locked_scenario2_split_hash',actual_split==EXPECTED_SPLIT,{'expected':EXPECTED_SPLIT,'actual':actual_split})
integrity={}
for category,keys in [('scenario_1_artifacts',['sar_baseline','optical_baseline','fusion_baseline','dual_stream_baseline','asynchronous_dual_stream','asynchronous_dual_stream_clean','transformer_self_attention','transformer_cross_attention']),
                     ('scenario_2_artifacts',['async_dual_stream','sync_dual_stream','cross_attention'])]:
    for key in keys:
        entry=inventory[category][key]
        for part in ['checkpoint','metrics_json','training_history_csv','per_city_metrics_csv']:
            if part not in entry: continue
            item=entry[part]; p=ROOT/item['path']; got=sha(p) if p.exists() else None
            integrity[item['path']]={'expected_sha256':item['sha256'],'actual_sha256':got,'unchanged':got==item['sha256']}
check('original_checkpoint_and_metric_artifacts_unchanged',all(v['unchanged'] for v in integrity.values()),integrity)

before=loadc(OUT/'BEFORE_AFTER_RESULTS.csv'); comp=loadc(OUT/'CNN_TRANSFORMER_COMPARISON.csv')
agg=loadc(OUT/'AGGREGATE_RESULTS.csv'); reg=loadc(OUT/'TRAINING_RUNS.csv')
cnn_test=loadc(OUT/'cnn_frozen_test_thresholds.csv'); cnn_sweeps=loadc(OUT/'cnn_validation_threshold_sweeps.csv')
check('expected_output_row_counts',len(before)==24 and len(comp)==24 and len(agg)==19 and len(reg)==32,
      {'before_after':len(before),'comparison':len(comp),'aggregates':len(agg),'registry':len(reg)})

# Metric identities and totals for every row with full confusion counts.
metric_rows=[]
for table_name,rows in [('before_after',before),('comparison',comp),('cnn_frozen_test',cnn_test)]:
    for i,r in enumerate(rows):
        if any(r.get(k) in (None,'','N/A') for k in ['tp','fp','tn','fn']): continue
        tp,fp,tn,fn=(int(float(r[k])) for k in ['tp','fp','tn','fn'])
        p=tp/(tp+fp) if tp+fp else None; rec=tp/(tp+fn) if tp+fn else None
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        iou=tp/(tp+fp+fn) if tp+fp+fn else None
        vals=[(r.get('precision'),p),(r.get('recall'),rec),(r.get('f1'),f1),(r.get('iou'),iou)]
        if all(a not in (None,'','N/A') and b is not None and near(a,b) for a,b in vals): metric_rows.append(f'{table_name}:{i}')
        else: failures.append(f'metric_formula:{table_name}:{i}')
check('metric_formulas_match_confusion_counts',not any(x.startswith('metric_formula:') for x in failures),{'verified_rows':len(metric_rows)})

# Test split and test-selection discipline.
expected_oscd_test=['brasilia','montpellier','norcia','rio','saclay_w','valencia','dubai','lasvegas','milano','chongqing']
all_cities_ok=True
for r in cnn_test:
    all_cities_ok &= r['test_cities'].split(';')==expected_oscd_test
check('oscd_test_city_membership',all_cities_ok,{'expected':expected_oscd_test})
no_test_select=all(str(r.get('test_used_for_selection','')).lower()=='false' for r in before+comp+cnn_test)
check('test_not_used_for_selection_claims',no_test_select,{'new_cnn_threshold_sweeps':'validation-only; test inference after frozen tau'})

# Verify each selected CNN threshold is the maximum validation F1 on the saved sweep.
selected=loadj(OUT/'cnn_validation_selected_thresholds.json')['selected']
sel_ok=True
for model,info in selected.items():
    rows=[r for r in cnn_sweeps if r['model']==model]
    best=max(rows,key=lambda r:(float(r['f1']) if r['f1'] else -1,float(r['threshold'])))
    sel_ok &= near(best['threshold'],info['threshold']) and near(best['f1'],info['validation_metrics']['f1'])
check('cnn_thresholds_match_validation_maximum',sel_ok,{k:v['threshold'] for k,v in selected.items()})

# Ensure invalidated/failed Transformer runs are retained but not used for tuned aggregates.
bad=[r['run_id'] for r in reg if 'invalidated' in r.get('status','') or 'failed_' in r.get('status','')]
used={r['run_id'] for r in before if r.get('cohort')=='tuned'}
bad_excluded=not bool(used.intersection(bad)) and len(bad)>=5
check('failed_and_invalidated_runs_excluded',bad_excluded,{'failed_or_invalidated':bad,'used_tuned_runs':sorted(used)})
unique_before={(r['dataset'],r['model'],r['run_id'],r['threshold_policy']) for r in before}
check('no_duplicate_before_after_rows',len(unique_before)==len(before),{'rows':len(before),'unique_keys':len(unique_before)})

# Recompute all aggregate rows from their source tables.
def metric_item(dataset,model,cohort,policy):
    if cohort=='tuned':
        return [r for r in before if ('OSCD' if r['dataset']=='oscd' else 'Scenario2_proxy')==dataset
                and r['model']==model and r['cohort']=='tuned' and r['threshold_policy']==policy]
    if cohort=='standard_sync_all_pixel_3_seeds':
        out=[]
        for seed in [42,43,44]:
            if seed==42:
                d=loadj(ROOT/'experiments/scenario_2/sync_dual_stream/metrics.json')['test_counts']
                m={'precision':d['tp']/(d['tp']+d['fp']),'recall':d['tp']/(d['tp']+d['fn']),
                   'f1':2*d['tp']/(2*d['tp']+d['fp']+d['fn']),'iou':d['tp']/(d['tp']+d['fp']+d['fn'])}
            else:
                d=loadj(ROOT/f'experiments/scenario_2_multiseed/sync_cnn_baseline_seed{seed}/metrics.json')['test_threshold_0_5']['all_pixel_protocol']; m=d
            out.append({'seed':str(seed),**m})
        return out
    if cohort=='existing_standard_sync_repeat':
        out=[]
        for seed in [43,44]:
            d=loadj(ROOT/f'experiments/scenario_2_multiseed/sync_cnn_baseline_seed{seed}/metrics.json')['test_threshold_0_5']['all_pixel_protocol'];out.append({'seed':str(seed),**d})
        return out
    return [r for r in comp if r.get('comparison_view')=='A_fixed_0.5'
            and r.get('dataset')==dataset and r.get('model')==model
            and r.get('cohort')==cohort and r.get('threshold_policy')==policy]

aggregate_ok=True; aggregate_failures=[]
for row in agg:
    items=metric_item(row['dataset'],row['model'],row['cohort'],row['threshold_policy'])
    aggregate_ok &= len(items)==int(row['n_unique_runs'])
    aggregate_ok &= len({str(x['seed']) for x in items})==len(items)
    for m in ['precision','recall','f1','iou']:
        vals=[float(x[m]) for x in items]
        mean=statistics.fmean(vals) if vals else float('nan')
        sd=statistics.stdev(vals) if len(vals)>1 else None
        aggregate_ok &= near(mean,row[f'{m}_mean'])
        if sd is None: aggregate_ok &= row[f'{m}_sample_sd']=='N/A'
        else: aggregate_ok &= near(sd,row[f'{m}_sample_sd'])
    if not aggregate_ok: aggregate_failures.append((row['dataset'],row['model'],row['cohort'],row['threshold_policy']))
check('aggregate_means_and_sample_stds_recomputed',aggregate_ok,{'groups':len(agg),'failures':aggregate_failures})

# Audit static safeguards in the training/evaluation code and mask preprocessing.
pipe=(ROOT/'experiments/transformer_tuning_2026-10-15/tuning_pipeline.py').read_text(encoding='utf-8')
check('transformer_validation_eval_mode_fixed','model.eval()' in pipe and '@torch.inference_mode()' in pipe,
      'source contains model.eval() and inference_mode validation')
ds=(ROOT/'src/datasets/scenario2_dataset.py').read_text(encoding='utf-8')
check('nan_mask_precedes_nan_replacement',ds.index('valid_mask = ~np.any(np.isnan(arr)') < ds.index('arr = np.nan_to_num(arr, nan=0.0)'),
      'valid mask computed from raw NaNs before nan_to_num')

# Metric rows remain traceable to configuration, checkpoint, and metric artifacts.
def exists_reference(value):
    if value in (None,'','N/A'): return False
    return Path(value).exists() or (ROOT/Path(value)).exists()
trace_ok=True
for row in before:
    trace_ok &= exists_reference(row.get('checkpoint_path'))
    trace_ok &= exists_reference(row.get('metrics_source'))
    trace_ok &= exists_reference(row.get('config_path'))
check('transformer_metrics_trace_to_checkpoint_config_and_metrics',trace_ok,{'rows':len(before)})

plots=['before_after_tuning.png','cnn_transformer_comparison.png','per_seed_f1_variation.png','validation_precision_recall.png']
check('required_reports_tables_and_plots_exist',all((OUT/p).is_file() for p in plots) and
      all((OUT/p).is_file() for p in ['AUDIT_REPORT.md','EXPERIMENT_PROTOCOL.md','FINAL_REPORT.md',
        'BEFORE_AFTER_RESULTS.csv','CNN_TRANSFORMER_COMPARISON.csv','AGGREGATE_RESULTS.csv','TRAINING_RUNS.csv']),
      {'plots':plots})

# Verify prior release by its own published checksum list without changing it.
release=ROOT/'artifacts/final_release_2026-10-15'; checksum_file=release/'SHA256SUMS.txt'
release_checks=[]
if checksum_file.exists():
    for line in checksum_file.read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line: continue
        pieces=line.split(None,1)
        if len(pieces)!=2: continue
        expected,path=pieces[0],pieces[1].lstrip('* ')
        p=release/path
        if p.exists(): release_checks.append({'path':path,'expected':expected,'actual':sha(p),'matches':sha(p)==expected})
check('existing_release_checksums_match',bool(release_checks) and all(x['matches'] for x in release_checks),
      {'files_checked':len(release_checks),'mismatches':[x['path'] for x in release_checks if not x['matches']]})

manifest_path=OUT/'sha256_manifest.json'
manifest_ok=manifest_path.exists(); manifest_detail={'generated_files':0,'checkpoint_references':0,'other_references':0}
if manifest_ok:
    manifest=loadj(manifest_path)
    for rel,expected in manifest.get('generated_files',{}).items():
        p=OUT/rel; manifest_detail['generated_files']+=1
        manifest_ok &= p.is_file() and sha(p)==expected
    for group in ['checkpoint_references','metric_and_run_references','configuration_references','source_and_split_references']:
        for rel,expected in manifest.get(group,{}).items():
            p=ROOT/rel if not Path(rel).is_absolute() else Path(rel)
            if group=='checkpoint_references': manifest_detail['checkpoint_references']+=1
            else: manifest_detail['other_references']+=1
            manifest_ok &= p.is_file() and sha(p)==expected
check('sha256_manifest_matches_outputs_and_references',manifest_ok,manifest_detail)

# Test suite evidence: pytest unavailable; repository verification scripts were executed separately.
checks['test_execution']={'pytest':'not run: pytest module is not installed in project venv',
 'src/training/test_device.py':'passed with PYTHONIOENCODING=utf-8; initial default encoding attempt failed on Windows arrow character',
 'src/training/verify_backward_compat.py':'passed',
 'src/training/verify_masked_loss.py':'passed',
 'src/training/smoke_test_transformers.py':'passed'}
checks['validation_passed']=not failures
checks['failures']=failures
(OUT/'validation_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'validation_passed':not failures,'failures':failures,
                  'checks':{k:v.get('passed') for k,v in checks.items() if isinstance(v,dict) and 'passed' in v}},indent=2))
if failures: raise SystemExit(1)
