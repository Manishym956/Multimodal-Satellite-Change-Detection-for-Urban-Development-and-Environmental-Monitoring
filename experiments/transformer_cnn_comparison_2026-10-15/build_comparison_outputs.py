"""Build additive comparison tables and plots from verified existing artifacts."""
import csv, json, statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
TUNE=ROOT/'experiments/transformer_tuning_2026-10-15'

def readcsv(path):
    with path.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))
def writecsv(name, rows):
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r.keys()))
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def jsonread(path): return json.loads(path.read_text(encoding='utf-8'))
def counts_metrics(counts):
    tp,fp,tn,fn=(int(counts[k]) for k in ('tp','fp','tn','fn'))
    p=tp/(tp+fp) if tp+fp else None; r=tp/(tp+fn) if tp+fn else None
    return {'precision':p,'recall':r,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'iou':tp/(tp+fp+fn) if tp+fp+fn else None,
            'false_alarm_rate':fp/(fp+tn) if fp+tn else None}

oscd_baselines={
 'sar_unet':'experiments/sar_baseline/metrics.json',
 'optical_unet':'experiments/optical_baseline/metrics.json',
 'early_fusion_unet':'experiments/fusion_baseline/metrics.json',
 'synchronous_dual_stream_cnn':'experiments/dual_stream_baseline/metrics.json',
 'naive_async_dual_stream_cnn':'experiments/asynchronous_dual_stream/metrics.json',
 'clean_async_dual_stream_cnn':'experiments/asynchronous_dual_stream_clean/metrics.json',
 'self_attention_transformer':'experiments/transformer_self_attention/metrics.json',
 'cross_attention_transformer':'experiments/transformer_cross_attention/metrics.json',
}
s2_baselines={
 'async_dual_stream_cnn':'experiments/scenario_2/async_dual_stream/metrics.json',
 'sync_dual_stream_cnn':'experiments/scenario_2/sync_dual_stream/metrics.json',
 'cross_attention_transformer':'experiments/scenario_2/cross_attention/metrics.json',
}

def baseline_record(dataset, model_name, rel):
    d=jsonread(ROOT/rel); rep=d.get('reproducibility',{}); c=d['test_counts']; m=counts_metrics(c)
    yaml_config=rel.replace('metrics.json','config.yaml')
    config_path=yaml_config if (ROOT/yaml_config).exists() else rel
    return {'dataset':dataset,'model':model_name,'cohort':'original_baseline','run_id':rel.removesuffix('/metrics.json'),
      'seed':rep.get('seed',42),'threshold_policy':'common_0.5','threshold':d.get('threshold',.5),**m,**c,
      'evaluated_pixels':sum(c.values()),'loss':rep.get('loss_function','BCEWithLogitsLoss'),
      'pos_weight':rep.get('pos_weight',1),'learning_rate':rep.get('learning_rate'),
      'weight_decay':rep.get('weight_decay','N/A'),'batch_size':rep.get('batch_size'),
      'epochs':rep.get('epochs'),'optimizer':rep.get('optimizer','Adam'),
      'model_selection_criterion':rep.get('checkpoint_selection','highest validation F1 at threshold 0.5'),
      'best_epoch':d.get('validation_best_epoch'),
      'runtime_seconds':d.get('training_time_seconds'),'peak_vram_mb':rep.get('peak_gpu_memory_mb','N/A'),
      'parameter_count':rep.get('parameter_count'),'input_channels':rep.get('total_input_channels',rep.get('input_channels')),
      'modality':rep.get('modality'),'checkpoint_path':rel.replace('experiments/','checkpoints/').replace('/metrics.json','/best.pt'),
      'metrics_path':rel,'config_path':config_path,'validation_threshold':'N/A','test_used_for_selection':False}

# Before/after rows: preserve saved original metrics at .5, and all selected tuned seeds.
tune_rows=readcsv(TUNE/'FINAL_TUNING_RESULTS.csv')
before_after=[]
base_metrics={
 ('oscd','self'):jsonread(ROOT/'experiments/transformer_self_attention/metrics.json'),
 ('oscd','cross'):jsonread(ROOT/'experiments/transformer_cross_attention/metrics.json'),
 ('s2','cross'):jsonread(ROOT/'experiments/scenario_2/cross_attention/metrics.json'),
}
for r in tune_rows:
    ds,arch=r['dataset'],r['architecture']; is_base=r['run_id']=='original_checkpoint'
    cfg=(base_metrics[(ds,arch)].get('reproducibility',{}) if is_base else
         jsonread(TUNE/'runs'/r['run_id']/'config.json'))
    if is_base and r['threshold']=='0.5':
        d=base_metrics[(ds,arch)]; c=d['test_counts']; mm=counts_metrics(c)
        vals={**r,**mm,**c,'test_f1':mm['f1'],'test_precision':mm['precision'],'test_recall':mm['recall'],'test_iou':mm['iou']}
    else: vals=r
    base_f1=base_metrics[(ds,arch)]['final_test_f1']
    f1_value=float(vals['test_f1'])
    delta=(f1_value-base_f1) if not is_base else None
    before_after.append({'dataset':ds,'model':f'{arch}_attention_transformer',
      'cohort':'original' if is_base else 'tuned','run_id':r['run_id'],'seed':r['seed'],
      'threshold_policy':'common_0.5' if float(r['threshold'])==.5 else 'validation_selected',
      'threshold':r['threshold'],'precision':vals['test_precision'],'recall':vals['test_recall'],
      'f1':vals['test_f1'],'iou':vals['test_iou'],'far':vals.get('test_fp_rate',vals.get('false_alarm_rate')),
      'delta_f1_vs_original_common_0_5':delta,
      'relative_delta_f1_percent_vs_original_common_0_5':(100*delta/base_f1 if delta is not None and base_f1 else None),
      'validation_precision':r.get('validation_precision'),'validation_recall':r.get('validation_recall'),
      'validation_f1':r.get('validation_f1'),'validation_iou':r.get('validation_iou'),
      'tp':vals['tp'],'fp':vals['fp'],'tn':vals['tn'],'fn':vals['fn'],
      'evaluated_pixels':vals.get('evaluated_pixels'),'valid_pixels':vals.get('valid_pixels'),
      'loss':r['loss'],'pos_weight':r['pos_weight'],'learning_rate':r['learning_rate'],
      'weight_decay':r['weight_decay'],'optimizer':cfg.get('optimizer','Adam'),
      'batch_size':cfg.get('batch_size','N/A'),'epochs':cfg.get('epochs',cfg.get('epochs_requested','N/A')),
      'model_selection_criterion':cfg.get('selection',cfg.get('checkpoint_selection','highest validation pooled F1')),
      'seed_count':'N/A','best_epoch':r['checkpoint_epoch'],
      'training_seconds':r['training_seconds'],'peak_vram_mb':r['peak_vram_mb'],
      'parameter_count':r['parameter_count'],'input_channels':15 if ds=='oscd' else 14,
      'modality':'SAR T1+T2 plus historical optical' if ds in ['oscd','s2'] else 'N/A',
      'metrics_source':r['metrics_artifact'],
      'config_path':(r['metrics_artifact'].replace('result.json','config.json') if 'result.json' in r['metrics_artifact'] else r['metrics_artifact']),
      'checkpoint_path':r['checkpoint_path'],'baseline_rerun_discrepancy':
       'saved-vs-rerun difference recorded' if ds=='oscd' and arch=='cross' and is_base else 'N/A',
      'test_used_for_selection':False})
writecsv('BEFORE_AFTER_RESULTS.csv',before_after)

# CNN and Transformer matrix: common threshold rows for all established CNN baselines,
# then a validation-calibrated view for primary competitors.
oscd_models={'sar_unet':'SAR-only U-Net','optical_unet':'Optical-only U-Net',
 'early_fusion_unet':'Early-fusion U-Net','synchronous_dual_stream_cnn':'Synchronous dual-stream CNN',
 'naive_async_dual_stream_cnn':'Naive asynchronous CNN','clean_async_dual_stream_cnn':'Clean asynchronous CNN'}
matrix=[]
for key,rel in oscd_baselines.items():
    if key.startswith(('self_attention','cross_attention')): continue
    matrix.append({'dataset':'OSCD','model':oscd_models[key],**baseline_record('OSCD',oscd_models[key],rel),
      'comparison_view':'A_fixed_0.5','metric_provenance':'original metrics.json'})
for r in before_after:
    if r['dataset']!='oscd' or r['threshold_policy']!='common_0.5' or r['seed']!='42': continue
    label='Self-attention Transformer' if r['model'].startswith('self') else 'Cross-attention Transformer'
    matrix.append({**r,'dataset':'OSCD','model':label,'comparison_view':'A_fixed_0.5',
      'metric_provenance':r['metrics_source']})

cnn_test=readcsv(OUT/'cnn_frozen_test_thresholds.csv')
cnn_val=jsonread(OUT/'cnn_validation_selected_thresholds.json')['selected']
for key,label in [('clean_async_cnn','Clean asynchronous CNN'),('sync_dual_stream_cnn','Synchronous dual-stream CNN')]:
    ctest=next(x for x in cnn_test if x['model']==key and x['threshold_role']=='validation_selected_threshold')
    bkey='clean_async_dual_stream_cnn' if key.startswith('clean') else 'synchronous_dual_stream_cnn'
    rel=oscd_baselines[bkey]
    matrix.append({'dataset':'OSCD','model':label,**baseline_record('OSCD',label,rel),
       **{k:ctest[k] for k in ['threshold','precision','recall','f1','iou','false_alarm_rate','tp','fp','tn','fn','pixels']},
       'comparison_view':'B_validation_selected','threshold_policy':'validation_selected',
       'validation_threshold':ctest['threshold'],'validation_f1':cnn_val[key]['validation_metrics']['f1'],
       'metric_provenance':'new validation-only probability sweep and frozen threshold; single test inference pass'})
for r in before_after:
    if r['dataset']!='oscd' or r['threshold_policy']!='validation_selected' or r['seed']!='42': continue
    label='Self-attention Transformer' if r['model'].startswith('self') else 'Cross-attention Transformer'
    matrix.append({**r,'dataset':'OSCD','model':label,'comparison_view':'B_validation_selected',
      'validation_threshold':r['threshold'],'validation_f1':next((x['validation_f1'] for x in tune_rows if x['run_id']==r['run_id'] and x['dataset']==r['dataset'] and x['architecture']==r['model'].split('_')[0] and x['threshold']==r['threshold']),None),
      'metric_provenance':r['metrics_source']})

# Scenario 2 fixed and historical calibrated views. Prior calibration is explicitly identified.
s2labels={'async_dual_stream_cnn':'Asynchronous CNN','sync_dual_stream_cnn':'Synchronous CNN','cross_attention_transformer':'Cross-attention Transformer'}
s2old={k:baseline_record('Scenario2_proxy',label,rel) for k,label,rel in [(k,s2labels[k],v) for k,v in s2_baselines.items()]}
for key,label in s2labels.items():
    if key!='cross_attention_transformer':
        matrix.append({'dataset':'Scenario2_proxy','model':label,**s2old[key],
          'comparison_view':'A_fixed_0.5','metric_provenance':'original metrics.json; building-probability XOR proxy'})
    else:
        for r in before_after:
            if r['dataset']=='s2' and r['model']=='cross_attention_transformer' and r['threshold_policy']=='common_0.5' and r['seed']=='42':
                matrix.append({**r,'dataset':'Scenario2_proxy','model':label,'comparison_view':'A_fixed_0.5','metric_provenance':r['metrics_source']})
cal=jsonread(ROOT/'experiments/scenario_2/threshold_calibration/threshold_calibration_results.json')
calmap={'Asynchronous CNN':'Async Dual-Stream CNN','Synchronous CNN':'Sync Dual-Stream CNN'}
for key,label,field in [('async_dual_stream_cnn','Asynchronous CNN','Async Dual-Stream CNN'),('sync_dual_stream_cnn','Synchronous CNN','Sync Dual-Stream CNN')]:
    c=cal[field]; m=c['test_calibrated']; vth=c['best_validation_tau']
    matrix.append({'dataset':'Scenario2_proxy','model':label,'comparison_view':'B_validation_selected',
      'threshold_policy':'validation_selected','threshold':vth,'validation_threshold':vth,
      'validation_f1':c['best_validation_f1'],'precision':m['precision'],'recall':m['recall'],
      'f1':m['f1'],'iou':m['iou'],'far':m['false_alarm_rate'],
      'tp':m['tp'],'fp':m['fp'],'tn':m['tn'],'fn':m['fn'],'evaluated_pixels':sum(m[k] for k in ['tp','fp','tn','fn']),
      'seed':42,'loss':'BCEWithLogitsLoss(pos_weight=94)','pos_weight':94,
      'optimizer':s2old[key]['optimizer'],'batch_size':s2old[key]['batch_size'],'epochs':s2old[key]['epochs'],
      'model_selection_criterion':s2old[key]['model_selection_criterion'],
      'input_channels':s2old[key]['input_channels'],'modality':s2old[key]['modality'],
      'checkpoint_path':s2old[key]['checkpoint_path'],'metrics_path':s2old[key]['metrics_path'],
      'config_path':s2old[key]['config_path'],
      'test_used_for_selection':False,
      'metric_provenance':'existing experiments/scenario_2/threshold_calibration/threshold_calibration_results.json; all-pixel policy; test historically accessed'})
for r in before_after:
    if r['dataset']=='s2' and r['model']=='cross_attention_transformer' and r['threshold_policy']=='validation_selected' and r['seed']=='42':
        matrix.append({**r,'dataset':'Scenario2_proxy','model':'Cross-attention Transformer','comparison_view':'B_validation_selected',
          'validation_threshold':r['threshold'],'metric_provenance':r['metrics_source']})
writecsv('CNN_TRANSFORMER_COMPARISON.csv',matrix)

# Existing Scenario 2 standard synchronous CNN repeats use a separately reported all-pixel protocol.
aggregate_inputs=[]
for x in matrix:
    if x.get('comparison_view')=='A_fixed_0.5' and x.get('cohort')!='tuned':
        aggregate_inputs.append({'dataset':x['dataset'],'model':x['model'],'cohort':x.get('cohort','fixed_baseline'),
          'threshold_policy':'common_0.5','seed':int(x.get('seed',42)),**{k:float(x[k]) for k in ['precision','recall','f1','iou']}})
for r in before_after:
    if r['cohort']=='tuned':
        aggregate_inputs.append({'dataset':'OSCD' if r['dataset']=='oscd' else 'Scenario2_proxy',
          'model':r['model'],'cohort':'tuned','threshold_policy':r['threshold_policy'],
          'seed':int(r['seed']),**{k:float(r[k]) for k in ['precision','recall','f1','iou']}})
for seed in (43,44):
    d=jsonread(ROOT/f'experiments/scenario_2_multiseed/sync_cnn_baseline_seed{seed}/metrics.json')
    m=d['test_threshold_0_5']['all_pixel_protocol']
    aggregate_inputs.append({'dataset':'Scenario2_proxy','model':'Synchronous CNN','cohort':'existing_standard_sync_repeat',
      'threshold_policy':'common_0.5_all_pixel','seed':seed,**{k:m[k] for k in ['precision','recall','f1','iou']}})
# Also report the three-seed standard synchronous Scenario 2 group as one cohort.
for seed in (42,43,44):
    if seed==42:
        m=jsonread(ROOT/'experiments/scenario_2/sync_dual_stream/metrics.json')
        m=counts_metrics(m['test_counts'])
    else:
        m=jsonread(ROOT/f'experiments/scenario_2_multiseed/sync_cnn_baseline_seed{seed}/metrics.json')['test_threshold_0_5']['all_pixel_protocol']
    aggregate_inputs.append({'dataset':'Scenario2_proxy','model':'Synchronous CNN','cohort':'standard_sync_all_pixel_3_seeds',
      'threshold_policy':'common_0.5_all_pixel','seed':seed,**{k:m[k] for k in ['precision','recall','f1','iou']}})
groups={}
for x in aggregate_inputs:
    key=(x['dataset'],x['model'],x['cohort'],x['threshold_policy']); groups.setdefault(key,[]).append(x)
aggregates=[]
for (ds,model,cohort,policy),items in sorted(groups.items()):
    row={'dataset':ds,'model':model,'cohort':cohort,'threshold_policy':policy,
         'n_unique_runs':len({(x['seed'],x.get('run_id',x['seed'])) for x in items}),
         'seeds':';'.join(str(x['seed']) for x in items)}
    for metric in ['precision','recall','f1','iou']:
        vals=[x[metric] for x in items]
        row[f'{metric}_mean']=statistics.fmean(vals)
        row[f'{metric}_sample_sd']=statistics.stdev(vals) if len(vals)>1 else 'N/A'
    aggregates.append(row)
writecsv('AGGREGATE_RESULTS.csv',aggregates)

# Comprehensive registry: prior Transformer trial directory registry plus immutable originals.
registry=readcsv(TUNE/'TRAINING_RUNS.csv')
for ds,models in [('OSCD',oscd_baselines),('Scenario2_proxy',s2_baselines)]:
    for key,rel in models.items():
        d=jsonread(ROOT/rel); rep=d.get('reproducibility',{})
        registry.append({'run_id':rel.removesuffix('/metrics.json'),'status':'completed_fixed_baseline',
          'dataset':ds,'architecture':key,'seed':rep.get('seed',42),'learning_rate':rep.get('learning_rate'),
          'weight_decay':rep.get('weight_decay','N/A'),'loss':rep.get('loss_function','BCEWithLogitsLoss'),
          'pos_weight':rep.get('pos_weight',1),'epochs_requested':rep.get('epochs'),
          'batch_size':rep.get('batch_size'),'optimizer':rep.get('optimizer','Adam'),
          'model_selection_criterion':rep.get('checkpoint_selection','highest validation F1 at threshold 0.5'),
          'input_channels':rep.get('total_input_channels',rep.get('input_channels')),
          'parameter_count':rep.get('parameter_count'),
          'best_epoch':d.get('validation_best_epoch'),'validation_f1_at_0_5':d.get('validation_best_f1'),
          'test_f1_at_0_5':d.get('final_test_f1'),'training_seconds':d.get('training_time_seconds'),
          'peak_vram_mb':rep.get('peak_gpu_memory_mb','N/A'),'test_evaluated':True,
          'failure_reason':'','artifact_path':rel,'config_path':rel,
          'checkpoint_path':rel.replace('experiments/','checkpoints/').replace('/metrics.json','/best.pt')})
for p in sorted((ROOT/'experiments/scenario_2_multiseed').glob('*/metrics.json')):
    d=jsonread(p); conf=d['config']; registry.append({'run_id':str(p.parent.relative_to(ROOT)),
      'status':'completed_existing_comparable_all_pixel' if conf.get('loss_type')=='standard' else 'completed_existing_noncomparable_masked_loss',
      'dataset':'Scenario2_proxy','architecture':'sync_cnn','seed':d.get('seed'),'learning_rate':conf.get('lr'),
      'weight_decay':'N/A','loss':conf.get('loss_type'),'pos_weight':conf.get('pos_weight'),
      'epochs_requested':conf.get('epochs'),'batch_size':conf.get('batch_size'),'optimizer':'Adam',
      'model_selection_criterion':conf.get('checkpoint_selection','highest validation F1 under valid-pixel protocol'),
      'input_channels':conf.get('total_input_channels'),'parameter_count':conf.get('parameter_count'),
      'best_epoch':d.get('validation_best_epoch'),
      'validation_f1_at_0_5':d.get('validation_best_f1_valid'),'test_f1_at_0_5':d.get('test_threshold_0_5',{}).get('all_pixel_protocol',{}).get('f1'),
      'training_seconds':d.get('training_time_seconds'),'peak_vram_mb':d.get('peak_gpu_memory_mb'),
      'test_evaluated':True,'failure_reason':'','artifact_path':str(p.relative_to(ROOT)),
      'config_path':str((p.parent/'config.yaml').relative_to(ROOT)),
      'checkpoint_path':str((ROOT/'checkpoints/scenario_2_multiseed'/p.parent.name/'best.pt').relative_to(ROOT))})
for row in registry:
    if not row.get('config_path') and row.get('run_id'):
        candidate=TUNE/'runs'/row['run_id']/'config.json'
        if candidate.exists(): row['config_path']=str(candidate.relative_to(ROOT))
    if not row.get('checkpoint_path') and row.get('run_id'):
        candidate=TUNE/'runs'/row['run_id']/'best.pt'
        if candidate.exists(): row['checkpoint_path']=str(candidate.relative_to(ROOT))
    cp=row.get('config_path')
    cp_path=Path(cp) if cp else None
    if cp_path and not cp_path.is_absolute(): cp_path=ROOT/cp_path
    if cp_path and cp_path.exists() and cp_path.suffix.lower()=='.json':
        raw=jsonread(cp_path); cfg=raw.get('reproducibility',raw)
        for dest,source in [('optimizer','optimizer'),('batch_size','batch_size'),
                            ('epochs_requested','epochs'),('input_channels','input_channels'),
                            ('parameter_count','parameter_count')]:
            if not row.get(dest) and cfg.get(source) is not None: row[dest]=cfg[source]
        if not row.get('model_selection_criterion'):
            row['model_selection_criterion']=cfg.get('selection',cfg.get('checkpoint_selection','N/A'))
writecsv('TRAINING_RUNS.csv',registry)

# Publication-style figure set.
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.dpi':120})
fig,ax=plt.subplots(figsize=(9,4.8)); labels=['OSCD self','OSCD cross','S2 cross']; xs=list(range(3)); width=.25
for j,(cohort,color,title) in enumerate([('original','#5b7184','Original @ .5'),('tuned','#e39b36','Tuned seed 42 @ .5')]):
 vals=[]
 for ds,arch in [('oscd','self'),('oscd','cross'),('s2','cross')]:
  rr=[x for x in before_after if x['dataset']==ds and x['model']==f'{arch}_attention_transformer' and x['cohort']==cohort and x['threshold_policy']=='common_0.5' and str(x['seed'])=='42']
  vals.append(float(rr[0]['f1']) if rr else float('nan'))
 ax.bar([x+(j-.5)*width for x in xs],vals,width,label=title,color=color)
ax.set_xticks(xs,labels);ax.set_ylabel('Pooled test F1');ax.set_title('Transformer test F1 before and after tuning (common threshold .5)');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'before_after_tuning.png',dpi=300);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
for ax,view in zip(axs,['A_fixed_0.5','B_validation_selected']):
 rows=[x for x in matrix if x.get('dataset')=='OSCD' and x.get('comparison_view')==view and x.get('model') in ['Clean asynchronous CNN','Synchronous dual-stream CNN','Self-attention Transformer','Cross-attention Transformer']]
 chosen=[]
 for name in ['Synchronous dual-stream CNN','Clean asynchronous CNN','Self-attention Transformer','Cross-attention Transformer']:
  candidates=[x for x in rows if x.get('model')==name]
  if view=='A_fixed_0.5': candidates=[x for x in candidates if x.get('cohort') in ['tuned','original_baseline'] or x.get('cohort')=='original']
  # Keep CNN fixed rows and both original/tuned transformer cohorts for view A.
  if view=='A_fixed_0.5' and name.endswith('Transformer'):
   candidates=[x for x in candidates if x.get('cohort') in ['original','tuned']]
  chosen.extend(candidates)
 names=[]; vals=[]
 for x in chosen:
  nm=x['model']+(' tuned' if x.get('cohort')=='tuned' else '')
  names.append(nm); vals.append(float(x['f1']))
 ax.barh(names,vals,color=['#476d85' if 'CNN' in n else '#d98c38' for n in names]); ax.set_title('View A: common threshold .5' if view.startswith('A') else 'View B: validation-selected threshold'); ax.set_xlabel('Pooled test F1'); ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'cnn_transformer_comparison.png',dpi=300);plt.close(fig)

fig,ax=plt.subplots(figsize=(9,4.8))
for ds,arch,label,color in [('oscd','self','OSCD self Transformer','#33658a'),('oscd','cross','OSCD cross Transformer','#f6ae2d'),('s2','cross','Scenario 2 cross Transformer','#758e4f')]:
 rows=[x for x in before_after if x['dataset']==ds and x['model']==f'{arch}_attention_transformer' and x['cohort']=='tuned' and x['threshold_policy']=='common_0.5']
 vals=[float(x['f1']) for x in sorted(rows,key=lambda x:int(x['seed']))]
 ax.plot([42,43,44],vals,marker='o',label=label,color=color)
ax.set_xticks([42,43,44]);ax.set_xlabel('Seed');ax.set_ylabel('Pooled test F1 at .5');ax.set_title('Tuned Transformer seed variability');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/'per_seed_f1_variation.png',dpi=300);plt.close(fig)

# Validation PR trade-offs: curves use validation only, not test scores.
pr=[]
for r in readcsv(TUNE/'threshold_validation_sweep.csv'):
 if r['dataset']=='oscd' and r['architecture'] in ['self','cross']:
  pr.append((f"{r['architecture']} Transformer original",float(r['threshold']),float(r['recall']),float(r['precision']),float(r['f1'])))
for name,run in [('self Transformer tuned','oscd_self_lr1e-4_wd1e-4_valid'),('cross Transformer tuned','oscd_cross_lr1e-4_wd1e-4_valid')]:
 for r in readcsv(TUNE/'runs'/run/'validation_threshold_sweep.csv'):
  pr.append((name,float(r['threshold']),float(r['recall']),float(r['precision']),float(r['f1'])))
for r in readcsv(OUT/'cnn_validation_threshold_sweeps.csv'):
 if r['model'] in ['clean_async_cnn','sync_dual_stream_cnn'] and r['precision'] and r['recall'] and r['f1']:
  label={'clean_async_cnn':'Clean asynchronous CNN','sync_dual_stream_cnn':'Synchronous dual-stream CNN'}[r['model']]
  pr.append((label,float(r['threshold']),float(r['recall']),float(r['precision']),float(r['f1'])))
fig,ax=plt.subplots(figsize=(7.5,5.5))
for name in sorted({x[0] for x in pr}):
 curve=[x for x in pr if x[0]==name]
 curve.sort(key=lambda x:x[1])
 best=max(curve,key=lambda x:x[4])
 ax.plot([x[3] for x in curve],[x[2] for x in curve],label=f'{name} (val tau*={best[1]:.3f})',lw=1.8)
ax.set_xlabel('Validation precision');ax.set_ylabel('Validation recall');ax.set_title('OSCD validation precision–recall operating points');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'validation_precision_recall.png',dpi=300);plt.close(fig)
print(json.dumps({'before_after_rows':len(before_after),'comparison_rows':len(matrix),'aggregate_rows':len(aggregates),'registry_rows':len(registry),'plots':['before_after_tuning.png','cnn_transformer_comparison.png','per_seed_f1_variation.png','validation_precision_recall.png']}))
