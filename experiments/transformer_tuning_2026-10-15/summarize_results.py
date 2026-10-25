from pathlib import Path
import csv,json,statistics
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
BASES={
 'oscd_self':('oscd','self',ROOT/'experiments/transformer_self_attention/metrics.json',ROOT/'checkpoints/transformer_self_attention/best.pt'),
 'oscd_cross':('oscd','cross',ROOT/'experiments/transformer_cross_attention/metrics.json',ROOT/'checkpoints/transformer_cross_attention/best.pt'),
 's2_cross':('s2','cross',ROOT/'experiments/scenario_2/cross_attention/metrics.json',ROOT/'checkpoints/scenario_2/cross_attention/best.pt'),
}
def readcsv(p): return list(csv.DictReader(p.open(encoding='utf-8')))
def wr(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def close(a,b):return abs(float(a)-float(b))<1e-8

test_rows=readcsv(HERE/'final_test_comparison.csv')
base_val=readcsv(HERE/'threshold_validation_sweep.csv')
selected_base=json.loads((HERE/'selected_baseline_thresholds.json').read_text(encoding='utf-8'))['thresholds']
frozen=json.loads((HERE/'selected_tuning_configs.json').read_text(encoding='utf-8'))['selected']
summary=[]
def from_count(r):
 return {k:(float(r[k]) if r.get(k) not in (None,'') else None) for k in ['precision','recall','f1','iou','false_alarm_rate']}
def numeric_test(r):
 out=dict(r)
 for k in ['precision','recall','f1','iou','false_alarm_rate','predicted_positive_fraction']:
  if out.get(k) not in (None,''): out[k]=float(out[k])
 for k in ['tp','fp','tn','fn','evaluated_pixels','valid_pixels']:
  if out.get(k) not in (None,''): out[k]=int(out[k])
 return out
def addrow(ds,arch,label,rid,seed,threshold,v,test,epoch,lr,wd,dropout,loss,pos,params,secs,vram,basef1,metric_source,ckpt_path):
  delta=(test.get('f1')-basef1) if (test.get('f1') is not None and basef1 is not None) else None
  rel=100*delta/basef1 if delta is not None and basef1 not in (None,0) else None
  summary.append({'dataset':ds,'architecture':arch,'comparison':label,'run_id':rid,'seed':seed,'checkpoint_epoch':epoch,
   'learning_rate':lr,'weight_decay':wd,'dropout':dropout,'loss':loss,'pos_weight':pos,'threshold':threshold,
   'validation_precision':v.get('precision'),'validation_recall':v.get('recall'),'validation_f1':v.get('f1'),'validation_iou':v.get('iou'),
   'test_precision':test.get('precision'),'test_recall':test.get('recall'),'test_f1':test.get('f1'),'test_iou':test.get('iou'),
   'test_fp_rate':test.get('false_alarm_rate'),'tp':test.get('tp'),'fp':test.get('fp'),'tn':test.get('tn'),'fn':test.get('fn'),
   'evaluated_pixels':test.get('evaluated_pixels'),'valid_pixels':test.get('valid_pixels'),
   'predicted_positive_fraction':test.get('predicted_positive_fraction'),'delta_f1_vs_original_test_at_0_5':delta,
   'relative_delta_percent_vs_original_test_at_0_5':rel,'parameter_count':params,'training_seconds':secs,'peak_vram_mb':vram,
   'metrics_artifact':metric_source,'checkpoint_path':ckpt_path})
for key,(ds,arch,mp,cp) in BASES.items():
 md=json.loads(mp.read_text(encoding='utf-8')); ck=torchload=None
 # Read the original checkpoint epoch from metrics; metric JSON is authoritative for the historic run.
 saved=md['final_test_f1']; bthr=selected_base[key]['threshold']; bval=selected_base[key]['validation_metrics']
 eval0=next(r for r in test_rows if r['run_id']=='original_checkpoint' and r['dataset']==ds and r['architecture']==arch and close(r['threshold'],.5))
 evalsel=next(r for r in test_rows if r['run_id']=='original_checkpoint' and r['dataset']==ds and r['architecture']==arch and close(r['threshold'],bthr))
 val0=next(r for r in base_val if r['dataset']==ds and r['architecture']==arch and r['split']=='validation' and close(r['threshold'],.5))
 # Original checkpoint at original and validation-selected operating points.
 for label,tr,vr,te in [('original_checkpoint_original_threshold',.5,val0,eval0),('original_checkpoint_validation_threshold',bthr,bval,evalsel)]:
  te=numeric_test(te); addrow(ds,arch,label,'original_checkpoint',42,tr,vr,te,md.get('validation_best_epoch'),md.get('reproducibility',{}).get('learning_rate'),md.get('reproducibility',{}).get('weight_decay',0),md.get('reproducibility',{}).get('dropout',0),md.get('reproducibility',{}).get('loss_function'),md.get('reproducibility',{}).get('pos_weight'),md.get('reproducibility',{}).get('parameter_count'),md.get('training_time_seconds'),None,saved,str(mp.relative_to(ROOT)),str(cp))
  # Keep saved-vs-rerun details available as separate columns.
  summary[-1]['saved_test_f1_at_0_5']=saved
  summary[-1]['rerun_minus_saved_test_f1_at_0_5']=float(eval0['f1'])-saved
 # Selected configuration and repeats.
 sel=frozen[key]; rid0=sel['selected_run_id']; run_ids=[rid0]
 for p in (HERE/'runs').iterdir():
  q=p/'result.json'
  if not q.exists():continue
  d=json.loads(q.read_text(encoding='utf-8'))
  if d['dataset']==ds and d['architecture']==arch and d['lr']==sel['lr'] and d['weight_decay']==sel['weight_decay'] and d['seed'] in [43,44]:run_ids.append(d['run_id'])
 for rid in run_ids:
  run=HERE/'runs'/rid; res=json.loads((run/'result.json').read_text(encoding='utf-8')); cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
  sweep=readcsv(run/'validation_threshold_sweep.csv'); tv=res['validation_selected_threshold']; tsel=float(tv['threshold']); vsel=tv['metrics']; vhalf=res['validation_at_0_5']
  basecomp=eval0['f1']
  for label,t,v in [('tuned_checkpoint_original_threshold',.5,vhalf),('tuned_checkpoint_validation_threshold',tsel,vsel)]:
   te=next(x for x in test_rows if x['run_id']==rid and x['dataset']==ds and x['architecture']==arch and close(x['threshold'],t))
   addrow(ds,arch,label,rid,cfg['seed'],t,v,numeric_test(te),res['best_epoch'],cfg['lr'],cfg['weight_decay'],cfg['dropout'],cfg['loss'],cfg['pos_weight'],res['parameters'],res['training_seconds'],res['peak_vram_mb'],float(basecomp),str((run/'result.json').relative_to(ROOT)),str(run/'best.pt'))
wr(HERE/'FINAL_TUNING_RESULTS.csv',summary)

# Descriptive seed means/sample SD for selected configs only; threshold-selected values use validation threshold per seed.
repeat=[]
for key,(ds,arch,*_) in BASES.items():
 for method,threshold_col in [('common_threshold_0.5','original'),('per_seed_validation_selected','selected')]:
  vals=[]
  for r in summary:
   if r['dataset']==ds and r['architecture']==arch and r['comparison']==('tuned_checkpoint_original_threshold' if method.startswith('common') else 'tuned_checkpoint_validation_threshold'):
    vals.append((int(r['seed']),float(r['test_f1'])))
  xs=[v for _,v in vals]
  repeat.append({'dataset':ds,'architecture':arch,'metric':'test F1','protocol':method,'n':len(xs),'seeds':';'.join(str(s) for s,_ in vals),'mean':statistics.fmean(xs) if xs else None,'sample_std':statistics.stdev(xs) if len(xs)>1 else None})
wr(HERE/'REPEATABILITY_SUMMARY.csv',repeat)

# concise tuning configuration table (validation-only ranking, includes invalidated attempts separately in run inventory).
runs=[]
for p in sorted((HERE/'runs').iterdir()):
 cp=p/'config.json';rp=p/'result.json';fail=p/'failure.json'
 if not cp.exists():continue
 cfg=json.loads(cp.read_text(encoding='utf-8'));r={'run_id':p.name,'status':cfg.get('status','valid'),'dataset':cfg['dataset'],'architecture':cfg['architecture'],'seed':cfg['seed'],'lr':cfg['lr'],'weight_decay':cfg['weight_decay'],'dropout':cfg['dropout'],'loss':cfg['loss'],'pos_weight':cfg['pos_weight'],'epochs_requested':cfg['epochs'],'parameters':None,'best_epoch':None,'validation_f1_at_0_5':None,'training_seconds':None,'peak_vram_mb':None,'test_evaluated':False,'failure_reason':''}
 if rp.exists():
  d=json.loads(rp.read_text(encoding='utf-8'));r.update({'parameters':d['parameters'],'best_epoch':d['best_epoch'],'validation_f1_at_0_5':d['validation_at_0_5']['f1'],'training_seconds':d['training_seconds'],'peak_vram_mb':d['peak_vram_mb'],'test_evaluated':d.get('test_evaluated',False)})
 if fail.exists():r['failure_reason']=json.loads(fail.read_text(encoding='utf-8')).get('reason','')
 runs.append(r)
wr(HERE/'TRAINING_RUNS.csv',runs)

# Plot validation selection vs final test comparisons without hiding seed spread.
labels=[('OSCD self','oscd','self'),('OSCD cross','oscd','cross'),('S2 cross','s2','cross')]
fig,axs=plt.subplots(1,2,figsize=(11,4.5))
for ax,title,labelkey in [(axs[0],'Seed-42 checkpoint: threshold decomposition','seed42'),(axs[1],'Tuned selected configuration: test seed variability','repeat')]:pass
xpos=range(len(labels)); w=.18
for j,(name,ds,arch) in enumerate(labels):
 rows=[r for r in summary if r['dataset']==ds and r['architecture']==arch]
 original=next(r for r in rows if r['comparison']=='original_checkpoint_original_threshold')['test_f1']
 baseline_sel=next(r for r in rows if r['comparison']=='original_checkpoint_validation_threshold')['test_f1']
 tuned=next(r for r in rows if r['comparison']=='tuned_checkpoint_original_threshold' and r['seed']==42)['test_f1']
 tuned_sel=next(r for r in rows if r['comparison']=='tuned_checkpoint_validation_threshold' and r['seed']==42)['test_f1']
 axs[0].bar(j-1.5*w,original,w,label='Original @ 0.5' if j==0 else None,color='#8797a6')
 axs[0].bar(j-.5*w,baseline_sel,w,label='Original @ val threshold' if j==0 else None,color='#4c78a8')
 axs[0].bar(j+.5*w,tuned,w,label='Tuned @ 0.5' if j==0 else None,color='#f2a541')
 axs[0].bar(j+1.5*w,tuned_sel,w,label='Tuned @ val threshold' if j==0 else None,color='#d95f59')
 vals=[]
 for r in rows:
  if r['comparison']=='tuned_checkpoint_original_threshold': vals.append((r['seed'],r['test_f1']))
 mean=statistics.fmean(v for _,v in vals);sd=statistics.stdev(v for _,v in vals)
 axs[1].errorbar(j,mean,yerr=sd,fmt='o',capsize=5,color='#315f7d'); axs[1].scatter([j]*len(vals),[v for _,v in vals],s=24,alpha=.55,color='#75a3bd')
axs[0].set_xticks(list(xpos),[x[0] for x in labels]);axs[0].set_ylabel('Pooled test F1');axs[0].set_title('Four requested comparison conditions');axs[0].legend(fontsize=8)
axs[1].set_xticks(list(xpos),[x[0] for x in labels]);axs[1].set_ylabel('Pooled test F1 @ threshold 0.5');axs[1].set_title('Seed 42/43/44; mean ± sample SD');
fig.tight_layout();fig.savefig(HERE/'test_f1_comparison.png',dpi=180);plt.close(fig)
print('rows',len(summary),'repeat_summaries',len(repeat),'runs',len(runs))
