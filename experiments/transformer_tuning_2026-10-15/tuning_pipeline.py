"""Additive threshold/tuning driver for the existing Transformer models.

Run modes are deliberately separated: `thresholds` evaluates original weights,
`train` writes validation-selected checkpoints only, and `evaluate` is reserved
for frozen configurations. The locked test split is never read in `train`.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, statistics, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import counts_from_probabilities, rates_from_counts, FAR_DEFINITION
from src.models.transformer_self_attention import TransformerSelfAttention, count_parameters
from src.models.transformer_cross_attention import TransformerCrossAttention
from src.training.train_sar_baseline import set_seed, train_one_epoch

EXPECTED_SPLIT = "5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27"
THRESHOLDS = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80]
GROUPS = {
    ("oscd","self"): {"baseline": ROOT/"checkpoints/transformer_self_attention/best.pt", "metric": ROOT/"experiments/transformer_self_attention/metrics.json"},
    ("oscd","cross"): {"baseline": ROOT/"checkpoints/transformer_cross_attention/best.pt", "metric": ROOT/"experiments/transformer_cross_attention/metrics.json"},
    ("s2","cross"): {"baseline": ROOT/"checkpoints/scenario_2/cross_attention/best.pt", "metric": ROOT/"experiments/scenario_2/cross_attention/metrics.json"},
}

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def verify_split() -> str:
    p=ROOT/"experiments/scenario_2/configs/split_definition.json"
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != EXPECTED_SPLIT: raise RuntimeError(f"Locked split hash mismatch: {digest}")
    return digest

def environment() -> dict:
    return {"python":sys.version,"torch":str(torch.__version__),"cuda":torch.cuda.is_available(),
            "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_gb":round(torch.cuda.get_device_properties(0).total_memory/1024**3,2) if torch.cuda.is_available() else None}

def make_model(dataset:str, arch:str, dropout:float=0.0, embed_dim:int=128, heads:int=4):
    sar,opt=(2,13) if dataset=="oscd" else (4,10)
    if arch=="self":
        return TransformerSelfAttention(sar_channels=sar,opt_channels=opt,embed_dim=embed_dim,
            patch_stride=4,num_blocks=2,num_heads=heads,mlp_ratio=4,dropout=dropout)
    return TransformerCrossAttention(sar_channels=sar,opt_channels=opt,embed_dim=embed_dim,
        patch_stride=4,num_self_blocks=2,num_cross_blocks=1,num_heads=heads,mlp_ratio=4,dropout=dropout)

def oscd_objects():
    from src.training.train_transformer_self_attention import TransformerAsyncDataset,_RandomCropDataset
    from src.training.train_sar_baseline import CitySubset,split_train_cities,assert_city_separation
    train=TransformerAsyncDataset(split="train"); test=TransformerAsyncDataset(split="test")
    train_cities,val_cities=split_train_cities(train.cities); test_cities=list(test.cities)
    assert_city_separation(train_cities,val_cities,test_cities)
    return train,test,train_cities,val_cities,test_cities,_RandomCropDataset,CitySubset

def scenario_objects(split:str,return_mask:bool=False):
    from src.datasets.scenario2_dataset import Scenario2Dataset
    return Scenario2Dataset(split=split,mode="async",return_mask=return_mask)

def city_sample(ds, city:str):
    from src.datasets.oscd_dataset import ORBITS
    from src.training.train_transformer_self_attention import _sample_at_orbit
    return _sample_at_orbit(ds,ds.cities.index(city),int(ORBITS[city][0]))

def count_rows_for_split(model, dataset:str, split:str, thresholds:list[float],device):
    """Run inference once per image/site and aggregate threshold confusion counts."""
    model.eval(); totals={t:{"tp":0,"tn":0,"fp":0,"fn":0} for t in thresholds}
    n_pixels=0; n_valid=0; positive=0; warnings=[]
    if dataset=="oscd":
        tr,te,tc,vc,xc,*_=oscd_objects(); ds=tr if split=="validation" else te; ids=vc if split=="validation" else xc
        samples=[(c,city_sample(ds,c)) for c in ids]
        valid_policy="all target pixels; OSCD loader has no explicit valid mask"
    else:
        ds=scenario_objects("validation" if split=="validation" else "test",return_mask=True)
        samples=[(ds.sites[i],ds[i]) for i in range(len(ds))]
        ids=[x for x,_ in samples]
        valid_policy="all pixels for primary metrics; raw SAR NaN-derived valid count recorded separately"
    with torch.inference_mode():
        for name,s in samples:
            x=s["image"].unsqueeze(0).to(device); y=s["label"].unsqueeze(0).to(device)
            logits=model(x)
            if logits.shape != y.shape: raise ValueError(f"{name}: output/label shape mismatch {logits.shape} vs {y.shape}")
            if not torch.isfinite(logits).all(): warnings.append(f"{name}: non-finite logits")
            probs=torch.sigmoid(logits)
            n_pixels += y.numel(); positive += int((y>=.5).sum().item())
            n_valid += int(s["valid_mask"].sum().item()) if "valid_mask" in s else y.numel()
            for t in thresholds:
                c=counts_from_probabilities(probs,y,t)
                for k in totals[t]: totals[t][k]+=c[k]
    rows=[]
    for t,c in totals.items():
        rates=rates_from_counts(c)
        rows.append({"dataset":dataset,"split":split,"threshold":t,**c,**rates,
            "predicted_positive_fraction":(c["tp"]+c["fp"])/n_pixels if n_pixels else None,
            "evaluated_pixels":n_pixels,"valid_pixels":n_valid,"target_positive_pixels":positive,
            "valid_pixel_policy":valid_policy,"sites_or_cities":";".join(ids),"warnings":";".join(warnings)})
    return rows

def thresholds_command():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; aborting planned full-resolution tuning workload")
    verify_split(); device=torch.device("cuda"); all_rows=[]; selected={}; test_rows=[]
    for (dataset,arch),paths in GROUPS.items():
        cfg=json.loads(paths["metric"].read_text(encoding="utf-8")); ck=torch.load(paths["baseline"],map_location="cpu",weights_only=False)
        repro=cfg.get("reproducibility",{}); model=make_model(dataset,arch)
        model.load_state_dict(ck["model_state_dict"],strict=True); model.to(device)
        base_threshold=float(cfg.get("threshold",.5))
        val=count_rows_for_split(model,dataset,"validation",THRESHOLDS,device)
        best=max(val,key=lambda r:(r["f1"] if r["f1"] is not None else -1, -abs(r["threshold"]-base_threshold)))
        fine=sorted(set([r["threshold"] for r in val]+[round(best["threshold"]+d,4) for d in [-.05,-.025,-.0125,.0125,.025,.05] if 0<best["threshold"]+d<1]))
        if len(fine)>len(val): val=count_rows_for_split(model,dataset,"validation",fine,device)
        # Deterministic tie break toward higher precision, then threshold nearest legacy .5.
        best=max(val,key=lambda r:(r["f1"] if r["f1"] is not None else -1,r["precision"] if r["precision"] is not None else -1,-abs(r["threshold"]-.5)))
        selected[f"{dataset}_{arch}"]={"threshold":best["threshold"],"validation_metrics":best,
            "baseline_threshold":base_threshold,"baseline_checkpoint":str(paths["baseline"]),"checkpoint_epoch":ck.get("epoch"),
            "baseline_test_f1_at_0_5":cfg.get("final_test_f1"),"baseline_saved_metrics":str(paths["metric"]),
            "architecture":arch,"dataset":dataset,"selection":"maximum pooled validation F1; ties: precision then proximity to 0.5"}
        for r in val: all_rows.append({"architecture":arch,"checkpoint":"original","checkpoint_epoch":ck.get("epoch"),**r})
        # Test only after this checkpoint's threshold is selected on validation.
        tr=count_rows_for_split(model,dataset,"test",[.5,float(best["threshold"])],device)
        for r in tr: test_rows.append({"architecture":arch,"checkpoint":"original","checkpoint_epoch":ck.get("epoch"),**r})
        del model; torch.cuda.empty_cache()
    write_csv(HERE/"threshold_validation_sweep.csv",all_rows)
    write_csv(HERE/"baseline_test_thresholds.csv",test_rows)
    (HERE/"selected_baseline_thresholds.json").write_text(json.dumps({"environment":environment(),"split_sha256":verify_split(),"thresholds":selected},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:{"threshold":v["threshold"],"val_f1":v["validation_metrics"]["f1"]} for k,v in selected.items()},indent=2))

def training_command(dataset,arch,lr,wd,seed,run_id,epochs=50):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    split_hash=verify_split(); set_seed(seed); device=torch.device("cuda")
    run=HERE/"runs"/run_id
    if run.exists(): raise FileExistsError(f"Refusing to overwrite {run}")
    run.mkdir(parents=True)
    if dataset=="oscd":
        train,test,train_ids,val_ids,test_ids,Crop,Subset=oscd_objects()
        cropset=Crop(Subset(train,train_ids),32,64)
        val_ds=train
    else:
        from src.datasets.scenario2_dataset import RandomCropScenario2,load_split
        split=load_split(); train_ids=split["train"]; val_ids=split["validation"]; test_ids=split["test"]
        train=scenario_objects("train"); val_ds=scenario_objects("validation"); test=scenario_objects("test")
        cropset=RandomCropScenario2(train,32,64)
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(cropset,batch_size=64,shuffle=True,num_workers=0,generator=gen,pin_memory=True)
    model=make_model(dataset,arch).to(device); params=count_parameters(model)
    pos=41.0 if dataset=="oscd" else 94.0
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos],device=device))
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    from src.evaluation.metrics import counts_from_logits,rates_from_counts
    @torch.inference_mode()
    def val_eval():
        model.eval(); c={"tp":0,"tn":0,"fp":0,"fn":0}
        if dataset=="oscd":
            for city in val_ids:
                s=city_sample(val_ds,city); x=s["image"].unsqueeze(0).to(device); y=s["label"].unsqueeze(0).to(device)
                cc=counts_from_logits(model(x),y)
                for k in c:c[k]+=cc[k]
        else:
            for city in val_ids:
                s=val_ds[val_ds.sites.index(city)]; x=s["image"].unsqueeze(0).to(device); y=s["label"].unsqueeze(0).to(device)
                cc=counts_from_logits(model(x),y)
                for k in c:c[k]+=cc[k]
        return {**c,**rates_from_counts(c)}
    code_paths=[ROOT/"src/models/transformer_self_attention.py",ROOT/"src/models/transformer_cross_attention.py",ROOT/"src/evaluation/metrics.py",ROOT/"src/training/train_sar_baseline.py",ROOT/"src/training/train_transformer_self_attention.py",ROOT/"src/training/train_transformer_cross_attention.py",ROOT/"src/datasets/scenario2_dataset.py"]
    code_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    code_hashes[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cfg={"run_id":run_id,"dataset":dataset,"architecture":arch,"seed":seed,"lr":lr,"weight_decay":wd,
         "dropout":0.0,"embed_dim":128,"heads":4,"depth_self_per_modality":2,"cross_blocks":1 if arch=="cross" else 0,
         "patch_stride":4,"mlp_ratio":4,"epochs":epochs,"batch_size":64,"crop_size":32,"crops_per_city_or_site":64,
         "loss":"BCEWithLogitsLoss","pos_weight":pos,"selection":"highest pooled validation F1 at 0.5; no test used during training",
         "optimizer":"Adam","split_sha256":split_hash,"train_ids":train_ids,"validation_ids":val_ids,"test_ids":test_ids,
         "input_channels":15 if dataset=="oscd" else 14,"channel_order":"OSCD S1 T1 VV,S1 T2 VV,S2 T1 B01-B12; S2 S1 T1 VV,VH,T2 VV,VH,S2 T1 ten configured bands",
         "environment":environment(),"source_sha256":code_hashes,"start_utc":datetime.now(timezone.utc).isoformat()}
    (run/"config.json").write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")
    best_f1=-1.; best_epoch=None; hist=[]; t0=time.perf_counter()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    for ep in range(1,epochs+1):
        model.train(); loss=train_one_epoch(model,loader,criterion,optimizer,device); vm=val_eval(); row={"epoch":ep,"train_loss":loss,**{f"val_{k}":v for k,v in vm.items()}}; hist.append(row)
        shown_f1 = "undefined" if vm["f1"] is None else f"{vm['f1']:.6f}"
        print(f"{run_id} epoch={ep} loss={loss:.5f} val_f1={shown_f1}",flush=True)
        torch.save({"model_state_dict":model.state_dict(),"epoch":ep,"validation_f1":vm["f1"],"config":cfg},run/"last.pt")
        if vm["f1"] is not None and vm["f1"]>best_f1:
            best_f1=vm["f1"];best_epoch=ep;torch.save({"model_state_dict":model.state_dict(),"epoch":ep,"validation_f1":vm["f1"],"config":cfg},run/"best.pt")
        write_csv(run/"training_history.csv",hist)
    elapsed=time.perf_counter()-t0
    if best_epoch is None:
        # Preserve the completed run but report that checkpoint selection had
        # no defined validation F1; do not silently evaluate its test split.
        (run/"failure.json").write_text(json.dumps({"reason":"validation F1 undefined at every epoch; no checkpoint selected","epochs_completed":len(hist)},indent=2)+"\n",encoding="utf-8")
        raise RuntimeError(f"{run_id}: validation F1 remained undefined; test was not evaluated")
    # Reload the checkpoint explicitly selected by validation.
    best=torch.load(run/"best.pt",map_location="cpu",weights_only=False); model.load_state_dict(best["model_state_dict"]);model.to(device)
    final_val=val_eval()
    result={"run_id":run_id,"dataset":dataset,"architecture":arch,"seed":seed,"lr":lr,"weight_decay":wd,
        "best_epoch":best_epoch,"validation_at_0_5":final_val,"training_seconds":elapsed,
        "peak_vram_mb":round(torch.cuda.max_memory_allocated()/1024**2,1),"best_checkpoint":"best.pt",
        "test_evaluated":False,"end_utc":datetime.now(timezone.utc).isoformat(),"parameters":params}
    (run/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))

def load_run_model(run_id):
    run=HERE/"runs"/run_id; cfg=json.loads((run/"config.json").read_text(encoding="utf-8"))
    ck=torch.load(run/"best.pt",map_location="cpu",weights_only=False)
    model=make_model(cfg["dataset"],cfg["architecture"],dropout=cfg["dropout"],embed_dim=cfg["embed_dim"],heads=cfg["heads"])
    model.load_state_dict(ck["model_state_dict"],strict=True)
    return run,cfg,ck,model

def threshold_run(run_id):
    run,cfg,ck,model=load_run_model(run_id); device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model.to(device)
    rows=count_rows_for_split(model,cfg["dataset"],"validation",THRESHOLDS,device)
    coarse=max(rows,key=lambda r:(r["f1"] if r["f1"] is not None else -1,-abs(r["threshold"]-.5)))
    fine=sorted(set([r["threshold"] for r in rows]+[round(coarse["threshold"]+d,4) for d in [-.05,-.025,-.0125,.0125,.025,.05] if 0<coarse["threshold"]+d<1]))
    if len(fine)>len(rows): rows=count_rows_for_split(model,cfg["dataset"],"validation",fine,device)
    best=max(rows,key=lambda r:(r["f1"] if r["f1"] is not None else -1,r["precision"] if r["precision"] is not None else -1,-abs(r["threshold"]-.5)))
    write_csv(run/"validation_threshold_sweep.csv",rows)
    result=json.loads((run/"result.json").read_text(encoding="utf-8")); result["validation_selected_threshold"]={"threshold":best["threshold"],"metrics":best,"rule":"maximum pooled validation F1; ties prefer precision, then proximity to 0.5"}
    (run/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(run_id,json.dumps({"threshold":best["threshold"],"validation_f1":best["f1"]}))

def select_tuning():
    selected={}
    for key,paths in GROUPS.items():
        dataset,arch=key; candidates=[]
        for p in (HERE/"runs").iterdir():
            rp=p/"result.json";cp=p/"config.json"
            if not rp.exists() or not cp.exists(): continue
            cfg=json.loads(cp.read_text(encoding="utf-8"));res=json.loads(rp.read_text(encoding="utf-8"))
            if cfg.get("status","valid")!="valid" or cfg["dataset"]!=dataset or cfg["architecture"]!=arch or cfg["seed"]!=42: continue
            candidates.append(res)
        if not candidates: raise RuntimeError(f"No valid candidate runs for {key}")
        winner=max(candidates,key=lambda r:r["validation_at_0_5"]["f1"] if r["validation_at_0_5"]["f1"] is not None else -1)
        base=json.loads(paths["metric"].read_text(encoding="utf-8")); base_val=base["validation_best_f1"]
        selected[f"{dataset}_{arch}"]={"dataset":dataset,"architecture":arch,"selected_run_id":winner["run_id"],
            "selected_run_validation_f1_at_0_5":winner["validation_at_0_5"]["f1"],"baseline_validation_f1_at_0_5":base_val,
            "delta_validation_f1":winner["validation_at_0_5"]["f1"]-base_val,"lr":winner["lr"],"weight_decay":winner["weight_decay"],
            "seed":42,"selection":"maximum validation pooled F1 at 0.5 among two preregistered tuning settings; test not accessed",
            "candidate_run_ids":[r["run_id"] for r in candidates]}
    doc={"split_sha256":verify_split(),"selection_utc":datetime.now(timezone.utc).isoformat(),"selected":selected}
    (HERE/"selected_tuning_configs.json").write_text(json.dumps(doc,indent=2)+"\n",encoding="utf-8")
    for v in selected.values(): threshold_run(v["selected_run_id"])
    print(json.dumps(selected,indent=2))

def final_evaluate():
    frozen=json.loads((HERE/"selected_tuning_configs.json").read_text(encoding="utf-8"))["selected"]
    outrows=[]
    # Existing baseline checkpoints: evaluate both legacy and validation-selected thresholds.
    for (ds,arch),paths in GROUPS.items():
        name=f"{ds}_{arch}"; model=make_model(ds,arch);ck=torch.load(paths["baseline"],map_location="cpu",weights_only=False);model.load_state_dict(ck["model_state_dict"]);model.to("cuda")
        baseline_threshold=float(json.loads((HERE/"selected_baseline_thresholds.json").read_text(encoding="utf-8"))["thresholds"][name]["threshold"])
        rows=count_rows_for_split(model,ds,"test",[.5,baseline_threshold],torch.device("cuda"))
        for r in rows: outrows.append({"dataset":ds,"architecture":arch,"run_id":"original_checkpoint","checkpoint":"original","seed":42,"validation_selected_threshold":baseline_threshold,**r})
        del model;torch.cuda.empty_cache()
    # Tuned seed-42 winners plus repeat seeds. The selection is frozen before this function runs.
    for name,sel in frozen.items():
        run_id=sel["selected_run_id"]; reps=[run_id]
        for p in (HERE/"runs").iterdir():
            q=p/"result.json"
            if not q.exists(): continue
            d=json.loads(q.read_text(encoding="utf-8"))
            if d["dataset"]==sel["dataset"] and d["architecture"]==sel["architecture"] and d["lr"]==sel["lr"] and d["weight_decay"]==sel["weight_decay"] and d["seed"] in (43,44): reps.append(d["run_id"])
        for rid in reps:
            run,cfg,ck,model=load_run_model(rid);model.to("cuda")
            if not (run/"validation_threshold_sweep.csv").exists(): threshold_run(rid)
            res=json.loads((run/"result.json").read_text(encoding="utf-8")); threshold=float(res["validation_selected_threshold"]["threshold"])
            rows=count_rows_for_split(model,cfg["dataset"],"test",sorted(set([.5,threshold])),torch.device("cuda"))
            for r in rows: outrows.append({"dataset":cfg["dataset"],"architecture":cfg["architecture"],"run_id":rid,"checkpoint":"tuned","seed":cfg["seed"],"validation_selected_threshold":threshold,**r})
            res["test_evaluated"]=True;res["final_test_rows"]=rows
            (run/"result.json").write_text(json.dumps(res,indent=2)+"\n",encoding="utf-8")
            del model;torch.cuda.empty_cache()
    write_csv(HERE/"final_test_comparison.csv",outrows)
    print(f"Wrote {len(outrows)} frozen test rows after validation selection")

def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="command",required=True)
    sp.add_parser("thresholds")
    sp.add_parser("select")
    sp.add_parser("final-evaluate")
    pr=sp.add_parser("threshold-run");pr.add_argument("--run-id",required=True)
    p=sp.add_parser("train");p.add_argument("--dataset",choices=["oscd","s2"],required=True);p.add_argument("--arch",choices=["self","cross"],required=True);p.add_argument("--lr",type=float,required=True);p.add_argument("--weight-decay",type=float,required=True);p.add_argument("--seed",type=int,default=42);p.add_argument("--run-id",required=True);p.add_argument("--epochs",type=int,default=50)
    a=ap.parse_args()
    if a.command=="thresholds": thresholds_command()
    elif a.command=="select": select_tuning()
    elif a.command=="final-evaluate": final_evaluate()
    elif a.command=="threshold-run": threshold_run(a.run_id)
    else: training_command(a.dataset,a.arch,a.lr,a.weight_decay,a.seed,a.run_id,a.epochs)
if __name__=="__main__": main()
