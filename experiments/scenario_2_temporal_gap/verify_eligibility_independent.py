import csv, hashlib, json
from pathlib import Path
from collections import Counter, defaultdict
import rasterio

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/sen12_mt_urbanmapping/multimodal_cd_dataset"
SPLIT = ROOT / "experiments/scenario_2/configs/split_definition.json"
OUT = ROOT / "experiments/scenario_2_temporal_gap"
TARGET = (2019, 12)
EXPECTED_HASH = "5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27"

def datekey(r): return (int(r["year"]), int(r["month"]))
def month_gap(a,b): return (b[0]*12+b[1])-(a[0]*12+a[1])
def fmt(k): return f"{k[0]:04d}-{k[1]:02d}"
def file_for(site, mod, k):
    prefix = {"s1":"s1", "s2":"s2", "buildings":"buildings"}[mod]
    return DATA/site/mod/f"{prefix}_{site}_{k[0]}_{k[1]:02d}.tif"
def inspect(path):
    if not path.is_file(): return {"exists":False,"bytes":None,"opens":False,"error":"missing"}
    size=path.stat().st_size
    if size<=0: return {"exists":True,"bytes":size,"opens":False,"error":"empty"}
    try:
        with rasterio.open(path) as ds:
            return {"exists":True,"bytes":size,"opens":True,"bands":ds.count,"height":ds.height,"width":ds.width,"dtypes":list(ds.dtypes)}
    except Exception as e:
        return {"exists":True,"bytes":size,"opens":False,"error":type(e).__name__+": "+str(e)}

raw=SPLIT.read_bytes(); split=json.loads(raw)
meta=json.loads((DATA/"metadata.json").read_text(encoding="utf-8"))
bad=json.loads((DATA/"bad_data.json").read_text(encoding="utf-8"))
split_hash=hashlib.sha256(raw).hexdigest()
sites=[]
for part in ("train","validation","test"):
    for site in split[part]: sites.append((site,part))
assert len(sites)==28 and len({s for s,_ in sites})==28


months=set(); source_notes=[]; indexed={}
for site,part in sites:
    records=meta.get(site)
    if records is None: source_notes.append(f"{site}: absent metadata site"); records=[]
    ix={}
    for i,r in enumerate(records):
        k=datekey(r)
        if k in ix: source_notes.append(f"{site}: duplicate metadata date {fmt(k)} (indices {ix[k]['index']},{i})")
        else: ix[k]={"index":i,**r}
        if k<TARGET: months.add(k)
    indexed[site]=ix
    folder=DATA/site/"s2"
    if folder.is_dir():
        for p in folder.glob(f"s2_{site}_*.tif"):
            try:
                tail=p.stem[len(f"s2_{site}_"):]; y,m=map(int,tail.split("_"))
                if 1<=m<=12 and (y,m)<TARGET: months.add((y,m))
            except Exception: source_notes.append(f"{site}: unparsed optical filename {p.name}")
months=sorted(months)
bad_index_notes=[]
for site,_ in sites:
    for mod in ("S1","S2"):
        for i in bad.get(site,{}).get(mod,[]):
            if not isinstance(i,int) or i<0 or i>=len(meta.get(site,[])):
                bad_index_notes.append(f"{site}: bad_data {mod} index out of range/noninteger: {i}")

def comp(site,k,mod):
    f=inspect(file_for(site,mod,k)); rec=indexed[site].get(k)
    flagname={"s1":"s1","s2":"s2","buildings":"buildings"}[mod]
    bkey="S1" if mod=="s1" else "S2"
    bi=bool(rec is not None and rec["index"] in set(bad.get(site,{}).get(bkey,[])))
    masked=bool(rec and rec.get("masked",False))
    flag=None if rec is None else bool(rec.get(flagname,False))
    metadata_usable=bool(rec is not None and flag and not masked and (not bi if mod in ("s1","s2") else True))
    # A disagreement is unresolved because the local source provides no
    # authoritative reason for false/bad/masked and raster values cannot settle it.
    usable=(f["exists"] and f["opens"] and metadata_usable)
    return {"file":str(file_for(site,mod,k)),"file_exists":f["exists"],"file_nonempty":bool(f.get("bytes",0) or 0),"file_opens":f["opens"],"file_bytes":f["bytes"],"raster":f,"metadata_record":rec is not None,"metadata_flag":flag,"bad_index":bi,"masked":masked,"metadata_usable_by_repository_convention":metadata_usable,"confirmed_component":usable,"conflict":bool(f["exists"] != metadata_usable or (f["exists"] and not f["opens"]))}

rows=[]
for site,part in sites:
    for k in months:
        c={m:comp(site,k,m) for m in ("s2","s1","buildings")}
        target_sar=comp(site,TARGET,"s1"); target_lbl=comp(site,TARGET,"buildings")
        # Variable-gap pair requires candidate optical + candidate SAR + candidate
        # buildings + fixed target SAR + fixed target buildings.
        comps={"historical_optical":c["s2"],"historical_sar":c["s1"],"historical_buildings":c["buildings"],"target_sar":target_sar,"target_buildings":target_lbl}
        conflicts=[name for name,v in comps.items() if v["conflict"]]
        missing=[name for name,v in comps.items() if not v["file_exists"]]
        corrupt=[name for name,v in comps.items() if v["file_exists"] and not v["file_opens"]]
        # Ineligible only where required file is absent/corrupt AND no metadata
        # usability support. Conflicts in present, readable files remain ambiguous.
        hard_absence=[name for name,v in comps.items() if (not v["file_exists"] or not v["file_opens"]) and not v["metadata_usable_by_repository_convention"]]
        if hard_absence:
            classification="ineligible"
        elif conflicts:
            classification="ambiguous"
        else:
            classification="confirmed_eligible"
        reasons=[]
        for name,v in comps.items():
            if not v["file_exists"]: reasons.append(name+" file missing")
            if v["file_exists"] and not v["file_opens"]: reasons.append(name+" file unreadable")
            if v["conflict"]:
                reasons.append(name+" file/metadata-bad-masked disagreement; flag semantics do not establish raster validity")
            if not v["metadata_record"]: reasons.append(name+" metadata record missing")
            if v["bad_index"]: reasons.append(name+" bad-index flag")
            if v["masked"]: reasons.append(name+" masked flag")
            if v["metadata_flag"] is False: reasons.append(name+" metadata availability flag false")
        rows.append({"site_id":site,"split":part,"historical_optical_date":fmt(k),"target_sar_date":fmt(TARGET),"gap_months":month_gap(k,TARGET),"classification":classification,"reasons":"; ".join(dict.fromkeys(reasons)),"components_json":json.dumps(comps,separators=(",",":"))})

# fixed set uses confirmed eligibility at every one of the 23 gaps, but March 2018
# remains the existing experiment and is separately reported.
candidate_months=[m for m in months if m!=(2018,3)]
by_site_date={(r["site_id"],r["historical_optical_date"]):r for r in rows}
fixed_sites=[s for s,_ in sites if all(by_site_date[(s,fmt(k))]["classification"]=="confirmed_eligible" for k in candidate_months)]
counts=Counter(r["classification"] for r in rows)
by_month=[]
for k in months:
    subset=[r for r in rows if r["historical_optical_date"]==fmt(k)]
    item={"historical_optical_date":fmt(k),"gap_months":month_gap(k,TARGET),"role":"existing_trained_interval" if k==(2018,3) else "candidate_untrained","sites":len(subset)}
    for cl in ("confirmed_eligible","ambiguous","ineligible"):
        item[cl]=sum(r["classification"]==cl for r in subset)
        for part in ("train","validation","test"):
            item[cl+"_"+part]=sum(r["classification"]==cl and r["split"]==part for r in subset)
    item["eligible_sites"]={part:[r["site_id"] for r in subset if r["split"]==part and r["classification"]=="confirmed_eligible"] for part in ("train","validation","test")}
    by_month.append(item)

disagree=[r for r in rows if json.loads(r["components_json"])["historical_optical"]["conflict"]]
march=[r for r in rows if r["historical_optical_date"]=="2018-03"]
report={"purpose":"Read-only independent eligibility audit; no model training or performance metrics.","method":{"gap_convention":"calendar-month index difference: (2019*12+12)-(year*12+month); March 2018 to December 2019 = 21 months.","candidate_universe":"Union of pre-target metadata dates and optical filenames across locked sites.","pair_requirements":["candidate S2","candidate S1","candidate buildings","2019-12 S1","2019-12 buildings"],"classification":"confirmed requires nonempty readable files and metadata convention usable; any present-file versus metadata/bad/masked conflict is ambiguous; missing or unreadable required file is ineligible only when metadata also does not support usability. Raster plausibility is not interpreted as quality.","metadata_semantics_limit":"Repository docs/code describe Boolean availability flags, bad indices as known-bad exclusions, and masked=true exclusion, but no authoritative upstream definition/reason was located for each flag. Conflicts are therefore left ambiguous."},"quality_flag_evidence":[{"source":str(ROOT/"src/training/temporal_pairing_analysis.py"),"evidence":"is_s1_usable/is_s2_usable require metadata Boolean true, no bad_data index, and masked=false; is_bld_usable requires buildings=true and masked=false."},{"source":str(ROOT/"src/datasets/scenario2_dataset.py"),"evidence":"preprocess_site reads fixed-date files directly and does not consult metadata.json or bad_data.json; optical is read without a quality mask; SAR pixel validity is based on NaN absence."},{"source":str(ROOT/"docs/SCENARIO_2_DATASET_INSPECTION.md"),"evidence":"Sections 9-10 describe temporal availability and known-bad indices, but do not define why individual flags are false or establish an upstream reason for masked=true."}],"sources":{"split":str(SPLIT),"split_sha256":split_hash,"expected_sha256":EXPECTED_HASH,"hash_matches":split_hash==EXPECTED_HASH,"metadata":str(DATA/"metadata.json"),"bad_data":str(DATA/"bad_data.json"),"loader":str(ROOT/"src/datasets/scenario2_dataset.py"),"quality_flag_code":str(ROOT/"src/training/temporal_pairing_analysis.py"),"inspection_doc":str(ROOT/"docs/SCENARIO_2_DATASET_INSPECTION.md")},"split_counts":{k:len(split[k]) for k in ("train","validation","test")},"candidate_months":[fmt(k) for k in months],"counts":{"rows":len(rows),"sites":len(sites),"months":len(months),**dict(counts),"historical_optical_file_metadata_disagreements":len(disagree),"march_2018_confirmed":sum(r["classification"]=="confirmed_eligible" for r in march),"march_2018_ambiguous":sum(r["classification"]=="ambiguous" for r in march),"march_2018_ineligible":sum(r["classification"]=="ineligible" for r in march)},"by_month":by_month,"fixed_site_intersection":{"months_in_intersection":len(candidate_months),"definition":"Sites confirmed eligible for every untrained candidate month (excluding the existing 2018-03 trained interval).","count":len(fixed_sites),"by_split":{p:[s for s in fixed_sites if dict(sites)[s]==p] for p in ("train","validation","test")},"variable_site_comparison_warning":"Per-gap comparisons with variable sites confound temporal gap with site composition."},"disagreement_records":[{"site_id":r["site_id"],"split":r["split"],"date":r["historical_optical_date"],"details":json.loads(r["components_json"])["historical_optical"]} for r in disagree],"source_notes":source_notes+bad_index_notes,"row_uniqueness":{"all_rows_unique":len({(r['site_id'],r['historical_optical_date']) for r in rows})==len(rows),"all_rows_classified_once":all(r["classification"] in ("confirmed_eligible","ambiguous","ineligible") for r in rows)},"recommendation":"Resolve metadata conflicts with dataset-provenance/authoritative quality documentation before training. If unresolved, design a predeclared common-site subset or report variable-site composition and perform sensitivity analyses; do not call this inventory a trained temporal-gap result."}
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/"eligibility_verification.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(OUT/"eligibility_verification.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
(OUT/"fixed_site_intersection.json").write_text(json.dumps(report["fixed_site_intersection"],indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps({"split_sha256":split_hash,"rows":len(rows),"months":len(months),"counts":report["counts"],"fixed_site_intersection":report["fixed_site_intersection"],"outputs":[str(OUT/"eligibility_verification.csv"),str(OUT/"eligibility_verification.json"),str(OUT/"fixed_site_intersection.json")]},indent=2))
