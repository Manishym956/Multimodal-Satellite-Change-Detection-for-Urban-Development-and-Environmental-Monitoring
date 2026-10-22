cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
until grep -q "P1V DONE\|MISMATCH" logs/robust/p1_verify2.log; do sleep 10; done
grep -q MISMATCH logs/robust/p1_verify2.log && { echo "STOP: reeval mismatch"; exit 1; }
.venv/Scripts/python.exe -m src.training.train_robust --name codecheck_hafner_full --protocol hafner --seed 1 --eval-every 390 2>&1 | tail -1
.venv/Scripts/python.exe - <<'PY'
import json, torch
a = torch.load("checkpoints/robust/hafner_full_seed1.pt", map_location="cpu", weights_only=False)["model_state_dict"]
b = torch.load("checkpoints/robust/codecheck_hafner_full_seed1.pt", map_location="cpu", weights_only=False)["model_state_dict"]
same_w = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
ma = json.load(open("experiments/robust/hafner_full_seed1/metrics.json", encoding="utf-8"))
mb = json.load(open("experiments/robust/codecheck_hafner_full_seed1/metrics.json", encoding="utf-8"))
same_m = all(ma["test"][c] == mb["test"][c] for c in ma["test"])
print(f"CODECHECK weights_identical={same_w} stored_metrics_identical={same_m}")
PY
.venv/Scripts/python.exe gap_analysis_robust.py
echo CC DONE
