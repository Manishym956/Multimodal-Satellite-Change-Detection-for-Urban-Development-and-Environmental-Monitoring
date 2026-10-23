# Step 6 parity: train_rrd.py (--arch dsunet --sar hafner_vv) with the B mixture must reproduce p2_dropout_seed1 exactly.
cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
test -e checkpoints/robust/parity_rrd_seed1.pt && { echo "parity checkpoint exists, refusing"; exit 1; }
date -Iseconds
.venv/Scripts/python.exe -m src.training.train_rrd --name parity_rrd --arch dsunet --sar hafner_vv --seed 1 --steps 2500 --eval-every 25 \
    --train-conditions full=0.5,no_o2=0.25,no_opt=0.25 2>&1 | tail -1
.venv/Scripts/python.exe - <<'PY'
import json, torch
a = torch.load("checkpoints/robust/p2_dropout_seed1.pt", map_location="cpu", weights_only=False)
b = torch.load("checkpoints/robust/parity_rrd_seed1.pt", map_location="cpu", weights_only=False)
same_w = a["model_state_dict"].keys() == b["model_state_dict"].keys() and all(torch.equal(a["model_state_dict"][k], b["model_state_dict"][k]) for k in a["model_state_dict"])
ma = json.load(open("experiments/robust/p2_dropout_seed1/metrics.json", encoding="utf-8"))
mb = json.load(open("experiments/robust/parity_rrd_seed1/metrics.json", encoding="utf-8"))
same_hist = ma["history"] == mb["history"]
same_sel = (a["step"], a["thresholds"]) == (b["step"], b["thresholds"]) and ma["selected_step"] == mb["selected_step"]
same_test = all(ma["test"][c] == mb["test"][c] for c in ma["test"])
print(f"PARITY weights_identical={same_w} val_history_identical={same_hist} selection_identical={same_sel} stored_test_identical={same_test}")
PY
date -Iseconds
echo PARITY DONE
