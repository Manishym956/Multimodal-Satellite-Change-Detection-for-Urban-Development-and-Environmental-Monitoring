# Step 7 (RRD RQ3): train the four Step 7 arms, seeds 1-10, then the registered evaluation, analysis and audit
# (PHASE7_PREREGISTRATION.md). Gates: Step 6 finished, smoke test 0 failed, every pre-registered hash unchanged.
cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
grep -q "P6 DONE" logs/robust/p6_run.log || { echo "STOP: Step 6 not finished"; exit 1; }
grep -q '"n_failed": 0' experiments/robust/phase7/smoke/smoke_results.json || { echo "STOP: smoke test not passed"; exit 1; }
.venv/Scripts/python.exe verify_prereg_hashes.py experiments/robust/PHASE7_PREREGISTRATION.md | tail -1
.venv/Scripts/python.exe verify_prereg_hashes.py experiments/robust/PHASE7_PREREGISTRATION.md > /dev/null || { echo "STOP: pre-registered hash changed"; exit 1; }
for s in 1 2 3 4 5 6 7 8 9 10; do
  for arm in "p7_unet_vh unet_sar vh" "p7_unet_vvvh unet_sar vvvh" "p7_dsunet_vh dsunet vh" "p7_dsunet_vvvh dsunet vvvh"; do
    set -- $arm
    test -f experiments/robust/$1_seed$s/metrics.json && { echo "$1 seed $s exists, skip"; continue; }
    echo "== $1 seed $s $(date -Iseconds)"
    .venv/Scripts/python.exe -m src.training.train_rrd --name $1 --arch $2 --sar $3 --seed $s --steps 2500 --eval-every 25 \
        --train-conditions full=1 2>&1 | tail -1
    test -f experiments/robust/$1_seed$s/metrics.json || { echo "STOP: $1 seed $s did not finish"; exit 1; }
  done
done
echo "== eval $(date -Iseconds)"
.venv/Scripts/python.exe eval_phase7.py || { echo "STOP: eval_phase7 failed (anchor or error)"; exit 1; }
echo "== analyze $(date -Iseconds)"
.venv/Scripts/python.exe analyze_phase7.py > logs/robust/p7_analyze.log 2>&1 || { echo "STOP: analysis failed"; exit 1; }
echo "== audit $(date -Iseconds)"
.venv/Scripts/python.exe audit_p7.py > logs/robust/p7_audit.log 2>&1; tail -1 logs/robust/p7_audit.log
echo "P7 DONE $(date -Iseconds)"
