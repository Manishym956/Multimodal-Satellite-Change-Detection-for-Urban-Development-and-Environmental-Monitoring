# Step 6 (RRD RQ1, RQ4): train the four Step 6 arms, seeds 1-10, then the registered analysis and audit (PHASE6_PREREGISTRATION.md).
# Gates: parity passed, smoke test 0 failed, every pre-registered hash unchanged. Any failure stops the chain.
cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
grep -q "PARITY weights_identical=True val_history_identical=True selection_identical=True stored_test_identical=True" logs/robust/p6_parity.log \
    || { echo "STOP: parity not passed"; exit 1; }
grep -q '"n_failed": 0' experiments/robust/phase6/smoke/smoke_results.json || { echo "STOP: smoke test not passed"; exit 1; }
.venv/Scripts/python.exe verify_prereg_hashes.py experiments/robust/PHASE6_PREREGISTRATION.md | tail -1
.venv/Scripts/python.exe verify_prereg_hashes.py experiments/robust/PHASE6_PREREGISTRATION.md > /dev/null || { echo "STOP: pre-registered hash changed"; exit 1; }
for s in 1 2 3 4 5 6 7 8 9 10; do
  for arm in "p6_unet_sar unet_sar full=1" "p6_unet_opt unet_opt full=1" "p6_unet_fusion unet_fusion full=1" "p6_async dsunet no_o2=1"; do
    set -- $arm
    test -f experiments/robust/$1_seed$s/metrics.json && { echo "$1 seed $s exists, skip"; continue; }
    echo "== $1 seed $s $(date -Iseconds)"
    .venv/Scripts/python.exe -m src.training.train_rrd --name $1 --arch $2 --sar hafner_vv --seed $s --steps 2500 --eval-every 25 \
        --train-conditions $3 2>&1 | tail -1
    test -f experiments/robust/$1_seed$s/metrics.json || { echo "STOP: $1 seed $s did not finish"; exit 1; }
  done
done
echo "== analyze $(date -Iseconds)"
.venv/Scripts/python.exe analyze_phase6.py > logs/robust/p6_analyze.log 2>&1 || { echo "STOP: analysis failed"; exit 1; }
echo "== audit $(date -Iseconds)"
.venv/Scripts/python.exe audit_p6.py > logs/robust/p6_audit.log 2>&1; tail -1 logs/robust/p6_audit.log
echo "P6 DONE $(date -Iseconds)"
