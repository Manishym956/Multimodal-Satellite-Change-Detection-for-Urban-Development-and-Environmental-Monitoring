# Step 4B: train p4b_spatial seeds 1-10, then the registered evaluation and analysis (PHASE4B_PREREGISTRATION.md).
# Gates: parity passed, smoke test 0 failed, every pre-registered hash unchanged. Any failure stops the chain.
cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
grep -q "PARITY weights_identical=True val_history_identical=True selection_identical=True stored_test_identical=True" logs/robust/p4b_parity.log \
    || { echo "STOP: parity not passed"; exit 1; }
grep -q '"n_failed": 0' experiments/robust/phase4b/smoke/smoke_results.json || { echo "STOP: smoke test not passed"; exit 1; }
.venv/Scripts/python.exe verify_prereg_hashes.py experiments/robust/PHASE4B_PREREGISTRATION.md | tail -1
.venv/Scripts/python.exe verify_prereg_hashes.py experiments/robust/PHASE4B_PREREGISTRATION.md > /dev/null || { echo "STOP: pre-registered hash changed"; exit 1; }
for s in 1 2 3 4 5 6 7 8 9 10; do
  test -f experiments/robust/p4b_spatial_seed$s/metrics.json && { echo "p4b_spatial seed $s exists, skip"; continue; }
  echo "== p4b_spatial seed $s $(date -Iseconds)"
  .venv/Scripts/python.exe -m src.training.train_spatial --name p4b_spatial --seed $s --steps 2500 --eval-every 25 \
      --train-conditions full=0.5,spatial=0.25,no_o2=0.125,no_opt=0.125 2>&1 | tail -1
  test -f experiments/robust/p4b_spatial_seed$s/metrics.json || { echo "STOP: seed $s did not finish"; exit 1; }
done
echo "== eval $(date -Iseconds)"
.venv/Scripts/python.exe eval_phase4b.py || { echo "STOP: eval_phase4b failed (anchor or error)"; exit 1; }
echo "== analyze $(date -Iseconds)"
.venv/Scripts/python.exe analyze_phase4b.py > logs/robust/p4b_analyze.log 2>&1 || { echo "STOP: analysis failed"; exit 1; }
echo "P4B DONE $(date -Iseconds)"
