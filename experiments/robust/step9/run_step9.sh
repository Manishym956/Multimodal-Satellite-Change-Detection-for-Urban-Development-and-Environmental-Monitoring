# Step 9: train E (early-fusion U-Net, modality dropout), seeds 1-10, then the registered analysis (STEP9_PREREGISTRATION.md).
# Gates: pre-registration committed and unchanged, smoke test 0 failed, every pre-registered hash unchanged. Any failure stops.
cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
P=experiments/robust/step9
git diff --quiet HEAD -- $P/STEP9_PREREGISTRATION.md && git ls-files --error-unmatch $P/STEP9_PREREGISTRATION.md > /dev/null \
    || { echo "STOP: pre-registration not committed or changed since commit"; exit 1; }
grep -q "^n_failed 0" $P/smoke_step9.log || { echo "STOP: smoke test not passed"; exit 1; }
.venv/Scripts/python.exe verify_prereg_hashes.py $P/STEP9_PREREGISTRATION.md | tail -1
.venv/Scripts/python.exe verify_prereg_hashes.py $P/STEP9_PREREGISTRATION.md > /dev/null || { echo "STOP: pre-registered hash changed"; exit 1; }
for s in 1 2 3 4 5 6 7 8 9 10; do
  test -f experiments/robust/p9_fusion_dropout_seed$s/metrics.json && { echo "seed $s exists, skip"; continue; }
  echo "== p9_fusion_dropout seed $s $(date -Iseconds)"
  .venv/Scripts/python.exe -m src.training.train_rrd --name p9_fusion_dropout --arch unet_fusion --sar hafner_vv --seed $s \
      --steps 2500 --eval-every 25 --train-conditions full=0.5,no_o2=0.25,no_opt=0.25 > /dev/null 2>&1
  test -f experiments/robust/p9_fusion_dropout_seed$s/metrics.json || { echo "STOP: seed $s did not finish"; exit 1; }
done
echo "== analyze $(date -Iseconds)"
.venv/Scripts/python.exe $P/analyze_step9.py || { echo "STOP: analysis failed"; exit 1; }
echo "P9 DONE $(date -Iseconds)"
