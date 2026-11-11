# Step 10: retrain E and the SAR-only U-Net on seeds 11-20, per-city evaluation, registered analysis (STEP10_PREREGISTRATION.md).
# Gates: pre-registration committed and unchanged, smoke test 0 failed, every pre-registered hash unchanged. Any failure stops.
cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
P=experiments/robust/step10
git ls-files --error-unmatch $P/STEP10_PREREGISTRATION.md > /dev/null && git diff --quiet HEAD -- $P/STEP10_PREREGISTRATION.md \
    || { echo "STOP: pre-registration not committed or changed since commit"; exit 1; }
grep -q "^n_failed 0" $P/smoke_step10.log || { echo "STOP: smoke test not passed"; exit 1; }
.venv/Scripts/python.exe verify_prereg_hashes.py $P/STEP10_PREREGISTRATION.md | tail -1
.venv/Scripts/python.exe verify_prereg_hashes.py $P/STEP10_PREREGISTRATION.md > /dev/null || { echo "STOP: pre-registered hash changed"; exit 1; }
for s in 11 12 13 14 15 16 17 18 19 20; do
  for arm in "p10_fusion_dropout unet_fusion full=0.5,no_o2=0.25,no_opt=0.25" "p10_unet_sar unet_sar full=1"; do
    set -- $arm
    test -f experiments/robust/$1_seed$s/metrics.json && { echo "$1 seed $s exists, skip"; continue; }
    echo "== $1 seed $s $(date -Iseconds)"
    .venv/Scripts/python.exe -m src.training.train_rrd --name $1 --arch $2 --sar hafner_vv --seed $s --steps 2500 --eval-every 25 \
        --train-conditions $3 > /dev/null 2>&1
    test -f experiments/robust/$1_seed$s/metrics.json || { echo "STOP: $1 seed $s did not finish"; exit 1; }
  done
done
for arm in p10_fusion_dropout p10_unet_sar; do
  echo "== city eval $arm $(date -Iseconds)"
  .venv/Scripts/python.exe $P/eval_step10_cities.py --arm $arm --seeds 11-20 || { echo "STOP: city evaluation failed"; exit 1; }
done
echo "== analyze $(date -Iseconds)"
.venv/Scripts/python.exe $P/analyze_step10.py || { echo "STOP: analysis failed"; exit 1; }
echo "P10 DONE $(date -Iseconds)"
