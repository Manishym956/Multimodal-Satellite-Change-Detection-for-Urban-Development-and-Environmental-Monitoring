# Step 11: evaluate frozen A, B, SAR-only on real cloudy scenes, then the registered analysis (STEP11_PREREGISTRATION.md).
# Gates: pre-registration committed and unchanged, smoke test 0 failed, every pre-registered hash unchanged. Any failure stops.
cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
P=experiments/robust/realcloud
git ls-files --error-unmatch $P/STEP11_PREREGISTRATION.md > /dev/null && git diff --quiet HEAD -- $P/STEP11_PREREGISTRATION.md \
    || { echo "STOP: pre-registration not committed or changed since commit"; exit 1; }
grep -q "^n_failed 0" $P/smoke_real_cloud.log || { echo "STOP: smoke test not passed"; exit 1; }
.venv/Scripts/python.exe verify_prereg_hashes.py $P/STEP11_PREREGISTRATION.md | tail -1
.venv/Scripts/python.exe verify_prereg_hashes.py $P/STEP11_PREREGISTRATION.md > /dev/null || { echo "STOP: pre-registered hash changed"; exit 1; }
echo "== eval $(date -Iseconds)"
.venv/Scripts/python.exe $P/eval_real_cloud.py || { echo "STOP: evaluation failed"; exit 1; }
echo "== analyze $(date -Iseconds)"
.venv/Scripts/python.exe $P/analyze_real_cloud.py || { echo "STOP: analysis failed"; exit 1; }
echo "P11 DONE $(date -Iseconds)"
