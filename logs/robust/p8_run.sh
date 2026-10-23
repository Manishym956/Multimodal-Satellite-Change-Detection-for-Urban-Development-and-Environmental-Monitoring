# Step 8 (RRD RQ1, RQ2, RQ5): registered evaluation of the frozen checkpoints with historical Sentinel-2 images, then
# analysis and audit (PHASE8_PREREGISTRATION.md). Gates: smoke test 0 failed, every pre-registered hash unchanged.
cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
grep -q '"n_failed": 0' experiments/robust/phase8/smoke/smoke_results.json || { echo "STOP: smoke test not passed"; exit 1; }
.venv/Scripts/python.exe verify_prereg_hashes.py experiments/robust/PHASE8_PREREGISTRATION.md | tail -1
.venv/Scripts/python.exe verify_prereg_hashes.py experiments/robust/PHASE8_PREREGISTRATION.md > /dev/null || { echo "STOP: pre-registered hash changed"; exit 1; }
echo "== eval $(date -Iseconds)"
.venv/Scripts/python.exe eval_phase8.py || { echo "STOP: eval_phase8 failed (anchor or error)"; exit 1; }
echo "== analyze $(date -Iseconds)"
.venv/Scripts/python.exe analyze_phase8.py > logs/robust/p8_analyze.log 2>&1 || { echo "STOP: analysis failed"; exit 1; }
echo "== audit $(date -Iseconds)"
.venv/Scripts/python.exe audit_p8.py > logs/robust/p8_audit.log 2>&1; tail -1 logs/robust/p8_audit.log
echo "P8 DONE $(date -Iseconds)"
