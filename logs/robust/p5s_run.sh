# Step 5S: gated evaluation, analysis and audit (PHASE5S_PREREGISTRATION.md, commit cde5434).
cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe verify_prereg_hashes.py experiments/robust/PHASE5S_PREREGISTRATION.md | tail -1
.venv/Scripts/python.exe verify_prereg_hashes.py experiments/robust/PHASE5S_PREREGISTRATION.md > /dev/null || { echo "STOP: pre-registered hash changed"; exit 1; }
echo "== eval $(date -Iseconds)"
.venv/Scripts/python.exe eval_phase5s.py || { echo "STOP: eval failed (anchor or error)"; exit 1; }
echo "== analyze $(date -Iseconds)"
.venv/Scripts/python.exe analyze_phase5s.py > logs/robust/p5s_analyze.log 2>&1 || { echo "STOP: analysis failed"; exit 1; }
echo "== audit $(date -Iseconds)"
.venv/Scripts/python.exe audit_p5s.py | tail -5
echo "P5S DONE $(date -Iseconds)"
