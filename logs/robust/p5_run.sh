# Step 5: gated test evaluation (PHASE5_PREREGISTRATION.md, commits d9e29ff and 5f5be9c). Analysis and audit run separately after the anchors are checked.
cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
.venv/Scripts/python.exe verify_prereg_hashes.py experiments/robust/PHASE5_PREREGISTRATION.md | tail -1
.venv/Scripts/python.exe verify_prereg_hashes.py experiments/robust/PHASE5_PREREGISTRATION.md > /dev/null || { echo "STOP: pre-registered hash changed"; exit 1; }
echo "== eval $(date -Iseconds)"
.venv/Scripts/python.exe eval_phase5.py || { echo "STOP: eval failed (anchor or error)"; exit 1; }
echo "P5 EVAL STEP DONE $(date -Iseconds)"
