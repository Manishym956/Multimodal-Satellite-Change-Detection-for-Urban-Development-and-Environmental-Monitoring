cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
while powershell -c "exit [int](-not (Get-Process -Id 19592 -EA SilentlyContinue))"; do sleep 10; done
echo "seed4 orphan exited; metrics present: $(test -f experiments/robust/hafner_full_seed4/metrics.json && echo yes || echo NO)"
.venv/Scripts/python.exe reeval_robust.py || { echo "REEVAL MISMATCH - stopping"; exit 1; }
for s in 5 6 7 8 9 10; do
  test -f experiments/robust/hafner_full_seed$s/metrics.json && { echo "seed $s exists, skip"; continue; }
  .venv/Scripts/python.exe -m src.training.train_robust --name hafner_full --protocol hafner --seed $s --eval-every 390 2>&1 | tail -1
done
.venv/Scripts/python.exe reeval_robust.py || { echo "REEVAL MISMATCH - stopping"; exit 1; }
echo P1V DONE
