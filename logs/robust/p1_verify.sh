cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8
until grep -q "P1S DONE" logs/robust/p1_steps.log; do sleep 10; done
.venv/Scripts/python.exe reeval_robust.py
for s in 4 5 6 7 8 9 10; do
  .venv/Scripts/python.exe -m src.training.train_robust --name hafner_full --protocol hafner --seed $s --eval-every 390 2>&1 | tail -1
done
echo P1V DONE
