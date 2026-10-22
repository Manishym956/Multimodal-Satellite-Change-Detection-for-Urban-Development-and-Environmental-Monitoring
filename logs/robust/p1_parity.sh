cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8
for s in 1 2 3; do
  .venv/Scripts/python.exe -m src.training.train_robust --name hafner_full --protocol hafner --seed $s --eval-every 390 2>&1 | tail -1
done
for s in 1 2 3; do
  .venv/Scripts/python.exe -m src.training.train_robust --name holdout_full --seed $s 2>&1 | tail -1
done
echo P1 DONE
