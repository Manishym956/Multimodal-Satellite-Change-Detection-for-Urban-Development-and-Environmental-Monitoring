cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8
for n in 1000 2500; do for s in 1 2 3; do
  echo "steps $n seed $s"
  .venv/Scripts/python.exe -m src.training.train_robust --name holdout_full_s$n --seed $s --steps $n --eval-every 25 2>&1 | tail -1
done; done
echo P1S DONE
