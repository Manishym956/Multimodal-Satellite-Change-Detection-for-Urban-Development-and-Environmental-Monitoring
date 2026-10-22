cd "/c/Users/Manishym/Researchpapers/satellite‑change‑detection/projects/multimodal-satellite-change-detection"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
for s in 1 2 3 4 5 6 7 8 9 10; do
  for arm in "p2_full full=1" "p2_dropout full=0.5,no_o2=0.25,no_opt=0.25"; do
    set -- $arm
    test -f experiments/robust/$1_seed$s/metrics.json && { echo "$1 seed $s exists, skip"; continue; }
    echo "== $1 seed $s"
    .venv/Scripts/python.exe -m src.training.train_robust --name $1 --seed $s --steps 2500 --eval-every 25 --train-conditions $2 2>&1 | tail -1
  done
done
echo P2 DONE
