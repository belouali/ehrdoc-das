#!/bin/bash
# XGBoost (38_iecv.py --spec v7, 2026-11-11): v6 with gradient boosting in place of the logistic regression. Writes results/v7_2026-11-11/.
set -e
cd "$(dirname "$0")/.."
O=v7_2026-11-11; A="--spec v7 --out $O"; L=logs/iecv_v7; R=results/$O
caffeinate -i -w $$ &
bash scripts/iecv.sh loop --half development --sample all $A > ${L}_dev_all.log 2>&1
bash scripts/iecv.sh loop --half development --sample with_medication $A > ${L}_dev_withmed.log 2>&1
bash scripts/iecv.sh pool $A > ${L}_screen.log 2>&1
grep "screening" ${L}_screen.log || true
for S in all with_medication; do
  if grep -q "decision: lock" $R/iecv_locked_choice_$S.yaml; then
    bash scripts/iecv.sh loop --half held_out --sample $S $A > ${L}_held_$S.log 2>&1
  fi
done
bash scripts/iecv.sh pool $A > ${L}_pool.log 2>&1
cat $R/iecv_locked_choice_*.yaml
echo "v7 chain finished $(date)"
