#!/bin/bash
# Discharge-time outcomes (38_iecv.py --spec v6, 2026-11-11): v3 plus ICU readmission and discharge to a care facility,
# predicted at ICU discharge from the whole ICU stay; DAS minimum 30 records per condition. Writes results/v6_2026-11-11/.
set -e
cd "$(dirname "$0")/.."
O=v6_2026-11-11; A="--spec v6 --out $O"; L=logs/iecv_v6; R=results/$O
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
echo "v6 chain finished $(date)"
