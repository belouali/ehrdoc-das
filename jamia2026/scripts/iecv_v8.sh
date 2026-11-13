#!/bin/bash
# 38_iecv.py --spec v8 (2026-11-11): see the script docstring. Writes results/v8_2026-11-11/.
set -e
cd "$(dirname "$0")/.."
O=v8_2026-11-11; A="--spec v8 --out $O"; L=logs/iecv_v8; R=results/$O
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
echo "v8 chain finished $(date)"
