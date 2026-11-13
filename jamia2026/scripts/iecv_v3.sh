#!/bin/bash
# Documentation-aware site-selection rerun (38_iecv.py --spec v3, 2026-11-10): v2 plus the 12 condition flags limited to
# the first 24 h (46_flags_24h.py). Same screening rule. eICU only. Writes results/v3_2026-11-10/.
set -e
cd "$(dirname "$0")/.."
O=v3_2026-11-10; A="--spec v3 --out $O"; L=logs/iecv_v3; R=results/$O
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
echo "v3 chain finished $(date)"
