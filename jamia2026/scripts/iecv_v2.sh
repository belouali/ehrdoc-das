#!/bin/bash
# Corrected site-selection rerun (38_iecv.py --spec v2, 2026-11-10): first-day covariates only, DAS-eligible pool for
# every arm, one training size for all sets, per-trim percentiles. Writes results/v2_2026-11-10/; v1 results untouched.
# Development loops, screening, held-out loops (only if screening locks), pooling, MIMIC-IV (all four outcomes).
set -e
cd "$(dirname "$0")/.."
O=v2_2026-11-10; A="--spec v2 --out $O"; L=logs/iecv_v2; R=results/$O
caffeinate -i -w $$ &
bash scripts/iecv.sh loop --half development --sample all $A > ${L}_dev_all.log 2>&1
bash scripts/iecv.sh loop --half development --sample with_medication $A > ${L}_dev_withmed.log 2>&1
bash scripts/iecv.sh pool $A > ${L}_screen.log 2>&1
grep "screening" ${L}_screen.log
if grep -q "decision: lock" $R/iecv_locked_choice_all.yaml; then
  bash scripts/iecv.sh loop --half held_out --sample all $A > ${L}_held_all.log 2>&1
  bash scripts/iecv.sh mimic $A > ${L}_mimic.log 2>&1
fi
if grep -q "decision: lock" $R/iecv_locked_choice_with_medication.yaml; then
  bash scripts/iecv.sh loop --half held_out --sample with_medication $A > ${L}_held_withmed.log 2>&1
fi
bash scripts/iecv.sh pool $A > ${L}_pool.log 2>&1
echo "v2 chain finished $(date)"
