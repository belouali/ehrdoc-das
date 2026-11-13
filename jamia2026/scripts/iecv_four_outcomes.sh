#!/bin/bash
# Four outcomes (death, ICU stay > 5 and > 7 days, hospital stay > 14 days) through every step.
# Completed targets and outcomes are skipped, so only ICU stay > 5 days is computed for the earlier ones.
set -e
cd "$(dirname "$0")/.."
R=bash; L=logs/iecv4
$R scripts/iecv.sh loop --half development --sample all > ${L}_dev_all.log 2>&1
$R scripts/iecv.sh loop --half development --sample with_medication > ${L}_dev_withmed.log 2>&1
$R scripts/iecv.sh pool > ${L}_screen.log 2>&1
grep "screening" ${L}_screen.log
$R scripts/iecv.sh loop --half held_out --sample all > ${L}_held_all.log 2>&1
$R scripts/iecv.sh loop --half held_out --sample with_medication > ${L}_held_withmed.log 2>&1
M=$(grep best_measure results/iecv_locked_choice_all.yaml | awk '{print $2}')
if [ "$M" = "mean" ]; then T="--tasks A7_los_gt5d"; else T=""; fi
echo "locked measure $M; mimic $T"
$R scripts/iecv.sh mimic $T > ${L}_mimic.log 2>&1
$R scripts/iecv.sh pool > ${L}_pool.log 2>&1
echo "finished $(date)"
