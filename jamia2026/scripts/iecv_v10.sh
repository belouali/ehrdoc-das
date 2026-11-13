#!/bin/bash
# v10 (2026-11-12): v6 design without facility discharge, plus a combined ranking (average of the mean-DAS and membership
# ranks). Held-out loops and MIMIC-IV (mean DAS, both pools) only; development files copied from v6. Writes results/v10_2026-11-12/.
set -e
cd "$(dirname "$0")/.."
A="--spec v10 --out v10_2026-11-12"; L=logs/iecv_v10
caffeinate -i -w $$ &
for S in all with_medication; do bash scripts/iecv.sh loop --half held_out --sample $S $A > ${L}_held_$S.log 2>&1; done
bash scripts/iecv.sh pool $A > ${L}_pool.log 2>&1
bash scripts/iecv.sh mimic $A --measure mean --jobs 6 > ${L}_mimic.log 2>&1
bash scripts/iecv.sh mimic $A --measure mean --sample with_medication --jobs 6 > ${L}_mimic_withmed.log 2>&1
echo "v10 finished $(date)"
