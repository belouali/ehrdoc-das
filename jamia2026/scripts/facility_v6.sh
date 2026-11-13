#!/bin/bash
# Discharge to a facility with the harmonized definition (2026-11-11): reruns only that outcome in results/v6_2026-11-11/
# (eICU development and held-out loops, ceiling, MIMIC-IV for diabetes and mean DAS in both pools), then pools.
set -e
cd "$(dirname "$0")/.."
A="--spec v6 --out v6_2026-11-11"; L=logs/facility_v6
caffeinate -i -w $$ &
for H in development held_out; do for S in all with_medication; do
  bash scripts/iecv.sh loop --half $H --sample $S $A > ${L}_${H}_$S.log 2>&1
done; done
for S in all with_medication; do bash scripts/ceiling.sh --sample $S > ${L}_ceiling_$S.log 2>&1; done
bash scripts/ceiling.sh --pool-only > ${L}_ceiling_pool.log 2>&1
for X in "" "--sample with_medication"; do for MS in "" "--measure mean"; do
  bash scripts/iecv.sh mimic $A $X $MS --tasks disch_facility --jobs 6 >> ${L}_mimic.log 2>&1
done; done
bash scripts/iecv.sh pool $A > ${L}_pool.log 2>&1
echo "facility rerun finished $(date)"
