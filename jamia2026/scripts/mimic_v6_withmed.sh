#!/bin/bash
# MIMIC-IV v6 test with the medication-records candidate pool, locked measure then mean DAS (2026-11-11).
cd "$(dirname "$0")/.."
while kill -0 69707 2>/dev/null; do sleep 60; done
caffeinate -i -w $$ &
bash scripts/iecv.sh mimic --spec v6 --out v6_2026-11-11 --sample with_medication --jobs 5 > logs/iecv_v6_mimic_withmed.log 2>&1
bash scripts/iecv.sh mimic --spec v6 --out v6_2026-11-11 --sample with_medication --measure mean --jobs 5 > logs/iecv_v6_mimic_withmed_mean.log 2>&1
echo "with-medication MIMIC-IV finished $(date)"
