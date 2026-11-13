#!/bin/bash
# MIMIC-IV external test of the v6 locked choice (diabetes DAS, ICU readmission) and the other five outcomes (2026-11-11).
# Run after the split-half ceiling.
cd "$(dirname "$0")/.."
caffeinate -i -w $$ &
bash scripts/iecv.sh mimic --spec v6 --out v6_2026-11-11 > logs/iecv_v6_mimic.log 2>&1
echo "v6 MIMIC-IV finished $(date) exit $?"
