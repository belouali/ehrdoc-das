#!/bin/bash
# Exploratory mean-DAS variants screen (38_iecv.py --spec v5, 2026-11-11): development loops and screening only.
set -e
cd "$(dirname "$0")/.."
O=v5_2026-11-11; A="--spec v5 --out $O"; L=logs/iecv_v5; R=results/$O
caffeinate -i -w $$ &
bash scripts/iecv.sh loop --half development --sample all $A > ${L}_dev_all.log 2>&1
bash scripts/iecv.sh loop --half development --sample with_medication $A > ${L}_dev_withmed.log 2>&1
bash scripts/iecv.sh pool $A > ${L}_screen.log 2>&1
cat $R/iecv_locked_choice_*.yaml
echo "v5 screen finished $(date)"
