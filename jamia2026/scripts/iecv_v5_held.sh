#!/bin/bash
# Held-out test of the v5 weighted mean for ICU stay over 5 days (rule fixed before the v5 screen; author's go-ahead 2026-11-11).
# Second look at the held-out set. Writes results/v5_2026-11-11/.
set -e
cd "$(dirname "$0")/.."
O=v5_2026-11-11; A="--spec v5 --out $O"; L=logs/iecv_v5
caffeinate -i -w $$ &
for S in all with_medication; do
  bash scripts/iecv.sh loop --half held_out --sample $S $A > ${L}_held_$S.log 2>&1
done
bash scripts/iecv.sh pool $A > ${L}_pool.log 2>&1
echo "v5 held-out finished $(date)"
