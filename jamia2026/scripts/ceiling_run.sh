#!/bin/bash
# Split-half ceiling on the v6 design, development half, both samples (2026-11-11).
set -e
cd "$(dirname "$0")/.."
caffeinate -i -w $$ &
bash scripts/ceiling.sh --sample all > logs/ceiling_v6_all.log 2>&1
bash scripts/ceiling.sh --sample with_medication > logs/ceiling_v6_withmed.log 2>&1
bash scripts/ceiling.sh --pool-only > logs/ceiling_v6_pool.log 2>&1
echo "ceiling finished $(date)"
