#!/bin/bash
# After the development runs and screening finish: held-out test (both samples), pooling, MIMIC-IV.
# Each step refuses to run if its sample's screening stopped (38_iecv.py asserts), so the chain just moves on.
cd "$(dirname "$0")/.."
until [ -f results/iecv_locked_choice_all.yaml ] && [ -f results/iecv_locked_choice_with_medication.yaml ] \
      && grep -q "screening (with_medication)" logs/iecv_pool_dev.log 2>/dev/null; do sleep 60; done
bash scripts/iecv.sh loop --half held_out --sample all > logs/iecv_held_all.log 2>&1
bash scripts/iecv.sh loop --half held_out --sample with_medication > logs/iecv_held_withmed.log 2>&1
bash scripts/iecv.sh pool > logs/iecv_pool_all.log 2>&1
bash scripts/iecv.sh mimic > logs/iecv_mimic.log 2>&1
echo "overnight chain finished $(date)" >> logs/iecv_overnight.log
