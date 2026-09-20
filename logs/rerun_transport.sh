#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
echo "=== $(date '+%F %T') 09 rerun (random-k, calibration) 4 conditions in parallel"
pids=()
for c in diabetes hypertension chf af; do
  $PY -u scripts/09_xgboost_prediction.py --conditions $c > logs/09_xgboost_prediction_$c.log 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || { echo "FAILED: 09"; exit 1; }; done
echo "=== $(date '+%F %T') 17 rerun (LOHO with random-k and pooled baseline)"
$PY -u scripts/17_zero_source_sensitivity.py --skip-regression > logs/17_zero_source.log 2>&1 || { echo "FAILED: 17"; exit 1; }
echo "=== $(date '+%F %T') transport rerun complete"
