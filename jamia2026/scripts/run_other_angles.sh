#!/bin/bash
# W_mimic_other_angles (configs/ceiling_devconfirm_settings.yaml): scripts 24-26 for three more angles, both stay outcomes.
cd "$(dirname "$0")/.."
UV="uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn --with joblib --with statsmodels python"
for c in hypertension af combined; do
  for t in los_gt7d hosp_los_gt14d; do
    for s in 24_w1_mimic 25_w1_pooled 26_w1_proxy; do
      echo "=== $s $c $t $(date +%H:%M)"
      $UV scripts/$s.py --config configs/external.yaml --condition $c --task $t || echo "FAILED $s $c $t"
    done
  done
done
echo "=== done $(date +%H:%M)"
