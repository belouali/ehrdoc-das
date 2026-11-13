#!/bin/bash
# W_mimic_aligned (configs/ceiling_devconfirm_settings.yaml): scripts 24-26, five angles, three outcomes.
cd "$(dirname "$0")/.."
UV="uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn --with joblib --with statsmodels python"
for t in death los_gt7d hosp_los_gt14d; do
  for c in chf diabetes hypertension af combined; do
    for s in 24_w1_mimic 25_w1_pooled 26_w1_proxy; do
      echo "=== $s $c $t $(date +%H:%M)"
      $UV scripts/$s.py --config configs/external.yaml --condition $c --task $t || echo "FAILED $s $c $t"
    done
  done
done
echo "=== done $(date +%H:%M)"
