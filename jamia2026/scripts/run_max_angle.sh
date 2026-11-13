#!/bin/bash
# S_max_angle (configs/screen_settings.yaml): eICU replay (script 37), then MIMIC-IV script 24 with the max angle.
cd "$(dirname "$0")/.."
UV="uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn --with joblib --with statsmodels python"
J=$(( $(sysctl -n hw.ncpu) / 2 ))
echo "=== 37 eICU $(date +%H:%M)"
$UV scripts/37_max_angle_eicu.py --config configs/external.yaml --jobs $J || echo "FAILED 37"
for t in death los_gt7d hosp_los_gt14d; do
  echo "=== 24 max $t $(date +%H:%M)"
  $UV scripts/24_w1_mimic.py --config configs/external.yaml --condition max --task $t || echo "FAILED 24 $t"
done
echo "=== done $(date +%H:%M)"
