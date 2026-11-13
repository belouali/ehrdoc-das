#!/bin/bash
# S_membership_comparator (configs/screen_settings.yaml): eICU then MIMIC-IV.
cd "$(dirname "$0")/.."
UV="uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn --with joblib --with statsmodels python"
J=$(( $(sysctl -n hw.ncpu) / 2 ))
$UV scripts/35_membership_eicu.py --config configs/external.yaml --jobs $J && \
$UV scripts/36_membership_mimic.py --config configs/external.yaml --jobs $J
