#!/usr/bin/env bash
# Stage 4 for the eICU paper's model (M1): published construct first, then DAM.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=(uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn --with joblib python)
for c in DHM DAM; do
  echo "=== $(date '+%H:%M') M1 $c"
  "${PY[@]}" scripts/05_external_models.py --config configs/external.yaml --settings "configs/stage4_settings_M1_$c.yaml" \
    > "logs/05_external_models_M1_$c.log" 2>&1 || { echo "FAILED M1 $c (see logs)"; exit 1; }
done
echo "=== $(date '+%H:%M') done"
