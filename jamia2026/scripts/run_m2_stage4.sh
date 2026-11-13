#!/usr/bin/env bash
# Stage 4 for M2 (paper's model + Charlson): published construct first, then DAM.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=(uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn --with joblib python)
for c in DHM DAM; do
  echo "=== $(date '+%H:%M') M2 $c"
  "${PY[@]}" scripts/05_external_models.py --config configs/external.yaml --settings "configs/stage4_settings_M2_$c.yaml" \
    > "logs/05_external_models_M2_$c.log" 2>&1 || { echo "FAILED M2 $c (see logs)"; exit 1; }
done
echo "=== $(date '+%H:%M') done"
