#!/usr/bin/env bash
# Past-history pipeline, run once MIMIC-IV-Note is fully downloaded:
#   08 extract history from the full notes file (SHA-256 checked)
#   02 stage 2 with the history constructs and the noise floor
#   05 stage 4 for DHM and DHAM, each with its own locked settings file
#
#   bash scripts/run_history_pipeline.sh      (from mimic_external/; logs in logs/)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=(uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with numpy
    --with scikit-learn --with joblib python)
LOG=logs; mkdir -p "$LOG"

run() {  # run <name> <args...>
  local name=$1; shift
  echo "=== $(date '+%F %T') $name"
  "${PY[@]}" "$@" > "$LOG/$name.log" 2>&1 || { echo "FAILED: $name (see $LOG/$name.log)"; exit 1; }
}

run 08_extract_history scripts/08_extract_history.py --config configs/external.yaml
run 02_das_external_history scripts/02_das_external.py --config configs/external.yaml
run 05_external_models_DHM scripts/05_external_models.py --config configs/external.yaml --settings configs/stage4_settings_DHM.yaml
run 05_external_models_DHAM scripts/05_external_models.py --config configs/external.yaml --settings configs/stage4_settings_DHAM.yaml
echo "=== $(date '+%F %T') done"
