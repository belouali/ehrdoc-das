#!/bin/bash
# Run one step of scripts/49_ceiling_iecv.py (S_site_selection_iecv) from mimic_external/.
cd "$(dirname "$0")/.."
exec uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
    --with joblib --with xgboost --with statsmodels --with scipy python scripts/49_ceiling_iecv.py "$@"
