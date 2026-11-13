#!/bin/bash
cd "$(dirname "$0")/.."
uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn --with joblib --with statsmodels --with scipy python scripts/40_das_variation.py "$@"
