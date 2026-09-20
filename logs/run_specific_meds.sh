#!/bin/bash
# Medication-specific sensitivity: flags -> vectors -> distances -> concordance -> zero-source concordance -> RI regression
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
P=configs/paths_specific_meds.yaml
set -e
echo "=== $(date '+%F %T') 01 flags"; $PY scripts/01_build_flags.py --paths $P --phenotypes configs/phenotypes_specific_meds.yaml --conditions all
echo "=== $(date '+%F %T') 02 vectors"; $PY scripts/02_build_vectors.py --paths $P --conditions all
echo "=== $(date '+%F %T') 03 distances"; $PY scripts/03_compute_distances.py --paths $P --conditions all
echo "=== $(date '+%F %T') 04 concordance"; $PY scripts/04_concordance_heatmap.py --paths $P
echo "=== $(date '+%F %T') 17 concordance excl zero-source"; $PY scripts/17_zero_source_sensitivity.py --paths $P --skip-loho --skip-regression
echo "=== $(date '+%F %T') 08 regression (RI variance)"; $PY scripts/08_regression_models.py --paths $P --models M1 M2 M6 M8 M18 M19
echo "=== $(date '+%F %T') specific-meds run complete"
