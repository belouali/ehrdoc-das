#!/usr/bin/env bash
# Frozen, reproducible run of the full pipeline.
#
#   bash run_all.sh                # full run (several hours: LOHO experiments dominate)
#   FAST=1 bash run_all.sh         # skip the slow transportability scripts (09, 13-16)
#
# Every artifact is regenerated from scratch under ../artifacts, and
# ../artifacts/RUN_INFO.json records when, with which scripts/configs (sha256),
# and with which package versions the run was made. Report numbers only from a
# run whose RUN_INFO.json you can cite.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
ART=$($PY -c "import yaml;print(yaml.safe_load(open('configs/paths.yaml'))['outputs']['artifacts'])")
LOG=logs; mkdir -p "$LOG"
COND=(diabetes hypertension chf af)

STARTED=${SKIP_TO:+0}   # SKIP_TO=<step name> resumes a failed run from that step
run() {  # run <name> <cmd...>
  local name=$1; shift
  if [ "${STARTED:-1}" = "0" ]; then
    if [ "$name" = "$SKIP_TO" ]; then STARTED=1; else echo "skip $name (already done)"; return; fi
  fi
  echo "=== $(date '+%F %T') $name"
  "$@" > "$LOG/$name.log" 2>&1 || { echo "FAILED: $name (see $LOG/$name.log)"; exit 1; }
}

mkdir -p "$ART"
[ -n "${SKIP_TO:-}" ] || $PY - <<'EOF'
import hashlib, json, platform, subprocess, sys, datetime, pathlib, yaml
art = pathlib.Path(yaml.safe_load(open("configs/paths.yaml"))["outputs"]["artifacts"])
files = sorted(list(pathlib.Path("scripts").glob("*.py")) + list(pathlib.Path("src").rglob("*.py")) + list(pathlib.Path("configs").glob("*.yaml")))
sha = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in files}
pk = {}
for m in ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "matplotlib", "tableone"]:
    try:
        from importlib.metadata import version; pk[m] = version(m)
    except Exception: pk[m] = None
info = {"run_started": datetime.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
        "platform": platform.platform(), "packages": pk, "file_sha256_16": sha}
(art / "RUN_INFO.json").write_text(json.dumps(info, indent=2))
print("wrote", art / "RUN_INFO.json")
EOF

# ── Phase A: sequential core pipeline ──
run 01_flags        $PY scripts/01_build_flags.py --conditions all
run 02_vectors      $PY scripts/02_build_vectors.py --conditions all
run 03_distances    $PY scripts/03_compute_distances.py --conditions all
run 04_concordance  $PY scripts/04_concordance_heatmap.py
run 05_assets       $PY scripts/05_make_amia_assets.py --all-conditions --concordance-csv "$ART/concordance/concordance_rho.csv"
run 06_thresholds   $PY scripts/06_sensitivity_thresholds.py
run 07_chars        $PY scripts/07_characteristic_scatter.py
run 08_regression   $PY scripts/08_regression_models.py
run 11_fragility    $PY scripts/11_phenotype_fragility.py
run 12_transfer     $PY scripts/12_transfer_decomposition.py
run 10_apache       $PY scripts/10_apache_sensitivity.py
run 18_flow_table1  $PY scripts/18_cohort_flow_table1.py

# ── Phase B: slow transportability experiments, one process per condition ──
if [ -z "${FAST:-}" ] && [ "${STARTED:-1}" != "0" ]; then
  for s in 09_xgboost_prediction 14_inference_transportability 15_pairwise_coefficient_similarity 16_ensemble_prediction 13_small_hospital_experiment; do
    echo "=== $(date '+%F %T') $s (4 conditions in parallel)"
    pids=()
    for c in "${COND[@]}"; do
      $PY scripts/$s.py --conditions $c > "$LOG/${s}_$c.log" 2>&1 & pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || { echo "FAILED: $s (see $LOG/${s}_*.log)"; exit 1; }; done
  done
fi

# ── Phase C: zero-source sensitivity (needs 08/09 outputs for comparison columns) ──
if [ -z "${FAST:-}" ]; then
  run 17_zero_source  $PY scripts/17_zero_source_sensitivity.py
else
  run 17_zero_source  $PY scripts/17_zero_source_sensitivity.py --skip-loho
fi

run 21_reference    $PY scripts/21_reference_sensitivity.py
run 20_capture_fig  $PY scripts/20_figure_capture_by_das.py
run 22_figure1      $PY scripts/22_figure1_composite.py --conditions diabetes hypertension chf af
run 23_table1       $PY scripts/23_table1_by_condition.py
run 24_figure3      $PY scripts/24_figure3_transport.py
run 25_size_adj     $PY scripts/25_transport_size_adjustment.py
run 26_permanova_mv $PY scripts/26_permanova_multivariable.py
run 19_collect      $PY scripts/19_collect_results.py

$PY - <<'EOF'
import json, datetime, pathlib, yaml
art = pathlib.Path(yaml.safe_load(open("configs/paths.yaml"))["outputs"]["artifacts"])
p = art / "RUN_INFO.json"; info = json.loads(p.read_text())
info["run_finished"] = datetime.datetime.now().isoformat(timespec="seconds")
p.write_text(json.dumps(info, indent=2)); print("run complete:", info["run_finished"])
EOF
