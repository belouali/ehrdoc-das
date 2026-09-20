#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
echo "=== $(date '+%F %T') 13_small_hospital_experiment (4 conditions in parallel, resumed)"
pids=()
for c in diabetes hypertension chf af; do
  $PY scripts/13_small_hospital_experiment.py --conditions $c > logs/13_small_hospital_experiment_$c.log 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || { echo "FAILED: 13_small_hospital_experiment"; exit 1; }; done
echo "=== $(date '+%F %T') 17_zero_source"
$PY scripts/17_zero_source_sensitivity.py > logs/17_zero_source.log 2>&1 || { echo "FAILED: 17_zero_source"; exit 1; }
echo "=== $(date '+%F %T') 19_collect"
$PY scripts/19_collect_results.py > logs/19_collect.log 2>&1 || { echo "FAILED: 19_collect"; exit 1; }
$PY - <<'PYEOF'
import json, datetime, pathlib, yaml
art = pathlib.Path(yaml.safe_load(open("configs/paths.yaml"))["outputs"]["artifacts"])
p = art / "RUN_INFO.json"; info = json.loads(p.read_text())
info["run_finished"] = datetime.datetime.now().isoformat(timespec="seconds")
info["notes"] = "Run resumed after fixes to scripts 12 (matplotlib boxplot API) and 13 (duplicate hospitalid merge); all earlier steps unchanged."
p.write_text(json.dumps(info, indent=2)); print("run complete:", info["run_finished"])
PYEOF
