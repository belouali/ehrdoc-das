#!/bin/bash
# Runs v8 (splines) then v9 (last-day SOFA) once the MIMIC-IV v6 run (pid 59100) ends.
cd "$(dirname "$0")/.."
while kill -0 59100 2>/dev/null; do sleep 60; done
bash scripts/iecv_v8.sh
bash scripts/iecv_v9.sh
